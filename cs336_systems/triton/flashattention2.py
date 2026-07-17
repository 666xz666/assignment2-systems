import triton
import triton.language as tl
import torch
from einops import rearrange
import math


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,  # 1 / tl.sqrt(D)
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    r"""
    Triton 实现 FlashAttention 前向分块计算内核
    采用 **在线 Softmax 分块累加** 算法，避免完整 Attention 矩阵显存占用，
    通过 KV 分块滑动、数值重缩放保证数值等价全局 Softmax。

    ## 核心数学公式
    原始注意力打分：

    $$S = \frac{QK^\top}{\sqrt{D}}$$
    
    在线 Softmax 迭代更新（标准 FA 递推式）：
    $$
    \begin{align}
    m_{new} &= \max(m_{old}, \max_{row}(S_{curr})) \\
    scale_{old} &= e^{m_{old} - m_{new}} \\
    l_{new} &= l_{old} \cdot scale_{old} + \sum(\exp(S_{curr}-m_{new})) \\
    O_{new} &= O_{old} \cdot scale_{old} + \exp(S_{curr}-m_{new})V_{curr}
    \end{align}
    $$
    最终归一化输出与 LogSumExp：

    $$O = \frac{O}{l},\quad LSE = m + \log(l)$$

    ## Args
        Q_ptr: Query 输入张量指针，shape [B, N_QUERIES, D]
        K_ptr: Key 输入张量指针，shape [B, N_KEYS, D]
        V_ptr: Value 输入张量指针，shape [B, N_KEYS, D]
        O_ptr: 注意力输出张量指针，shape [B, N_QUERIES, D]
        L_ptr: 每行 LogSumExp 输出指针，shape [B, N_QUERIES]

        stride_qb/qq/qd: Q 的 batch/query/feature 维度步长
        stride_kb/kk/kd: K 的 batch/key/feature 维度步长
        stride_vb/vk/vd: V 的 batch/key/feature 维度步长
        stride_ob/oq/od: O 的 batch/query/feature 维度步长
        stride_lb/lq: LSE 的 batch/query 维度步长

        N_QUERIES: 单样本 Query 序列长度
        N_KEYS: 单样本 Key 序列长度
        scale: 注意力打分缩放系数 $1/\sqrt{D}$

        D: 头维度，编译期常量
        Q_TILE_SIZE: 单次迭代处理的 Query 行数
        K_TILE_SIZE: 单次迭代处理的 Key 行数
        is_causal: 是否开启因果掩码（自回归单向注意力），编译期常量

    ## Returns
        无返回值，通过指针原地写入 O（注意力输出）与 L（LogSumExp）
    """
    # ====================== 1. 程序并行索引 ======================
    # 0维：切分 query 分块；1维：遍历 batch
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # ====================== 2. 加载 Query Tile（常驻 SMEM） ======================
    # Q 按列优先加载，提升访存合并效率，单次加载常驻片上内存
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    # 越界补0保证访存安全，末尾不完整tile无非法内存访问
    Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # ====================== 3. 初始化 K/V 滑动指针 ======================
    # K/V 从序列起始位置开始，逐块向后滑动遍历所有 key
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # ====================== 4. 输出结果写回指针初始化 ======================
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    # 初始化输出累加缓冲区
    O = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    # ====================== 5. LogSumExp 输出指针初始化 ======================
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # ====================== 6. 在线 Softmax 迭代变量初始化 ======================
    # m: 每行历史全局最大值，初始 -inf（首轮自动覆盖）
    # l: 每行历史归一化分母累加和，初始 0
    m = tl.full((Q_TILE_SIZE,), value=-float("inf"), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    # KV 遍历全局偏移，用于掩码全局索引计算
    k_offset = 0
    # 当前 Query Tile 的全局起始索引，用于因果掩码计算
    q_base = query_tile_index * Q_TILE_SIZE

    # ====================== 7. KV 分块滑动主循环 ======================
    for _ in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # 加载当前块 K/V，越界补0保证访存合法
        K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # 注意力分数计算: S = QK^T / sqrt(D)
        S = tl.dot(Q, tl.trans(K)) * scale

        # ---------------- 7.1 KV 边界越界掩码（解决padding数值污染） ----------------
        # shape k_indices: [K_TILE_SIZE]
        # 越界 key 列置 -inf，exp(-inf)=0，不参与 softmax 分母
        k_indices = k_offset + tl.arange(0, K_TILE_SIZE)
        key_mask = tl.where(k_indices < N_KEYS, 0.0, -float("inf"))
        # [K_T] -> [1, K_T] 广播，整列所有 query 统一 mask
        S += key_mask[None, :]

        # ---------------- 7.2 因果掩码（自回归注意力） ----------------
        # 约束：query 位置只能看到 key 位置 <= query 位置
        if is_causal:
            # q_idx: [Q_T, 1] 每行query全局索引
            q_idx = q_base + tl.arange(0, Q_TILE_SIZE)[:, None]
            # k_idx: [1, K_T] 每列key全局索引
            k_idx = k_offset + tl.arange(0, K_TILE_SIZE)[None, :]
            # 广播比较得到 [Q_T, K_T] 完整掩码矩阵
            causal_mask = tl.where(q_idx >= k_idx, 0.0, -float("inf"))
            S += causal_mask

        # ---------------- 7.3 在线 Softmax 递推更新核心逻辑 ----------------
        # 取当前块每行最大值，更新全局最大偏移
        cur_m = tl.maximum(tl.max(S, axis=-1), m)
        # 旧最大值缩放系数：用【更新前的旧m】计算
        scale_old = tl.exp(m - cur_m)

        # 滚动更新全局最大值
        m = cur_m
        # 数值稳定偏移，防止 exp 溢出
        P = tl.exp(S - cur_m[:, None])
        # P 强制 cast 为 V 的 dtype
        P_cast = P.to(V.dtype)

        # 分母、分子累加缩放更新
        l = scale_old * l + tl.sum(P, axis=-1)
        # O = scale_old[:, None] * O + tl.dot(P, V)
        O *= scale_old[:, None]
        O = tl.dot(P_cast, V, acc=O)

        # ---------------- 7.4 KV 指针滑动偏移 ----------------
        k_offset += K_TILE_SIZE
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # ====================== 8. 最终归一化 & LogSumExp 计算 ======================
    # 广播除分母，每行独立归一化
    O = O / l[:, None]
    # 标准在线 softmax LSE 公式: LSE = max_val + log(sum(exp(x-max_val)))
    L = m + tl.log(l)

    # ====================== 9. 结果写回显存 ======================
    out_dtype = O_block_ptr.type.element_ty
    tl.store(O_block_ptr, O.to(out_dtype), boundary_check=(0, 1))
    tl.store(L_block_ptr, L, boundary_check=(0,))


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # rearrange
        output_shape = Q.shape[:-2]
        Q = rearrange(Q, "... q d -> (...) q d")
        K = rearrange(K, "... k d -> (...) k d")
        V = rearrange(V, "... k d -> (...) k d")

        # 1. 读取维度 + 自定义Tile尺寸（满足>=16×16，选32可自行修改）
        bh, n_querys, d = Q.shape
        n_keys = K.shape[-2]
        Q_TILE_SIZE = 32  # Query分块大小
        K_TILE_SIZE = 32  # KV分块大小

        # 算子输出
        O = torch.empty_like(Q)
        L = torch.empty((bh, n_querys), device=Q.device, dtype=Q.dtype)

        # 启动算子
        flash_fwd_kernel[(triton.cdiv(n_querys, Q_TILE_SIZE), bh)](
            Q,
            K,
            V,
            O,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            n_querys,
            n_keys,
            1 / math.sqrt(d),
            d,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            is_causal,
        )

        # 保存计算图依赖
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.Q_TILE_SIZE, ctx.K_TILE_SIZE = Q_TILE_SIZE, K_TILE_SIZE
        ctx.n_querys = n_querys
        ctx.n_keys = n_keys
        ctx.head_dim = d

        O = O.view(*output_shape, n_querys, d)
        return O

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError

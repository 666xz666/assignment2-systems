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
    d: tl.constexpr,
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

        d: 头维度，编译期常量
        Q_TILE_SIZE: 单次迭代处理的 Query 行数
        K_TILE_SIZE: 单次迭代处理的 Key 行数
        is_causal: 是否开启因果掩码（自回归单向注意力），编译期常量

    ## Returns
        无返回值，通过指针原地写入 O（注意力输出）与 L（LogSumExp）
    """
    # ====================== 1. 程序并行索引 ======================
    # 0维：切分 query 分块；1维：遍历 batch
    batch_index = tl.program_id(0)
    query_tile_index = tl.program_id(1)

    # ====================== 2. 加载 Query Tile（常驻 SMEM） ======================
    # Q 按列优先加载，提升访存合并效率，单次加载常驻片上内存
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, d),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )
    # 越界补0保证访存安全，末尾不完整tile无非法内存访问
    Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # ====================== 3. 初始化 K/V 滑动指针 ======================
    # K/V 从序列起始位置开始，逐块向后滑动遍历所有 key
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, d),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, d),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )

    # ====================== 4. 输出结果写回指针初始化 ======================
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, d),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )
    # 初始化输出累加缓冲区
    O = tl.zeros((Q_TILE_SIZE, d), dtype=tl.float32)

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


@triton.jit
def load_kv_related_bwd(
    K_ptr,
    V_ptr,
    batch_idx,
    kv_start,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    N_KEYS,
    d: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    """
    加载当前 KV tile 的 K 和 V 矩阵块。

    Args:
        K_ptr, V_ptr      : K 和 V 的全局指针（已包含 batch 偏移）
        batch_idx         : 当前 batch 索引
        kv_start          : 当前 KV tile 的起始行索引
        stride_kb, ...    : K 的步长（batch, query, dim）
        stride_vb, ...    : V 的步长
        N_KEYS            : key 总数（用于边界检查）
        d                 : 隐藏维度（编译时常量）
        K_TILE_SIZE       : tile 行大小（编译时常量）
    Returns:
        K, V : 两个形状为 (K_TILE_SIZE, d) 的张量
    """
    # ----- K 加载 -----
    K = tl.load(
        tl.make_block_ptr(
            base=K_ptr + batch_idx * stride_kb,
            shape=(N_KEYS, d),
            strides=(stride_kk, stride_kd),
            offsets=(kv_start, 0),
            block_shape=(K_TILE_SIZE, d),
            order=(1, 0),  # 列主序，兼容行连续存储
        ),
        boundary_check=(0, 1),  # 检查行和列边界
        padding_option="zero",  # 越界填充零
    )

    # ----- V 加载 -----
    V = tl.load(
        tl.make_block_ptr(
            base=V_ptr + batch_idx * stride_vb,
            shape=(N_KEYS, d),
            strides=(stride_vk, stride_vd),
            offsets=(kv_start, 0),
            block_shape=(K_TILE_SIZE, d),
            order=(1, 0),
        ),
        boundary_check=(0, 1),
        padding_option="zero",
    )

    return K, V


@triton.jit
def load_q_related_bwd(
    Q_ptr,
    O_ptr,
    dO_ptr,
    L_ptr,
    batch_idx,
    q_start,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_lb,
    stride_lq,
    N_QUERIES,
    d: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
):
    """
    加载当前 Q tile 对应的 Q, O, dO, L 矩阵/向量块。
    返回四个张量：Q (Q_TILE_SIZE, d), O (Q_TILE_SIZE, d),
                  dO (Q_TILE_SIZE, d), L (Q_TILE_SIZE,)
    """
    # ----- Q 加载 -----
    Q = tl.load(
        tl.make_block_ptr(
            base=Q_ptr + batch_idx * stride_qb,
            shape=(N_QUERIES, d),
            strides=(stride_qq, stride_qd),
            offsets=(q_start, 0),
            block_shape=(Q_TILE_SIZE, d),
            order=(1, 0),
        ),
        boundary_check=(0, 1),
        padding_option="zero",
    )

    # ----- O 加载 -----
    O = tl.load(
        tl.make_block_ptr(
            base=O_ptr + batch_idx * stride_ob,
            shape=(N_QUERIES, d),
            strides=(stride_oq, stride_od),
            offsets=(q_start, 0),
            block_shape=(Q_TILE_SIZE, d),
            order=(1, 0),
        ),
        boundary_check=(0, 1),
        padding_option="zero",
    )

    # ----- dO 加载 -----
    dO = tl.load(
        tl.make_block_ptr(
            base=dO_ptr + batch_idx * stride_dob,
            shape=(N_QUERIES, d),
            strides=(stride_doq, stride_dod),
            offsets=(q_start, 0),
            block_shape=(Q_TILE_SIZE, d),
            order=(1, 0),
        ),
        boundary_check=(0, 1),
        padding_option="zero",
    )

    # ----- L 加载（一维向量） -----
    L = tl.load(
        tl.make_block_ptr(
            base=L_ptr + batch_idx * stride_lb,
            shape=(N_QUERIES,),  # 注意是一维
            strides=(stride_lq,),
            offsets=(q_start,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),  # 一维时 ordr 需为 (0,)
        ),
        boundary_check=(0,),  # 只检查行边界
        padding_option="zero",
    )

    return Q, O, dO, L


@triton.jit
def flash_bwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    dO_ptr,
    dQ_ptr,
    dK_ptr,
    dV_ptr,
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
    stride_dob,
    stride_doq,
    stride_dod,
    stride_dqb,
    stride_dqq,
    stride_dqd,
    stride_dkb,
    stride_dkk,
    stride_dkd,
    stride_dvb,
    stride_dvk,
    stride_dvd,
    N_QUERIES,
    N_KEYS,
    scale,
    d: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    mode: tl.constexpr,
    out_type: tl.constexpr = tl.float32,  # 新增：输出数据类型，应与 dO 一致
):
    batch_idx = tl.program_id(0)

    if mode == "q":
        q_tile_idx = tl.program_id(1)
        q_start = q_tile_idx * Q_TILE_SIZE

        # 加载 Q, O, dO, L
        Q, O, dO, L = load_q_related_bwd(
            Q_ptr,
            O_ptr,
            dO_ptr,
            L_ptr,
            batch_idx,
            q_start,
            stride_qb,
            stride_qq,
            stride_qd,
            stride_ob,
            stride_oq,
            stride_od,
            stride_dob,
            stride_doq,
            stride_dod,
            stride_lb,
            stride_lq,
            N_QUERIES,
            d,
            Q_TILE_SIZE,
        )

        D = tl.sum(O * dO, axis=-1)
        dQ = tl.zeros((Q_TILE_SIZE, d), dtype=tl.float32)
        num_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)

        for j in range(num_k_tiles):
            kv_start = j * K_TILE_SIZE

            # 加载 K, V
            K, V = load_kv_related_bwd(
                K_ptr,
                V_ptr,
                batch_idx,
                kv_start,
                stride_kb,
                stride_kk,
                stride_kd,
                stride_vb,
                stride_vk,
                stride_vd,
                N_KEYS,
                d,
                K_TILE_SIZE,
            )

            S = tl.dot(Q, tl.trans(K)) * scale
            if is_causal:
                q_idx = q_start + tl.arange(0, Q_TILE_SIZE)[:, None]
                k_idx = kv_start + tl.arange(0, K_TILE_SIZE)[None, :]
                S = tl.where(q_idx >= k_idx, S, -float("inf"))
            P = tl.exp(S - L[:, None])
            dP = tl.dot(dO, tl.trans(V))
            dS = P * (dP - D[:, None])
            dQ = tl.dot(dS * scale, K, acc=dQ)

        # 循环外写入 dQ
        tl.store(
            tl.make_block_ptr(
                base=dQ_ptr + batch_idx * stride_dqb,
                shape=(N_QUERIES, d),
                strides=(stride_dqq, stride_dqd),
                offsets=(q_start, 0),
                block_shape=(Q_TILE_SIZE, d),
                order=(1, 0),
            ),
            dQ.to(out_type),
            boundary_check=(0, 1),
        )

    elif mode == "kv":
        kv_tile_idx = tl.program_id(1)
        kv_start = kv_tile_idx * K_TILE_SIZE

        # 加载 K, V（外循环唯一一次）
        K, V = load_kv_related_bwd(
            K_ptr,
            V_ptr,
            batch_idx,
            kv_start,
            stride_kb,
            stride_kk,
            stride_kd,
            stride_vb,
            stride_vk,
            stride_vd,
            N_KEYS,
            d,
            K_TILE_SIZE,
        )

        dK = tl.zeros((K_TILE_SIZE, d), dtype=tl.float32)
        dV = tl.zeros((K_TILE_SIZE, d), dtype=tl.float32)
        num_q_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)

        for j in range(num_q_tiles):
            q_start = j * Q_TILE_SIZE

            # 加载 Q, O, dO, L（内循环每次重加载）
            Q, O, dO, L = load_q_related_bwd(
                Q_ptr,
                O_ptr,
                dO_ptr,
                L_ptr,
                batch_idx,
                q_start,
                stride_qb,
                stride_qq,
                stride_qd,
                stride_ob,
                stride_oq,
                stride_od,
                stride_dob,
                stride_doq,
                stride_dod,
                stride_lb,
                stride_lq,
                N_QUERIES,
                d,
                Q_TILE_SIZE,
            )

            D = tl.sum(O * dO, axis=-1)

            S = tl.dot(Q, tl.trans(K)) * scale
            if is_causal:
                q_idx = q_start + tl.arange(0, Q_TILE_SIZE)[:, None]
                k_idx = kv_start + tl.arange(0, K_TILE_SIZE)[None, :]
                S = tl.where(q_idx >= k_idx, S, -float("inf"))
            P = tl.exp(S - L[:, None])
            dP = tl.dot(dO, tl.trans(V))
            dS = P * (dP - D[:, None])

            dK = tl.dot(tl.trans(dS) * scale, Q, acc=dK)
            dV = tl.dot(tl.trans(P), dO, acc=dV)

        # 循环外写入 dK, dV
        tl.store(
            tl.make_block_ptr(
                base=dK_ptr + batch_idx * stride_dkb,
                shape=(N_KEYS, d),
                strides=(stride_dkk, stride_dkd),
                offsets=(kv_start, 0),
                block_shape=(K_TILE_SIZE, d),
                order=(1, 0),
            ),
            dK.to(out_type),
            boundary_check=(0, 1),
        )
        tl.store(
            tl.make_block_ptr(
                base=dV_ptr + batch_idx * stride_dvb,
                shape=(N_KEYS, d),
                strides=(stride_dvk, stride_dvd),
                offsets=(kv_start, 0),
                block_shape=(K_TILE_SIZE, d),
                order=(1, 0),
            ),
            dV.to(out_type),
            boundary_check=(0, 1),
        )


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

        ptrs = (
            Q,
            K,
            V,
            O,
            L,
        )
        strides = (
            *Q.stride(),
            *K.stride(),
            *V.stride(),
            *O.stride(),
            *L.stride(),
        )
        # 启动算子
        flash_fwd_kernel[(triton.cdiv(n_querys, Q_TILE_SIZE), bh)](
            *ptrs,
            *strides,
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
        ctx.output_shape = output_shape

        O = O.view(*output_shape, n_querys, d)
        return O

    @staticmethod
    def backward(ctx, grad_O):
        # 取出前向保存的张量与超参
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        Q_TILE_SIZE = ctx.Q_TILE_SIZE
        K_TILE_SIZE = ctx.K_TILE_SIZE
        n_querys = ctx.n_querys
        n_keys = ctx.n_keys
        d = ctx.head_dim
        output_shape = ctx.output_shape

        # 1. 把上游梯度 展平成和前向一致的 (bh, nq, d)
        dO = rearrange(grad_O, "... nq d -> (...) nq d")
        bh = dO.shape[0]

        # 2. 初始化三个梯度张量，和输入同形状
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        # 3. 构建反向网格 和前向完全一致 (q_tile_num, batch_head)
        scale = 1.0 / math.sqrt(d)

        # 4. 调用 Triton 反向kernel，按签名顺序填入所有指针+stride+常量
        ptrs = (
            Q,
            K,
            V,
            O,
            L,
            dO,
            dQ,
            dK,
            dV,
        )
        strides = (
            *Q.stride(),
            *K.stride(),
            *V.stride(),
            *O.stride(),
            *L.stride(),
            *dO.stride(),
            *dQ.stride(),
            *dK.stride(),
            *dV.stride(),
        )
        modes = ("q", "kv")
        grids = (
            (bh, triton.cdiv(n_querys, Q_TILE_SIZE)),
            (bh, triton.cdiv(n_keys, K_TILE_SIZE)),
        )
        for mode, grid in zip(modes, grids):
            flash_bwd_kernel[grid](
                *ptrs,
                *strides,
                n_querys,
                n_keys,
                scale,
                d=d,
                Q_TILE_SIZE=Q_TILE_SIZE,
                K_TILE_SIZE=K_TILE_SIZE,
                is_causal=is_causal,
                mode=mode,
                # dtype=dO.dtype,
            )

        # 5. 将展平梯度还原回原始前缀维度
        dQ = dQ.view(*output_shape, n_querys, d)
        dK = dK.view(*output_shape, n_keys, d)
        dV = dV.view(*output_shape, n_keys, d)

        # forward 入参顺序: Q, K, V, is_causal
        # is_causal 是布尔常量无梯度，返回None占位
        return dQ, dK, dV, None

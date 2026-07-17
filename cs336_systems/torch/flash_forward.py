import torch
import torch.autograd as autograd
from einops import rearrange


def flash_attn_ref_naive(Q, K, V, is_causal=False):
    """原生标准缩放点积注意力，用作结果校验"""
    d = Q.size(-1)
    S = Q @ K.transpose(-2, -1) / (d**0.5)
    if is_causal:
        mask = torch.triu(torch.ones_like(S), diagonal=1)
        S.masked_fill_(mask.bool(), -1e9)
    P = torch.softmax(S, dim=-1)
    O = P @ V
    # 计算LogSumExp L（逐行）
    L = torch.logsumexp(S, dim=-1)
    return O, L


class FlashAttention2Forward(autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        严格复刻 Algorithm 1 FlashAttention-2 前向分块递推
        支持输入任意前缀批量/多头维度：如 [B, H, Nq, D] / [B, G, H, Nq, D]
        内部自动展平为二维分块计算，输出还原输入原始shape
        """
        # ========== 新增：保存原始前缀维度，展平前缀所有维度 ==========
        prefix_shape = Q.shape[:-2]
        # ... 匹配任意多前缀维度，合并为单一batch维度
        Q_flat = rearrange(Q, "... nq d -> (...) nq d")
        K_flat = rearrange(K, "... nk d -> (...) nk d")
        V_flat = rearrange(V, "... nk d -> (...) nk d")

        # 展平后固定格式：[total_batch, n_q, d]
        total_batch, n_q, d = Q_flat.shape
        n_k = K_flat.size(-2)
        Q_TILE_SIZE = 32  # Query分块大小
        K_TILE_SIZE = 32  # KV分块大小

        # 计算分块总数 Tq, Tk
        n_q_tiles = (n_q + Q_TILE_SIZE - 1) // Q_TILE_SIZE
        n_k_tiles = (n_k + K_TILE_SIZE - 1) // K_TILE_SIZE

        # 初始化最终输出O、LogSumExp L，shape [total_batch, n_q, d]
        O_total = torch.zeros_like(Q_flat)
        L_total = torch.zeros(
            (total_batch, n_q), device=Q_flat.device, dtype=Q_flat.dtype
        )

        # 遍历每一个合并后的batch/多头样本
        for b_idx in range(total_batch):
            Q_2d = Q_flat[b_idx]  # [n_q, d]
            K_2d = K_flat[b_idx]  # [n_k, d]
            V_2d = V_flat[b_idx]  # [n_k, d]
            O_2d = torch.zeros_like(Q_2d)
            L_2d = torch.zeros(n_q, device=Q_2d.device, dtype=Q_2d.dtype)

            # 外层循环：遍历每一个Query分块 Qi (对应伪代码第4行)
            for i in range(n_q_tiles):
                # 伪代码第5行：加载Qi
                q_start = i * Q_TILE_SIZE
                q_end = min(q_start + Q_TILE_SIZE, n_q)
                Qi = Q_2d[q_start:q_end, :]
                cur_Bq = q_end - q_start

                # 伪代码第6行：初始化 Oi^{(0)}, li^{(0)}, mi^{(0)}
                O_prev = torch.zeros(cur_Bq, d, device=Q_2d.device, dtype=Q_2d.dtype)
                l_prev = torch.zeros(cur_Bq, device=Q_2d.device, dtype=Q_2d.dtype)
                m_prev = torch.full(
                    (cur_Bq,), -float("inf"), device=Q_2d.device, dtype=Q_2d.dtype
                )

                # 内层循环：遍历每一个KV分块 K(j), V(j) 伪代码第7行
                for j in range(n_k_tiles):
                    # 加载当前KV tile
                    kv_start = j * K_TILE_SIZE
                    kv_end = min(kv_start + K_TILE_SIZE, n_k)
                    Kj = K_2d[kv_start:kv_end, :]
                    Vj = V_2d[kv_start:kv_end, :]
                    cur_Bk = kv_end - kv_start

                    # 9: 计算得分矩阵 S_i^{(j)} = Qi @ Kj.T / sqrt(d)
                    S_ij = Qi @ Kj.T / (d**0.5)

                    # 10: 行max更新全局最大值 m_i^{(j)}
                    row_max_S = S_ij.max(dim=-1).values  # [cur_Bq]
                    m_curr = torch.maximum(m_prev, row_max_S)

                    # 11: 减去当前全局最大值，指数得到未归一化权重 P~
                    P_tilde = torch.exp(S_ij - m_curr.unsqueeze(-1))  # [Bq, Bk]

                    # 12: 递推更新 l_i^{(j)}
                    scale = torch.exp(m_prev - m_curr)
                    l_curr = scale * l_prev + P_tilde.sum(dim=-1)

                    # 13: 递推更新输出 O_i^{(j)}
                    O_curr = scale.unsqueeze(-1) * O_prev + P_tilde @ Vj

                    # 迭代变量滚动赋值
                    m_prev = m_curr
                    l_prev = l_curr
                    O_prev = O_curr

                # 15: 遍历所有KV块结束，归一化得到最终Qi对应的输出
                O_final = O_prev / l_prev.unsqueeze(-1)
                # 16: 计算该行LogSumExp L_i = m + log(l)
                L_final = m_prev + torch.log(l_prev)

                # 回填到单batch结果
                O_2d[q_start:q_end, :] = O_final
                L_2d[q_start:q_end] = L_final

            # 回填全局batch维度
            O_total[b_idx] = O_2d
            L_total[b_idx] = L_2d

        # 保存上下文，用于后续反向传播
        ctx.save_for_backward(Q, K, V, O_total, L_total)
        # 存入原始前缀形状，反向用来复原梯度维度
        ctx.prefix_shape = prefix_shape
        ctx.is_causal = is_causal
        ctx.Bq, ctx.Bk = Q_TILE_SIZE, K_TILE_SIZE

        # 直接view复原维度，prefix_shape是最开始Q.shape[:-2]
        O_out = O_total.view(*prefix_shape, n_q, d)
        return O_out

    @staticmethod
    def backward(ctx, grad_O):
        """题目要求先占位抛出未实现异常，后续可基于FlashAttn反向重计算逻辑补全"""
        raise NotImplementedError(
            "Backward pass not implemented for this assignment skeleton"
        )


# 包装成可直接调用的函数
flash_attention2_forward = FlashAttention2Forward.apply

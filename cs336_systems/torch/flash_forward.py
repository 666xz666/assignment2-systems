import torch
import torch.autograd as autograd


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
        Q: [Nq, d]
        K: [Nk, d]
        V: [Nk, d]
        is_causal: 本题可忽略
        return O[Nq, d], 同时ctx保存Q,K,V,O,L供反向使用
        """
        # 1. 读取维度 + 自定义Tile尺寸（满足>=16×16，选32可自行修改）
        n_q, d = Q.shape
        n_k, _ = K.shape
        Q_TILE_SIZE = 32  # Query分块大小
        K_TILE_SIZE = 32  # KV分块大小

        # 计算分块总数 Tq, Tk
        n_q_tiles = (n_q + Q_TILE_SIZE - 1) // Q_TILE_SIZE
        n_k_tiles = (n_k + K_TILE_SIZE - 1) // K_TILE_SIZE

        # 初始化最终输出O、LogSumExp L
        O_total = torch.zeros_like(Q)
        L_total = torch.zeros(n_q, device=Q.device, dtype=Q.dtype)

        # 外层循环：遍历每一个Query分块 Qi (对应伪代码第4行)
        for i in range(n_q_tiles):
            # 伪代码第5行：加载Qi
            q_start = i * Q_TILE_SIZE
            q_end = min(q_start + Q_TILE_SIZE, n_q)
            Qi = Q[q_start:q_end, :]
            cur_Bq = q_end - q_start

            # 伪代码第6行：初始化 Oi^{(0)}, li^{(0)}, mi^{(0)}
            O_prev = torch.zeros(cur_Bq, d, device=Q.device, dtype=Q.dtype)
            l_prev = torch.zeros(cur_Bq, device=Q.device, dtype=Q.dtype)
            m_prev = torch.full(
                (cur_Bq,), -float("inf"), device=Q.device, dtype=Q.dtype
            )

            # 内层循环：遍历每一个KV分块 K(j), V(j) 伪代码第7行
            for j in range(n_k_tiles):
                # 加载当前KV tile
                kv_start = j * K_TILE_SIZE
                kv_end = min(kv_start + K_TILE_SIZE, n_k)
                Kj = K[kv_start:kv_end, :]
                Vj = V[kv_start:kv_end, :]
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

            # 回填到全局结果
            O_total[q_start:q_end, :] = O_final
            L_total[q_start:q_end] = L_final

        # 保存上下文，用于后续反向传播
        ctx.save_for_backward(Q, K, V, O_total, L_total)
        ctx.is_causal = is_causal
        ctx.Bq, ctx.Bk = Q_TILE_SIZE, K_TILE_SIZE
        return O_total

    @staticmethod
    def backward(ctx, grad_O):
        """题目要求先占位抛出未实现异常，后续可基于FlashAttn反向重计算逻辑补全"""
        raise NotImplementedError(
            "Backward pass not implemented for this assignment skeleton"
        )


# 包装成可直接调用的函数
flash_attention2_forward = FlashAttention2Forward.apply


## 单元测试：验证前向结果与原生Attention完全一致
if __name__ == "__main__":
    # 构造2的幂维度输入（题目保证输入均为2幂且>=16）
    Nq, Nk, d = 64, 64, 32
    Q = torch.randn(Nq, d, device="cuda", dtype=torch.float32, requires_grad=True)
    K = torch.randn(Nk, d, device="cuda", dtype=torch.float32, requires_grad=True)
    V = torch.randn(Nk, d, device="cuda", dtype=torch.float32, requires_grad=True)

    # 自定义FlashAttention2前向
    O_flash = flash_attention2_forward(Q, K, V, False)
    # 原生标准Attention参考结果
    O_ref, L_ref = flash_attn_ref_naive(Q, K, V, False)

    # 数值误差校验
    max_err = (O_flash - O_ref).abs().max().item()
    print(f"最大绝对误差: {max_err:.2e}")
    torch.testing.assert_close(O_flash, O_ref, rtol=1e-4, atol=1e-4)
    print("✅ FlashAttention-2 纯PyTorch前向结果与原生注意力完全匹配")

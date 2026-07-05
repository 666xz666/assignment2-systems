import torch
import torch.nn as nn
import einops

from .softmax import Softmax


class ScaledDotProductAttention(nn.Module):
    r"""
    缩放点积注意力封装层，无可训练参数，仅做前向运算，兼容3维、4维多头输入格式
    计算公式 LaTeX：
    $$
    \text{Attention}(Q,K,V,\text{mask})
    = \text{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} + \text{mask} \right) V
    $$
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            Q: Query 张量，形状 (..., q_seq_len, d_k)
            K: Key 张量，形状 (..., k_seq_len, d_k)
            V: Value 张量，形状 (..., k_seq_len, d_v)
            mask: 可选注意力掩码张量，形状 (..., q_seq_len, k_seq_len)

        Returns:
            FloatTensor: 注意力加权输出，维度 (..., q_seq_len, d_v)
        """
        d_k = K.size()[-1]
        # 小写tenser才能传device和dtype,且第一个参数是传值
        # 大写Tensor第一个参数是传size
        d_k = torch.tensor(d_k, device=K.device, dtype=K.dtype)

        # 关系矩阵及掩码
        q_kt = einops.einsum(
            Q, K, "... q_seq_len d_k, ... k_seq_len d_k -> ... q_seq_len k_seq_len"
        )
        if mask is not None:
            # False → -inf(取一个极小数-1e9)，True → 0
            # -inf过softmax之后会趋近0，相当于被掩盖了
            mask_fill = torch.where(
                mask,
                torch.tensor(0.0, device=Q.device, dtype=Q.dtype),
                torch.tensor(-1e9, device=Q.device, dtype=Q.dtype),
            )
            # 广播加到注意力分数上
            q_kt = q_kt + mask_fill

        # "The attention probabilities of positions with a mask value of True should collectively sum to 1" 针对同一个 Query 位置，它对应的所有 Key 位置注意力权重总和必须等于 1
        softmax = Softmax(-1)
        softmaxed_val = softmax(q_kt / torch.sqrt(d_k))

        res = einops.einsum(
            softmaxed_val,
            V,
            "... q_seq_len k_seq_len, ... k_seq_len d_v -> ... q_seq_len d_v",
        )
        return res

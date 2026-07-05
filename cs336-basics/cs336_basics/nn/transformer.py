import torch
import torch.nn as nn

from .rmsnorm import RMSNorm
from .multihead_self_attention import MultiheadSelfAttention
from .swiglu import SwiGLUFeedForward


class TransformerBlock(nn.Module):
    r"""
    Pre-Norm 结构 Transformer 解码器块（因果多头自注意力 + SwiGLU 前馈网络）
    使用 RMSNorm 作为前置归一化，标准两层残差结构，数学流程：
    $$
    \begin{align*}
    x_1 &= x + \text{MultiHeadSelfAttention}\big(\text{RMSNorm}_1(x)\big) \\
    x_{\text{out}} &= x_1 + \text{SwiGLUFFN}\big(\text{RMSNorm}_2(x_1)\big)
    \end{align*}
    $$
    两层独立残差分支：
    1. 输入先经过 RMSNorm 归一化 → 因果多头自注意力 → 残差回加原输入
    2. 中间特征再次独立 RMSNorm 归一化 → SwiGLU 位置前馈网络 → 残差回加

    NOTE： Pre-Norm降低了梯度爆炸的风险

    Args:
        d_model: int
            Transformer 块输入/输出特征整体维度 $d_\text{model}$
        num_heads: int
            多头自注意力头数量
        d_ff: int
            SwiGLU 前馈网络内部隐藏维度
        device: torch.device
            模块所有参数与张量运算设备
        dtype: torch.dtype
            权重与前向计算张量精度类型
        theta: float | None = None
            RoPE 旋转位置编码角度超参；传入 None 则禁用 RoPE
        max_seq_len: int = 2048
            RoPE 预计算三角函数支持的最大序列长度
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device: torch.device,
        dtype: torch.dtype,
        theta: float | None = None,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        # 注意力前置归一化（独立参数）
        self.norm_attn = RMSNorm(d_model, device=device, dtype=dtype)
        # 因果多头自注意力
        self.attn = MultiheadSelfAttention(
            d_model, num_heads, device, dtype, theta, max_seq_len
        )
        # FFN前置归一化（独立参数，不能和上面共用）
        self.norm_ffn = RMSNorm(d_model, device=device, dtype=dtype)
        # SwiGLU 激活前馈网络
        self.ffn = SwiGLUFeedForward(d_model, d_ff, device, dtype)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""
        Transformer Block 前向传播

        Args:
            x: torch.Tensor
                输入特征张量，形状 $(\dots,\ \text{seq\_len},\ d_\text{model})$
            token_positions: torch.Tensor | None
                RoPE 所需位置下标张量；未启用 RoPE 时传 None 即可自动内部适配占位

        Returns:
            torch.Tensor
                块输出特征张量，形状与输入 x 完全一致 $(\dots,\ \text{seq\_len},\ d_\text{model})$
        """
        # Pre-Norm + 多头注意力 + 残差连接
        residual = x
        x_norm = self.norm_attn(x)
        x_attn = self.attn(x_norm, token_positions)
        x = residual + x_attn

        # Pre-Norm + SwiGLU前馈 + 残差连接
        residual = x
        x_norm = self.norm_ffn(x)
        x_ffn = self.ffn(x_norm)
        x = residual + x_ffn

        return x
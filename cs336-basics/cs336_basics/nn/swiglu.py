import torch
from torch import nn
import math

from cs336_basics.nn import SiLU, Linear


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        """
        Construct position-wise SwiGLU feed-forward network.
        Inner hidden dimension d_ff ≈ (8/3)*d_model, rounded up to nearest multiple of 64.

        :param d_model: int, model hidden dimension (input & output dimension)
        :param d_ff: int, output dimension of ffn
        :param device: torch.device | None = None, parameter storage device
        :param dtype: torch.dtype | None = None, parameter data type
        """
        super().__init__()
        if not d_ff:
            # 输出维度 d_ff
            raw_ff = (8 / 3) * d_model
            # 向上取整再对齐64，"a multiple of 64 to make good use of your hardware."
            d_ff = math.ceil(raw_ff / 64) * 64

        # 严格对齐公式符号 W1, W2, W3
        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)

        # 激活函数
        self.act = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Position-wise SwiGLU forward pass, operates independently on every token position.
        Input shape:  (batch_size, sequence_length, d_model)
        Output shape: (batch_size, sequence_length, d_model)

        $$
        \operatorname{SwiGLU}(x,W_1,W_2,W_3) = W_2\Big(\operatorname{SiLU}(W_1 x)\odot W_3 x\Big)
        $$
        Auxiliary definition:
        $$
        \sigma(z) = \frac{1}{1+e^{-z}},\quad \operatorname{SiLU}(z) = z\cdot\sigma(z)
        $$
        Note: Use torch.sigmoid for numerical stability as required.
        """
        y = self.W1(x)
        # ⊙ 逐元素相乘
        gate = self.act(y) * self.W3(x)
        return self.W2(gate)

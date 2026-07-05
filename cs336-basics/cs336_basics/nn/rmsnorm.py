import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module. This function should accept the following parameters:
        :param d_model: int, Hidden dimension of the model
        :param eps: float = 1e-5, Epsilon value for numerical stability
        :param device: torch.device | None = None, Device to store the parameters on
        :param dtype: torch.dtype | None = None, Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # 初始化可训练参数g_i, 一开始全设置为1
        g = torch.ones(d_model, device=device, dtype=dtype)
        self.g = nn.Parameter(g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape.
        Note: Upcast input to torch.float32 before normalization computation,
        then downcast result back to original input dtype before returning.
        """
        # 转换成32位浮点数避免溢出
        in_dtype = x.dtype
        x = x.to(torch.float32)

        x_sq = x**2
        sum_sq = x_sq.sum(-1, keepdim=True)
        """
        $$\operatorname{RMS}(a) = \sqrt{\frac{1}{d_{\text{model}}}\sum_{i=1}^{d_{\text{model}}} a_i^2 + \varepsilon}$$
        
        rms shape: (batch_size, max_len, 1)
        """
        rms = torch.sqrt(1.0 / self.d_model * sum_sq + self.eps)
        """
        $$\operatorname{RMSNorm}(a_i) = \frac{a_i}{\operatorname{RMS}(a)}\,g_i$$
        """
        result = (x / rms) * self.g
        return result.to(in_dtype)

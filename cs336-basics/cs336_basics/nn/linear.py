import torch
from torch import nn
from einops import einsum


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,  # final dimension of the input
        out_features: int,  # final dimension of the output
        device: torch.device | None = None,  # Device to store the parameters on
        dtype: torch.dtype | None = None,  # Data type of the parameters
    ):
        super(Linear, self).__init__()

        # 参数初始化
        weight = torch.empty((out_features, in_features), device=device, dtype=dtype)
        sigma = torch.sqrt(torch.tensor(2.0 / (in_features + out_features))).item()
        nn.init.trunc_normal_(weight, 0, sigma, -3 * sigma, 3 * sigma)
        self.W = nn.Parameter(weight)

    def _set_w(self, w: torch.Tensor):
        """[TEST]set W for test"""
        self.W = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... in, out in -> ... out")

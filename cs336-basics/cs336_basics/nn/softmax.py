import torch
import torch.nn as nn
import einops

from ..utils.function import softmax

class Softmax(nn.Module):
    r"""
    自定义 Softmax 网络层，对齐 PyTorch nn.Softmax 设计
    Softmax 计算公式 LaTeX：
    $$
    \text{softmax}(\boldsymbol{x})_j = \frac{\exp\left(x_j-\max(\boldsymbol{x})\right)}{\sum_{k}\exp\left(x_k-\max(\boldsymbol{x})\right)}
    $$
    初始化时固定归一化维度，前向传播仅传入特征张量；内置最大值平移防止数值溢出。
    """
    def __init__(self, dim: int):
        super().__init__()
        # 初始化阶段固定归一化维度，与原生 nn.Softmax 保持一致
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 任意形状输入特征张量

        Returns:
            归一化后同形状张量
        """
        return softmax(x, self.dim)
        
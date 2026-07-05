import torch
import einops


def softmax(x: torch.Tensor, dim: int = -1):
    r"""
    Softmax 计算公式 LaTeX：
    $$
    \text{softmax}(\boldsymbol{x})_j = \frac{\exp\left(x_j-\max(\boldsymbol{x})\right)}{\sum_{k}\exp\left(x_k-\max(\boldsymbol{x})\right)}
    $$

    Args:
        x: torch.Tensor， 任意形状输入特征张量
        dim: int = -1， 在哪一维进行
    Returns:
        归一化后同形状张量

    NOTE: exp（x）在较大值时可以变为inf（则inf / inf = NaN）
    softmax 操作对对所有输入添加任意常数 c 保持不变。
    通常，我们会从v的所有元素中减去v中最大的元素，使得新的最大元素为0。
    """
    # shape: (..., dim, ...) 形状不变
    # keepdim=False 时: (..., 1, ...)
    # .values返回张量结果
    # .indices返回下标
    max_val = x.max(dim=dim, keepdim=True).values
    y = x - max_val
    exp = torch.exp(y)
    exp_sum = exp.sum(dim=dim, keepdim=True)
    return exp / exp_sum

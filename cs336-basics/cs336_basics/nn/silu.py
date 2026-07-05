import torch
from torch import nn


class SiLU(nn.Module):
    def __init__(self):
        r"""
        Sigmoid Linear Unit (SiLU) activation function.
        Formula:
        $$
        \operatorname{SiLU}(x) = x \cdot \sigma(x),\quad \sigma(x)=\frac{1}{1+e^{-x}}
        $$
        Uses torch.sigmoid internally for numerical stability.
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Element-wise SiLU activation.
        $$
        \text{SiLU}(x) = x \cdot \sigma(x),\quad \sigma(x)=\frac{1}{1+e^{-x}}
        $$
        :param x: arbitrary-shape input tensor
        :return: same-shape activated tensor
        
        这里直接使用torch的实现保证数值稳定性

        ## 1. 自定义反向公式，减少运算步数
        SiLU 导数公式：
        $$
        \text{SiLU}'(x) = \sigma(x)\cdot\big(1 + x\cdot(1-\sigma(x))\big)
        $$
        PyTorch 内置算子优势：为 $\sigma(x)$ 手写反向核，直接复用前向计算结果 $\sigma(x)$ 求解梯度，无需重复计算指数 $\exp$，运算量更低、浮点误差更小。

        ## 2. 梯度裁剪保护，杜绝极端梯度爆炸 / NaN
        当 $|x|$ 极大时，$\sigma'(x)\approx0$（梯度消失区间），原生算子内置边界数值钳位逻辑；手写实现遇到极端输入容易出现梯度断崖、产生 NaN。

        ## 3. CUDA 向量化算子融合优化
        GPU 下 `torch.sigmoid` 为单融合内核（Kernel fusion）批量运算；
        手写 $\exp+\text{add}+\text{div}$ 会拆分为三次独立算子调用，中间张量频繁读写显存，速度更慢，额外引入浮点舍入噪声，梯度累积偏差更大。
        """
        sigma = torch.sigmoid(x)
        return x * sigma

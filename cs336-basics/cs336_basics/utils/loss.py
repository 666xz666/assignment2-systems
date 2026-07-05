import torch
import einops
from torch import Tensor
from jaxtyping import Float, Int


def cross_entropy(
    logits: Float[Tensor, "*batch vocab_size"],
    target: Int[Tensor, "*batch"]
) -> Float[Tensor, ""]:
    r"""
    根据模型原始输出 logits $o_i$ 与目标 token 索引 $x_{i+1}$，计算平均交叉熵（负对数似然）损失。
    采用每行减去最大值的 log-sum-exp 方案保证数值稳定性；
    对 $\log(\exp/\sum\exp)$ 做代数化简，抵消嵌套的对数与指数运算；
    自动兼容任意前置批量维度，最终返回全部元素的平均损失。

    单个位置损失定义：
    $$
    \ell_i = -\log\big(\text{softmax}(o_i)\big)\big[x_{i+1}\big]
    $$
    其中 softmax 概率表达式：
    $$
    \text{softmax}(o_i)[x_{i+1}]
    = \frac{\exp\big(o_i[x_{i+1}]\big)}{\sum_{a=1}^{\text{vocab\_size}} \exp\big(o_i[a]\big)}
    $$
    代入化简，并对每行 logits 减去该行最大值做数值稳定后的损失公式：
    $$
    \ell_i
    = -\bigg( \big(o_i[x_{i+1}] - \max(o_i)\big)
    - \log\bigg(\sum_{a=1}^{\text{vocab\_size}} \exp\big(o_i[a] - \max(o_i)\big)\bigg) \bigg)
    $$
    最终总损失为所有批量位置损失的平均值：
    $$
    \mathcal{L} = \frac{1}{N}\sum_{i}\ell_i
    $$
    $N$ 为所有前置批量维度展平后的总元素数量。

    NOTE: 这里不能先softmax再log的原因，softmax结果取值范围(0, 1)
    结果趋近0时取对数会出现-inf，所以要用化简后的LogSumExp。此时log里内容是exp求和，且求和项中至少有一项（取最大项时）为1，所以求和结果肯定大于1，对数项不会出现-inf；因为只有一项等于1，其他介于0和1之间，所以求和项小于vocab_size，故可推导损失函数有上下界，不会梯度爆炸
    NOTE：为什么要取对数：为了放大惩罚

    Args:
        logits: Float[Tensor, "*batch vocab_size"]
            模型未归一化原始输出得分 $o_i$，前面可以是任意批量维度，最后一维为词表大小
        target: Int[Tensor, "*batch"]
            真实下一个 token 的索引 $x_{i+1}$，取值范围 $0 \le target < vocab\_size$，形状与 logits 的所有前置批量维度完全匹配

    Returns:
        Float[Tensor, ""]
            标量张量，所有批量位置交叉熵损失的平均值
    """
    # 每行最大值移位，和softmax稳定逻辑一致
    max_vals = logits.max(dim=-1, keepdim=True).values
    shifted_logits = logits - max_vals

    # einops 扩维 gather 取出正确类别的移位logit
    target_exp = einops.rearrange(target, "... -> ... 1")
    correct_shifted = torch.gather(shifted_logits, dim=-1, index=target_exp)
    correct_shifted = einops.rearrange(correct_shifted, "... 1 -> ...")

    # log(sum(exp(shifted_logits)))
    sum_exp = torch.exp(shifted_logits).sum(dim=-1)
    log_sum_exp = torch.log(sum_exp)

    # 化简后无log(prob)，彻底规避log(0)无穷
    per_sample_loss = -(correct_shifted - log_sum_exp)

    # 取平均值
    return torch.mean(per_sample_loss)
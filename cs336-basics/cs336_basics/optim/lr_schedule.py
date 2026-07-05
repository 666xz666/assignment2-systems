import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


def get_lr_cosine_schedule(
    t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int
) -> float:
    r"""
    实现带warmup的余弦退火学习率调度策略，分三个阶段计算迭代t时刻的学习率$\alpha_t$

    调度规则分三段：
    1. **预热阶段(Warm-up)**：$t < T_w$

       $$\alpha_t = \frac{t}{T_w}\alpha_{\max}$$

       学习率从0线性增长到最大学习率$\alpha_{\max}$

    2. **余弦退火阶段(Cosine annealing)**：$T_w \le t \le T_c$
       $$\alpha_t = \alpha_{\min} + \frac{1}{2}\left(1 + \cos\left(\frac{t-T_w}{T_c-T_w}\pi\right)\right)(\alpha_{\max}-\alpha_{\min})$$

       以余弦曲线形式从$\alpha_{\max}$平滑下降到最小学习率$\alpha_{\min}$

    3. **退火后稳定阶段(Post-annealing)**：$t > T_c$
       $$\alpha_t = \alpha_{\min}$$
       学习率固定保持最小值不再变化

    Args:
        t: int, 当前迭代步数
        alpha_max: float, 学习率峰值$\alpha_{\max}$
        alpha_min: float, 学习率最终下限$\alpha_{\min}$
        T_w: int, 预热阶段总迭代步数
        T_c: int, 余弦退火结束的最终迭代步数

    Returns:
        float: 当前迭代t对应的学习率$\alpha_t$
    """
    if t < T_w:
        # 线性预热阶段
        return (t / T_w) * alpha_max
    elif T_w <= t <= T_c:
        # 余弦退火阶段
        cos_term = math.cos(((t - T_w) / (T_c - T_w)) * math.pi)
        return alpha_min + 0.5 * (1 + cos_term) * (alpha_max - alpha_min)
    else:
        # 退火后恒定最小学习率
        return alpha_min


class CosineAnnealingWarmupLR(_LRScheduler):
    r"""
    PyTorch 调度器子类：带线性预热的余弦退火学习率调度器
    内部复用 get_lr_cosine_schedule 完成学习率计算，兼容自定义/原生 Optimizer

    三段学习率公式同 get_lr_cosine_schedule:
    $$
    \alpha_t=
    \begin{cases}
    \displaystyle\frac{t}{T_w}\alpha_{\max}, & t<T_w \\[6pt]
    \displaystyle\alpha_{\min}+\frac12\left(1+\cos\left(\frac{t-T_w}{T_c-T_w}\pi\right)\right)(\alpha_{\max}-\alpha_{\min}),
    & T_w\le t\le T_c \\[6pt]
    \alpha_{\min}, & t>T_c
    \end{cases}
    $$
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        alpha_max: float,
        alpha_min: float,
        T_w: int,
        T_c: int,
        last_epoch: int = -1,
    ):
        r"""
        Args:
        optimizer: 绑定的优化器实例（继承 torch.optim.Optimizer）
        alpha_max: 学习率峰值 $\alpha_{\max}$
        alpha_min: 学习率下限 $\alpha_{\min}$
        T_w: 预热总步数
        T_c: 余弦退火终止步数
        last_epoch: 初始迭代计数，默认 -1（调度器标准初始化配置）
        """
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.T_w = T_w
        self.T_c = T_c
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        # 取当前迭代步数 t
        current_t = self.last_epoch
        lr = get_lr_cosine_schedule(
            t=current_t,
            alpha_max=self.alpha_max,
            alpha_min=self.alpha_min,
            T_w=self.T_w,
            T_c=self.T_c,
        )
        # 适配多 param_group，每组赋予相同计算出的学习率
        return [lr for _ in self.base_lrs]

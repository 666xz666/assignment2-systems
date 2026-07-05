import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    Custom implementation of AdamW optimizer, subclass of torch.optim.Optimizer
    Strictly follows the AdamW pseudocode algorithm from course material.

    NOTE: 一个可行的理解路线：
    - https://www.bilibili.com/video/BV1NZ421s75D： 理解指数加权平均的概念，时间上越近影响权重越大
    - https://www.bilibili.com/video/BV1X5EFzmEAW： SGD->动量法->RMSProp->Adam->AdamW

    NOTE: 适配了混合精度的思想
    Solution: mixed precision training  [Micikevicius+ 2017]:
    - Use bf16 for parameters, activations, and gradients
    - Use fp32 for optimizer states
    """

    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
    ):
        r"""
        Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate $\alpha$
        betas (tuple[float, float]): $(\beta_1, \beta_2)$, coefficients for first/second moment exponential moving average
        eps (float): $\varepsilon$, numerical stability term added to denominator to avoid division by zero
        weight_decay (float): $\lambda$, weight decay coefficient for AdamW decoupled regularization
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step (one full iteration of AdamW update rule).

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        Returns:
            Optional loss value evaluated by closure if provided
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Iterate over all parameter groups maintained by base Optimizer
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lam = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.to(torch.float32)
                orig_dtype = p.data.dtype
                theta = p.data.to(torch.float32)

                # Retrieve or initialize per-parameter state (m, v, iteration counter t)
                state = self.state[p]

                if len(state) == 0:
                    # Initialize first moment $m$, second moment $v$, step counter $t$
                    # 强制动量存储在 float32，适配 bf16/fp16 混合精度
                    state["m"] = torch.zeros_like(p.data, dtype=torch.float32)
                    state["v"] = torch.zeros_like(p.data, dtype=torch.float32)
                    state["t"] = 0

                m = state["m"]
                v = state["v"]
                t = state["t"]
                # NOTE: t从1开始
                t += 1

                # --------------------------
                # AdamW update steps, strictly match pseudocode formula one by one
                # 1. Compute bias-corrected learning rate:

                # $$\alpha_t \leftarrow \alpha \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}$$

                # 这一步其实化简后得来的，相当于分别对第3步的m和第4步的v除以一个对应项
                # 在步数比较小时起到了加速的作用。步数越大，这项作用越小。
                # numerator = torch.sqrt(
                #     torch.tensor(
                #         1.0 - beta2**t, device=theta.device, dtype=torch.float32
                #     )
                # )
                numerator = (1.0 - beta2**t) ** 0.5  # 直接标量运算不用考虑dtype
                denominator = 1.0 - beta1**t
                alpha_t = lr * numerator / denominator

                # 2. Decoupled weight decay step:
                # $$\theta \leftarrow \theta - \alpha\lambda \theta$$
                # 直接的权重衰减，没有参与梯度计算。理论认为权重越小泛化性越强
                theta = theta - lr * lam * theta

                # 3. Update first moment estimate:
                # $$m \leftarrow \beta_1 m + (1-\beta_1)g$$
                # 源自动量法，保留了上一次梯度的影响，避免陷入局部最优
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 4. Update second moment estimate:
                # $$v \leftarrow \beta_2 v + (1-\beta_2)g^2$$
                # 源自RMS Prop，思想是除以g的长度单位（开根号后），缩小更新步长，抑制震荡
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 5. Parameter update with bias-corrected moments:
                # $$\theta \leftarrow \theta - \alpha_t \frac{m}{\sqrt{v+\varepsilon}}$$

                # eps防止零除
                theta = theta - alpha_t * m / (torch.sqrt(v + eps))

                # Write updated data back to parameter
                p.data = theta.to(orig_dtype)

                # Update iteration counter stored in state
                state["t"] = t

        return loss

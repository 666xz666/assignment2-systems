from typing import Iterable
import torch


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6
) -> None:
    r"""
    按总$\ell_2$范数执行全局梯度裁剪，就地修改梯度。

    ### Algorithm Definition
    Let the concatenated global gradient vector of all trainable parameters be $\boldsymbol{g}$.
    Compute its global $\ell_2$ norm:
    $$
    \|\boldsymbol{g}\|_2 = \sqrt{\sum_{i} g_i^2}
    $$
    Scaling rule for all gradient tensors:
    $$
    \text{scale} =
    \begin{cases}
    1, & \|\boldsymbol{g}\|_2 \le \text{max\_l2\_norm} \\[4pt]
    \displaystyle \frac{\text{max\_l2\_norm}}{\|\boldsymbol{g}\|_2 + \varepsilon},
    & \|\boldsymbol{g}\|_2 > \text{max\_l2\_norm}
    \end{cases}
    $$
    Every gradient element is updated in-place:
    $$
    \boldsymbol{g} \leftarrow \text{scale} \cdot \boldsymbol{g}
    $$
    $\varepsilon=10^{-6}$ is added to denominator for numerical stability against division by zero.

    Args:
        parameters: Iterable of model parameters (`torch.nn.Parameter`), collect their existing gradients
        max_l2_norm: Upper bound $M$ for allowed global gradient $\ell_2$ norm
        eps: Small offset for numerical stability in denominator, default $10^{-6}$

    Returns:
        None: Gradients are modified in-place on parameter `.grad` attributes
    """
    grads = []
    for p in parameters:
        if p.grad is not None:
            grads.append(p.grad)
    if not grads:
        return

    # Flatten all gradients into one single 1D vector
    flat = torch.cat([g.flatten() for g in grads])
    # Compute global L2 norm of concatenated gradient
    global_norm = torch.norm(flat, p=2)

    if global_norm > max_l2_norm:
        scale = max_l2_norm / (global_norm + eps)
        # In-place scale every gradient tensor
        for g in grads:
            g.mul_(scale)

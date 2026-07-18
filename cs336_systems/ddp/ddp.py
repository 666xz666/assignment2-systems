import torch
from torch import nn
import torch.distributed as dist


class DDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """在所有 rank 上对模型参数的梯度进行 all-reduce 平均。"""
        # 使用 no_grad 避免干扰 autograd 图（尽管梯度张量是叶子节点，但习惯上安全操作）
        with torch.no_grad():
            for param in self.module.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

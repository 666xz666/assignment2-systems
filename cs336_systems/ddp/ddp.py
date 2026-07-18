import torch
from torch import nn
import torch.distributed as dist


class DDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._handles = []
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param, src=0)
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(self._grad_hook)

    def _grad_hook(self, param: torch.Tensor):
        """
        梯度就绪后的钩子函数。
        发起异步 all-reduce（平均操作），并将返回的句柄保存。
        """
        if param.grad is None:
            return
        # 异步 all-reduce，屏蔽通信延迟
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
        self._handles.append(handle)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """
        等待所有异步 all-reduce 操作完成。
        必须在 optimizer.step() 之前调用，以确保梯度已同步。
        """
        for handle in self._handles:
            handle.wait()
        self._handles.clear()

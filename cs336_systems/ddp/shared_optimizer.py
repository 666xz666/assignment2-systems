import torch
import torch.distributed as dist
from typing import Any, Dict, Type


class ShardedOptimizer(torch.optim.Optimizer):
    """
    优化器状态分片包装器。
    每个 rank 仅维护自己负责的那部分参数的优化器状态，并在每次 step() 后通过 broadcast
    同步所有参数，使得所有 rank 持有完整的模型参数。
    """

    def __init__(
        self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any
    ):
        """
        Args：
        params: 一个可迭代对象，包含要优化的参数（或参数组字典的列表）。
        optimizer_cls: 要包装的优化器类，例如 torch.optim.AdamW。
        **kwargs: 传给 optimizer_cls 构造函数的其他关键字参数（如 lr, weight_decay 等）。
        """
        # 获取分布式信息
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # 全局参数映射
        self._param_to_global_id: Dict[int, int] = {}
        self._global_id_to_param: Dict[int, torch.Tensor] = {}
        self._next_global_id: int = 0

        # 将关键字参数作为父类的 defaults（用于填充参数组缺省键）
        defaults = kwargs.copy()
        super().__init__(params, defaults)

        # 此时父类的 __init__ 已经调用了 add_param_group，且 self.param_groups 只包含当前 rank 的参数
        # 创建内部优化器，只维护本 rank 参数的优化器状态
        self.optim = optimizer_cls(self.param_groups, **kwargs)

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """
        添加一组参数，并对这些参数进行分片。
        只有属于当前 rank 的参数会被保留在 self.param_groups 和内部优化器中。
        """
        params = param_group["params"]
        sharded_params = []

        for param in params:
            param_id = id(param)
            if param_id not in self._param_to_global_id:
                # 为新参数分配全局 ID
                global_id = self._next_global_id
                self._next_global_id += 1
                self._param_to_global_id[param_id] = global_id
                self._global_id_to_param[global_id] = param
            else:
                # 一般不会出现重复添加
                global_id = self._param_to_global_id[param_id]

            # 只保留属于当前 rank 的参数
            if global_id % self.world_size == self.rank:
                sharded_params.append(param)

        # 构建分片后的参数组字典
        sharded_group = {k: v for k, v in param_group.items() if k != "params"}
        sharded_group["params"] = sharded_params

        # 添加到父类的 param_groups（优化器框架需要）
        super().add_param_group(sharded_group)

        # 同时添加到内部优化器
        if hasattr(self, "optim"):
            self.optim.add_param_group(sharded_group)

    def step(self, closure=None, **kwargs):
        """
        执行一步优化：
        1. 更新当前 rank 维护的参数（使用内部优化器）。
        2. 通过 broadcast 将每个参数的最新值从其 owner rank 传播到所有 rank。
        """
        # 更新分片参数
        loss = self.optim.step(closure, **kwargs)

        # 同步所有参数：每个参数由其 owner rank 广播新值
        with torch.no_grad():
            for global_id, param in self._global_id_to_param.items():
                owner_rank = global_id % self.world_size
                dist.broadcast(param.data, src=owner_rank)

        return loss

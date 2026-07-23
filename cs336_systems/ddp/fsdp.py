from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from cs336_basics.nn import Embedding as BaseEmbedding
from cs336_basics.nn import Linear as BaseLinear


def _get_weight(module: nn.Module) -> torch.Tensor:
    """Read a weight tensor from either course or PyTorch modules."""
    if hasattr(module, "W"):
        return module.W.detach()
    if hasattr(module, "weight"):
        return module.weight.detach()
    raise AttributeError(f"{type(module).__name__} has no weight or W attribute")


def _get_weight_name(module: nn.Module) -> str:
    """Return the original registered parameter name for the module weight."""
    if hasattr(module, "W"):
        return "W"
    if hasattr(module, "weight"):
        return "weight"
    raise AttributeError(f"{type(module).__name__} has no weight or W attribute")


def _shard_range(total_rows: int, rank: int, world_size: int) -> tuple[int, int]:
    """Return [start, end) for an approximately even row-wise shard."""
    base, remainder = divmod(total_rows, world_size)
    start = rank * base + min(rank, remainder)
    size = base + (1 if rank < remainder else 0)
    return start, start + size


def _pad_rows(tensor: torch.Tensor, target_rows: int) -> torch.Tensor:
    """Pad only dimension zero to target_rows."""
    pad_rows = target_rows - tensor.shape[0]
    if pad_rows < 0:
        raise ValueError("target_rows must be at least tensor.shape[0]")
    if pad_rows == 0:
        return tensor

    output = torch.zeros(
        (target_rows, *tensor.shape[1:]),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    output[: tensor.shape[0]].copy_(tensor)
    return output


class GatherWeights(torch.autograd.Function):
    """All-gather parameter shards in forward and reduce gradients in backward."""

    @staticmethod
    def forward(
        ctx,
        shard: torch.Tensor,
        shard_sizes: list[int],
        max_shard_size: int,
        rank: int,
    ) -> torch.Tensor:
        world_size = len(shard_sizes)
        ctx.shard_sizes = shard_sizes
        ctx.rank = rank

        padded_shard = _pad_rows(shard, max_shard_size)
        gathered_padded = torch.empty(
            (world_size * max_shard_size, *shard.shape[1:]),
            device=shard.device,
            dtype=shard.dtype,
        )

        work = dist.all_gather_into_tensor(
            gathered_padded,
            padded_shard,
            async_op=True,
        )
        work.wait()

        parts = []
        for source_rank, size in enumerate(shard_sizes):
            block_start = source_rank * max_shard_size
            parts.append(gathered_padded[block_start : block_start + size])
        return torch.cat(parts, dim=0)

    @staticmethod
    def backward(ctx, grad_full: torch.Tensor):
        shard_sizes = ctx.shard_sizes
        rank = ctx.rank
        world_size = len(shard_sizes)

        reduced_grad = grad_full.contiguous()
        work = dist.all_reduce(reduced_grad, op=dist.ReduceOp.SUM, async_op=True)
        work.wait()
        reduced_grad.div_(world_size)

        start = sum(shard_sizes[:rank])
        end = start + shard_sizes[rank]
        grad_shard = reduced_grad[start:end].contiguous()

        return grad_shard, None, None, None


class _FSDPWeightModule(nn.Module):
    """Shared row-sharding and full-weight reconstruction implementation."""

    def __init__(self, weight: torch.Tensor, rank: int, world_size: int):
        super().__init__()
        self.rank = rank
        self.world_size = world_size

        start, end = _shard_range(weight.shape[0], rank, world_size)
        shard = weight[start:end].clone().contiguous()
        self.weight_shard = nn.Parameter(shard)
        self.weight_shard._fsdp_sharded = True

        local_size = torch.tensor(
            [shard.shape[0]],
            device=weight.device,
            dtype=torch.long,
        )
        all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size)

        sizes = torch.cat(all_sizes).cpu()
        self.register_buffer("_sizes", sizes, persistent=False)
        self._max_size = int(sizes.max().item())

    def _full_weight(self) -> torch.Tensor:
        return GatherWeights.apply(
            self.weight_shard,
            self._sizes.tolist(),
            self._max_size,
            self.rank,
        )

    @torch.no_grad()
    def gather_weight(self) -> torch.Tensor:
        """Synchronously reconstruct this module's complete weight."""
        shard = self.weight_shard.detach()
        padded_shard = _pad_rows(shard, self._max_size)
        gathered_padded = torch.empty(
            (self.world_size * self._max_size, *shard.shape[1:]),
            device=shard.device,
            dtype=shard.dtype,
        )
        dist.all_gather_into_tensor(gathered_padded, padded_shard)

        parts = []
        for source_rank, size in enumerate(self._sizes.tolist()):
            start = source_rank * self._max_size
            parts.append(gathered_padded[start : start + size])
        return torch.cat(parts, dim=0)


class FSDPLinear(_FSDPWeightModule):
    def __init__(
        self,
        linear: nn.Module,
        rank: int,
        world_size: int,
        compute_dtype: torch.dtype | None = None,
    ):
        self._original_weight_name = _get_weight_name(linear)
        weight = _get_weight(linear)
        super().__init__(weight, rank, world_size)

        self.compute_dtype = compute_dtype
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]

        original_bias = getattr(linear, "bias", None)
        if original_bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(original_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._full_weight()
        bias = self.bias

        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)
            weight = weight.to(self.compute_dtype)
            if bias is not None:
                bias = bias.to(self.compute_dtype)

        return F.linear(x, weight, bias)


class FSDPEmbedding(_FSDPWeightModule):
    def __init__(
        self,
        embedding: nn.Module,
        rank: int,
        world_size: int,
        compute_dtype: torch.dtype | None = None,
    ):
        self._original_weight_name = _get_weight_name(embedding)
        weight = _get_weight(embedding)
        super().__init__(weight, rank, world_size)

        self.compute_dtype = compute_dtype
        self.num_embeddings = weight.shape[0]
        self.embedding_dim = weight.shape[1]
        self.padding_idx = getattr(embedding, "padding_idx", None)
        self.max_norm = getattr(embedding, "max_norm", None)
        self.norm_type = getattr(embedding, "norm_type", 2.0)
        self.scale_grad_by_freq = getattr(embedding, "scale_grad_by_freq", False)
        self.sparse = getattr(embedding, "sparse", False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._full_weight()
        if self.compute_dtype is not None:
            weight = weight.to(self.compute_dtype)

        return F.embedding(
            x,
            weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )


class FSDP(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before FSDP")

        self.module = module
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._gradient_work_handles: list[object] = []

        self._replace_layers(self.module)
        self._register_unsharded_gradient_hooks()

    def _replace_layers(self, module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, (BaseLinear, nn.Linear)):
                setattr(
                    module,
                    name,
                    FSDPLinear(
                        child,
                        self.rank,
                        self.world_size,
                        self.compute_dtype,
                    ),
                )
            elif isinstance(child, (BaseEmbedding, nn.Embedding)):
                setattr(
                    module,
                    name,
                    FSDPEmbedding(
                        child,
                        self.rank,
                        self.world_size,
                        self.compute_dtype,
                    ),
                )
            else:
                self._replace_layers(child)

    def _register_unsharded_gradient_hooks(self) -> None:
        for parameter in self.module.parameters():
            if not parameter.requires_grad or getattr(
                parameter, "_fsdp_sharded", False
            ):
                continue
            parameter.register_hook(self._make_average_gradient_hook())

    def _make_average_gradient_hook(self):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if grad is None or self.world_size == 1:
                return grad

            synced_grad = grad.contiguous()
            dist.all_reduce(synced_grad, op=dist.ReduceOp.AVG)
            return synced_grad

        return hook

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        """Synchronize gradients before the optimizer step."""
        self._gradient_work_handles.clear()

    @torch.no_grad()
    def gather_full_params(self) -> dict[str, torch.Tensor]:
        """Return full parameter tensors using the original module parameter names."""
        full_params: dict[str, torch.Tensor] = {}

        for module_name, child in self.module.named_modules():
            if isinstance(child, (FSDPLinear, FSDPEmbedding)):
                weight_name = child._original_weight_name
                name = f"{module_name}.{weight_name}" if module_name else weight_name
                full_params[name] = child.gather_weight()

                if isinstance(child, FSDPLinear) and child.bias is not None:
                    bias_name = f"{module_name}.bias" if module_name else "bias"
                    full_params[bias_name] = child.bias.detach().clone()

        for name, parameter in self.module.named_parameters():
            if name.endswith(".weight_shard") or name in full_params:
                continue
            full_params[name] = parameter.detach().clone()

        return full_params

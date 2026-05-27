from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import os
from typing import TYPE_CHECKING
from typing import Literal

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class DistributedContext:
    backend: Literal["none", "fsdp"]
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_distributed(self) -> bool:
        return self.backend != "none"

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(args) -> DistributedContext:
    import torch
    import torch.distributed as dist

    if args.distributed_backend == "none":
        return DistributedContext(backend="none")

    if args.distributed_backend != "fsdp":
        raise ValueError(f"Unsupported distributed backend: {args.distributed_backend}")
    if not torch.cuda.is_available():
        raise RuntimeError("FSDP training requires CUDA.")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        raise RuntimeError(
            "distributed_backend=fsdp requires torchrun. "
            "Example: torchrun --standalone --nproc_per_node=8 scripts/train_robotwin_vla.py ..."
        )
    if args.dist_timeout_minutes <= 0:
        raise ValueError(
            "dist_timeout_minutes must be positive, "
            f"but got {args.dist_timeout_minutes}."
        )

    if not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=args.dist_timeout_minutes),
        )

    return DistributedContext(
        backend="fsdp",
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )


def reduce_mean(value: "torch.Tensor", dist_ctx: DistributedContext) -> "torch.Tensor":
    import torch.distributed as dist

    if not dist_ctx.is_distributed:
        return value.detach()
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist_ctx.world_size
    return reduced


def destroy_distributed(dist_ctx: DistributedContext, *, success: bool) -> None:
    import torch.distributed as dist

    if dist_ctx.is_distributed and dist.is_initialized():
        if success:
            dist.barrier()
        dist.destroy_process_group()

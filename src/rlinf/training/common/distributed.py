from __future__ import annotations

"""本地训练 / FSDP 训练共用的分布式上下文工具。"""

from dataclasses import dataclass
from datetime import timedelta
import os
from typing import TYPE_CHECKING
from typing import Literal

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class DistributedContext:
    """当前训练进程的分布式状态快照。"""

    backend: Literal["none", "fsdp"]
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_distributed(self) -> bool:
        """是否处在 torchrun / FSDP 这类多进程训练模式下。"""
        return self.backend != "none"

    @property
    def is_main(self) -> bool:
        """rank 0 负责主日志、保存等只需要执行一次的操作。"""
        return self.rank == 0


def init_distributed(args) -> DistributedContext:
    """根据启动参数和 torchrun 环境变量初始化分布式进程组。"""
    import torch
    import torch.distributed as dist

    if args.distributed_backend == "none":
        return DistributedContext(backend="none")

    if args.distributed_backend != "fsdp":
        raise ValueError(f"Unsupported distributed backend: {args.distributed_backend}")
    if not torch.cuda.is_available():
        raise RuntimeError("FSDP training requires CUDA.")

    # torchrun 会注入这些环境变量；这里统一读出来形成 DistributedContext。
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
        # 每个进程先绑定自己的 local GPU，再初始化 NCCL，避免不同进程抢同一张卡。
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
    """在所有 rank 上求均值；非分布式模式下直接返回 detach 后的值。"""
    import torch.distributed as dist

    if not dist_ctx.is_distributed:
        return value.detach()
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist_ctx.world_size
    return reduced


def destroy_distributed(dist_ctx: DistributedContext, *, success: bool) -> None:
    """训练结束时销毁进程组；成功结束时先 barrier，避免某些 rank 提前退出。"""
    import torch.distributed as dist

    if dist_ctx.is_distributed and dist.is_initialized():
        if success:
            dist.barrier()
        dist.destroy_process_group()

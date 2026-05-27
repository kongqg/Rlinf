from __future__ import annotations

from typing import TYPE_CHECKING

from rlinf.training.common.distributed import DistributedContext

if TYPE_CHECKING:
    import torch


def device_of(device_name: str) -> "torch.device":
    import torch

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return device


def device_for_training(device_name: str, dist_ctx: DistributedContext) -> "torch.device":
    import torch

    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if dist_ctx.is_distributed and device_name == "cuda":
        return torch.device("cuda", dist_ctx.local_rank)
    return torch.device(device_name)

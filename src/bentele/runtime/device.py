from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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


def device_of(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return device


def device_for_training(device_name: str, dist_ctx: DistributedContext) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if dist_ctx.is_distributed and device_name == "cuda":
        return torch.device("cuda", dist_ctx.local_rank)
    return torch.device(device_name)

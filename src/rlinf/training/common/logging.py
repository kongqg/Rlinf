from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np

from rlinf.training.common.distributed import DistributedContext


class WandbLogger:
    def __init__(self, run) -> None:
        self._run = run

    def log(self, payload: dict[str, float], *, step: int) -> None:
        if payload:
            self._run.log(payload, step=step)

    def finish(self) -> None:
        self._run.finish()


def log_stage(
    message: str,
    dist_ctx: DistributedContext | None = None,
    *,
    enabled: bool,
    all_ranks: bool = False,
    prefix: str = "training",
) -> None:
    if not enabled:
        return
    rank = 0 if dist_ctx is None else dist_ctx.rank
    if dist_ctx is not None and not all_ranks and not dist_ctx.is_main:
        return
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{prefix}][{now}][rank {rank}] {message}", flush=True)


def json_ready(value: Any) -> Any:
    try:
        import torch
    except ModuleNotFoundError:
        torch = None

    if isinstance(value, Path):
        return str(value)
    if torch is not None and isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value

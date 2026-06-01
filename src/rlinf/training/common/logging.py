from __future__ import annotations

"""训练日志与 JSON 序列化辅助工具。"""

from pathlib import Path
import time
from typing import Any

import numpy as np

from rlinf.training.common.distributed import DistributedContext


class WandbLogger:
    """对 wandb run 做一层很薄的封装，方便 runner 按统一接口调用。"""

    def __init__(self, run) -> None:
        self._run = run

    def log(self, payload: dict[str, float], *, step: int) -> None:
        """只有 payload 非空时才写入，避免产生空日志记录。"""
        if payload:
            self._run.log(payload, step=step)

    def finish(self) -> None:
        """显式结束 wandb run，确保远端日志正常 flush。"""
        self._run.finish()


def log_stage(
    message: str,
    dist_ctx: DistributedContext | None = None,
    *,
    enabled: bool,
    all_ranks: bool = False,
    prefix: str = "training",
) -> None:
    """打印训练阶段日志；默认只在 main rank 输出，避免多卡重复刷屏。"""
    if not enabled:
        return
    rank = 0 if dist_ctx is None else dist_ctx.rank
    if dist_ctx is not None and not all_ranks and not dist_ctx.is_main:
        return
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{prefix}][{now}][rank {rank}] {message}", flush=True)


def json_ready(value: Any) -> Any:
    """递归把 Path / Tensor / ndarray 等对象转换成 JSON 可写类型。"""
    try:
        import torch
    except ModuleNotFoundError:
        torch = None

    if isinstance(value, Path):
        return str(value)
    if torch is not None and isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        # 多元素 tensor 先 detach 到 CPU，再转成 list，避免序列化时保留计算图。
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value

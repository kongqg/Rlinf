from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


class WandbLogger:
    def __init__(self, run) -> None:
        self._run = run

    def log(self, payload: dict[str, float], *, step: int) -> None:
        if payload:
            self._run.log(payload, step=step)

    def finish(self) -> None:
        self._run.finish()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
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


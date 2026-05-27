from __future__ import annotations

from typing import Any

import torch


def loss_from_output(losses: Any, device: torch.device) -> torch.Tensor:
    if isinstance(losses, torch.Tensor):
        return losses
    if isinstance(losses, (list, tuple)):
        tensors = [loss for loss in losses if isinstance(loss, torch.Tensor)]
        if not tensors:
            return torch.tensor(0.0, device=device)
        return sum(tensors) / len(tensors)
    raise TypeError(f"Unsupported loss output type: {type(losses)!r}")


from __future__ import annotations

from typing import Any

import torch


def loss_from_output(losses: Any, device: torch.device) -> torch.Tensor:
    if isinstance(losses, (list, tuple)):
        losses = torch.stack(list(losses))
    elif not isinstance(losses, torch.Tensor):
        losses = torch.tensor(losses, dtype=torch.float32, device=device)
    return losses.float().mean()

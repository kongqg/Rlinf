from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SFTTrainingArgs:
    batch_size: int = 8
    grad_accum_steps: int = 1
    train_steps: int = 20000
    learning_rate: float = 2.5e-5
    min_learning_rate: float = 2.5e-6
    lr_warmup_steps: int = 1000
    lr_total_steps: int = 30000
    lr_num_cycles: float = 0.5
    lr_scheduler: Literal["none", "cosine"] = "cosine"
    weight_decay: float = 1.0e-10
    clip_grad_norm: float = 1.0
    log_every: int = 20
    save_every: int = 200
    seed: int = 1234
    device: str = "cuda"


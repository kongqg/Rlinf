from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RLTrainingArgs:
    seed: int = 1234
    total_steps: int = 4000
    total_num_envs: int = 256
    rollout_epoch: int = 4
    global_batch_size: int = 2048
    micro_batch_size: int = 32
    update_epoch: int = 5
    log_every: int = 10
    save_every: int = 100
    eval_every: int = 100


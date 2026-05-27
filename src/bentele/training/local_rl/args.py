from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Args:
    assets_path: str
    model_path: str
    output_dir: str
    config_name: str = "robotwin_place_phone_stand_ppo_openpi_pi05"
    device: str = "cuda"
    seed: int = 1234
    total_steps: int = 4000
    total_num_envs: int = 256
    rollout_epoch: int = 4
    max_steps_per_rollout_epoch: int = 200
    max_episode_steps: int = 200
    global_batch_size: int = 2048
    micro_batch_size: int = 32
    update_epoch: int = 5
    actor_lr: float = 5.0e-6
    value_lr: float = 1.0e-4
    weight_decay: float = 1.0e-2
    clip_grad_norm: float = 1.0
    entropy_bonus: float = 0.0
    log_every: int = 10
    save_every: int = 100
    eval_every: int = 100
    eval_num_envs: int = 128
    eval_rollout_epochs: int = 1
    eval_device: Literal["cpu", "cuda", "auto"] = "cuda"
    save_optimizer_state: bool = False
    wandb_enabled: bool = True
    wandb_project: str = "bentele"
    wandb_run_name: str | None = None
    wandb_log_dir: str | None = None
    wandb_proxy: str | None = None
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    debug_stage_logs: bool = False


__all__ = ["Args"]

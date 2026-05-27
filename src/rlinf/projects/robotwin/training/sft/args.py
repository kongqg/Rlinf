from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Args:
    dataset_root: str
    train_repo_ids: Sequence[str]
    model_path: str
    output_dir: str
    train_episode_indices: tuple[int, ...] = ()
    train_num_episodes_limit: int = 0
    train_sample_limit: int = 0
    config_name: str = "pi05_aloha_robotwin"
    eval_config_name: str = "robotwin_place_phone_stand_ppo_openpi_pi05"
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
    save_step_checkpoints: bool = False
    num_workers: int = 0
    seed: int = 1234
    device: str = "cuda"
    use_quantile_norm: bool = False
    is_lora: bool = False
    lora_rank: int = 32
    lora_path: str | None = None
    assets_path: str | None = None
    eval_every: int = 0
    eval_num_envs: int = 1
    eval_rollout_epochs: int = 1
    eval_seeds_path: str | None = None
    eval_step_limit: int | None = None
    eval_max_chunk_steps: int | None = None
    eval_action_exec_horizon: int | None = None
    eval_record_progress: bool = True
    eval_visualize: bool = False
    eval_render_freq: int = 10
    eval_sleep_between_chunks: float = 0.0
    eval_hold_viewer_seconds: float = 0.0
    eval_device: str = "cpu"
    eval_debug_log_chunks: bool = False
    eval_debug_action_steps: int = 0
    action_probe_every: int = 100
    action_probe_num_samples: int = 5
    wandb_enabled: bool = False
    wandb_project: str = "bentele"
    wandb_run_name: str | None = None
    wandb_group: str | None = None
    wandb_log_dir: str | None = None
    wandb_proxy: str | None = None
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_tags: tuple[str, ...] = ()
    distributed_backend: Literal["none", "fsdp"] = "none"
    dist_timeout_minutes: int = 60
    fsdp_sharding_strategy: Literal["full_shard", "shard_grad_op"] = "full_shard"
    fsdp_mixed_precision: Literal["bf16", "fp32"] = "bf16"
    fsdp_cpu_offload: bool = False
    fsdp_use_orig_params: bool = True
    fsdp_lora_auto_wrap: bool = False
    save_optimizer_state: bool = False
    debug_stage_logs: bool = True
    debug_all_ranks: bool = False
    noise_level: float = 0.5


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


__all__ = ["Args", "DistributedContext"]

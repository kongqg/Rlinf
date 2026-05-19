from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


def validate_cfg(cfg: DictConfig) -> DictConfig:
    """Apply small scaffold defaults and verify the expected sections exist."""

    required_sections = [
        "cluster",
        "runner",
        "algorithm",
        "env",
        "rollout",
        "actor",
        "reward",
        "critic",
    ]
    missing = [name for name in required_sections if name not in cfg]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    if "per_worker_log_path" not in cfg.runner:
        cfg.runner.per_worker_log_path = cfg.runner.logger.log_path

    if "task_type" not in cfg.runner:
        cfg.runner.task_type = "embodied"

    return cfg


def to_resolved_dict(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)

from __future__ import annotations

from rlinf.projects.robotwin.adapters.rl_task import RobotwinRLTaskSpec
from rlinf.projects.robotwin.training.local_rl.args import Args
from rlinf.training.rl.runner import run_rl_training as _run_generic_rl_training


_HELPER_EXPORTS = {
    "CONFIG_DIR",
    "_copy_if_exists",
    "_init_wandb_logger",
    "append_history",
    "build_optimizer",
    "build_wandb_payload",
    "collect_rollout",
    "compose_cfg",
    "compute_loss_mask",
    "env_interact_step",
    "evaluate_current_policy",
    "log_stage",
    "main",
    "make_initial_env_output",
    "prepare_output_dir",
    "prepare_rollout_batch",
    "process_nested_dict_for_adv",
    "process_nested_dict_for_train",
    "run_update",
    "save_checkpoint",
    "summarize_env_metrics",
}


def run_local_rl_training(args: Args) -> None:
    _run_generic_rl_training(args, RobotwinRLTaskSpec())


def __getattr__(name: str):
    if name not in _HELPER_EXPORTS:
        raise AttributeError(name)
    from rlinf.projects.robotwin.adapters import _rl_impl

    return getattr(_rl_impl, name)


__all__ = [
    "RobotwinRLTaskSpec",
    *_HELPER_EXPORTS,
    "run_local_rl_training",
]


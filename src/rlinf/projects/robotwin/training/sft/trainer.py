from __future__ import annotations

from rlinf.projects.robotwin.adapters.sft_task import RobotwinSFTTaskSpec
from rlinf.projects.robotwin.training.sft.args import Args
from rlinf.training.sft.trainer import run_sft_training as _run_generic_sft_training


_HELPER_EXPORTS = {
    "_apply_openpi_lora",
    "_build_fsdp_model",
    "_build_loader",
    "_build_model",
    "_build_openpi_model_for_fsdp",
    "_build_runtime_cfg",
    "_build_train_config",
    "_copy_if_exists",
    "_count_parameters",
    "_export_eval_model_dir",
    "_learning_rate_for_update",
    "_load_openpi_weights",
    "_loss_from_output",
    "_move_observation_to_device",
    "_run_periodic_eval",
    "_save_checkpoint",
    "_set_optimizer_learning_rate",
    "_tag_vlm_subtree",
    "_write_model_dir",
    "main",
}


def run(args: Args) -> None:
    _run_generic_sft_training(args, RobotwinSFTTaskSpec())


def run_sft_training(args: Args) -> None:
    run(args)


def __getattr__(name: str):
    if name not in _HELPER_EXPORTS:
        raise AttributeError(name)
    from rlinf.projects.robotwin.adapters import _sft_impl

    return getattr(_sft_impl, name)


__all__ = [
    "RobotwinSFTTaskSpec",
    *_HELPER_EXPORTS,
    "run",
    "run_sft_training",
]


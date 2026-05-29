"""Supervised fine-tuning entrypoint for RoboTwin place_phone_stand on pi0.5.

This script keeps the training path intentionally small:

- load a PyTorch OpenPI/pi0.5 checkpoint
- read one or more local LeRobot-format RoboTwin datasets
- reuse the existing RLinf/OpenPI RoboTwin ALOHA dataconfig
- run pure VLA imitation learning (no RL / no value head)
- save a model directory that `scripts/eval_robotwin_policy.py` can read
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
import json
import math
import os
import shutil
import traceback
from dataclasses import asdict
from pathlib import Path
import time
from typing import Any, Literal

from rlinf.projects.robotwin.data.robotwin_v3_dataset import RoboTwinV3LocalDataset
from rlinf.projects.robotwin.adapters.probes import (
    RolloutProbeSample,
    _build_rollout_probe_samples,
    _compute_rollout_probe_metrics,
)
from rlinf.projects.robotwin.runtime.env import ensure_torch_transformers_runtime
from rlinf.projects.robotwin.runtime.paths import REPO_ROOT
from rlinf.projects.robotwin.training.sft.args import Args, DistributedContext
from rlinf.training.common.device import device_for_training as _device_of
from rlinf.training.common.distributed import init_distributed as _common_init_distributed
from rlinf.training.common.distributed import reduce_mean as _common_reduce_mean
from rlinf.training.common.lr_scheduler import cosine_learning_rate_for_update
from rlinf.training.common.logging import WandbLogger as _WandbLogger
from rlinf.training.common.optim import set_optimizer_learning_rate
from rlinf.training.common.seed import _set_seed
from rlinf.training.common.utils import copy_if_exists

# Force transformers/openpi onto the torch path before importing any model code.
ensure_torch_transformers_runtime()

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    CPUOffload,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.utils import _pytree
from torch.utils.data import ConcatDataset
import tyro

import openpi.shared.normalize as normalize
from rlinf.projects.robotwin.utils.robotwin_eval import run_robotwin_policy_eval, save_eval_result
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.utils.pytree import register_pytree_dataclasses


def _log_stage(
    message: str,
    dist_ctx: DistributedContext,
    *,
    enabled: bool,
    all_ranks: bool = False,
) -> None:
    if not enabled:
        return
    if not all_ranks and not dist_ctx.is_main:
        return
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[train_robotwin_vla][{now}][rank {dist_ctx.rank}] {message}", flush=True)


def _init_distributed(args: Args) -> DistributedContext:
    # gpu size
    return _common_init_distributed(args)


def _init_wandb_logger(
    args: Args,
    dist_ctx: DistributedContext,
    output_dir: Path,
) -> _WandbLogger | None:
    if (
        not args.wandb_enabled
        or args.wandb_mode == "disabled"
        or not dist_ctx.is_main
    ):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed in the current environment. "
            "Install it first or run with --wandb-enabled False."
        ) from exc

    settings = None
    if args.wandb_proxy:
        settings = wandb.Settings(https_proxy=args.wandb_proxy)

    wandb_log_dir = (
        Path(args.wandb_log_dir).expanduser().resolve()
        if args.wandb_log_dir is not None
        else output_dir / "wandb"
    )
    wandb_log_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.wandb_run_name or output_dir.name
    run = wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=asdict(args),
        settings=settings,
        dir=str(wandb_log_dir),
        group=args.wandb_group,
        tags=list(args.wandb_tags),
        mode=args.wandb_mode,
        reinit=True,
    )
    return _WandbLogger(run)


def _build_train_config(
    *,
    config_name: str,
    model_path: str,
    repo_id: str,
    batch_size: int,
    use_quantile_norm: bool,
):
    return get_openpi_config(
        config_name,
        model_path=model_path,
        batch_size=batch_size,
        repo_id=repo_id,
        data_kwargs={"use_quantile_norm": use_quantile_norm},
    )


def _build_loader(args: Args, dist_ctx: DistributedContext):
    import openpi.training.data_loader as openpi_data

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    if not args.train_repo_ids:
        raise ValueError("train_repo_ids must not be empty.")
    if args.train_episode_indices and args.train_num_episodes_limit > 0:
        raise ValueError(
            "Use either train_episode_indices or train_num_episodes_limit, not both."
        )

    train_config = _build_train_config(
        config_name=args.config_name,
        model_path=args.model_path,
        repo_id=args.train_repo_ids[0],
        batch_size=args.batch_size,
        use_quantile_norm=args.use_quantile_norm,
    )

    datasets = []
    data_config = None
    for repo_id in args.train_repo_ids:
        repo_train_config = _build_train_config(
            config_name=args.config_name,
            model_path=args.model_path,
            repo_id=repo_id,
            batch_size=args.batch_size,
            use_quantile_norm=args.use_quantile_norm,
        )
        data_config = repo_train_config.data.create(
            repo_train_config.assets_dirs, repo_train_config.model
        )
        raw_dataset = RoboTwinV3LocalDataset(
            dataset_root / repo_id,
            action_horizon=repo_train_config.model.action_horizon,
            episode_indices=args.train_episode_indices or None,
            episode_limit=(
                args.train_num_episodes_limit
                if args.train_num_episodes_limit > 0
                else None
            ),
            max_samples=args.train_sample_limit if args.train_sample_limit > 0 else None,
        )
        if dist_ctx.is_main:
            episode_summary = (
                raw_dataset.selected_episode_indices
                if len(raw_dataset.selected_episode_indices) <= 10
                else [
                    *raw_dataset.selected_episode_indices[:5],
                    "...",
                    *raw_dataset.selected_episode_indices[-3:],
                ]
            )
            print(
                f"[train_robotwin_vla] repo_id={repo_id} "
                f"selected_episodes={episode_summary} "
                f"num_selected_episodes={len(raw_dataset.selected_episode_indices)} "
                f"raw_samples={len(raw_dataset.sample_indices)}"
            )
        dataset = openpi_data.transform_dataset(raw_dataset, data_config)
        datasets.append(dataset)

    merged = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    sampler = None
    if dist_ctx.is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            merged,
            num_replicas=dist_ctx.world_size,
            rank=dist_ctx.rank,
            shuffle=True,
            drop_last=True,
        )
    torch_loader = openpi_data.TorchDataLoader(
        merged,
        local_batch_size=args.batch_size,
        shuffle=True,
        sampler=sampler,
        num_batches=None,
        num_workers=args.num_workers,
        seed=args.seed,
        framework="pytorch",
    )
    loader = openpi_data.DataLoaderImpl(data_config, torch_loader)
    if dist_ctx.is_main:
        global_batch_size = args.batch_size * args.grad_accum_steps * dist_ctx.world_size
        print(
            f"[train_robotwin_vla] dataset_root={dataset_root} "
            f"train_repo_ids={list(args.train_repo_ids)} "
            f"num_datasets={len(datasets)} total_samples={len(merged)} "
            f"distributed_backend={args.distributed_backend} "
            f"world_size={dist_ctx.world_size} "
            f"local_batch_size={args.batch_size} "
            f"effective_global_batch_size={global_batch_size}"
        )
    return train_config, loader


def _build_fsdp_model(
    model: torch.nn.Module,
    args: Args,
    device: torch.device,
    dist_ctx: DistributedContext,
) -> FSDP:
    if device.type == "cuda":
        torch.cuda.set_device(device)
    _log_stage(
        "fsdp wrap start",
        dist_ctx,
        enabled=args.debug_stage_logs,
        all_ranks=args.debug_all_ranks,
    )
    # The OpenPI/pi0.5 checkpoint intentionally keeps some parameters in
    # bfloat16 and others in float32. Classic FSDP cannot flatten mixed-dtype
    # parameters inside one wrapped module, so normalize everything back to
    # float32 before wrapping. Runtime compute still uses autocast bf16.
    model.to(device=device, dtype=torch.float32)

    sharding_strategy = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    }[args.fsdp_sharding_strategy]

    mixed_precision = None
    if args.fsdp_mixed_precision == "bf16":
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

    cpu_offload = CPUOffload(offload_params=args.fsdp_cpu_offload)
    auto_wrap_policy = None
    if args.is_lora and args.fsdp_lora_auto_wrap:
        from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=model,
            config={},
            is_lora=True,
            model_type="openpi",
        )
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=device,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision,
        cpu_offload=cpu_offload,
        use_orig_params=args.fsdp_use_orig_params,
        sync_module_states=True,
        limit_all_gathers=True,
    )
    _log_stage(
        "fsdp wrap done",
        dist_ctx,
        enabled=args.debug_stage_logs,
        all_ranks=args.debug_all_ranks,
    )
    return model


def _build_runtime_cfg(args: Args):
    # Keep the standalone SFT runtime aligned with the project pi0.5 embodied
    # model defaults. If this drifts from `configs/embodiment/model/pi0_5.yaml`,
    # training and evaluation may silently use different OpenPI behaviors.
    return OmegaConf.create(
        {
            "model_path": str(Path(args.model_path).expanduser().resolve()),
            "is_lora": args.is_lora,
            "lora_rank": args.lora_rank,
            "lora_path": (
                str(Path(args.lora_path).expanduser().resolve())
                if args.lora_path is not None
                else None
            ),
            "openpi": {
                "config_name": args.config_name,
                "num_images_in_input": 3,
                "noise_level": args.noise_level,
                "action_chunk": 50,
                "num_steps": 5,
                "train_expert_only": False,
                "add_value_head": False,
                "action_env_dim": 14,
                "value_after_vlm": False,
                "value_vlm_mode": "mean_token",
                "detach_critic_input": True,
            },
            "openpi_data": {
                "use_quantile_norm": args.use_quantile_norm,
            },
        }
    )


def _load_openpi_weights(model: torch.nn.Module, checkpoint_dir: str) -> None:
    import glob

    import safetensors

    full_weights_path = os.path.join(checkpoint_dir, "model_state_dict", "full_weights.pt")
    actor_full_weights_path = os.path.join(
        checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"
    )

    if os.path.exists(full_weights_path):
        model_state_dict = torch.load(full_weights_path, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=False)
        return
    if os.path.exists(actor_full_weights_path):
        model_state_dict = torch.load(actor_full_weights_path, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=False)
        return

    weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if not weight_paths:
        weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]
    all_state_dict = {}
    for weight_path in weight_paths:
        state_dict = safetensors.torch.load_file(weight_path, device="cpu")
        all_state_dict.update(state_dict)
    model.load_state_dict(all_state_dict, strict=False)


def _tag_vlm_subtree(model: torch.nn.Module, is_vlm: bool) -> None:
    for _, module in model.named_modules():
        setattr(module, "_to_lora", is_vlm)


def _apply_openpi_lora(
    model: torch.nn.Module,
    *,
    lora_rank: int,
    lora_path: str | None,
) -> torch.nn.Module:
    """Apply the RLinf/OpenPI LoRA layout to the PaliGemma VLM subtree."""
    if getattr(model, "_rlinf_lora_applied", False):
        return model
    if lora_rank <= 0:
        raise ValueError(f"lora_rank must be positive, got {lora_rank}.")
    if not hasattr(model, "paligemma_with_expert") or not hasattr(
        model.paligemma_with_expert,
        "paligemma",
    ):
        raise AttributeError(
            "OpenPI LoRA expects model.paligemma_with_expert.paligemma to exist."
        )

    from peft import LoraConfig, PeftModel, get_peft_model

    target_modules = [
        "proj",
        "qkv",
        "fc1",
        "fc2",
        "q",
        "kv",
        "fc3",
        "out_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ]

    _tag_vlm_subtree(model, False)
    module_to_lora = model.paligemma_with_expert.paligemma
    if lora_path is None:
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            lora_dropout=0.0,
            target_modules=target_modules,
            init_lora_weights="gaussian",
        )
        module_to_lora = get_peft_model(module_to_lora, lora_config)
    else:
        module_to_lora = PeftModel.from_pretrained(
            module_to_lora,
            lora_path,
            is_trainable=True,
        )
    _tag_vlm_subtree(module_to_lora, True)
    model.paligemma_with_expert.paligemma = module_to_lora
    setattr(model, "_rlinf_lora_applied", True)
    return model


def _count_parameters(model: torch.nn.Module) -> dict[str, float | int]:
    total = 0
    trainable = 0
    lora = 0
    trainable_lora = 0
    for name, param in model.named_parameters():
        n_params = param.numel()
        total += n_params
        if param.requires_grad:
            trainable += n_params
        if "lora_" in name.lower():
            lora += n_params
            if param.requires_grad:
                trainable_lora += n_params
    return {
        "total": total,
        "trainable": trainable,
        "lora": lora,
        "trainable_lora": trainable_lora,
        "trainable_ratio": (trainable / total) if total else 0.0,
    }


def _build_openpi_model_for_fsdp(
    args: Args,
    dist_ctx: DistributedContext,
):
    import openpi.shared.download as download
    import openpi.transforms as transforms
    from openpi.training import checkpoints as _checkpoints

    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0Config,
        OpenPi0ForRLActionPrediction,
    )

    cfg = _build_runtime_cfg(args)
    config_name = getattr(cfg.openpi, "config_name", None)
    data_kwargs = getattr(cfg, "openpi_data", None)
    actor_train_config = get_openpi_config(
        config_name,
        model_path=cfg.model_path,
        data_kwargs=data_kwargs,
    )
    actor_model_config = OpenPi0Config(**actor_train_config.model.__dict__)
    for key, val in cfg.openpi.items():
        actor_model_config.__dict__[key] = val

    checkpoint_dir = download.maybe_download(str(cfg.model_path))
    data_config = actor_train_config.data.create(
        actor_train_config.assets_dirs, actor_model_config
    )

    model: OpenPi0ForRLActionPrediction = OpenPi0ForRLActionPrediction(actor_model_config)
    if actor_model_config.train_expert_only:
        model.freeze_vlm()

    _log_stage(
        "checkpoint load plan: rank0 loads weights, other ranks use fsdp state sync",
        dist_ctx,
        enabled=args.debug_stage_logs,
        all_ranks=False,
    )
    if dist_ctx.is_main:
        _log_stage(
            "rank0 loading checkpoint weights",
            dist_ctx,
            enabled=args.debug_stage_logs,
            all_ranks=False,
        )
        _load_openpi_weights(model, checkpoint_dir)
        _log_stage(
            "rank0 checkpoint weights loaded",
            dist_ctx,
            enabled=args.debug_stage_logs,
            all_ranks=False,
        )

    if args.is_lora and not getattr(model, "_rlinf_lora_applied", False):
        resolved_lora_path = (
            str(Path(args.lora_path).expanduser().resolve())
            if args.lora_path is not None
            else None
        )
        _log_stage(
            f"applying LoRA after base checkpoint load rank={args.lora_rank} lora_path={resolved_lora_path}",
            dist_ctx,
            enabled=args.debug_stage_logs,
            all_ranks=args.debug_all_ranks,
        )
        model = _apply_openpi_lora(
            model,
            lora_rank=args.lora_rank,
            lora_path=resolved_lora_path,
        )

    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)

    repack_transforms = transforms.Group()
    model.setup_wrappers(
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
    )
    return model


def _build_model(args: Args, device: torch.device, dist_ctx: DistributedContext):
    cfg = _build_runtime_cfg(args)
    if not dist_ctx.is_distributed:
        from rlinf.models.embodiment.openpi import get_model

        _log_stage(
            "get_model start",
            dist_ctx,
            enabled=args.debug_stage_logs,
            all_ranks=args.debug_all_ranks,
        )
        model = get_model(cfg)
        _log_stage(
            "get_model done",
            dist_ctx,
            enabled=args.debug_stage_logs,
            all_ranks=args.debug_all_ranks,
        )
        model.to(device)
        model.train()
        return model

    _log_stage(
        "get_model start",
        dist_ctx,
        enabled=args.debug_stage_logs,
        all_ranks=args.debug_all_ranks,
    )
    model = _build_openpi_model_for_fsdp(args, dist_ctx)
    _log_stage(
        "get_model done",
        dist_ctx,
        enabled=args.debug_stage_logs,
        all_ranks=args.debug_all_ranks,
    )
    model.to(device)
    if dist_ctx.is_distributed:
        model = _build_fsdp_model(model, args, device, dist_ctx)
    model.train()
    return model


def _move_observation_to_device(observation, device: torch.device):
    register_pytree_dataclasses(observation)
    return _pytree.tree_map(
        lambda x: (
            x.to(device=device, dtype=torch.float32, non_blocking=True)
            if torch.is_tensor(x) and torch.is_floating_point(x)
            else x.to(device=device, non_blocking=True)
            if torch.is_tensor(x)
            else x
        )
        if x is not None
        else x,
        observation,
    )


def _loss_from_output(losses: Any, device: torch.device) -> torch.Tensor:
    if isinstance(losses, (list, tuple)):
        losses = torch.stack(list(losses))
    elif not isinstance(losses, torch.Tensor):
        losses = torch.tensor(losses, dtype=torch.float32, device=device)
    return losses.float().mean()


def _learning_rate_for_update(args: Args, update_step: int) -> float:
    return cosine_learning_rate_for_update(args, update_step)


def _set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    set_optimizer_learning_rate(optimizer, learning_rate)


def _copy_if_exists(src: Path, dst: Path) -> None:
    copy_if_exists(src, dst)


def _write_model_dir(
    *,
    output_dir: Path,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any] | None,
    step: int,
    args: Args,
    data_config,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "model_state_dict"
    weights_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model_state, weights_dir / "full_weights.pt")
    torch.save(
        {
            "step": step,
            "optimizer": optimizer_state,
            "args": asdict(args),
        },
        output_dir / "training_state.pt",
    )

    if data_config.norm_stats is not None and data_config.asset_id is not None:
        normalize.save(output_dir / data_config.asset_id, data_config.norm_stats)

    model_path = Path(args.model_path).expanduser().resolve()
    _copy_if_exists(model_path / "config.json", output_dir / "config.json")
    _copy_if_exists(model_path / "assets", output_dir / "assets")

    metadata = {
        "config_name": args.config_name,
        "base_model_path": str(model_path),
        "train_repo_ids": list(args.train_repo_ids),
        "train_episode_indices": list(args.train_episode_indices),
        "train_num_episodes_limit": args.train_num_episodes_limit,
        "train_sample_limit": args.train_sample_limit,
        "use_quantile_norm": args.use_quantile_norm,
        "is_lora": args.is_lora,
        "lora_rank": args.lora_rank,
        "lora_path": (
            str(Path(args.lora_path).expanduser().resolve())
            if args.lora_path is not None
            else None
        ),
        "saved_step": step,
    }
    (output_dir / "sft_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _save_checkpoint(
    *,
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: Args,
    data_config,
    dist_ctx: DistributedContext,
) -> None:
    if dist_ctx.is_distributed:
        state_dict_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        _log_stage(
            f"step {step}: gather full model state start",
            dist_ctx,
            enabled=args.debug_stage_logs,
            all_ranks=args.debug_all_ranks,
        )
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            state_dict_cfg,
            optim_cfg,
        ):
            model_state = model.state_dict()
        _log_stage(
            f"step {step}: gather full model state done",
            dist_ctx,
            enabled=args.debug_stage_logs,
            all_ranks=args.debug_all_ranks,
        )
        optimizer_state = None
        if args.save_optimizer_state:
            _log_stage(
                f"step {step}: gather optimizer state start",
                dist_ctx,
                enabled=args.debug_stage_logs,
                all_ranks=args.debug_all_ranks,
            )
            optimizer_state = (
                FSDP.full_optim_state_dict(model, optimizer) if dist_ctx.is_main else None
            )
            _log_stage(
                f"step {step}: gather optimizer state done",
                dist_ctx,
                enabled=args.debug_stage_logs,
                all_ranks=args.debug_all_ranks,
            )
        if not dist_ctx.is_main:
            dist.barrier()
            return
    else:
        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict() if args.save_optimizer_state else None

    _write_model_dir(
        output_dir=output_dir,
        model_state=model_state,
        optimizer_state=optimizer_state,
        step=step,
        args=args,
        data_config=data_config,
    )
    if dist_ctx.is_main and args.save_step_checkpoints:
        _write_model_dir(
            output_dir=output_dir / "checkpoints" / f"step_{step:07d}",
            model_state=model_state,
            optimizer_state=optimizer_state,
            step=step,
            args=args,
            data_config=data_config,
        )
    if dist_ctx.is_distributed:
        dist.barrier()


def _export_eval_model_dir(
    *,
    eval_dir: Path,
    model: torch.nn.Module,
    step: int,
    args: Args,
    data_config,
    dist_ctx: DistributedContext,
) -> Path | None:
    if dist_ctx.is_distributed:
        state_dict_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        _log_stage(
            f"step {step}: eval gather model state start",
            dist_ctx,
            enabled=args.debug_stage_logs,
            all_ranks=args.debug_all_ranks,
        )
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            state_dict_cfg,
            optim_cfg,
        ):
            model_state = model.state_dict()
        _log_stage(
            f"step {step}: eval gather model state done",
            dist_ctx,
            enabled=args.debug_stage_logs,
            all_ranks=args.debug_all_ranks,
        )
        if not dist_ctx.is_main:
            dist.barrier()
            return None
    else:
        model_state = model.state_dict()

    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    _write_model_dir(
        output_dir=eval_dir,
        model_state=model_state,
        optimizer_state=None,
        step=step,
        args=args,
        data_config=data_config,
    )
    if dist_ctx.is_distributed:
        dist.barrier()
    return eval_dir


def _run_periodic_eval(
    *,
    args: Args,
    step: int,
    model_dir: Path,
    output_root: Path,
    dist_ctx: DistributedContext,
    wandb_logger: _WandbLogger | None,
) -> dict[str, Any] | None:
    if not dist_ctx.is_main:
        if dist_ctx.is_distributed:
            dist.barrier()
        return None

    if args.assets_path is None:
        raise ValueError("assets_path is required when eval_every > 0.")

    print(
        f"[train_robotwin_vla] eval start step={step} "
        f"model_dir={model_dir} num_envs={args.eval_num_envs} "
        f"rollout_epochs={args.eval_rollout_epochs}"
    )
    eval_start = time.time()
    try:
        result = run_robotwin_policy_eval(
            config_name=args.eval_config_name,
            assets_path=args.assets_path,
            model_path=str(model_dir),
            num_envs=args.eval_num_envs,
            eval_rollout_epochs=args.eval_rollout_epochs,
            eval_step_limit=args.eval_step_limit,
            max_chunk_steps=args.eval_max_chunk_steps,
            action_exec_horizon=args.eval_action_exec_horizon,
            visualize=args.eval_visualize,
            render_freq=args.eval_render_freq,
            sleep_between_chunks=args.eval_sleep_between_chunks,
            hold_viewer_seconds=args.eval_hold_viewer_seconds,
            device=args.eval_device,
            eval_seeds_path=args.eval_seeds_path,
            debug_log_chunks=args.eval_debug_log_chunks,
            debug_action_steps=args.eval_debug_action_steps,
            record_progress=args.eval_record_progress,
        )
    except Exception:
        print(
            "[train_robotwin_vla] "
            f"eval failed at step={step} model_dir={model_dir}"
        )
        traceback.print_exc()
        raise
    eval_runtime = time.time() - eval_start
    eval_json_path = output_root / "evals" / f"step_{step:07d}.json"
    save_eval_result(result, eval_json_path)
    metrics = result["metrics"]
    progress_aggregate = result.get("progress", {}).get("aggregate", {})
    stage_counts = progress_aggregate.get("stage_counts", {})
    print(
        "[train_robotwin_vla] "
        f"eval step={step} "
        f"success_at_end={metrics.get('success_at_end', 0.0)} "
        f"success_once={metrics.get('success_once', 0.0)} "
        f"return={metrics.get('return', 0.0)} "
        f"episode_len={metrics.get('episode_len', 0.0)} "
        f"mean_min_tcp_phone_l2={progress_aggregate.get('mean_min_tcp_phone_l2', 'n/a')} "
        f"mean_max_phone_displacement={progress_aggregate.get('mean_max_phone_displacement', 'n/a')} "
        f"stage_counts={json.dumps(stage_counts, ensure_ascii=False)} "
        f"runtime_seconds={eval_runtime:.2f}"
    )
    if wandb_logger is not None:
        eval_log_payload: dict[str, float] = {
            "eval/success_at_end": float(metrics.get("success_at_end", 0.0)),
            "eval/success_once": float(metrics.get("success_once", 0.0)),
            "eval/return": float(metrics.get("return", 0.0)),
            "eval/reward": float(metrics.get("reward", 0.0)),
            "eval/episode_len": float(metrics.get("episode_len", 0.0)),
            "eval/num_trajectories": float(metrics.get("num_trajectories", 0.0)),
            "eval/runtime_seconds": float(eval_runtime),
        }
        for key in (
            "mean_initial_l2_distance",
            "mean_min_l2_distance",
            "mean_final_l2_distance",
            "mean_initial_tcp_phone_l2",
            "mean_min_tcp_phone_l2",
            "mean_final_tcp_phone_l2",
            "mean_max_phone_displacement",
            "mean_max_phone_height_gain",
        ):
            if key in progress_aggregate:
                eval_log_payload[f"eval_progress/{key}"] = float(progress_aggregate[key])
        for stage_name in (
            "success",
            "moved_phone",
            "approached_phone",
            "weak_approach",
            "no_meaningful_approach",
        ):
            eval_log_payload[f"eval_progress/stage_count/{stage_name}"] = float(
                stage_counts.get(stage_name, 0)
            )
        wandb_logger.log(eval_log_payload, step=step)
    if dist_ctx.is_distributed:
        dist.barrier()
    return result


def _reduce_mean(value: torch.Tensor, dist_ctx: DistributedContext) -> torch.Tensor:
    return _common_reduce_mean(value, dist_ctx)


def run(args: Args) -> None:
    from rlinf.projects.robotwin.adapters.sft_task import RobotwinSFTTaskSpec
    from rlinf.training.sft.trainer import run_sft_training

    run_sft_training(args, RobotwinSFTTaskSpec())


def main(
    dataset_root: str,
    train_repo_ids: Sequence[str],
    model_path: str,
    output_dir: str,
    train_episode_indices: tuple[int, ...] = (),
    train_num_episodes_limit: int = 0,
    train_sample_limit: int = 0,
    config_name: str = "pi05_aloha_robotwin",
    eval_config_name: str = "robotwin_place_phone_stand_ppo_openpi_pi05",
    batch_size: int = 8,
    grad_accum_steps: int = 1,
    train_steps: int = 20000,
    learning_rate: float = 2.5e-5,
    min_learning_rate: float = 2.5e-6,
    lr_warmup_steps: int = 1000,
    lr_total_steps: int = 30000,
    lr_num_cycles: float = 0.5,
    lr_scheduler: Literal["none", "cosine"] = "cosine",
    weight_decay: float = 1.0e-10,
    clip_grad_norm: float = 1.0,
    log_every: int = 20,
    save_every: int = 200,
    save_step_checkpoints: bool = False,
    num_workers: int = 0,
    seed: int = 1234,
    device: str = "cuda",
    use_quantile_norm: bool = False,
    is_lora: bool = False,
    lora_rank: int = 32,
    lora_path: str | None = None,
    assets_path: str | None = None,
    eval_every: int = 0,
    eval_num_envs: int = 1,
    eval_rollout_epochs: int = 1,
    eval_seeds_path: str | None = None,
    eval_step_limit: int | None = None,
    eval_max_chunk_steps: int | None = None,
    eval_action_exec_horizon: int | None = None,
    eval_record_progress: bool = True,
    eval_visualize: bool = False,
    eval_render_freq: int = 10,
    eval_sleep_between_chunks: float = 0.0,
    eval_hold_viewer_seconds: float = 0.0,
    eval_device: str = "cpu",
    eval_debug_log_chunks: bool = False,
    eval_debug_action_steps: int = 0,
    action_probe_every: int = 100,
    action_probe_num_samples: int = 5,
    wandb_enabled: bool = False,
    wandb_project: str = "bentele",
    wandb_run_name: str | None = None,
    wandb_group: str | None = None,
    wandb_log_dir: str | None = None,
    wandb_proxy: str | None = None,
    wandb_mode: Literal["online", "offline", "disabled"] = "online",
    wandb_tags: tuple[str, ...] = (),
    distributed_backend: Literal["none", "fsdp"] = "none",
    dist_timeout_minutes: int = 60,
    fsdp_sharding_strategy: Literal["full_shard", "shard_grad_op"] = "full_shard",
    fsdp_mixed_precision: Literal["bf16", "fp32"] = "bf16",
    fsdp_cpu_offload: bool = False,
    fsdp_use_orig_params: bool = True,
    save_optimizer_state: bool = False,
    debug_stage_logs: bool = True,
    debug_all_ranks: bool = False,
    noise_level: float = 0.5,
) -> None:
    run(
        Args(
            dataset_root=dataset_root,
            train_repo_ids=train_repo_ids,
            model_path=model_path,
            output_dir=output_dir,
            train_episode_indices=train_episode_indices,
            train_num_episodes_limit=train_num_episodes_limit,
            train_sample_limit=train_sample_limit,
            config_name=config_name,
            eval_config_name=eval_config_name,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            train_steps=train_steps,
            learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            lr_warmup_steps=lr_warmup_steps,
            lr_total_steps=lr_total_steps,
            lr_num_cycles=lr_num_cycles,
            lr_scheduler=lr_scheduler,
            weight_decay=weight_decay,
            clip_grad_norm=clip_grad_norm,
            log_every=log_every,
            save_every=save_every,
            save_step_checkpoints=save_step_checkpoints,
            num_workers=num_workers,
            seed=seed,
            device=device,
            use_quantile_norm=use_quantile_norm,
            is_lora=is_lora,
            lora_rank=lora_rank,
            lora_path=lora_path,
            assets_path=assets_path,
            eval_every=eval_every,
            eval_num_envs=eval_num_envs,
            eval_rollout_epochs=eval_rollout_epochs,
            eval_seeds_path=eval_seeds_path,
            eval_step_limit=eval_step_limit,
            eval_max_chunk_steps=eval_max_chunk_steps,
            eval_action_exec_horizon=eval_action_exec_horizon,
            eval_record_progress=eval_record_progress,
            eval_visualize=eval_visualize,
            eval_render_freq=eval_render_freq,
            eval_sleep_between_chunks=eval_sleep_between_chunks,
            eval_hold_viewer_seconds=eval_hold_viewer_seconds,
            eval_device=eval_device,
            eval_debug_log_chunks=eval_debug_log_chunks,
            eval_debug_action_steps=eval_debug_action_steps,
            action_probe_every=action_probe_every,
            action_probe_num_samples=action_probe_num_samples,
            wandb_enabled=wandb_enabled,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
            wandb_group=wandb_group,
            wandb_log_dir=wandb_log_dir,
            wandb_proxy=wandb_proxy,
            wandb_mode=wandb_mode,
            wandb_tags=wandb_tags,
            distributed_backend=distributed_backend,
            dist_timeout_minutes=dist_timeout_minutes,
            fsdp_sharding_strategy=fsdp_sharding_strategy,
            fsdp_mixed_precision=fsdp_mixed_precision,
            fsdp_cpu_offload=fsdp_cpu_offload,
            fsdp_use_orig_params=fsdp_use_orig_params,
            save_optimizer_state=save_optimizer_state,
            debug_stage_logs=debug_stage_logs,
            debug_all_ranks=debug_all_ranks,
            noise_level=noise_level,
        )
    )


def run_sft_training(args: Args) -> None:
    run(args)


if __name__ == "__main__":
    tyro.cli(main)

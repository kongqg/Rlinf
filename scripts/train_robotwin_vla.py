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
import json
import os
import random
import shutil
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Literal

# Force transformers/openpi onto the torch path before importing any model code.
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
from omegaconf import OmegaConf
import pyarrow as pa
import pyarrow.parquet as pq
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

from bentele.utils.robotwin_eval import run_robotwin_policy_eval, save_eval_result
import openpi.shared.normalize as normalize
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.utils.pytree import register_pytree_dataclasses


REPO_ROOT = Path(__file__).resolve().parents[1]

FPS = 30.0
CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


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
    train_steps: int = 2000
    learning_rate: float = 5.0e-6
    weight_decay: float = 1.0e-2
    clip_grad_norm: float = 1.0
    log_every: int = 20
    save_every: int = 200
    num_workers: int = 0
    seed: int = 1234
    device: str = "cuda"
    use_quantile_norm: bool = False
    assets_path: str | None = None
    eval_every: int = 0
    eval_num_envs: int = 1
    eval_rollout_epochs: int = 1
    eval_seeds_path: str | None = None
    eval_max_chunk_steps: int | None = None
    eval_action_exec_horizon: int | None = None
    eval_visualize: bool = False
    eval_render_freq: int = 10
    eval_sleep_between_chunks: float = 0.0
    eval_hold_viewer_seconds: float = 0.0
    eval_device: Literal["cpu", "cuda", "auto"] = "cpu"
    eval_debug_log_chunks: bool = False
    eval_debug_action_steps: int = 0
    action_probe_every: int = 100
    action_probe_num_samples: int = 5
    wandb_enabled: bool = False
    wandb_project: str = "bentele"
    wandb_run_name: str | None = None
    wandb_log_dir: str | None = None
    wandb_proxy: str | None = None
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_tags: tuple[str, ...] = ()
    distributed_backend: Literal["none", "fsdp"] = "none"
    fsdp_sharding_strategy: Literal["full_shard", "shard_grad_op"] = "full_shard"
    fsdp_mixed_precision: Literal["bf16", "fp32"] = "bf16"
    fsdp_cpu_offload: bool = False
    fsdp_use_orig_params: bool = True
    save_optimizer_state: bool = False
    debug_stage_logs: bool = True
    debug_all_ranks: bool = False


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


class _WandbLogger:
    def __init__(self, run: Any):
        self.run = run

    def log(self, data: dict[str, Any], step: int) -> None:
        self.run.log(data, step=step)

    def finish(self) -> None:
        self.run.finish()


@dataclass(frozen=True)
class RolloutProbeSample:
    repo_id: str
    dataset_item: int
    prompt: str
    state: np.ndarray
    target_actions: np.ndarray
    main_image: np.ndarray
    wrist_images: np.ndarray


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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_of(device: str, dist_ctx: DistributedContext) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if dist_ctx.is_distributed and device == "cuda":
        return torch.device("cuda", dist_ctx.local_rank)
    return torch.device(device)


def _init_distributed(args: Args) -> DistributedContext:
    if args.distributed_backend == "none":
        return DistributedContext(backend="none")

    if args.distributed_backend != "fsdp":
        raise ValueError(f"Unsupported distributed backend: {args.distributed_backend}")
    if not torch.cuda.is_available():
        raise RuntimeError("FSDP training requires CUDA.")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        raise RuntimeError(
            "distributed_backend=fsdp requires torchrun. "
            "Example: torchrun --standalone --nproc_per_node=8 scripts/train_robotwin_vla.py ..."
        )

    if not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    return DistributedContext(
        backend="fsdp",
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )


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


class RoboTwinV3LocalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        action_horizon: int,
        *,
        episode_indices: Sequence[int] | None = None,
        episode_limit: int | None = None,
        max_samples: int | None = None,
    ):
        self.dataset_dir = dataset_dir
        self.action_horizon = action_horizon
        self._video_handles: dict[str, Any] = {}
        self._last_frame_index: dict[str, int] = {}

        data_tables = []
        for parquet_path in sorted((dataset_dir / "data").glob("chunk-*/file-*.parquet")):
            data_tables.append(pq.read_table(parquet_path))
        if not data_tables:
            raise FileNotFoundError(f"No parquet files found under {dataset_dir / 'data'}")
        merged = data_tables[0] if len(data_tables) == 1 else pa.concat_tables(data_tables)

        self.states = np.stack(merged.column("observation.state").to_pylist()).astype(np.float32)
        self.actions = np.stack(merged.column("action").to_pylist()).astype(np.float32)
        self.frame_index = np.asarray(merged.column("frame_index")).astype(np.int64)
        self.episode_index = np.asarray(merged.column("episode_index")).astype(np.int64)

        episode_tables = []
        for parquet_path in sorted((dataset_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
            episode_tables.append(pq.read_table(parquet_path))
        if not episode_tables:
            raise FileNotFoundError(f"No episode parquet files found under {dataset_dir / 'meta' / 'episodes'}")
        episodes = episode_tables[0] if len(episode_tables) == 1 else pa.concat_tables(episode_tables)

        self.episode_meta: dict[int, dict[str, Any]] = {}
        for row in episodes.to_pylist():
            episode_idx = int(row["episode_index"])
            self.episode_meta[episode_idx] = row

        if episode_indices and episode_limit is not None:
            raise ValueError("episode_indices and episode_limit cannot be used together.")

        available_episode_indices = sorted(self.episode_meta)
        if episode_indices:
            selected_episode_indices = sorted({int(idx) for idx in episode_indices})
            missing_episode_indices = [
                idx for idx in selected_episode_indices if idx not in self.episode_meta
            ]
            if missing_episode_indices:
                raise ValueError(
                    "Requested episode indices are missing from dataset "
                    f"{dataset_dir}: {missing_episode_indices}"
                )
        elif episode_limit is not None:
            if episode_limit <= 0:
                raise ValueError(f"episode_limit must be positive, but got {episode_limit}.")
            selected_episode_indices = available_episode_indices[:episode_limit]
        else:
            selected_episode_indices = available_episode_indices
        if not selected_episode_indices:
            raise ValueError(f"No episodes selected from dataset {dataset_dir}.")
        self.selected_episode_indices = selected_episode_indices

        self.sample_indices: list[int] = []
        self.samples_per_episode: dict[int, int] = {}
        for episode_idx in self.selected_episode_indices:
            meta = self.episode_meta[episode_idx]
            start = int(meta["dataset_from_index"])
            stop = int(meta["dataset_to_index"])
            last_valid = stop - self.action_horizon + 1
            if last_valid <= start:
                self.samples_per_episode[episode_idx] = 0
                continue
            episode_sample_indices = list(range(start, last_valid))
            self.samples_per_episode[episode_idx] = len(episode_sample_indices)
            self.sample_indices.extend(episode_sample_indices)
        if max_samples is not None and max_samples > 0:
            self.sample_indices = self.sample_indices[:max_samples]
        if not self.sample_indices:
            raise ValueError(
                f"No valid samples found in dataset {dataset_dir} for "
                f"episodes={self.selected_episode_indices} with action_horizon={action_horizon}."
            )

    def __len__(self) -> int:
        return len(self.sample_indices)

    def _open_video(self, video_path: Path):
        import av

        key = str(video_path)
        if key not in self._video_handles:
            container = av.open(key)
            stream = container.streams.video[0]
            self._video_handles[key] = (container, stream)
            self._last_frame_index[key] = -1
        return self._video_handles[key]

    def _read_video_frame(self, video_path: Path, frame_index: int) -> np.ndarray:
        container, stream = self._open_video(video_path)
        key = str(video_path)
        # The torch DataLoader may revisit the same sample or move backward in
        # time after shuffling. Reopen the container whenever access is not
        # strictly increasing to keep `__getitem__` idempotent.
        if frame_index <= self._last_frame_index[key]:
            container.close()
            del self._video_handles[key]
            container, stream = self._open_video(video_path)

        if frame_index > self._last_frame_index[key] + 1:
            avg_rate = float(stream.average_rate)
            time_base = float(stream.time_base)
            target_pts = int(frame_index / avg_rate / time_base)
            container.seek(target_pts, stream=stream, any_frame=False, backward=True)

        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            current_index = int(
                round(float(frame.pts) * float(stream.time_base) * float(stream.average_rate))
            )
            if current_index < frame_index:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            self._last_frame_index[key] = current_index
            if current_index == frame_index:
                return rgb

        raise RuntimeError(f"Failed to decode frame {frame_index} from {video_path}")

    def __getitem__(self, item: int) -> dict[str, Any]:
        sample_idx = self.sample_indices[item]
        episode_idx = int(self.episode_index[sample_idx])
        meta = self.episode_meta[episode_idx]

        prompt = meta["tasks"][0]
        obs_state = self.states[sample_idx]
        action_seq = self.actions[sample_idx : sample_idx + self.action_horizon]
        local_frame_index = int(self.frame_index[sample_idx])

        sample = {
            "observation.state": obs_state,
            "action": action_seq,
            "prompt": prompt,
        }

        for camera_key in CAMERA_KEYS:
            chunk_idx = int(meta[f"videos/{camera_key}/chunk_index"])
            file_idx = int(meta[f"videos/{camera_key}/file_index"])
            from_timestamp = float(meta[f"videos/{camera_key}/from_timestamp"])
            video_path = (
                self.dataset_dir
                / "videos"
                / camera_key
                / f"chunk-{chunk_idx:03d}"
                / f"file-{file_idx:03d}.mp4"
            )
            absolute_frame_index = int(round(from_timestamp * FPS)) + local_frame_index
            sample[camera_key] = self._read_video_frame(video_path, absolute_frame_index)

        return sample


def _evenly_spaced_positions(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    positions = np.linspace(0, length - 1, num=count, dtype=int)
    return [int(pos) for pos in positions.tolist()]


def _build_rollout_probe_samples(
    args: Args,
    *,
    action_horizon: int,
) -> list[RolloutProbeSample]:
    if args.action_probe_num_samples <= 0:
        return []

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    raw_datasets: list[tuple[str, RoboTwinV3LocalDataset]] = []
    for repo_id in args.train_repo_ids:
        raw_dataset = RoboTwinV3LocalDataset(
            dataset_root / repo_id,
            action_horizon=action_horizon,
            episode_indices=args.train_episode_indices or None,
            episode_limit=(
                args.train_num_episodes_limit
                if args.train_num_episodes_limit > 0
                else None
            ),
            max_samples=args.train_sample_limit if args.train_sample_limit > 0 else None,
        )
        if len(raw_dataset) > 0:
            raw_datasets.append((repo_id, raw_dataset))
    if not raw_datasets:
        return []

    per_repo_target = max(
        1,
        int(np.ceil(args.action_probe_num_samples / len(raw_datasets))),
    )
    selected_positions: list[tuple[str, RoboTwinV3LocalDataset, int]] = []
    for repo_id, raw_dataset in raw_datasets:
        for dataset_item in _evenly_spaced_positions(len(raw_dataset), per_repo_target):
            selected_positions.append((repo_id, raw_dataset, dataset_item))

    probe_samples: list[RolloutProbeSample] = []
    for repo_id, raw_dataset, dataset_item in selected_positions[: args.action_probe_num_samples]:
        sample = raw_dataset[dataset_item]
        wrist_images = np.stack(
            [
                sample["observation.images.cam_left_wrist"],
                sample["observation.images.cam_right_wrist"],
            ],
            axis=0,
        ).astype(np.uint8)
        probe_samples.append(
            RolloutProbeSample(
                repo_id=repo_id,
                dataset_item=dataset_item,
                prompt=sample["prompt"],
                state=sample["observation.state"].astype(np.float32),
                target_actions=sample["action"].astype(np.float32),
                main_image=sample["observation.images.cam_high"].astype(np.uint8),
                wrist_images=wrist_images,
            )
        )
    return probe_samples


def _build_probe_env_obs(
    sample: RolloutProbeSample,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "main_images": torch.from_numpy(sample.main_image).unsqueeze(0).to(device=device),
        "wrist_images": torch.from_numpy(sample.wrist_images).unsqueeze(0).to(device=device),
        "states": torch.from_numpy(sample.state).unsqueeze(0).to(
            device=device,
            dtype=torch.float32,
        ),
        "task_descriptions": [sample.prompt],
    }


def _compute_rollout_probe_metrics(
    model: torch.nn.Module,
    probe_samples: Sequence[RolloutProbeSample],
    device: torch.device,
) -> dict[str, float]:
    if not probe_samples:
        return {}

    target_model = getattr(model, "module", model)
    was_training = target_model.training
    target_model.eval()

    maes = []
    mses = []
    pred_abs_means = []
    target_abs_means = []
    try:
        with torch.no_grad():
            for sample in probe_samples:
                env_obs = _build_probe_env_obs(sample, device)
                predicted_actions, _ = target_model.predict_action_batch(
                    env_obs,
                    mode="eval",
                    compute_values=False,
                )
                predicted_actions = predicted_actions[0].detach().cpu().float()
                target_actions = torch.from_numpy(sample.target_actions).float()
                diff = predicted_actions - target_actions
                maes.append(float(diff.abs().mean().item()))
                mses.append(float(diff.square().mean().item()))
                pred_abs_means.append(float(predicted_actions.abs().mean().item()))
                target_abs_means.append(float(target_actions.abs().mean().item()))
    finally:
        if was_training:
            target_model.train()

    return {
        "probe/raw_train_rollout_action_mae": float(np.mean(maes)),
        "probe/raw_train_rollout_action_mse": float(np.mean(mses)),
        "probe/raw_train_rollout_pred_abs_mean": float(np.mean(pred_abs_means)),
        "probe/raw_train_rollout_target_abs_mean": float(np.mean(target_abs_means)),
        "probe/raw_train_rollout_num_samples": float(len(probe_samples)),
    }


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
    model.to(dtype=torch.float32)

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
    model = FSDP(
        model,
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
            "openpi": {
                "config_name": args.config_name,
                "num_images_in_input": 3,
                "noise_level": 0.3,
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


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


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
    print(
        "[train_robotwin_vla] "
        f"eval step={step} "
        f"success_at_end={metrics.get('success_at_end', 0.0)} "
        f"success_once={metrics.get('success_once', 0.0)} "
        f"return={metrics.get('return', 0.0)} "
        f"episode_len={metrics.get('episode_len', 0.0)} "
        f"runtime_seconds={eval_runtime:.2f}"
    )
    if wandb_logger is not None:
        wandb_logger.log(
            {
                "eval/success_at_end": float(metrics.get("success_at_end", 0.0)),
                "eval/success_once": float(metrics.get("success_once", 0.0)),
                "eval/return": float(metrics.get("return", 0.0)),
                "eval/reward": float(metrics.get("reward", 0.0)),
                "eval/episode_len": float(metrics.get("episode_len", 0.0)),
                "eval/num_trajectories": float(metrics.get("num_trajectories", 0.0)),
                "eval/runtime_seconds": float(eval_runtime),
            },
            step=step,
        )
    if dist_ctx.is_distributed:
        dist.barrier()
    return result


def _reduce_mean(value: torch.Tensor, dist_ctx: DistributedContext) -> torch.Tensor:
    if not dist_ctx.is_distributed:
        return value
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist_ctx.world_size
    return reduced


def run(args: Args) -> None:
    dist_ctx = _init_distributed(args)
    _log_stage(
        f"initialized distributed backend={dist_ctx.backend} world_size={dist_ctx.world_size} local_rank={dist_ctx.local_rank}",
        dist_ctx,
        enabled=args.debug_stage_logs,
        all_ranks=args.debug_all_ranks,
    )
    _set_seed(args.seed)
    device = _device_of(args.device, dist_ctx)
    _log_stage(
        f"using device={device}",
        dist_ctx,
        enabled=args.debug_stage_logs,
        all_ranks=args.debug_all_ranks,
    )

    _log_stage(
        "building data loader",
        dist_ctx,
        enabled=args.debug_stage_logs,
        all_ranks=args.debug_all_ranks,
    )
    train_config, data_loader = _build_loader(args, dist_ctx)
    _log_stage(
        "data loader ready",
        dist_ctx,
        enabled=args.debug_stage_logs,
        all_ranks=args.debug_all_ranks,
    )
    rollout_probe_samples = _build_rollout_probe_samples(
        args,
        action_horizon=train_config.model.action_horizon,
    )
    if dist_ctx.is_main and rollout_probe_samples:
        probe_preview = [
            {
                "repo_id": sample.repo_id,
                "dataset_item": sample.dataset_item,
                "prompt": sample.prompt,
            }
            for sample in rollout_probe_samples[: min(3, len(rollout_probe_samples))]
        ]
        print(
            "[train_robotwin_vla] "
            f"rollout_probe_num_samples={len(rollout_probe_samples)} "
            f"rollout_probe_every={args.action_probe_every} "
            f"rollout_probe_preview={json.dumps(probe_preview, ensure_ascii=False)}"
        )
    _log_stage(
        "building model",
        dist_ctx,
        enabled=args.debug_stage_logs,
        all_ranks=args.debug_all_ranks,
    )
    model = _build_model(args, device, dist_ctx)
    _log_stage(
        "model ready",
        dist_ctx,
        enabled=args.debug_stage_logs,
        all_ranks=args.debug_all_ranks,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1.0e-8,
    )

    amp_enabled = device.type == "cuda"
    loader_iter = iter(data_loader)
    optimizer.zero_grad(set_to_none=True)

    output_dir = Path(args.output_dir).expanduser().resolve()
    wandb_logger = _init_wandb_logger(args, dist_ctx, output_dir)
    if args.eval_every > 0 and args.assets_path is None:
        raise ValueError("assets_path must be set when eval_every > 0.")

    failure: BaseException | None = None
    try:
        for step in range(1, args.train_steps + 1):
            _log_stage(
                f"step {step}: fetching batch",
                dist_ctx,
                enabled=args.debug_stage_logs,
                all_ranks=args.debug_all_ranks,
            )
            observation, actions = next(loader_iter)
            _log_stage(
                f"step {step}: batch fetched",
                dist_ctx,
                enabled=args.debug_stage_logs,
                all_ranks=args.debug_all_ranks,
            )
            observation = _move_observation_to_device(observation, device)
            actions = actions.to(torch.float32).to(device, non_blocking=True)

            _log_stage(
                f"step {step}: forward start",
                dist_ctx,
                enabled=args.debug_stage_logs,
                all_ranks=args.debug_all_ranks,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                losses = model(
                    data={"observation": observation, "actions": actions},
                    forward_type=ForwardType.SFT,
                )
                loss = _loss_from_output(losses, device)
            _log_stage(
                f"step {step}: forward done loss={loss.detach().item():.6f}",
                dist_ctx,
                enabled=args.debug_stage_logs,
                all_ranks=args.debug_all_ranks,
            )

            (loss / args.grad_accum_steps).backward()
            _log_stage(
                f"step {step}: backward done",
                dist_ctx,
                enabled=args.debug_stage_logs,
                all_ranks=args.debug_all_ranks,
            )

            should_step = (step % args.grad_accum_steps == 0) or (step == args.train_steps)
            if should_step:
                _log_stage(
                    f"step {step}: optimizer step start",
                    dist_ctx,
                    enabled=args.debug_stage_logs,
                    all_ranks=args.debug_all_ranks,
                )
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.clip_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                _log_stage(
                    f"step {step}: optimizer step done",
                    dist_ctx,
                    enabled=args.debug_stage_logs,
                    all_ranks=args.debug_all_ranks,
                )
            else:
                grad_norm = None

            rollout_probe_metrics: dict[str, float] = {}
            should_probe_actions = (
                dist_ctx.is_main
                and bool(rollout_probe_samples)
                and args.action_probe_every > 0
                and (
                    step == 1
                    or step % args.action_probe_every == 0
                    or step == args.train_steps
                )
            )
            if should_probe_actions:
                _log_stage(
                    f"step {step}: rollout probe start",
                    dist_ctx,
                    enabled=args.debug_stage_logs,
                    all_ranks=False,
                )
                rollout_probe_metrics = _compute_rollout_probe_metrics(
                    model,
                    rollout_probe_samples,
                    device,
                )
                _log_stage(
                    f"step {step}: rollout probe done mae="
                    f"{rollout_probe_metrics.get('probe/raw_train_rollout_action_mae', float('nan')):.6f}",
                    dist_ctx,
                    enabled=args.debug_stage_logs,
                    all_ranks=False,
                )

            if step % args.log_every == 0 or step == 1:
                reduced_loss = _reduce_mean(loss.detach(), dist_ctx)
                train_log_payload: dict[str, float] = {
                    "train/loss": float(reduced_loss),
                    "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
                train_log_payload.update(rollout_probe_metrics)
                grad_text = "accumulating"
                if grad_norm is not None:
                    reduced_grad = _reduce_mean(grad_norm.detach().float(), dist_ctx)
                    grad_text = f"{float(reduced_grad):.4f}"
                    train_log_payload["train/grad_norm"] = float(reduced_grad)
                probe_text = ""
                if rollout_probe_metrics:
                    probe_text = (
                        " "
                        f"rollout_action_mae="
                        f"{rollout_probe_metrics['probe/raw_train_rollout_action_mae']:.6f}"
                    )
                if dist_ctx.is_main:
                    print(
                        f"[train_robotwin_vla] step={step}/{args.train_steps} "
                        f"loss={float(reduced_loss):.6f} grad_norm={grad_text}{probe_text}"
                    )
                    if wandb_logger is not None:
                        wandb_logger.log(train_log_payload, step=step)

            checkpoint_for_eval: Path | None = None
            if step % args.save_every == 0 or step == args.train_steps:
                _log_stage(
                    f"step {step}: checkpoint save start",
                    dist_ctx,
                    enabled=args.debug_stage_logs,
                    all_ranks=args.debug_all_ranks,
                )
                _save_checkpoint(
                    output_dir=output_dir,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    args=args,
                    data_config=data_loader.data_config(),
                    dist_ctx=dist_ctx,
                )
                _log_stage(
                    f"step {step}: checkpoint save done",
                    dist_ctx,
                    enabled=args.debug_stage_logs,
                    all_ranks=args.debug_all_ranks,
                )
                if dist_ctx.is_main:
                    print(f"[train_robotwin_vla] saved checkpoint to {output_dir}")
                checkpoint_for_eval = output_dir

            should_eval = args.eval_every > 0 and (
                step % args.eval_every == 0 or step == args.train_steps
            )
            if should_eval:
                if checkpoint_for_eval is None:
                    checkpoint_for_eval = _export_eval_model_dir(
                        eval_dir=output_dir / "_eval_runtime" / "latest",
                        model=model,
                        step=step,
                        args=args,
                        data_config=data_loader.data_config(),
                        dist_ctx=dist_ctx,
                    )
                _run_periodic_eval(
                    args=args,
                    step=step,
                    model_dir=checkpoint_for_eval or (output_dir / "_eval_runtime" / "latest"),
                    output_root=output_dir,
                    dist_ctx=dist_ctx,
                    wandb_logger=wandb_logger,
                )
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if wandb_logger is not None:
            wandb_logger.finish()
        _log_stage(
            "training shutdown",
            dist_ctx,
            enabled=args.debug_stage_logs,
            all_ranks=args.debug_all_ranks,
        )
        if dist_ctx.is_distributed and dist.is_initialized():
            if failure is None:
                dist.barrier()
            dist.destroy_process_group()


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
    train_steps: int = 2000,
    learning_rate: float = 5.0e-6,
    weight_decay: float = 1.0e-2,
    clip_grad_norm: float = 1.0,
    log_every: int = 20,
    save_every: int = 200,
    num_workers: int = 0,
    seed: int = 1234,
    device: str = "cuda",
    use_quantile_norm: bool = False,
    assets_path: str | None = None,
    eval_every: int = 0,
    eval_num_envs: int = 1,
    eval_rollout_epochs: int = 1,
    eval_seeds_path: str | None = None,
    eval_max_chunk_steps: int | None = None,
    eval_action_exec_horizon: int | None = None,
    eval_visualize: bool = False,
    eval_render_freq: int = 10,
    eval_sleep_between_chunks: float = 0.0,
    eval_hold_viewer_seconds: float = 0.0,
    eval_device: Literal["cpu", "cuda", "auto"] = "cpu",
    eval_debug_log_chunks: bool = False,
    eval_debug_action_steps: int = 0,
    action_probe_every: int = 100,
    action_probe_num_samples: int = 5,
    wandb_enabled: bool = False,
    wandb_project: str = "bentele",
    wandb_run_name: str | None = None,
    wandb_log_dir: str | None = None,
    wandb_proxy: str | None = None,
    wandb_mode: Literal["online", "offline", "disabled"] = "online",
    wandb_tags: tuple[str, ...] = (),
    distributed_backend: Literal["none", "fsdp"] = "none",
    fsdp_sharding_strategy: Literal["full_shard", "shard_grad_op"] = "full_shard",
    fsdp_mixed_precision: Literal["bf16", "fp32"] = "bf16",
    fsdp_cpu_offload: bool = False,
    fsdp_use_orig_params: bool = True,
    save_optimizer_state: bool = False,
    debug_stage_logs: bool = True,
    debug_all_ranks: bool = False,
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
            weight_decay=weight_decay,
            clip_grad_norm=clip_grad_norm,
            log_every=log_every,
            save_every=save_every,
            num_workers=num_workers,
            seed=seed,
            device=device,
            use_quantile_norm=use_quantile_norm,
            assets_path=assets_path,
            eval_every=eval_every,
            eval_num_envs=eval_num_envs,
            eval_rollout_epochs=eval_rollout_epochs,
            eval_seeds_path=eval_seeds_path,
            eval_max_chunk_steps=eval_max_chunk_steps,
            eval_action_exec_horizon=eval_action_exec_horizon,
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
            wandb_log_dir=wandb_log_dir,
            wandb_proxy=wandb_proxy,
            wandb_mode=wandb_mode,
            wandb_tags=wandb_tags,
            distributed_backend=distributed_backend,
            fsdp_sharding_strategy=fsdp_sharding_strategy,
            fsdp_mixed_precision=fsdp_mixed_precision,
            fsdp_cpu_offload=fsdp_cpu_offload,
            fsdp_use_orig_params=fsdp_use_orig_params,
            save_optimizer_state=save_optimizer_state,
            debug_stage_logs=debug_stage_logs,
            debug_all_ranks=debug_all_ranks,
        )
    )


if __name__ == "__main__":
    tyro.cli(main)

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.projects.robotwin.data.robotwin_v3_dataset import RoboTwinV3LocalDataset
from rlinf.projects.robotwin.training.sft.args import Args


@dataclass(frozen=True)
class RolloutProbeSample:
    repo_id: str
    dataset_item: int
    prompt: str
    state: np.ndarray
    target_actions: np.ndarray
    main_image: np.ndarray
    wrist_images: np.ndarray


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


__all__ = [
    "RolloutProbeSample",
    "_build_probe_env_obs",
    "_build_rollout_probe_samples",
    "_compute_rollout_probe_metrics",
    "_evenly_spaced_positions",
]

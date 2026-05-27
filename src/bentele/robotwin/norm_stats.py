"""Compute OpenPI-compatible norm stats for RoboTwin ALOHA datasets.

This script computes normalization statistics directly from local RoboTwin
LeRobot-v3.0 parquet files. It mirrors the important input-side transforms used
by the RLinf/OpenPI RoboTwin ALOHA pipeline:

- convert ALOHA state into Pi's internal state space
- convert ALOHA actions into Pi's internal action space
- convert absolute joint actions to delta joint actions

Unlike the official OpenPI script, this version does not depend on LeRobot's
metadata loader, which expects a different on-disk metadata layout.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads
import tqdm
import tyro

import openpi.shared.normalize as normalize
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.policies import aloha_policy


def _decode_state_batch(state: np.ndarray, *, adapt_to_pi: bool) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).copy()
    if adapt_to_pi:
        state = state * aloha_policy._joint_flip_mask()
        state[:, [6, 13]] = aloha_policy._gripper_to_angular(state[:, [6, 13]])
    return state


def _encode_actions_inv_batch(actions: np.ndarray, *, adapt_to_pi: bool) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32).copy()
    if adapt_to_pi:
        actions = actions * aloha_policy._joint_flip_mask()
        actions[:, [6, 13]] = aloha_policy._gripper_from_angular_inv(
            actions[:, [6, 13]]
        )
    return actions


def _delta_actions_batch(
    state: np.ndarray, actions: np.ndarray, *, extra_delta_transform: bool
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32).copy()
    if extra_delta_transform:
        delta_action_mask = np.array(
            [True] * 6 + [False] + [True] * 6 + [False],
            dtype=bool,
        )
        actions[:, delta_action_mask] -= state[:, delta_action_mask]
    return actions


def _iter_state_action_batches(
    dataset_dir: Path, *, batch_size: int, max_frames: int | None = None
):
    data_dir = dataset_dir / "data"
    dataset = pads.dataset(str(data_dir), format="parquet")
    scanner = dataset.scanner(
        columns=["observation.state", "action"],
        batch_size=batch_size,
    )
    remaining = max_frames
    for batch in scanner.to_batches():
        states = np.stack(batch.column("observation.state").to_pylist()).astype(np.float32)
        actions = np.stack(batch.column("action").to_pylist()).astype(np.float32)
        if remaining is not None:
            if remaining <= 0:
                break
            states = states[:remaining]
            actions = actions[:remaining]
            remaining -= len(states)
        if len(states) == 0:
            continue
        yield states, actions


@dataclasses.dataclass(frozen=True)
class Args:
    dataset_dirs: Sequence[str]
    output_dir: str
    config_name: str = "pi05_aloha_robotwin"
    batch_size: int = 512
    max_frames_per_dataset: int | None = None
    use_quantile_norm: bool = False


def run(args: Args) -> None:
    train_config = get_openpi_config(
        args.config_name,
        data_kwargs={"use_quantile_norm": args.use_quantile_norm},
    )
    data_cfg = train_config.data
    adapt_to_pi = getattr(data_cfg, "adapt_to_pi", True)
    extra_delta_transform = getattr(data_cfg, "extra_delta_transform", True)

    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }

    for dataset_dir_str in args.dataset_dirs:
        dataset_dir = Path(dataset_dir_str).expanduser().resolve()
        desc = f"Computing stats from {dataset_dir.name}"
        iterator = _iter_state_action_batches(
            dataset_dir,
            batch_size=args.batch_size,
            max_frames=args.max_frames_per_dataset,
        )
        for raw_state, raw_action in tqdm.tqdm(iterator, desc=desc):
            state = _decode_state_batch(raw_state, adapt_to_pi=adapt_to_pi)
            actions = _encode_actions_inv_batch(raw_action, adapt_to_pi=adapt_to_pi)
            actions = _delta_actions_batch(
                state, actions, extra_delta_transform=extra_delta_transform
            )
            stats["state"].update(state)
            stats["actions"].update(actions)

    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    output_dir = Path(args.output_dir).expanduser().resolve()
    print(f"Writing stats to: {output_dir}")
    normalize.save(output_dir, norm_stats)


def main(
    dataset_dirs: Sequence[str],
    output_dir: str,
    config_name: str = "pi05_aloha_robotwin",
    batch_size: int = 512,
    max_frames_per_dataset: int | None = None,
    use_quantile_norm: bool = False,
) -> None:
    run(
        Args(
            dataset_dirs=dataset_dirs,
            output_dir=output_dir,
            config_name=config_name,
            batch_size=batch_size,
            max_frames_per_dataset=max_frames_per_dataset,
            use_quantile_norm=use_quantile_norm,
        )
    )


if __name__ == "__main__":
    tyro.cli(main)

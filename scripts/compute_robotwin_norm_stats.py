"""Compute OpenPI-compatible norm stats for RoboTwin ALOHA datasets.

This script computes normalization statistics directly from local RoboTwin
LeRobot-v3.0 parquet files. It mirrors the important input-side transforms used
by the RLinf/OpenPI RoboTwin ALOHA pipeline:

- convert ALOHA state into Pi's internal state space
- convert ALOHA actions into Pi's internal action space
- build the same fixed-horizon action chunks used by training
- convert absolute joint action chunks to delta joint actions

Unlike the official OpenPI script, this version does not depend on LeRobot's
metadata loader, which expects a different on-disk metadata layout.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
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
        actions[..., [6, 13]] = aloha_policy._gripper_from_angular_inv(
            actions[..., [6, 13]]
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
        dims = delta_action_mask.shape[-1]
        state_delta = np.where(delta_action_mask, state[..., :dims], 0)
        if actions.ndim == state.ndim + 1:
            state_delta = np.expand_dims(state_delta, axis=-2)
        actions[..., :dims] -= state_delta
    return actions


def _load_dataset_table(dataset_dir: Path) -> pa.Table:
    data_dir = dataset_dir / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset data directory does not exist: {data_dir}")
    dataset = pads.dataset(str(data_dir), format="parquet")
    table = dataset.to_table(columns=["observation.state", "action"])
    if table.num_rows == 0:
        raise ValueError(f"No rows found under {data_dir}")
    return table


def _load_episode_rows(dataset_dir: Path) -> list[dict]:
    episodes_dir = dataset_dir / "meta" / "episodes"
    episode_paths = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(f"No episode parquet files found under {episodes_dir}")

    tables = [pq.read_table(path) for path in episode_paths]
    episodes = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    return sorted(episodes.to_pylist(), key=lambda row: int(row["episode_index"]))


def _valid_sample_indices(
    dataset_dir: Path,
    *,
    num_frames: int,
    action_horizon: int,
) -> list[int]:
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}.")

    indices: list[int] = []
    for row in _load_episode_rows(dataset_dir):
        start = int(row["dataset_from_index"])
        stop = int(row["dataset_to_index"])
        if start < 0 or stop > num_frames:
            raise ValueError(
                "Episode range is outside dataset frame count: "
                f"start={start} stop={stop} num_frames={num_frames}"
            )
        last_valid = stop - action_horizon + 1
        if last_valid <= start:
            continue
        indices.extend(range(start, last_valid))
    if not indices:
        raise ValueError(
            f"No valid training samples found in {dataset_dir} for action_horizon={action_horizon}."
        )
    return indices


def _iter_state_action_chunk_batches(
    dataset_dir: Path,
    *,
    action_horizon: int,
    batch_size: int,
    max_samples: int | None = None,
):
    table = _load_dataset_table(dataset_dir)
    states = np.stack(table.column("observation.state").to_pylist()).astype(np.float32)
    actions = np.stack(table.column("action").to_pylist()).astype(np.float32)
    sample_indices = _valid_sample_indices(
        dataset_dir,
        num_frames=len(states),
        action_horizon=action_horizon,
    )
    if max_samples is not None:
        sample_indices = sample_indices[:max_samples]

    for offset in range(0, len(sample_indices), batch_size):
        batch_indices = sample_indices[offset : offset + batch_size]
        if not batch_indices:
            continue
        batch_states = states[batch_indices]
        batch_actions = np.stack(
            [actions[index : index + action_horizon] for index in batch_indices],
            axis=0,
        ).astype(np.float32)
        yield batch_states, batch_actions


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
    action_horizon = int(getattr(train_config.model, "action_horizon", 1))

    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }

    for dataset_dir_str in args.dataset_dirs:
        dataset_dir = Path(dataset_dir_str).expanduser().resolve()
        desc = f"Computing stats from {dataset_dir.name}"
        iterator = _iter_state_action_chunk_batches(
            dataset_dir,
            action_horizon=action_horizon,
            batch_size=args.batch_size,
            max_samples=args.max_frames_per_dataset,
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

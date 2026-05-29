from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


FPS = 30.0
CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
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
        # OpenPI transforms mutate arrays in-place, so return copies rather than
        # views into the cached parquet arrays.
        obs_state = self.states[sample_idx].copy()
        action_seq = self.actions[
            sample_idx : sample_idx + self.action_horizon
        ].copy()
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


__all__ = ["CAMERA_KEYS", "FPS", "RoboTwinV3LocalDataset"]

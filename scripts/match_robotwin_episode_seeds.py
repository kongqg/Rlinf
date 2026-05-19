#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


FPS = 30.0
CAMERA_KEY_MAP = {
    "full_image": "observation.images.cam_high",
    "left_wrist_image": "observation.images.cam_left_wrist",
    "right_wrist_image": "observation.images.cam_right_wrist",
}


def _ensure_robotwin_import_path(robotwin_path: Path) -> None:
    robotwin_path = robotwin_path.expanduser().resolve()
    os.environ.setdefault("ROBOTWIN_PATH", str(robotwin_path))
    os.environ.setdefault("ASSETS_PATH", str(robotwin_path))
    if str(robotwin_path) not in sys.path:
        sys.path.insert(0, str(robotwin_path))


def _load_episode_meta(dataset_dir: Path) -> dict[int, dict[str, Any]]:
    episode_tables = []
    for parquet_path in sorted((dataset_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        episode_tables.append(pq.read_table(parquet_path))
    if not episode_tables:
        raise FileNotFoundError(f"No episode parquet files found under {dataset_dir / 'meta' / 'episodes'}")
    episodes = episode_tables[0] if len(episode_tables) == 1 else pa.concat_tables(episode_tables)
    return {int(row["episode_index"]): row for row in episodes.to_pylist()}


def _read_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    try:
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            current_index = int(
                round(float(frame.pts) * float(stream.time_base) * float(stream.average_rate))
            )
            if current_index < frame_index:
                continue
            return frame.to_ndarray(format="rgb24")
    finally:
        container.close()
    raise RuntimeError(f"Failed to decode frame {frame_index} from {video_path}")


def _load_episode_reference_images(
    dataset_dir: Path,
    episode_meta: dict[int, dict[str, Any]],
    episode_index: int,
) -> dict[str, np.ndarray]:
    meta = episode_meta[episode_index]
    images = {}
    for obs_key, camera_key in CAMERA_KEY_MAP.items():
        chunk_idx = int(meta[f"videos/{camera_key}/chunk_index"])
        file_idx = int(meta[f"videos/{camera_key}/file_index"])
        from_timestamp = float(meta[f"videos/{camera_key}/from_timestamp"])
        local_frame_index = 0
        absolute_frame_index = int(round(from_timestamp * FPS)) + local_frame_index
        video_path = (
            dataset_dir
            / "videos"
            / camera_key
            / f"chunk-{chunk_idx:03d}"
            / f"file-{file_idx:03d}.mp4"
        )
        images[obs_key] = _read_video_frame(video_path, absolute_frame_index)
    return images


def _build_task_config(
    *,
    robotwin_path: Path,
    task_config_path: Path,
    task_name: str,
    planner_backend: str,
) -> dict[str, Any]:
    with task_config_path.expanduser().resolve().open("r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["task_name"] = task_name
    args["planner_backend"] = planner_backend
    args.setdefault("step_lim", 200)
    return args


def _load_candidate_seeds(seeds_path: Path, task_name: str, limit: int) -> list[int]:
    data = json.loads(seeds_path.expanduser().resolve().read_text())
    seeds = list(map(int, data[task_name]["success_seeds"]))
    if limit > 0:
        seeds = seeds[:limit]
    if not seeds:
        raise ValueError(f"No candidate seeds found in {seeds_path} for task {task_name}.")
    return seeds


def _score_obs(
    env_obs: dict[str, np.ndarray],
    dataset_images: dict[str, np.ndarray],
) -> tuple[float, dict[str, float]]:
    total = 0.0
    details: dict[str, float] = {}
    for obs_key in CAMERA_KEY_MAP:
        env_img = env_obs[obs_key]
        ds_img = dataset_images[obs_key]
        ds_resized = cv2.resize(
            ds_img,
            (env_img.shape[1], env_img.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        diff = float(np.mean(np.abs(env_img.astype(np.float32) - ds_resized.astype(np.float32))))
        details[obs_key] = diff
        total += diff
    return total, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match LeRobot RoboTwin episodes to the closest current RoboTwin reset seeds "
            "by comparing the initial multi-camera images."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--robotwin-path", required=True)
    parser.add_argument("--task-config-path", required=True)
    parser.add_argument("--task-name", default="place_phone_stand")
    parser.add_argument("--planner-backend", default="mplib")
    parser.add_argument("--episode-indices", type=int, nargs="+", required=True)
    parser.add_argument("--candidate-seeds-path", required=True)
    parser.add_argument("--candidate-limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_root).expanduser().resolve() / args.repo_id
    robotwin_path = Path(args.robotwin_path).expanduser().resolve()
    task_config_path = Path(args.task_config_path).expanduser().resolve()
    candidate_seeds_path = Path(args.candidate_seeds_path).expanduser().resolve()

    _ensure_robotwin_import_path(robotwin_path)
    from robotwin.envs.vector_env import VectorEnv

    episode_meta = _load_episode_meta(dataset_dir)
    reference_images = {
        episode_index: _load_episode_reference_images(dataset_dir, episode_meta, episode_index)
        for episode_index in args.episode_indices
    }
    candidate_seeds = _load_candidate_seeds(
        candidate_seeds_path,
        args.task_name,
        args.candidate_limit,
    )
    task_config = _build_task_config(
        robotwin_path=robotwin_path,
        task_config_path=task_config_path,
        task_name=args.task_name,
        planner_backend=args.planner_backend,
    )

    ranked_matches: dict[int, list[dict[str, Any]]] = {
        episode_index: [] for episode_index in args.episode_indices
    }
    for start in range(0, len(candidate_seeds), args.batch_size):
        batch_seeds = candidate_seeds[start : start + args.batch_size]
        venv = VectorEnv(task_config=task_config, n_envs=len(batch_seeds), env_seeds=batch_seeds)
        try:
            venv.reset(env_seeds=batch_seeds)
            obs_list = venv.get_obs()
            for seed, obs in zip(batch_seeds, obs_list, strict=True):
                for episode_index, episode_images in reference_images.items():
                    total_diff, details = _score_obs(obs, episode_images)
                    ranked_matches[episode_index].append(
                        {
                            "seed": int(seed),
                            "total_diff": total_diff,
                            "camera_diffs": details,
                        }
                    )
        finally:
            venv.close(clear_cache=True)

    result = {
        "task_name": args.task_name,
        "repo_id": args.repo_id,
        "candidate_limit": args.candidate_limit,
        "top_k": args.top_k,
        "matches": {},
    }
    for episode_index, matches in ranked_matches.items():
        matches.sort(key=lambda item: item["total_diff"])
        result["matches"][str(episode_index)] = matches[: args.top_k]

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json is not None:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

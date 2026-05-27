from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# Force transformers/openpi onto the torch path before importing model code.
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_FLAX", "0")

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
from rlinf.models.embodiment.openpi import get_model
from rlinf.utils.metric_utils import compute_evaluate_metrics


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "src" / "bentele" / "configs" / "embodiment"


def ensure_robotwin_import_path() -> None:
    candidate = Path(
        os.environ.get("ROBOTWIN_PATH", Path.home() / "RoboTwin")
    ).expanduser()
    if candidate.exists():
        os.environ.setdefault("ROBOTWIN_PATH", str(candidate))
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def prepare_visualization_runtime() -> None:
    """Force SAPIEN to use the system NVIDIA Vulkan/EGL ICDs under VNC."""
    vulkan_icd = Path("/etc/vulkan/icd.d/nvidia_icd.json")
    egl_vendor = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")

    if vulkan_icd.exists() and not os.environ.get("VK_ICD_FILENAMES"):
        os.environ["VK_ICD_FILENAMES"] = str(vulkan_icd)
    if egl_vendor.exists() and not os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES"):
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(egl_vendor)

    print(
        "[visualize] DISPLAY="
        f"{os.environ.get('DISPLAY', '')} "
        "VK_ICD_FILENAMES="
        f"{os.environ.get('VK_ICD_FILENAMES', '')} "
        "__EGL_VENDOR_LIBRARY_FILENAMES="
        f"{os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES', '')}"
    )


def load_eval_cfg(
    *,
    config_name: str,
    assets_path: str,
    model_path: str | None = None,
    num_envs: int | None = None,
    eval_rollout_epochs: int | None = None,
    eval_seeds_path: str | None = None,
    eval_step_limit: int | None = None,
    visualize: bool = False,
    render_freq: int = 10,
):
    overrides = [
        f"env.eval.assets_path={assets_path}",
        "runner.logger.logger_backends=[]",
    ]
    if model_path is not None:
        overrides.append(f"actor.model.model_path={model_path}")
    if num_envs is not None:
        overrides.append(f"env.eval.total_num_envs={num_envs}")
    if eval_rollout_epochs is not None:
        overrides.append(f"algorithm.eval_rollout_epoch={eval_rollout_epochs}")

    with hydra.initialize_config_dir(
        version_base="1.3",
        config_dir=str(CONFIG_DIR.resolve()),
    ):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    metadata = _load_sft_metadata(model_path)
    if metadata.get("is_lora", False):
        with open_dict(cfg.actor.model):
            cfg.actor.model.is_lora = True
            cfg.actor.model.lora_rank = int(
                metadata.get(
                    "lora_rank",
                    getattr(cfg.actor.model, "lora_rank", 32),
                )
            )
            # RLinf SFT checkpoints store the LoRA module weights inside
            # model_state_dict/full_weights.pt, not as a PEFT adapter directory.
            cfg.actor.model.lora_path = None
    if eval_seeds_path is not None:
        with open_dict(cfg.env.eval):
            cfg.env.eval.seeds_path = str(Path(eval_seeds_path).expanduser().resolve())
            cfg.env.eval.use_fixed_reset_state_ids = True
    if eval_step_limit is not None:
        with open_dict(cfg.env.eval):
            cfg.env.eval.max_episode_steps = int(eval_step_limit)
            cfg.env.eval.max_steps_per_rollout_epoch = int(eval_step_limit)
            cfg.env.eval.task_config.step_lim = int(eval_step_limit)
    if visualize:
        with open_dict(cfg.env.eval):
            cfg.env.eval.visualize = True
            cfg.env.eval.auto_reset = False
            if num_envs is None:
                cfg.env.eval.total_num_envs = 1
        cfg.env.eval.task_config.render_freq = render_freq
        cfg.env.eval.task_config.eval_video_log = False
    return cfg


def _load_sft_metadata(model_path: str | None) -> dict[str, Any]:
    if model_path is None:
        return {}
    metadata_path = Path(model_path).expanduser().resolve() / "sft_metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid SFT metadata JSON: {metadata_path}") from exc


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        return value.detach().cpu().tolist()
    return value


def save_eval_result(result: dict[str, Any], output_json: str | Path) -> Path:
    output_path = Path(output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def _slice_metric_value(value: Any, mask: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value[mask].detach().cpu()
    if isinstance(value, list):
        return torch.as_tensor(value, dtype=torch.float32)[mask].cpu()
    return torch.as_tensor(value, dtype=torch.float32)[mask].cpu()


def _debug_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _debug_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_debug_value(v) for v in value]
    return value


def _to_rgb_frame(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        frame = image.detach().cpu().numpy()
    else:
        frame = np.asarray(image)
    if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.moveaxis(frame, 0, -1)
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _capture_obs_frames(obs: dict[str, Any]) -> list[np.ndarray]:
    if "main_images" not in obs or obs["main_images"] is None:
        return []
    return [_to_rgb_frame(frame) for frame in obs["main_images"]]


def _extract_place_phone_stand_progress(env: RoboTwinEnv) -> list[dict[str, Any]]:
    progress = []
    sub_envs = getattr(getattr(env, "venv", None), "envs", [])
    for env_idx, sub_env in enumerate(sub_envs):
        task = getattr(sub_env, "task", None)
        snapshot: dict[str, Any] = {"env_idx": env_idx}
        if task is None:
            progress.append(snapshot)
            continue

        phone = getattr(task, "phone", None)
        stand = getattr(task, "stand", None)
        if phone is not None and stand is not None:
            phone_pose = np.asarray(phone.get_pose().p, dtype=np.float64)[:3]
            phone_func_pose = np.asarray(phone.get_functional_point(0), dtype=np.float64)[:3]
            stand_func_pose = np.asarray(stand.get_functional_point(0), dtype=np.float64)[:3]
            delta = phone_func_pose - stand_func_pose
            snapshot["phone_pose"] = phone_pose.tolist()
            snapshot["phone_func_pose"] = phone_func_pose.tolist()
            snapshot["stand_func_pose"] = stand_func_pose.tolist()
            snapshot["xyz_abs_error"] = np.abs(delta).tolist()
            snapshot["l2_distance"] = float(np.linalg.norm(delta))
            snapshot["phone_height"] = float(phone_pose[2])

            robot = getattr(task, "robot", None)
            if robot is not None:
                try:
                    left_tcp = np.asarray(robot.get_left_tcp_pose()[:3], dtype=np.float64)
                    right_tcp = np.asarray(robot.get_right_tcp_pose()[:3], dtype=np.float64)
                    active_arm = "left" if float(phone_pose[0]) < 0 else "right"
                    active_tcp = left_tcp if active_arm == "left" else right_tcp
                    snapshot["active_arm"] = active_arm
                    snapshot["active_tcp_pose"] = active_tcp.tolist()
                    snapshot["tcp_phone_l2"] = float(np.linalg.norm(active_tcp - phone_pose))
                except Exception:
                    pass

        if hasattr(task, "is_left_gripper_open"):
            snapshot["left_gripper_open"] = bool(task.is_left_gripper_open())
        if hasattr(task, "is_right_gripper_open"):
            snapshot["right_gripper_open"] = bool(task.is_right_gripper_open())
        if hasattr(task, "episode_left_gripper_state"):
            snapshot["chunk_left_gripper_trace"] = [
                bool(v) for v in getattr(task, "episode_left_gripper_state", [])
            ]
        if hasattr(task, "episode_right_gripper_state"):
            snapshot["chunk_right_gripper_trace"] = [
                bool(v) for v in getattr(task, "episode_right_gripper_state", [])
            ]
        if hasattr(task, "run_steps"):
            snapshot["run_steps"] = int(task.run_steps)

        try:
            snapshot["success_now"] = bool(task.check_success())
        except Exception:
            pass

        progress.append(snapshot)
    return progress


def _append_progress_history(
    progress_history: list[list[dict[str, Any]]],
    chunk_idx: int,
    snapshots: list[dict[str, Any]],
) -> None:
    for snapshot in snapshots:
        env_idx = int(snapshot["env_idx"])
        event = {"chunk_idx": chunk_idx}
        event.update(snapshot)
        progress_history[env_idx].append(event)


def _summarize_progress_history(
    progress_history: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    per_env = []
    aggregate_initial_l2 = []
    aggregate_min_l2 = []
    aggregate_final_l2 = []
    aggregate_initial_tcp_phone = []
    aggregate_min_tcp_phone = []
    aggregate_final_tcp_phone = []
    aggregate_max_phone_displacement = []
    aggregate_max_phone_height_gain = []

    for env_idx, history in enumerate(progress_history):
        env_summary: dict[str, Any] = {
            "env_idx": env_idx,
            "num_snapshots": len(history),
            "history": history,
        }
        l2_values = [float(item["l2_distance"]) for item in history if "l2_distance" in item]
        tcp_phone_values = [float(item["tcp_phone_l2"]) for item in history if "tcp_phone_l2" in item]
        xyz_values = [np.asarray(item["xyz_abs_error"], dtype=np.float64) for item in history if "xyz_abs_error" in item]
        phone_pose_values = [np.asarray(item["phone_pose"], dtype=np.float64) for item in history if "phone_pose" in item]
        phone_height_values = [float(item["phone_height"]) for item in history if "phone_height" in item]
        left_values = [bool(item["left_gripper_open"]) for item in history if "left_gripper_open" in item]
        right_values = [bool(item["right_gripper_open"]) for item in history if "right_gripper_open" in item]
        success_values = [bool(item["success_now"]) for item in history if "success_now" in item]
        active_arm = next((str(item["active_arm"]) for item in history if "active_arm" in item), None)

        if l2_values:
            env_summary["initial_l2_distance"] = l2_values[0]
            env_summary["min_l2_distance"] = min(l2_values)
            env_summary["final_l2_distance"] = l2_values[-1]
            aggregate_initial_l2.append(l2_values[0])
            aggregate_min_l2.append(min(l2_values))
            aggregate_final_l2.append(l2_values[-1])
        if tcp_phone_values:
            env_summary["initial_tcp_phone_l2"] = tcp_phone_values[0]
            env_summary["min_tcp_phone_l2"] = min(tcp_phone_values)
            env_summary["final_tcp_phone_l2"] = tcp_phone_values[-1]
            env_summary["tcp_phone_delta_to_min"] = tcp_phone_values[0] - min(tcp_phone_values)
            aggregate_initial_tcp_phone.append(tcp_phone_values[0])
            aggregate_min_tcp_phone.append(min(tcp_phone_values))
            aggregate_final_tcp_phone.append(tcp_phone_values[-1])
        if xyz_values:
            stacked_xyz = np.stack(xyz_values, axis=0)
            env_summary["initial_xyz_abs_error"] = stacked_xyz[0].tolist()
            env_summary["min_xyz_abs_error"] = stacked_xyz.min(axis=0).tolist()
            env_summary["final_xyz_abs_error"] = stacked_xyz[-1].tolist()
        if phone_pose_values:
            phone_pose_stack = np.stack(phone_pose_values, axis=0)
            displacement = np.linalg.norm(phone_pose_stack - phone_pose_stack[0], axis=1)
            env_summary["initial_phone_pose"] = phone_pose_stack[0].tolist()
            env_summary["final_phone_pose"] = phone_pose_stack[-1].tolist()
            env_summary["max_phone_displacement"] = float(displacement.max())
            env_summary["final_phone_displacement"] = float(displacement[-1])
            aggregate_max_phone_displacement.append(float(displacement.max()))
        if phone_height_values:
            initial_height = phone_height_values[0]
            max_height = max(phone_height_values)
            env_summary["initial_phone_height"] = initial_height
            env_summary["max_phone_height"] = max_height
            env_summary["final_phone_height"] = phone_height_values[-1]
            env_summary["max_phone_height_gain"] = max_height - initial_height
            aggregate_max_phone_height_gain.append(max_height - initial_height)
        if left_values:
            env_summary["initial_left_gripper_open"] = left_values[0]
            env_summary["ever_left_gripper_open"] = any(left_values)
            env_summary["final_left_gripper_open"] = left_values[-1]
        if right_values:
            env_summary["initial_right_gripper_open"] = right_values[0]
            env_summary["ever_right_gripper_open"] = any(right_values)
            env_summary["final_right_gripper_open"] = right_values[-1]
        if success_values:
            env_summary["ever_success"] = any(success_values)
            env_summary["final_success"] = success_values[-1]
        if active_arm is not None:
            env_summary["active_arm"] = active_arm

        env_summary["stage"] = _classify_place_phone_stand_progress(env_summary)

        per_env.append(env_summary)

    aggregate: dict[str, Any] = {"num_envs": len(per_env)}
    if aggregate_initial_l2:
        aggregate["mean_initial_l2_distance"] = float(np.mean(aggregate_initial_l2))
        aggregate["mean_min_l2_distance"] = float(np.mean(aggregate_min_l2))
        aggregate["mean_final_l2_distance"] = float(np.mean(aggregate_final_l2))
    if aggregate_initial_tcp_phone:
        aggregate["mean_initial_tcp_phone_l2"] = float(np.mean(aggregate_initial_tcp_phone))
        aggregate["mean_min_tcp_phone_l2"] = float(np.mean(aggregate_min_tcp_phone))
        aggregate["mean_final_tcp_phone_l2"] = float(np.mean(aggregate_final_tcp_phone))
    if aggregate_max_phone_displacement:
        aggregate["mean_max_phone_displacement"] = float(
            np.mean(aggregate_max_phone_displacement)
        )
    if aggregate_max_phone_height_gain:
        aggregate["mean_max_phone_height_gain"] = float(
            np.mean(aggregate_max_phone_height_gain)
        )
    aggregate["stage_counts"] = dict(
        sorted(
            (
                (stage, sum(1 for env_summary in per_env if env_summary.get("stage") == stage))
                for stage in {env_summary.get("stage", "unknown") for env_summary in per_env}
            ),
            key=lambda item: item[0],
        )
    )

    return {"per_env": per_env, "aggregate": aggregate}


def _classify_place_phone_stand_progress(env_summary: dict[str, Any]) -> str:
    if env_summary.get("final_success") or env_summary.get("ever_success"):
        return "success"

    max_phone_displacement = float(env_summary.get("max_phone_displacement", 0.0))
    max_phone_height_gain = float(env_summary.get("max_phone_height_gain", 0.0))
    min_tcp_phone_l2 = float(env_summary.get("min_tcp_phone_l2", float("inf")))
    tcp_phone_delta_to_min = float(env_summary.get("tcp_phone_delta_to_min", 0.0))

    if max_phone_displacement > 0.02 or max_phone_height_gain > 0.01:
        return "moved_phone"
    if min_tcp_phone_l2 < 0.16 or tcp_phone_delta_to_min > 0.08:
        return "approached_phone"
    if min_tcp_phone_l2 < 0.24 or tcp_phone_delta_to_min > 0.02:
        return "weak_approach"
    return "no_meaningful_approach"


def _write_rollout_videos(
    *,
    video_dir: str | Path,
    video_frames: list[list[np.ndarray]],
    video_fps: float,
) -> list[str]:
    import cv2

    output_dir = Path(video_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[str] = []

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for env_idx, frames in enumerate(video_frames):
        if not frames:
            continue
        first = frames[0]
        height, width = first.shape[:2]
        output_path = output_dir / f"env{env_idx:02d}.mp4"
        writer = cv2.VideoWriter(str(output_path), fourcc, float(video_fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")
        try:
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        output_paths.append(str(output_path))

    return output_paths


def run_robotwin_policy_eval(
    *,
    config_name: str,
    assets_path: str,
    model_path: str,
    ckpt_path: str | None = None,
    num_envs: int | None = None,
    eval_rollout_epochs: int | None = None,
    eval_seeds_path: str | None = None,
    eval_step_limit: int | None = None,
    max_chunk_steps: int | None = None,
    action_exec_horizon: int | None = None,
    visualize: bool = False,
    render_freq: int = 10,
    sleep_between_chunks: float = 0.0,
    hold_viewer_seconds: float = 0.0,
    device: str | None = None,
    debug_log_chunks: bool = False,
    debug_action_steps: int = 0,
    record_progress: bool = False,
    save_video_dir: str | None = None,
    video_fps: float = 2.0,
) -> dict[str, Any]:
    ensure_robotwin_import_path()
    if visualize:
        prepare_visualization_runtime()

    cfg = load_eval_cfg(
        config_name=config_name,
        assets_path=assets_path,
        model_path=model_path,
        num_envs=num_envs,
        eval_rollout_epochs=eval_rollout_epochs,
        eval_seeds_path=eval_seeds_path,
        eval_step_limit=eval_step_limit,
        visualize=visualize,
        render_freq=render_freq,
    )

    if device is None or device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)
    resolved_model_path = Path(str(cfg.actor.model.model_path)).expanduser()
    if not resolved_model_path.exists():
        raise FileNotFoundError(
            "Model path does not exist for evaluation: "
            f"{resolved_model_path}. Pass model_path explicitly."
        )
    if ckpt_path is not None and not Path(ckpt_path).expanduser().exists():
        raise FileNotFoundError(f"Actor checkpoint path does not exist: {Path(ckpt_path).expanduser()}")

    env_cfg = cfg.env.eval
    actual_num_envs = env_cfg.total_num_envs
    if visualize and actual_num_envs != 1:
        raise ValueError(
            f"visualize currently requires num_envs=1, but got {actual_num_envs}."
        )
    actual_eval_rollout_epochs = cfg.algorithm.eval_rollout_epoch
    actual_action_exec_horizon = action_exec_horizon
    if actual_action_exec_horizon is None:
        actual_action_exec_horizon = (
            1 if visualize else cfg.actor.model.num_action_chunks
        )
    if actual_action_exec_horizon <= 0:
        raise ValueError(
            "action_exec_horizon must be positive, "
            f"but got {actual_action_exec_horizon}."
        )

    n_eval_chunk_steps = math.ceil(
        env_cfg.max_steps_per_rollout_epoch / actual_action_exec_horizon
    )
    if max_chunk_steps is not None:
        n_eval_chunk_steps = max_chunk_steps

    env = RoboTwinEnv(
        cfg=env_cfg,
        num_envs=actual_num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info={"rank": 0, "world_size": 1},
    )

    model = get_model(cfg.actor.model)
    model = model.to(torch_device)
    model.eval()

    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location=torch_device)
        model.load_state_dict(state_dict)

    eval_metrics = defaultdict(list)
    prev_done = torch.zeros(actual_num_envs, dtype=torch.bool)
    obs = None
    progress_history: list[list[dict[str, Any]]] = [[] for _ in range(actual_num_envs)]
    video_frames: list[list[np.ndarray]] = [[] for _ in range(actual_num_envs)]

    try:
        with torch.no_grad():
            for eval_rollout_epoch in range(actual_eval_rollout_epochs):
                episode_finished = False
                if (not env_cfg.auto_reset) or eval_rollout_epoch == 0:
                    env.is_start = True
                    prev_done = torch.zeros(actual_num_envs, dtype=torch.bool)
                    obs, _ = env.reset()
                    if save_video_dir is not None:
                        for env_idx, frame in enumerate(_capture_obs_frames(obs)):
                            video_frames[env_idx].append(frame)
                    if record_progress:
                        _append_progress_history(
                            progress_history,
                            chunk_idx=-1,
                            snapshots=_extract_place_phone_stand_progress(env),
                        )

                for chunk_idx in range(n_eval_chunk_steps):
                    actions, _result = model.predict_action_batch(
                        env_obs=obs,
                        mode="eval",
                        compute_values=False,
                    )
                    actions_to_execute = actions[:, :actual_action_exec_horizon, :]
                    next_obs, _rewards, terminations, truncations, infos = env.step(
                        actions_to_execute, auto_reset=env_cfg.auto_reset
                    )
                    if save_video_dir is not None:
                        for env_idx, frame in enumerate(_capture_obs_frames(next_obs)):
                            video_frames[env_idx].append(frame)
                    if record_progress:
                        _append_progress_history(
                            progress_history,
                            chunk_idx=chunk_idx,
                            snapshots=_extract_place_phone_stand_progress(env),
                        )

                    if visualize and sleep_between_chunks > 0:
                        time.sleep(sleep_between_chunks)

                    current_dones = torch.logical_or(terminations, truncations)
                    newly_done = current_dones & ~prev_done.to(current_dones.device)
                    prev_done = current_dones.clone()

                    if newly_done.any():
                        if "final_info" in infos and "episode" in infos["final_info"]:
                            metric_source = infos["final_info"]["episode"]
                        elif "episode" in infos:
                            metric_source = infos["episode"]
                        else:
                            metric_source = {}

                        for key, value in metric_source.items():
                            eval_metrics[key].append(
                                _slice_metric_value(value, newly_done)
                            )

                    if debug_log_chunks:
                        action_min = float(actions_to_execute.min().item())
                        action_max = float(actions_to_execute.max().item())
                        action_mean = float(actions_to_execute.mean().item())
                        print(
                            "[robotwin_eval] "
                            f"chunk={chunk_idx} "
                            f"action[min,max,mean]=({action_min:.4f}, {action_max:.4f}, {action_mean:.4f}) "
                            f"rewards={_debug_value(_rewards)} "
                            f"terminations={_debug_value(terminations)} "
                            f"truncations={_debug_value(truncations)}"
                        )
                        if "success" in infos:
                            print(
                                "[robotwin_eval] "
                                f"chunk={chunk_idx} success={_debug_value(infos['success'])}"
                            )
                        if "episode" in infos:
                            print(
                                "[robotwin_eval] "
                                f"chunk={chunk_idx} episode={json.dumps(_debug_value(infos['episode']), ensure_ascii=False)}"
                            )
                        if "final_info" in infos:
                            print(
                                "[robotwin_eval] "
                                f"chunk={chunk_idx} final_info={json.dumps(_debug_value(infos['final_info']), ensure_ascii=False)}"
                            )
                        if debug_action_steps > 0:
                            action_preview = actions_to_execute[
                                0, : min(debug_action_steps, actions_to_execute.shape[1]), :
                            ]
                            print(
                                "[robotwin_eval] "
                                f"chunk={chunk_idx} env0_actions_head="
                                f"{json.dumps(_debug_value(action_preview), ensure_ascii=False)}"
                            )

                    obs = next_obs
                    if current_dones.any() and not env_cfg.auto_reset:
                        episode_finished = True
                        break

                if episode_finished:
                    break

        concatenated_metrics = {}
        for key, values in eval_metrics.items():
            if not values:
                continue
            concatenated_metrics[key] = torch.cat(values, dim=0).contiguous().cpu()

        summary = compute_evaluate_metrics([concatenated_metrics])
        summary = {k: json_ready(v) for k, v in summary.items()}

        result = {
            "config_name": config_name,
            "assets_path": env_cfg.assets_path,
            "model_path": cfg.actor.model.model_path,
            "ckpt_path": ckpt_path,
            "num_envs": actual_num_envs,
            "eval_rollout_epochs": actual_eval_rollout_epochs,
            "chunk_steps_per_epoch": n_eval_chunk_steps,
            "action_exec_horizon": actual_action_exec_horizon,
            "num_action_chunks": cfg.actor.model.num_action_chunks,
            "is_lora": bool(getattr(cfg.actor.model, "is_lora", False)),
            "lora_rank": int(getattr(cfg.actor.model, "lora_rank", 0) or 0),
            "device": str(device),
            "visualize": visualize,
            "render_freq": env_cfg.task_config.render_freq,
            "metrics": summary,
        }
        if record_progress:
            result["progress"] = _summarize_progress_history(progress_history)
        if save_video_dir is not None:
            result["video_paths"] = _write_rollout_videos(
                video_dir=save_video_dir,
                video_frames=video_frames,
                video_fps=video_fps,
            )
    finally:
        if visualize and hold_viewer_seconds > 0:
            print(
                f"[visualize] Holding viewer open for {hold_viewer_seconds:.1f}s "
                "before closing..."
            )
            time.sleep(hold_viewer_seconds)
        env.close()

    return result

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
    if eval_seeds_path is not None:
        with open_dict(cfg.env.eval):
            cfg.env.eval.seeds_path = str(Path(eval_seeds_path).expanduser().resolve())
            cfg.env.eval.use_fixed_reset_state_ids = True
    if visualize:
        with open_dict(cfg.env.eval):
            cfg.env.eval.visualize = True
            cfg.env.eval.auto_reset = False
            if num_envs is None:
                cfg.env.eval.total_num_envs = 1
        cfg.env.eval.task_config.render_freq = render_freq
        cfg.env.eval.task_config.eval_video_log = False
    return cfg


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


def run_robotwin_policy_eval(
    *,
    config_name: str,
    assets_path: str,
    model_path: str,
    ckpt_path: str | None = None,
    num_envs: int | None = None,
    eval_rollout_epochs: int | None = None,
    eval_seeds_path: str | None = None,
    max_chunk_steps: int | None = None,
    action_exec_horizon: int | None = None,
    visualize: bool = False,
    render_freq: int = 10,
    sleep_between_chunks: float = 0.0,
    hold_viewer_seconds: float = 0.0,
    device: str | None = None,
    debug_log_chunks: bool = False,
    debug_action_steps: int = 0,
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

    try:
        with torch.no_grad():
            for eval_rollout_epoch in range(actual_eval_rollout_epochs):
                episode_finished = False
                if (not env_cfg.auto_reset) or eval_rollout_epoch == 0:
                    env.is_start = True
                    prev_done = torch.zeros(actual_num_envs, dtype=torch.bool)
                    obs, _ = env.reset()

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
            "device": str(device),
            "visualize": visualize,
            "render_freq": env_cfg.task_config.render_freq,
            "metrics": summary,
        }
    finally:
        if visualize and hold_viewer_seconds > 0:
            print(
                f"[visualize] Holding viewer open for {hold_viewer_seconds:.1f}s "
                "before closing..."
            )
            time.sleep(hold_viewer_seconds)
        env.close()

    return result

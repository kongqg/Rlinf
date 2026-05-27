#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from rlinf.projects.robotwin.runtime.paths import REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone policy evaluation for RoboTwin place_phone_stand. "
            "This bypasses the training runner and directly loops env + pi0.5 policy."
        )
    )
    parser.add_argument(
        "--config-name",
        default="robotwin_place_phone_stand_ppo_openpi_pi05",
    )
    parser.add_argument(
        "--assets-path",
        required=True,
        help="Path to RoboTwin assets root, e.g. ~/RoboTwin",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="OpenPI / pi0.5 checkpoint directory to evaluate.",
    )
    parser.add_argument(
        "--ckpt-path",
        default=None,
        help=(
            "Optional actor state_dict checkpoint to load on top of model_path. "
            "Useful for evaluating RL fine-tuned checkpoints."
        ),
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Override env.eval.total_num_envs. Defaults to config value.",
    )
    parser.add_argument(
        "--eval-rollout-epochs",
        type=int,
        default=None,
        help="Override algorithm.eval_rollout_epoch. Defaults to config value.",
    )
    parser.add_argument(
        "--eval-step-limit",
        type=int,
        default=None,
        help=(
            "Override env.eval.max_episode_steps, env.eval.max_steps_per_rollout_epoch, "
            "and env.eval.task_config.step_lim together."
        ),
    )
    parser.add_argument(
        "--eval-seeds-path",
        default=None,
        help=(
            "Optional JSON file overriding env.eval.seeds_path. "
            "Use this to evaluate on a custom seed list, such as the same seeds "
            "used during a small overfit experiment."
        ),
    )
    parser.add_argument(
        "--max-chunk-steps",
        type=int,
        default=None,
        help=(
            "Override the number of action chunks evaluated per rollout epoch. "
            "Defaults to env.eval.max_steps_per_rollout_epoch // action_exec_horizon."
        ),
    )
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "traces" / "robotwin_place_phone_stand_eval.json"),
        help="Where to save the evaluation summary JSON.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help=(
            "Open the RoboTwin viewer during evaluation. Run this from a graphical "
            "session such as the ms VNC desktop."
        ),
    )
    parser.add_argument(
        "--render-freq",
        type=int,
        default=10,
        help=(
            "Viewer refresh frequency passed into RoboTwin when --visualize is set. "
            "Smaller means smoother but slower."
        ),
    )
    parser.add_argument(
        "--sleep-between-chunks",
        type=float,
        default=0.0,
        help=(
            "Optional sleep in seconds between each chunk-level env.step call. "
            "Useful only with --visualize."
        ),
    )
    parser.add_argument(
        "--hold-viewer-seconds",
        type=float,
        default=10.0,
        help=(
            "How long to keep the viewer window open after evaluation finishes "
            "when --visualize is enabled."
        ),
    )
    parser.add_argument(
        "--action-exec-horizon",
        type=int,
        default=None,
        help=(
            "How many predicted actions to actually execute per env.step call. "
            "Defaults to the full policy chunk, but visualize mode defaults to 1 "
            "to make the motion observable."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Evaluation model device: auto, cuda, or cpu.",
    )
    parser.add_argument(
        "--debug-log-chunks",
        action="store_true",
        help="Print per-chunk action / reward / success diagnostics during eval.",
    )
    parser.add_argument(
        "--debug-action-steps",
        type=int,
        default=0,
        help="When debug logging is enabled, print the first N env actions of each chunk.",
    )
    parser.add_argument(
        "--record-progress",
        action="store_true",
        help=(
            "Record task-specific placement progress metrics such as phone-stand "
            "distance and gripper state after each chunk."
        ),
    )
    parser.add_argument(
        "--save-video-dir",
        default=None,
        help=(
            "Optional directory to save rollout videos. One mp4 is written per env "
            "using the observed head camera frames."
        ),
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=2.0,
        help="FPS used when writing rollout videos.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help=(
            "Training step to use for W&B logging. If omitted, the script tries "
            "to read sft_metadata.json or training_state.pt from model-path."
        ),
    )
    parser.add_argument("--wandb-enabled", action="store_true")
    parser.add_argument("--wandb-project", default="bentele")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-log-dir", default=None)
    parser.add_argument("--wandb-proxy", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    return parser.parse_args()


def _infer_step(model_path: str) -> int:
    model_dir = Path(model_path).expanduser().resolve()
    metadata_path = model_dir / "sft_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "saved_step" in metadata:
            return int(metadata["saved_step"])

    state_path = model_dir / "training_state.pt"
    if state_path.exists():
        import torch

        state = torch.load(state_path, map_location="cpu")
        if "step" in state:
            return int(state["step"])
    return 0


def _wandb_payload(result: dict[str, Any], runtime_seconds: float) -> dict[str, float]:
    metrics = result.get("metrics", {})
    progress_aggregate = result.get("progress", {}).get("aggregate", {})
    stage_counts = progress_aggregate.get("stage_counts", {})
    payload: dict[str, float] = {
        "eval/success_at_end": float(metrics.get("success_at_end", 0.0)),
        "eval/success_once": float(metrics.get("success_once", 0.0)),
        "eval/return": float(metrics.get("return", 0.0)),
        "eval/reward": float(metrics.get("reward", 0.0)),
        "eval/episode_len": float(metrics.get("episode_len", 0.0)),
        "eval/num_trajectories": float(metrics.get("num_trajectories", 0.0)),
        "eval/runtime_seconds": float(runtime_seconds),
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
            payload[f"eval_progress/{key}"] = float(progress_aggregate[key])
    for stage_name in (
        "success",
        "moved_phone",
        "approached_phone",
        "weak_approach",
        "no_meaningful_approach",
    ):
        payload[f"eval_progress/stage_count/{stage_name}"] = float(
            stage_counts.get(stage_name, 0)
        )
    return payload


def _log_wandb(args: argparse.Namespace, result: dict[str, Any], runtime_seconds: float) -> None:
    if not args.wandb_enabled or args.wandb_mode == "disabled":
        return
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed in the current environment. "
            "Install it first or omit --wandb-enabled."
        ) from exc

    settings = None
    if args.wandb_proxy:
        settings = wandb.Settings(https_proxy=args.wandb_proxy)

    output_path = Path(args.output_json).expanduser().resolve()
    wandb_log_dir = (
        Path(args.wandb_log_dir).expanduser().resolve()
        if args.wandb_log_dir is not None
        else output_path.parent / "wandb_eval"
    )
    wandb_log_dir.mkdir(parents=True, exist_ok=True)
    step = args.step if args.step is not None else _infer_step(args.model_path)
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"{Path(args.model_path).name}-eval",
        group=args.wandb_group,
        config=vars(args),
        settings=settings,
        dir=str(wandb_log_dir),
        tags=list(args.wandb_tags),
        mode=args.wandb_mode,
        reinit=True,
    )
    run.log(_wandb_payload(result, runtime_seconds), step=step)
    run.finish()


def main() -> None:
    args = parse_args()
    from rlinf.projects.robotwin.utils.robotwin_eval import run_robotwin_policy_eval, save_eval_result

    eval_start = time.time()
    result = run_robotwin_policy_eval(
        config_name=args.config_name,
        assets_path=args.assets_path,
        model_path=args.model_path,
        ckpt_path=args.ckpt_path,
        num_envs=args.num_envs,
        eval_rollout_epochs=args.eval_rollout_epochs,
        eval_seeds_path=args.eval_seeds_path,
        eval_step_limit=args.eval_step_limit,
        max_chunk_steps=args.max_chunk_steps,
        action_exec_horizon=args.action_exec_horizon,
        visualize=args.visualize,
        render_freq=args.render_freq,
        sleep_between_chunks=args.sleep_between_chunks,
        hold_viewer_seconds=args.hold_viewer_seconds,
        device=args.device,
        debug_log_chunks=args.debug_log_chunks,
        debug_action_steps=args.debug_action_steps,
        record_progress=args.record_progress,
        save_video_dir=args.save_video_dir,
        video_fps=args.video_fps,
    )
    runtime_seconds = time.time() - eval_start
    step = args.step if args.step is not None else _infer_step(args.model_path)
    result["step"] = step
    result["runtime_seconds"] = runtime_seconds
    save_eval_result(result, args.output_json)
    _log_wandb(args, result, runtime_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bentele.utils.robotwin_eval import run_robotwin_policy_eval, save_eval_result


REPO_ROOT = Path(__file__).resolve().parents[1]


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    save_eval_result(result, args.output_json)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

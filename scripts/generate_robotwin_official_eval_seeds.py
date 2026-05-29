#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate RoboTwin official-compatible eval success seeds using "
            "the expert-check logic from RoboTwin script/eval_policy.py."
        )
    )
    parser.add_argument("--task-name", default="place_phone_stand")
    parser.add_argument(
        "--task-config",
        choices=("demo_clean", "demo_randomized"),
        required=True,
    )
    parser.add_argument(
        "--robotwin-path",
        default=os.environ.get("ROBOTWIN_PATH", str(Path.home() / "RoboTwin")),
        help="Path to the official RoboTwin checkout.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--planner-backend",
        default="mplib",
        help=(
            "Planner backend passed to task_env.setup_demo. The official YAMLs "
            "do not include this field; mplib matches the current RLinf VectorEnv "
            "eval config and avoids requiring curobo."
        ),
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required RoboTwin YAML file does not exist: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.load(fh.read(), Loader=yaml.FullLoader)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in YAML file: {path}")
    return data


def _prepare_robotwin_imports(robotwin_root: Path) -> None:
    if not robotwin_root.is_dir():
        raise FileNotFoundError(
            f"RoboTwin path does not exist: {robotwin_root}. "
            "Pass --robotwin-path or set ROBOTWIN_PATH."
        )
    os.environ.setdefault("ROBOTWIN_PATH", str(robotwin_root))
    os.environ.setdefault("ASSETS_PATH", str(robotwin_root))
    for candidate in (
        robotwin_root,
        robotwin_root / "script",
        robotwin_root / "description" / "utils",
    ):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def _class_decorator(task_name: str):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
    except AttributeError as exc:
        raise SystemExit(f"No RoboTwin task class found for {task_name}") from exc
    return env_class()


def _get_embodiment_config(robot_file: str) -> dict[str, Any]:
    robot_config_file = Path(robot_file) / "config.yml"
    return _load_yaml(robot_config_file)


def _build_task_args(
    *,
    robotwin_root: Path,
    task_name: str,
    task_config: str,
) -> dict[str, Any]:
    config_dir = robotwin_root / "task_config"
    task_args = _load_yaml(config_dir / f"{task_config}.yml")
    task_args["task_name"] = task_name
    task_args["task_config"] = task_config
    task_args["eval_mode"] = True

    embodiment_types = task_args.get("embodiment")
    if not isinstance(embodiment_types, list):
        raise ValueError(
            f"Official task config {task_config} must define list field 'embodiment'."
        )

    embodiment_config = _load_yaml(config_dir / "_embodiment_config.yml")
    camera_config = _load_yaml(config_dir / "_camera_config.yml")

    head_camera_type = task_args["camera"]["head_camera_type"]
    task_args["head_camera_h"] = camera_config[head_camera_type]["h"]
    task_args["head_camera_w"] = camera_config[head_camera_type]["w"]

    def get_embodiment_file(embodiment_type: str) -> str:
        robot_file = embodiment_config[embodiment_type]["file_path"]
        if robot_file is None:
            raise ValueError(f"No embodiment file configured for {embodiment_type}")
        return str(robotwin_root / robot_file)

    if len(embodiment_types) == 1:
        task_args["left_robot_file"] = get_embodiment_file(embodiment_types[0])
        task_args["right_robot_file"] = get_embodiment_file(embodiment_types[0])
        task_args["dual_arm_embodied"] = True
    elif len(embodiment_types) == 3:
        task_args["left_robot_file"] = get_embodiment_file(embodiment_types[0])
        task_args["right_robot_file"] = get_embodiment_file(embodiment_types[1])
        task_args["embodiment_dis"] = embodiment_types[2]
        task_args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    task_args["left_embodiment_config"] = _get_embodiment_config(
        task_args["left_robot_file"]
    )
    task_args["right_embodiment_config"] = _get_embodiment_config(
        task_args["right_robot_file"]
    )
    return task_args


def _safe_close(task_env, *, clear_cache: bool = False) -> None:
    try:
        task_env.close_env(clear_cache=clear_cache)
    except Exception:
        pass
    viewer = getattr(task_env, "viewer", None)
    if viewer is not None:
        try:
            viewer.close()
        except Exception:
            pass


def generate_success_seeds(args: argparse.Namespace) -> list[int]:
    robotwin_root = Path(args.robotwin_path).expanduser().resolve()
    _prepare_robotwin_imports(robotwin_root)

    from envs.utils.create_actor import UnStableError

    task_args = _build_task_args(
        robotwin_root=robotwin_root,
        task_name=args.task_name,
        task_config=args.task_config,
    )
    if args.planner_backend:
        task_args["planner_backend"] = args.planner_backend
    clear_cache_freq = int(task_args.get("clear_cache_freq", 5) or 5)
    success_seeds: list[int] = []
    now_seed = 100000 * (1 + int(args.seed))
    checked = 0

    while len(success_seeds) < args.num_episodes:
        task_env = _class_decorator(args.task_name)
        now_ep_num = len(success_seeds)
        try:
            task_env.setup_demo(
                now_ep_num=now_ep_num,
                seed=now_seed,
                is_test=True,
                **task_args,
            )
            task_env.play_once()
            success = bool(task_env.plan_success and task_env.check_success())
        except UnStableError:
            success = False
        except Exception as exc:
            print(
                "[generate_robotwin_official_eval_seeds] "
                f"seed={now_seed} failed with {type(exc).__name__}: {exc}",
                flush=True,
            )
            success = False
        finally:
            checked += 1
            _safe_close(task_env, clear_cache=(checked % clear_cache_freq == 0))

        if success:
            success_seeds.append(now_seed)
            print(
                "[generate_robotwin_official_eval_seeds] "
                f"accepted {len(success_seeds)}/{args.num_episodes}: seed={now_seed}",
                flush=True,
            )
        now_seed += 1

    return success_seeds


def main() -> None:
    args = parse_args()
    success_seeds = generate_success_seeds(args)
    payload = {
        args.task_name: {
            "task_name": args.task_name,
            "task_config": args.task_config,
            "seed": args.seed,
            "success_seeds": success_seeds,
        }
    }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote success seeds to: {output_path}")


if __name__ == "__main__":
    main()

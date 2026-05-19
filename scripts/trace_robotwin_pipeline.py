#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Any

# Force transformers/openpi onto the torch path before importing any model code.
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_FLAX", "0")

import hydra
import torch
from omegaconf import OmegaConf

from openpi.models.model import Observation
from rlinf.data.embodied_io_struct import EnvOutput, RolloutResult
from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
from rlinf.models.embodiment.openpi import get_model
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
)
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "src" / "bentele" / "configs" / "embodiment"
MAIN_CONFIG_FILE = (
    REPO_ROOT
    / "src"
    / "bentele"
    / "configs"
    / "embodiment"
    / "robotwin_place_phone_stand_ppo_openpi_pi05.yaml"
)
ENV_CONFIG_FILE = (
    REPO_ROOT
    / "src"
    / "bentele"
    / "configs"
    / "embodiment"
    / "env"
    / "robotwin_place_phone_stand.yaml"
)


def code_ref(obj: Any) -> str:
    source = inspect.getsourcefile(obj)
    _, line = inspect.getsourcelines(obj)
    if source is None:
        return f"<unknown>:{line}"
    path = Path(source).resolve()
    try:
        path = path.relative_to(REPO_ROOT)
    except ValueError:
        pass
    return f"{path}:{line}"


def literal_ref(path: str, line: int) -> str:
    return f"{path}:{line}"


def summarize_value(value: Any, depth: int = 0) -> Any:
    if depth >= 2:
        return type(value).__name__
    if isinstance(value, torch.Tensor):
        summary: dict[str, Any] = {
            "type": "torch.Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
        if value.numel() <= 16:
            summary["values"] = value.detach().cpu().tolist()
        elif value.numel() > 0 and value.dtype != torch.bool:
            summary["min"] = float(value.min().item())
            summary["max"] = float(value.max().item())
        return summary
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        if not value:
            return []
        if all(isinstance(item, str) for item in value):
            return {
                "type": "list[str]",
                "length": len(value),
                "sample": value[: min(2, len(value))],
            }
        return [summarize_value(item, depth + 1) for item in value[:2]]
    if isinstance(value, tuple):
        return [summarize_value(item, depth + 1) for item in value[:2]]
    if isinstance(value, dict):
        return {
            key: summarize_value(val, depth + 1)
            for key, val in list(value.items())[:30]
        }
    return repr(value)


def build_logger(log_path: Path | None) -> logging.Logger:
    logger = logging.getLogger("robotwin_pipeline_trace")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def log_block(
    logger: logging.Logger,
    title: str,
    refs: list[str],
    payload: dict[str, Any] | None = None,
) -> None:
    logger.info("")
    logger.info("=" * 100)
    logger.info(title)
    for ref in refs:
        logger.info("ref: %s", ref)
    if payload:
        logger.info(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            )
        )


def load_cfg(args: argparse.Namespace):
    overrides = [
        f"env.train.assets_path={args.assets_path}",
        f"env.eval.assets_path={args.assets_path}",
        f"actor.model.model_path={args.model_path}",
        "runner.logger.logger_backends=[]",
        f"env.train.total_num_envs={args.num_envs}",
        f"env.eval.total_num_envs={args.num_envs}",
    ]
    with hydra.initialize_config_dir(
        version_base="1.3", config_dir=str(CONFIG_DIR.resolve())
    ):
        cfg = hydra.compose(config_name=args.config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace the place_phone_stand data flow from config to action output."
    )
    parser.add_argument(
        "--config-name",
        default="robotwin_place_phone_stand_ppo_openpi_pi05",
    )
    parser.add_argument("--assets-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--mode",
        default="eval",
        choices=["train", "eval"],
    )
    parser.add_argument(
        "--log-file",
        default=str(REPO_ROOT / "traces" / "robotwin_place_phone_stand_pipeline.log"),
    )
    parser.add_argument(
        "--skip-step",
        action="store_true",
        help="Only trace until action output, do not feed actions back into env.step().",
    )
    args = parser.parse_args()

    logger = build_logger(Path(args.log_file))
    cfg = load_cfg(args)
    selected_env_cfg = cfg.env.eval if args.mode == "eval" else cfg.env.train
    selected_env_name = "env.eval" if args.mode == "eval" else "env.train"

    log_block(
        logger,
        "STEP 1: Load Hydra config and resolve task/model/env settings",
        refs=[
            f"{MAIN_CONFIG_FILE.relative_to(REPO_ROOT)}:1",
            f"{ENV_CONFIG_FILE.relative_to(REPO_ROOT)}:1",
            code_ref(main),
        ],
        payload={
            "config_name": args.config_name,
            "resolved_runner": summarize_value(cfg.runner),
            "resolved_actor_model": summarize_value(cfg.actor.model),
            "selected_env_name": selected_env_name,
            "selected_env": {
                "assets_path": selected_env_cfg.assets_path,
                "task_name": selected_env_cfg.task_config.task_name,
                "planner_backend": selected_env_cfg.task_config.planner_backend,
                "total_num_envs": selected_env_cfg.total_num_envs,
                "max_episode_steps": selected_env_cfg.max_episode_steps,
                "auto_reset": selected_env_cfg.auto_reset,
                "ignore_terminations": selected_env_cfg.ignore_terminations,
            },
            "resolved_sampling": summarize_value(cfg.algorithm.sampling_params),
        },
    )

    env = RoboTwinEnv(
        cfg=selected_env_cfg,
        num_envs=args.num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info={"rank": 0, "world_size": 1},
    )
    log_block(
        logger,
        f"STEP 2: Build RoboTwinEnv from {selected_env_name}",
        refs=[code_ref(RoboTwinEnv.__init__), code_ref(RoboTwinEnv._init_env)],
        payload={
            "num_envs": env.num_envs,
            "seed": env.seed,
            "task_name": env.task_name,
            "device": str(env.device),
            "auto_reset": env.auto_reset,
            "use_custom_reward": env.use_custom_reward,
            "use_rel_reward": env.use_rel_reward,
        },
    )

    reset_obs, reset_infos = env.reset()
    log_block(
        logger,
        "STEP 3: env.reset() returns extracted observation tensors",
        refs=[code_ref(RoboTwinEnv.reset), code_ref(RoboTwinEnv._extract_obs_image)],
        payload={
            "reset_obs": summarize_value(reset_obs),
            "reset_infos": summarize_value(reset_infos),
        },
    )

    env_output = EnvOutput(obs=reset_obs).to_dict()
    log_block(
        logger,
        "STEP 4: Wrap reset output into EnvOutput transport format",
        refs=[code_ref(EnvOutput.prepare_observations), code_ref(EnvOutput.to_dict)],
        payload={"env_output": summarize_value(env_output)},
    )

    data_kwargs = None
    if cfg.actor.model.get("openpi_data", None) is not None:
        data_kwargs = OmegaConf.to_container(
            cfg.actor.model.openpi_data, resolve=True
        )
    actor_train_cfg = get_openpi_config(
        cfg.actor.model.openpi.config_name,
        model_path=cfg.actor.model.model_path,
        data_kwargs=data_kwargs,
    )

    model: OpenPi0ForRLActionPrediction = get_model(cfg.actor.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    log_block(
        logger,
        "STEP 5: Build OpenPI/pi0.5 policy and its data adapter",
        refs=[
            literal_ref(
                "src/rlinf/models/embodiment/openpi/__init__.py",
                22,
            ),
            literal_ref(
                "src/rlinf/models/embodiment/openpi/dataconfig/__init__.py",
                439,
            ),
            code_ref(actor_train_cfg.data.create),
        ],
        payload={
            "openpi_config_name": cfg.actor.model.openpi.config_name,
            "train_config_model_type": type(actor_train_cfg.model).__name__,
            "data_factory": type(actor_train_cfg.data).__name__,
            "repo_id": actor_train_cfg.data.repo_id,
            "pytorch_weight_path": actor_train_cfg.pytorch_weight_path,
            "model_device": str(next(model.parameters()).device),
            "model_config": {
                "action_chunk": model.config.action_chunk,
                "action_env_dim": model.config.action_env_dim,
                "num_steps": model.config.num_steps,
                "add_value_head": model.config.add_value_head,
            },
        },
    )

    policy_input_obs = model.obs_processor(env_output["obs"])
    log_block(
        logger,
        "STEP 6: obs_processor maps env obs -> policy obs keys",
        refs=[code_ref(OpenPi0ForRLActionPrediction.obs_processor)],
        payload={"policy_input_obs": summarize_value(policy_input_obs)},
    )

    transformed_obs = model.input_transform(policy_input_obs, transpose=False)
    log_block(
        logger,
        "STEP 7: input_transform applies repack/data/model transforms",
        refs=[
            code_ref(OpenPi0ForRLActionPrediction.input_transform),
            code_ref(actor_train_cfg.data.create),
            code_ref(Observation.from_dict),
        ],
        payload={"transformed_obs": summarize_value(transformed_obs)},
    )

    precise_obs = model.precision_processor(transformed_obs)
    observation = Observation.from_dict(precise_obs)
    log_block(
        logger,
        "STEP 8: precision_processor + Observation.from_dict build structured model input",
        refs=[
            code_ref(OpenPi0ForRLActionPrediction.precision_processor),
            code_ref(Observation.from_dict),
        ],
        payload={
            "observation.images": summarize_value(observation.images),
            "observation.image_masks": summarize_value(observation.image_masks),
            "observation.state": summarize_value(observation.state),
            "tokenized_prompt": summarize_value(observation.tokenized_prompt),
            "tokenized_prompt_mask": summarize_value(observation.tokenized_prompt_mask),
        },
    )

    actions, result = model.predict_action_batch(env_output["obs"], mode=args.mode)
    log_block(
        logger,
        "STEP 9: predict_action_batch produces action chunks and PPO-side metadata",
        refs=[
            literal_ref(
                "src/rlinf/workers/rollout/hf/huggingface_worker.py",
                248,
            ),
            literal_ref(
                "src/rlinf/models/embodiment/openpi/openpi_action_model.py",
                518,
            ),
            literal_ref(
                "src/rlinf/models/embodiment/openpi/openpi_action_model.py",
                613,
            ),
            literal_ref(
                "src/rlinf/models/embodiment/openpi/openpi_action_model.py",
                294,
            ),
        ],
        payload={
            "actions": summarize_value(actions),
            "prev_logprobs": summarize_value(result.get("prev_logprobs")),
            "prev_values": summarize_value(result.get("prev_values")),
            "forward_inputs": summarize_value(result.get("forward_inputs")),
        },
    )

    rollout_result = RolloutResult(
        actions=actions,
        prev_logprobs=result.get("prev_logprobs"),
        prev_values=result.get("prev_values"),
        bootstrap_values=None,
        forward_inputs=result.get("forward_inputs", {}),
        versions=torch.zeros(actions.shape[0], actions.shape[1], dtype=torch.float32),
    )
    log_block(
        logger,
        "STEP 10: Package model outputs into RolloutResult like rollout worker does",
        refs=[
            literal_ref("src/rlinf/data/embodied_io_struct.py", 261),
            literal_ref(
                "src/rlinf/workers/rollout/hf/huggingface_worker.py",
                392,
            ),
        ],
        payload={"rollout_result": summarize_value(rollout_result.__dict__)},
    )

    if not args.skip_step:
        next_obs, rewards, terminations, truncations, step_infos = env.step(
            actions, auto_reset=False
        )
        log_block(
            logger,
            "STEP 11: Feed action chunks back into env.step()",
            refs=[code_ref(RoboTwinEnv.step)],
            payload={
                "next_obs": summarize_value(next_obs),
                "rewards": summarize_value(rewards),
                "terminations": summarize_value(terminations),
                "truncations": summarize_value(truncations),
                "step_infos": summarize_value(step_infos),
            },
        )

    env.close()
    log_block(
        logger,
        "DONE: pipeline trace completed",
        refs=[code_ref(RoboTwinEnv.close)],
        payload={"log_file": str(Path(args.log_file).resolve())},
    )


if __name__ == "__main__":
    main()

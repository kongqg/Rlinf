#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import shutil
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

# Keep transformers on the torch path and avoid the tensorflow import chain.
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_FLAX", "0")

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
import tyro

from bentele.utils.robotwin_eval import save_eval_result
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    EnvOutput,
    convert_trajectories_to_batch,
)
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
from rlinf.models import get_model
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.utils.utils import masked_mean


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "src" / "bentele" / "configs" / "embodiment"


@dataclass(frozen=True)
class Args:
    assets_path: str
    model_path: str
    output_dir: str
    config_name: str = "robotwin_place_phone_stand_ppo_openpi_pi05"
    device: str = "cuda"
    seed: int = 1234
    total_steps: int = 4000
    total_num_envs: int = 256
    rollout_epoch: int = 4
    max_steps_per_rollout_epoch: int = 200
    max_episode_steps: int = 200
    global_batch_size: int = 2048
    micro_batch_size: int = 32
    update_epoch: int = 5
    actor_lr: float = 5.0e-6
    value_lr: float = 1.0e-4
    weight_decay: float = 1.0e-2
    clip_grad_norm: float = 1.0
    entropy_bonus: float = 0.0
    log_every: int = 10
    save_every: int = 100
    eval_every: int = 100
    eval_num_envs: int = 128
    eval_rollout_epochs: int = 1
    eval_device: Literal["cpu", "cuda", "auto"] = "cuda"
    save_optimizer_state: bool = False
    wandb_enabled: bool = True
    wandb_project: str = "bentele"
    wandb_run_name: str | None = None
    wandb_log_dir: str | None = None
    wandb_proxy: str | None = None
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    debug_stage_logs: bool = False


class _WandbLogger:
    def __init__(self, run) -> None:
        self._run = run

    def log(self, payload: dict[str, float], *, step: int) -> None:
        if payload:
            self._run.log(payload, step=step)

    def finish(self) -> None:
        self._run.finish()


def log_stage(message: str, *, enabled: bool) -> None:
    if enabled:
        print(f"[local_rl] {message}", flush=True)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_wandb_logger(args: Args, output_dir: Path) -> _WandbLogger | None:
    if not args.wandb_enabled or args.wandb_mode == "disabled":
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed in the current environment. "
            "Install it first or disable wandb."
        ) from exc

    settings = None
    if args.wandb_proxy:
        settings = wandb.Settings(https_proxy=args.wandb_proxy)

    wandb_log_dir = (
        Path(args.wandb_log_dir).expanduser().resolve()
        if args.wandb_log_dir is not None
        else output_dir / "wandb"
    )
    wandb_log_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.wandb_run_name or output_dir.name
    run = wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=asdict(args),
        settings=settings,
        dir=str(wandb_log_dir),
        mode=args.wandb_mode,
        reinit=True,
    )
    return _WandbLogger(run)


def process_nested_dict_for_adv(
    nested_dict: dict[str, Any], rollout_epoch: int
) -> dict[str, Any]:
    ret_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, torch.Tensor):
            new_value = value.reshape(rollout_epoch, -1, *value.shape[1:])
            new_value = new_value.transpose(0, 1)
            new_value = new_value.reshape(new_value.shape[0], -1, *new_value.shape[3:])
            ret_dict[key] = new_value
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_adv(value, rollout_epoch)
    return ret_dict


def process_nested_dict_for_train(
    nested_dict: dict[str, Any], shuffle_id: torch.Tensor
) -> dict[str, Any]:
    ret_dict = {}
    for key, value in nested_dict.items():
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            value = value[:-1]
        if value is None:
            ret_dict[key] = None
        elif isinstance(value, torch.Tensor):
            ret_dict[key] = value.reshape(-1, *value.shape[2:])[shuffle_id]
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_train(value, shuffle_id)
    return ret_dict


def compute_loss_mask(dones: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, actual_bsz, num_action_chunks = dones.shape
    n_chunk_step = dones.shape[0] - 1
    flattened_dones = dones.transpose(1, 2).reshape(-1, actual_bsz)
    flattened_dones = flattened_dones[-(n_chunk_step * num_action_chunks + 1) :]
    flattened_loss_mask = (flattened_dones.cumsum(dim=0) == 0)[:-1]
    loss_mask = flattened_loss_mask.reshape(n_chunk_step, num_action_chunks, actual_bsz)
    loss_mask = loss_mask.transpose(1, 2)
    loss_mask_sum = loss_mask.sum(dim=(0, 2), keepdim=True)
    loss_mask_sum = loss_mask_sum.expand_as(loss_mask)
    return loss_mask, loss_mask_sum


def build_optimizer(
    model: torch.nn.Module, args: Args
) -> torch.optim.Optimizer:
    actor_params = []
    critic_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "value_head" in name or "model.value_head" in name:
            critic_params.append(param)
        else:
            actor_params.append(param)

    betas = (0.9, 0.95)
    param_groups = []
    if actor_params:
        param_groups.append(
            {"params": actor_params, "lr": args.actor_lr, "betas": betas}
        )
    if critic_params:
        param_groups.append(
            {"params": critic_params, "lr": args.value_lr, "betas": betas}
        )

    return torch.optim.AdamW(
        param_groups,
        eps=1.0e-8,
        weight_decay=args.weight_decay,
    )


def compose_cfg(args: Args):
    overrides = [
        f"env.train.total_num_envs={args.total_num_envs}",
        f"env.eval.total_num_envs={args.eval_num_envs}",
        f"env.train.assets_path={args.assets_path}",
        f"env.eval.assets_path={args.assets_path}",
        f"actor.model.model_path={args.model_path}",
        f"runner.logger.log_path={args.output_dir}",
        "runner.logger.logger_backends=[]",
        "runner.val_check_interval=-1",
        "runner.save_interval=-1",
        f"algorithm.rollout_epoch={args.rollout_epoch}",
        f"algorithm.eval_rollout_epoch={args.eval_rollout_epochs}",
        f"algorithm.update_epoch={args.update_epoch}",
        f"algorithm.entropy_bonus={args.entropy_bonus}",
        f"env.train.max_steps_per_rollout_epoch={args.max_steps_per_rollout_epoch}",
        f"env.train.max_episode_steps={args.max_episode_steps}",
        f"env.train.task_config.step_lim={args.max_episode_steps}",
        f"actor.global_batch_size={args.global_batch_size}",
        f"actor.micro_batch_size={args.micro_batch_size}",
        "env.train.video_cfg.save_video=false",
        "env.eval.video_cfg.save_video=false",
    ]
    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return hydra.compose(config_name=args.config_name, overrides=overrides)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists() or dst.exists():
        return
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def prepare_output_dir(base_model_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_if_exists(base_model_path / "config.json", output_dir / "config.json")
    _copy_if_exists(base_model_path / "assets", output_dir / "assets")
    _copy_if_exists(
        base_model_path / "physical-intelligence",
        output_dir / "physical-intelligence",
    )


def save_checkpoint(
    *,
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: Args,
) -> None:
    model_dir = output_dir / "model_state_dict"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "full_weights.pt")
    training_state = {"step": step}
    if args.save_optimizer_state:
        training_state["optimizer"] = optimizer.state_dict()
    torch.save(training_state, output_dir / "training_state.pt")
    metadata = {
        "config_name": args.config_name,
        "base_model_path": str(Path(args.model_path).expanduser().resolve()),
        "saved_step": step,
        "rl_local": True,
    }
    (output_dir / "sft_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def make_initial_env_output(cfg, obs: dict[str, Any]) -> EnvOutput:
    num_envs = cfg.env.train.total_num_envs
    num_action_chunks = cfg.actor.model.num_action_chunks
    zeros = torch.zeros((num_envs, num_action_chunks), dtype=torch.bool)
    return EnvOutput(
        obs=obs,
        final_obs=None,
        rewards=None,
        dones=zeros,
        terminations=zeros.clone(),
        truncations=zeros.clone(),
    )


def env_interact_step(cfg, env: RoboTwinEnv, raw_actions: torch.Tensor):
    chunk_actions = prepare_actions(
        raw_chunk_actions=raw_actions,
        env_type=cfg.env.train.env_type,
        model_type=cfg.actor.model.model_type,
        num_action_chunks=cfg.actor.model.num_action_chunks,
        action_dim=cfg.actor.model.action_dim,
        policy=cfg.actor.model.get("policy_setup", None),
        wm_env_type=cfg.env.train.get("wm_env_type", None),
    )
    obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = (
        env.chunk_step(chunk_actions)
    )
    extracted_obs = obs_list[-1] if isinstance(obs_list, (list, tuple)) else obs_list
    infos = infos_list[-1] if isinstance(infos_list, (list, tuple)) else infos_list
    chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
    final_obs = (
        infos["final_observation"]
        if isinstance(infos, dict) and "final_observation" in infos
        else None
    )

    env_metrics = {}
    if not cfg.env.train.auto_reset:
        if cfg.env.train.ignore_terminations:
            if chunk_truncations[:, -1].any() and "episode" in infos:
                env_metrics = {k: v.cpu() for k, v in infos["episode"].items()}
        elif "episode" in infos:
            env_metrics = {k: v.cpu() for k, v in infos["episode"].items()}

    env_output = EnvOutput(
        obs=extracted_obs,
        final_obs=final_obs,
        rewards=chunk_rewards,
        dones=chunk_dones,
        terminations=chunk_terminations,
        truncations=chunk_truncations,
    )
    return env_output, env_metrics


def collect_rollout(
    cfg,
    env: RoboTwinEnv,
    model: torch.nn.Module,
    *,
    debug_stage_logs: bool,
):
    rollout = EmbodiedRolloutResult(max_episode_length=cfg.env.train.max_episode_steps)
    env_metrics = defaultdict(list)
    n_train_chunk_steps = (
        cfg.env.train.max_steps_per_rollout_epoch // cfg.actor.model.num_action_chunks
    )

    with torch.no_grad():
        for epoch_idx in range(cfg.algorithm.rollout_epoch):
            log_stage(
                f"rollout epoch {epoch_idx + 1}/{cfg.algorithm.rollout_epoch}: reset env",
                enabled=debug_stage_logs,
            )
            env.is_start = True
            obs, _ = env.reset()
            current_env_output = make_initial_env_output(cfg, obs)

            for chunk_idx in range(n_train_chunk_steps):
                log_stage(
                    f"rollout epoch {epoch_idx + 1}: predict chunk {chunk_idx + 1}/{n_train_chunk_steps}",
                    enabled=debug_stage_logs,
                )
                actions, result = model.predict_action_batch(
                    env_obs=current_env_output.obs,
                    mode="train",
                    compute_values=True,
                )
                rollout.append_step_result(
                    ChunkStepResult(
                        actions=result["forward_inputs"].get("action", None),
                        prev_logprobs=result["prev_logprobs"],
                        prev_values=result["prev_values"],
                        forward_inputs=result["forward_inputs"],
                        rewards=current_env_output.rewards,
                        dones=current_env_output.dones,
                        truncations=current_env_output.truncations,
                        terminations=current_env_output.terminations,
                    )
                )

                current_env_output, current_env_metrics = env_interact_step(
                    cfg, env, actions
                )
                for key, value in current_env_metrics.items():
                    env_metrics[key].append(value)

            log_stage(
                f"rollout epoch {epoch_idx + 1}: bootstrap value for final obs",
                enabled=debug_stage_logs,
            )
            _, final_result = model.predict_action_batch(
                env_obs=current_env_output.obs,
                mode="train",
                compute_values=True,
            )
            rollout.append_step_result(
                ChunkStepResult(
                    prev_values=final_result["prev_values"],
                    rewards=current_env_output.rewards,
                    dones=current_env_output.dones,
                    truncations=current_env_output.truncations,
                    terminations=current_env_output.terminations,
                )
            )

            if epoch_idx != cfg.algorithm.rollout_epoch - 1:
                env.update_reset_state_ids()

    trajectory = rollout.to_splited_trajectories(split_size=1)[0]
    batch = convert_trajectories_to_batch([trajectory])
    return batch, env_metrics


def prepare_rollout_batch(cfg, rollout_batch: dict[str, Any]):
    rollout_batch = process_nested_dict_for_adv(
        rollout_batch, cfg.algorithm.rollout_epoch
    )
    if not cfg.env.train.auto_reset and not cfg.env.train.ignore_terminations:
        loss_mask, loss_mask_sum = compute_loss_mask(rollout_batch["dones"])
        if cfg.algorithm.reward_type == "chunk_level":
            loss_mask = loss_mask.any(dim=-1, keepdim=True)
            loss_mask_sum = loss_mask_sum[..., -1:]
        rollout_batch["loss_mask"] = loss_mask
        rollout_batch["loss_mask_sum"] = loss_mask_sum

    adv_kwargs = {
        "task_type": cfg.runner.task_type,
        "adv_type": cfg.algorithm.adv_type,
        "rewards": rollout_batch["rewards"],
        "dones": rollout_batch["dones"],
        "values": rollout_batch.get("prev_values", None),
        "gamma": cfg.algorithm.get("gamma", 1.0),
        "gae_lambda": cfg.algorithm.get("gae_lambda", 1.0),
        "group_size": cfg.algorithm.get("group_size", 1),
        "reward_type": cfg.algorithm.reward_type,
        "loss_mask": rollout_batch.get("loss_mask", None),
        "loss_mask_sum": rollout_batch.get("loss_mask_sum", None),
    }
    rollout_batch.update(calculate_adv_and_returns(**adv_kwargs))
    return rollout_batch


def summarize_env_metrics(env_metrics: dict[str, list[torch.Tensor]]) -> dict[str, Any]:
    summarized = {}
    for key, value in env_metrics.items():
        if value:
            summarized[key] = _json_ready(torch.cat(value, dim=0))
    return summarized


def build_wandb_payload(
    *,
    step: int,
    summary: dict[str, Any],
    env_metrics_summary: dict[str, Any],
) -> dict[str, float]:
    payload: dict[str, float] = {
        "rollout/reward_sum": float(summary["reward_sum"]),
        "rollout/adv_mean": float(summary["adv_mean"]),
        "rollout/return_mean": float(summary["return_mean"]),
    }
    for key, value in summary["train_metrics"].items():
        payload[f"train/{key}"] = float(value)

    for key, value in env_metrics_summary.items():
        if isinstance(value, list):
            if not value:
                continue
            payload[f"env/{key}"] = float(np.mean(value))
        elif isinstance(value, bool):
            payload[f"env/{key}"] = float(value)
        else:
            payload[f"env/{key}"] = float(value)

    payload["step"] = float(step)
    return payload


def run_update(
    cfg,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rollout_batch,
    *,
    debug_stage_logs: bool,
):
    rollout_size = (
        rollout_batch["prev_logprobs"].shape[0] * rollout_batch["prev_logprobs"].shape[1]
    )
    shuffle_id = torch.randperm(rollout_size)
    train_batch = process_nested_dict_for_train(rollout_batch, shuffle_id)

    batch_size_per_rank = cfg.actor.global_batch_size
    if rollout_size % batch_size_per_rank != 0:
        raise ValueError(
            f"rollout_size={rollout_size} is not divisible by actor.global_batch_size={batch_size_per_rank}"
        )
    if batch_size_per_rank % cfg.actor.micro_batch_size != 0:
        raise ValueError(
            "actor.global_batch_size must be divisible by actor.micro_batch_size "
            f"but got {batch_size_per_rank} and {cfg.actor.micro_batch_size}"
        )

    device = next(model.parameters()).device
    gradient_accumulation = batch_size_per_rank // cfg.actor.micro_batch_size
    metrics = defaultdict(list)

    model.train()
    for _ in range(cfg.algorithm.update_epoch):
        global_batches = split_dict_to_chunk(
            train_batch, rollout_size // batch_size_per_rank
        )
        for global_batch in global_batches:
            global_batch_size = global_batch["prev_logprobs"].shape[0]
            micro_batches = split_dict_to_chunk(
                global_batch, global_batch_size // cfg.actor.micro_batch_size
            )
            optimizer.zero_grad(set_to_none=True)
            for batch in micro_batches:
                batch = put_tensor_device(batch, device)
                forward_inputs = batch["forward_inputs"]
                log_stage("forward/backward on one micro batch", enabled=debug_stage_logs)
                output_dict = model(
                    forward_inputs=forward_inputs,
                    compute_logprobs=True,
                    compute_entropy=cfg.algorithm.entropy_bonus > 0,
                    compute_values=cfg.algorithm.adv_type == "gae",
                    use_cache=False,
                )

                loss_kwargs = {
                    "loss_type": cfg.algorithm.loss_type,
                    "logprob_type": cfg.algorithm.logprob_type,
                    "reward_type": cfg.algorithm.reward_type,
                    "single_action_dim": cfg.actor.model.action_dim,
                    "logprobs": output_dict["logprobs"],
                    "values": output_dict.get("values", None),
                    "old_logprobs": batch["prev_logprobs"],
                    "advantages": batch["advantages"],
                    "returns": batch.get("returns", None),
                    "prev_values": batch.get("prev_values", None),
                    "clip_ratio_high": cfg.algorithm.clip_ratio_high,
                    "clip_ratio_low": cfg.algorithm.clip_ratio_low,
                    "value_clip": cfg.algorithm.get("value_clip", None),
                    "huber_delta": cfg.algorithm.get("huber_delta", None),
                    "loss_mask": batch.get("loss_mask", None),
                    "loss_mask_sum": batch.get("loss_mask_sum", None),
                    "max_episode_steps": cfg.env.train.max_episode_steps,
                    "task_type": cfg.runner.task_type,
                    "critic_warmup": False,
                }
                loss, metric_dict = policy_loss(**loss_kwargs)

                entropy_loss = torch.tensor(0.0, device=device)
                if cfg.algorithm.entropy_bonus > 0 and "entropy" in output_dict:
                    entropy = output_dict["entropy"]
                    loss_mask = batch.get("loss_mask", None)
                    entropy_loss = masked_mean(entropy, loss_mask)
                    loss = loss - cfg.algorithm.entropy_bonus * entropy_loss
                metric_dict["actor/entropy_loss"] = float(entropy_loss.detach().item())

                for key, value in metric_dict.items():
                    metrics[key].append(float(value))

                (loss / gradient_accumulation).backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad],
                max_norm=cfg.actor.optim.clip_grad,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            metrics["actor/grad_norm"].append(float(grad_norm))
            metrics["actor/lr"].append(float(optimizer.param_groups[0]["lr"]))

    return {key: float(np.mean(values)) for key, values in metrics.items()}


def evaluate_current_policy(
    cfg,
    model: torch.nn.Module,
    *,
    device: torch.device,
    debug_stage_logs: bool,
) -> dict[str, Any]:
    del debug_stage_logs
    env_cfg = cfg.env.eval
    action_exec_horizon = cfg.actor.model.num_action_chunks
    n_eval_chunk_steps = env_cfg.max_steps_per_rollout_epoch // action_exec_horizon
    eval_env = RoboTwinEnv(
        cfg=env_cfg,
        num_envs=env_cfg.total_num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info={"rank": 0, "world_size": 1},
    )

    eval_metrics = defaultdict(list)
    prev_done = torch.zeros(env_cfg.total_num_envs, dtype=torch.bool)
    obs = None
    start_time = time.time()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for eval_rollout_epoch in range(cfg.algorithm.eval_rollout_epoch):
                episode_finished = False
                if (not env_cfg.auto_reset) or eval_rollout_epoch == 0:
                    eval_env.is_start = True
                    prev_done = torch.zeros(env_cfg.total_num_envs, dtype=torch.bool)
                    obs, _ = eval_env.reset()

                for _ in range(n_eval_chunk_steps):
                    actions, _result = model.predict_action_batch(
                        env_obs=obs,
                        mode="eval",
                        compute_values=False,
                    )
                    actions_to_execute = actions[:, :action_exec_horizon, :]
                    next_obs, _rewards, terminations, truncations, infos = eval_env.step(
                        actions_to_execute, auto_reset=env_cfg.auto_reset
                    )
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
                            if isinstance(value, torch.Tensor):
                                eval_metrics[key].append(value[newly_done].cpu())
                            else:
                                eval_metrics[key].append(
                                    torch.as_tensor(value)[newly_done].cpu()
                                )

                    obs = next_obs
                    if current_dones.any() and not env_cfg.auto_reset:
                        episode_finished = True
                        break

                if episode_finished:
                    break
    finally:
        eval_env.close()
        if was_training:
            model.train()

    concatenated_metrics = {}
    for key, values in eval_metrics.items():
        if values:
            concatenated_metrics[key] = torch.cat(values, dim=0).contiguous().cpu()
    summary = compute_evaluate_metrics([concatenated_metrics])

    return {
        "runtime_seconds": float(time.time() - start_time),
        "num_envs": env_cfg.total_num_envs,
        "eval_rollout_epochs": cfg.algorithm.eval_rollout_epoch,
        "chunk_steps_per_epoch": n_eval_chunk_steps,
        "action_exec_horizon": action_exec_horizon,
        "metrics": {k: _json_ready(v) for k, v in summary.items()},
        "device": str(device),
    }


def append_history(output_dir: Path, summary: dict[str, Any]) -> None:
    history_path = output_dir / "history.jsonl"
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_json_ready(summary), ensure_ascii=False) + "\n")


def main() -> None:
    args = tyro.cli(Args)
    seed_everything(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    base_model_path = Path(args.model_path).expanduser().resolve()
    prepare_output_dir(base_model_path, output_dir)
    wandb_logger = _init_wandb_logger(args, output_dir)

    log_stage("compose config", enabled=True)
    cfg = compose_cfg(args)
    cfg.actor.optim.lr = args.actor_lr
    cfg.actor.optim.value_lr = args.value_lr
    cfg.actor.optim.weight_decay = args.weight_decay
    cfg.actor.optim.clip_grad = args.clip_grad_norm

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log_stage(f"load model from {args.model_path}", enabled=True)
    model = get_model(cfg.actor.model).to(device)
    log_stage(f"model loaded on {device}", enabled=True)
    optimizer = build_optimizer(model, args)
    log_stage("optimizer built", enabled=True)

    log_stage("init RoboTwin env", enabled=True)
    env = RoboTwinEnv(
        cfg=cfg.env.train,
        num_envs=cfg.env.train.total_num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info={"rank": 0, "world_size": 1},
    )
    log_stage("env ready", enabled=True)

    last_summary: dict[str, Any] | None = None
    failure: BaseException | None = None
    start_time = time.time()
    try:
        for step in range(1, args.total_steps + 1):
            rollout_batch, env_metrics = collect_rollout(
                cfg, env, model, debug_stage_logs=args.debug_stage_logs
            )
            rollout_batch = prepare_rollout_batch(cfg, rollout_batch)
            train_metrics = run_update(
                cfg,
                model,
                optimizer,
                rollout_batch,
                debug_stage_logs=args.debug_stage_logs,
            )

            env_metrics_summary = summarize_env_metrics(env_metrics)
            summary = {
                "step": step,
                "elapsed_seconds": time.time() - start_time,
                "rollout_size": int(
                    rollout_batch["prev_logprobs"].shape[0]
                    * rollout_batch["prev_logprobs"].shape[1]
                ),
                "reward_sum": float(rollout_batch["rewards"].sum().item()),
                "adv_mean": float(rollout_batch["advantages"].mean().item()),
                "return_mean": float(rollout_batch["returns"].mean().item()),
                "train_metrics": train_metrics,
                "env_metrics": env_metrics_summary,
            }
            append_history(output_dir, summary)
            last_summary = summary

            if step % args.log_every == 0 or step == 1:
                print(json.dumps(_json_ready(summary), ensure_ascii=False), flush=True)

            if wandb_logger is not None:
                wandb_logger.log(
                    build_wandb_payload(
                        step=step,
                        summary=summary,
                        env_metrics_summary=env_metrics_summary,
                    ),
                    step=step,
                )

            if step % args.save_every == 0 or step == args.total_steps:
                log_stage(f"save checkpoint at step={step}", enabled=True)
                save_checkpoint(
                    output_dir=output_dir,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    args=args,
                )

            if step % args.eval_every == 0 or step == args.total_steps:
                log_stage(f"eval start step={step}", enabled=True)
                try:
                    eval_result = evaluate_current_policy(
                        cfg,
                        model,
                        device=device,
                        debug_stage_logs=args.debug_stage_logs,
                    )
                except Exception:
                    print(f"[local_rl] eval failed at step={step}", flush=True)
                    traceback.print_exc()
                    raise

                eval_json_path = output_dir / "evals" / f"step_{step:07d}.json"
                save_eval_result(eval_result, eval_json_path)
                metrics = eval_result["metrics"]
                print(
                    "[local_rl] "
                    f"eval step={step} "
                    f"success_at_end={metrics.get('success_at_end', 0.0)} "
                    f"success_once={metrics.get('success_once', 0.0)} "
                    f"return={metrics.get('return', 0.0)} "
                    f"episode_len={metrics.get('episode_len', 0.0)} "
                    f"runtime_seconds={eval_result['runtime_seconds']:.2f}",
                    flush=True,
                )
                if wandb_logger is not None:
                    wandb_logger.log(
                        {
                            "eval/success_at_end": float(
                                metrics.get("success_at_end", 0.0)
                            ),
                            "eval/success_once": float(
                                metrics.get("success_once", 0.0)
                            ),
                            "eval/return": float(metrics.get("return", 0.0)),
                            "eval/reward": float(metrics.get("reward", 0.0)),
                            "eval/episode_len": float(
                                metrics.get("episode_len", 0.0)
                            ),
                            "eval/num_trajectories": float(
                                metrics.get("num_trajectories", 0.0)
                            ),
                            "eval/runtime_seconds": float(
                                eval_result["runtime_seconds"]
                            ),
                        },
                        step=step,
                    )
    except BaseException as exc:
        failure = exc
        raise
    finally:
        env.close()
        if wandb_logger is not None:
            wandb_logger.finish()

        final_result = {
            "args": asdict(args),
            "config_name": args.config_name,
            "device": str(device),
            "completed_steps": 0 if last_summary is None else int(last_summary["step"]),
            "last_summary": _json_ready(last_summary) if last_summary is not None else None,
            "failed": failure is not None,
            "failure": repr(failure) if failure is not None else None,
        }
        result_path = output_dir / "local_rl_summary.json"
        result_path.write_text(
            json.dumps(final_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[local_rl] wrote summary to {result_path}", flush=True)


if __name__ == "__main__":
    main()

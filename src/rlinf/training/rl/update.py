from __future__ import annotations

"""Local RL optimization loop.

This module consumes a prepared rollout batch, reshapes it into shuffled global
and micro batches, recomputes current-policy logprobs, and applies the configured
policy/value loss with gradient accumulation.
"""

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from rlinf.algorithms.registry import policy_loss
from rlinf.training.rl.batch import process_nested_dict_for_train
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.utils.utils import masked_mean


def run_update(
    cfg,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rollout_batch: dict[str, Any],
    *,
    debug_stage_logs: bool,
):
    """Run PPO/GRPO-style updates over one rollout buffer."""
    rollout_size = (
        rollout_batch["prev_logprobs"].shape[0] * rollout_batch["prev_logprobs"].shape[1]
    )
    # Shuffle after rollout collection so each optimizer step mixes trajectories
    # instead of seeing environment streams in their original order.
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
        # First split into global optimizer batches, then into micro-batches for
        # memory-safe gradient accumulation.
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
                if debug_stage_logs:
                    print("[local_rl] forward/backward on one micro batch", flush=True)
                output_dict = model(
                    forward_inputs=forward_inputs,
                    compute_logprobs=True,
                    compute_entropy=cfg.algorithm.entropy_bonus > 0,
                    compute_values=cfg.algorithm.adv_type == "gae",
                    use_cache=False,
                )

                # Keep the update loop algorithm-agnostic: the registry decides
                # whether this is PPO, decoupled PPO, or another registered loss.
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
                    # Entropy is maximized, so subtract it from the minimized loss.
                    loss = loss - cfg.algorithm.entropy_bonus * entropy_loss
                metric_dict["actor/entropy_loss"] = float(entropy_loss.detach().item())

                for key, value in metric_dict.items():
                    metrics[key].append(float(value))

                # Divide by accumulation steps so the summed micro-batch gradients
                # match one global-batch gradient scale.
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

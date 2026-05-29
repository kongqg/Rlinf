from __future__ import annotations

from typing import Any

import torch

from rlinf.algorithms.registry import calculate_adv_and_returns


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

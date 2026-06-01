from __future__ import annotations

"""RL rollout 后处理用的 batch 形状整理工具。

rollout worker 收集到的张量通常是按 rollout 顺序排列的。计算 advantage
和训练之前，需要按 rollout epoch 重新折叠、对已经结束的 episode 做 mask，
再 flatten / shuffle 成适合 mini-batch 训练的形状。
"""

from typing import Any

import torch

from rlinf.algorithms.registry import calculate_adv_and_returns


def process_nested_dict_for_adv(
    nested_dict: dict[str, Any], rollout_epoch: int
) -> dict[str, Any]:
    """重新整理重复 rollout epoch，让 advantage 计算看到连续轨迹。"""
    ret_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, torch.Tensor):
            # 原始 layout 会把不同 rollout epoch 交错放在 batch 维里。
            # 这里先 reshape 再 transpose，把同一个环境流里的样本聚到一起，
            # 最后再把 epoch 维重新并回时间维。
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
    """递归 flatten rollout 张量，并应用 mini-batch shuffle 索引。"""
    ret_dict = {}
    for key, value in nested_dict.items():
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            # 这些张量包含额外的 bootstrap 时间点；actor / critic loss 只使用
            # 真实的 T 个 transition step。
            value = value[:-1]
        if value is None:
            ret_dict[key] = None
        elif isinstance(value, torch.Tensor):
            ret_dict[key] = value.reshape(-1, *value.shape[2:])[shuffle_id]
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_train(value, shuffle_id)
    return ret_dict


def compute_loss_mask(dones: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """为非 auto-reset 环境构造有效 step mask。"""
    _, actual_bsz, num_action_chunks = dones.shape
    n_chunk_step = dones.shape[0] - 1
    # 将 chunked done flag 展平到和 advantage 计算一致的时间顺序，
    # 同时保留最后额外的 done 边界点，供 cumsum 判断 episode 是否结束。
    flattened_dones = dones.transpose(1, 2).reshape(-1, actual_bsz)
    flattened_dones = flattened_dones[-(n_chunk_step * num_action_chunks + 1) :]
    # 一条轨迹 done 之后，后续 chunk 位置全部视为无效训练样本。
    flattened_loss_mask = (flattened_dones.cumsum(dim=0) == 0)[:-1]
    loss_mask = flattened_loss_mask.reshape(n_chunk_step, num_action_chunks, actual_bsz)
    loss_mask = loss_mask.transpose(1, 2)
    # loss_mask_sum 会 broadcast 回每个有效位置，用于 actor update 中按
    # 有效长度做 ratio-style 归一化。
    loss_mask_sum = loss_mask.sum(dim=(0, 2), keepdim=True)
    loss_mask_sum = loss_mask_sum.expand_as(loss_mask)
    return loss_mask, loss_mask_sum


def prepare_rollout_batch(cfg, rollout_batch: dict[str, Any]):
    """整理已经收集好的 rollout，并写入训练所需的 advantages / returns。"""
    rollout_batch = process_nested_dict_for_adv(
        rollout_batch, cfg.algorithm.rollout_epoch
    )
    if not cfg.env.train.auto_reset and not cfg.env.train.ignore_terminations:
        loss_mask, loss_mask_sum = compute_loss_mask(rollout_batch["dones"])
        if cfg.algorithm.reward_type == "chunk_level":
            # chunk-level reward 会折叠 action chunk，因此 mask 也必须用同样方式折叠，
            # 保持 reward / loss 的形状一致。
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
    # registry 内部会处理任务类型相关的预处理，所以 batch 准备阶段只需要
    # 这一个算法分发入口。
    rollout_batch.update(calculate_adv_and_returns(**adv_kwargs))
    return rollout_batch

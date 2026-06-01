# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RL loss 和 advantage 共享的张量形状适配工具。

大多数算法实现期望的是比较紧凑的数学形状，比如 ``[time, batch]``
或 ``[batch, chunk]``。真实的具身任务和 reasoning 任务 batch 形状不完全一致，
所以这里集中处理 reshape、transpose、score 汇总和日志后处理。
"""

from typing import Optional

import torch


def huber_loss(error: torch.Tensor, delta: float) -> torch.Tensor:
    """Huber 损失：小误差用二次项，大误差退化为线性项，减小离群值影响。"""
    return torch.where(
        error.abs() < delta, 0.5 * error**2, delta * (error.abs() - 0.5 * delta)
    )


def kl_penalty(
    logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty
) -> torch.FloatTensor:
    """根据当前策略和 reference 策略的 logprob 计算不同形式的 KL 惩罚。"""
    if kl_penalty in ("kl", "k1"):
        # k1 是带符号的 log-ratio 估计量；具体如何 reduce 由调用方决定。
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman 的低方差 KL 近似形式。
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # 先限制指数输入，避免策略和 reference 差距过大时 exp 溢出。
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # full KL 需要传入完整词表 logits，而不是已采样 token 的 logprob。
        raise NotImplementedError

    raise NotImplementedError


def preprocess_embodied_advantages_inputs(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    """计算 advantage / return 之前，把具身任务 batch 统一成算法期望形状。"""
    if kwargs["reward_type"] == "chunk_level":
        # chunk-level reward 会先折叠 action 维度，因为后续 advantage 估计器
        # 只处理每个环境 step 一个标量 reward。
        # rewards, dones, loss_mask, loss_mask_sum: [n_chunk_steps, bsz, num_action_chunks] -> [n_chunk_steps, bsz, 1]
        rewards = rewards.sum(dim=-1, keepdim=True)
        dones = dones.max(dim=-1, keepdim=True)[0]
        if loss_mask is not None:
            loss_mask = loss_mask.max(dim=-1, keepdim=True)[0]
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.max(dim=-1, keepdim=True)[0]

    num_chunk, bsz, chunk_size = rewards.shape
    n_steps = num_chunk * chunk_size
    kwargs.update(
        {
            "num_chunk": num_chunk,
            "batch_size": bsz,
            "chunk_size": chunk_size,
            "n_steps": n_steps,
        }
    )

    # 先把 batch 维和 chunk 内时间维交换，再把 chunk 时间展开成连续时间轴。
    # [num_chunk, bsz, chunk_size] -> [num_chunk, chunk_size, bsz] -> [n_steps, bsz]
    rewards = rewards.transpose(1, 2).reshape(n_steps, bsz)

    if loss_mask is not None:
        loss_mask = loss_mask.transpose(1, 2).reshape(n_steps, bsz)

    # dones 多一个 bootstrap 边界点，因此展平后保留最后 n_steps + 1 个位置。
    flattened_dones_full = dones.transpose(1, 2).reshape(
        (num_chunk + 1) * chunk_size, bsz
    )
    dones = flattened_dones_full[-(n_steps + 1) :]

    if kwargs["adv_type"] == "gae":
        # GAE 需要 V_t 和 V_{t+1}，所以 flatten 后要保留额外的 bootstrap value。
        flattened_values_full = values.transpose(1, 2).reshape(
            (num_chunk + 1) * chunk_size, bsz
        )
        values = flattened_values_full[: n_steps + 1]

    kwargs.update(
        {
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
        }
    )

    return kwargs


def calculate_scores(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    **kwargs,
) -> dict:
    """为 group-relative 方法汇总每条轨迹的无折扣 episodic score。"""
    scores = torch.zeros(kwargs["batch_size"])
    for step in reversed(range(kwargs["n_steps"])):
        # 如果下一个时间点已经 done，说明跨到了新 episode，需要重置后缀和。
        scores = scores * ~dones[step + 1]
        scores += rewards[step]
    scores = scores.reshape(-1, kwargs["group_size"])

    kwargs.update(
        {
            "rewards": scores,
            "dones": dones,
        }
    )

    return kwargs


def postprocess_embodied_advantages_outputs(
    advantages: torch.Tensor,
    num_chunk: int,
    chunk_size: int,
    returns: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    """把具身任务 advantage / return 从算法形状还原回训练 batch 形状。"""
    res = {}

    # 反向恢复 preprocess_embodied_advantages_inputs 的展平过程：[T, B] -> [chunk, B, chunk_size]。
    advantages = advantages.reshape(num_chunk, chunk_size, -1).transpose(1, 2)
    res.update({"advantages": advantages})

    if returns is not None:
        returns = returns.reshape(num_chunk, chunk_size, -1).transpose(1, 2)
        res.update({"returns": returns})

    return res


def preprocess_reasoning_advantages_inputs(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    logprob: Optional[torch.Tensor] = None,
    ref_logprob: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    # 为了和具身任务接口对齐，这里把 [bsz, seq_len] 转成 [seq_len, bsz]。

    bsz, seq_len = loss_mask.shape
    loss_mask = loss_mask.transpose(0, 1)  # [seq_len, bsz]

    assert rewards.ndim == 1, f"Unsupported reward shape {rewards.shape}"

    if kwargs["adv_type"] == "gae":
        # reasoning 任务通常只有最终 reward；这里把它放到最后一个 token 上，
        # 让 GAE 继续复用 [seq_len, batch] 接口。
        expanded_rewards = torch.zeros(
            (seq_len, bsz), dtype=rewards.dtype, device=rewards.device
        )
        expanded_rewards[-1] = rewards  # 只有最后一个 token 有 reward。
        kwargs.update({"rewards": expanded_rewards})

    elif kwargs["adv_type"] == "grpo":
        grouped_rewards = rewards.reshape(-1, kwargs["group_size"]).contiguous()
        kwargs.update(
            {
                "rewards": grouped_rewards,
            }
        )

    elif kwargs["adv_type"] == "grpo_dynamic":
        # dynamic GRPO 保留每个展平 turn / sequence 的 reward，后续 estimator
        # 再通过 idx_to_traj 还原 trajectory / question 分组关系。
        grouped_rewards = (
            rewards.reshape(-1, kwargs["num_sequence"]).transpose(0, 1).contiguous()
        )
        kwargs.update(
            {
                "rewards": grouped_rewards,
            }
        )

    elif kwargs["adv_type"] == "reinpp":
        kwargs.update({"rewards": rewards.unsqueeze(0)})

    elif kwargs["adv_type"] == "raw":
        kwargs.update({"rewards": rewards})

    else:
        assert False, f"Unsupported adv_type {kwargs['adv_type']}"

    if values is not None:  # [bsz, seq_len]
        assert values.ndim == 2, f"Unsupported values shape {values.shape}"
        values = values.transpose(0, 1)  # [seq_len, bsz]
        # 末尾补 0 作为 bootstrap value，保持 [seq_len + 1, bsz]。
        values = torch.cat(
            [
                values,
                torch.zeros(
                    (1, values.shape[-1]), dtype=values.dtype, device=values.device
                ),
            ],
            dim=0,
        )  # [seq_len+1, bsz]

        kwargs.update({"values": values})

    if logprob is not None:
        logprob = logprob.transpose(0, 1)
        kwargs.update({"logprob": logprob})

    if ref_logprob is not None:
        ref_logprob = ref_logprob.transpose(0, 1)
        kwargs.update({"ref_logprob": ref_logprob})

    # reasoning 任务视作在最后一个 token 结束 episode。
    dones = torch.zeros(seq_len + 1, bsz, dtype=torch.bool, device=rewards.device)
    dones[-1] = True
    kwargs.update(
        {
            "dones": dones,
            "loss_mask": loss_mask,
        }
    )

    return kwargs


def postprocess_reasoning_advantages_outputs(
    advantages: torch.Tensor,
    returns: Optional[torch.Tensor] = None,
) -> dict:
    """把 reasoning 任务的 advantage / return 转回 [bsz, seq_len]。"""

    # contiguous 很重要：这些 tensor 后续可能会通过 channel / IPC 传输。
    advantages = advantages.transpose(0, 1).contiguous()  # [bsz, seq_len]
    if returns is not None:
        returns = returns.transpose(0, 1).contiguous()  # [bsz, seq_len]

    return advantages, returns


def preprocess_loss_inputs(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    logprob_type: Optional[str] = None,
    single_action_dim: Optional[int] = None,
    loss_mask: Optional[torch.Tensor] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    values: Optional[torch.Tensor] = None,
    prev_values: Optional[torch.Tensor] = None,
    returns: Optional[torch.Tensor] = None,
    reward_type: Optional[str] = None,
    versions: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    """按配置的 logprob 粒度整理具身 actor loss 需要的张量。"""
    if reward_type == "chunk_level":
        # chunk-level reward 把整个 action chunk 当成一个训练样本，因此相关
        # 辅助张量也要 flatten 到和 logprob 一致的形状。
        advantages = advantages.flatten()
        if loss_mask is not None:
            loss_mask = loss_mask.flatten()
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.flatten()
        if values is not None:
            values = values.flatten()
        if prev_values is not None:
            prev_values = prev_values.flatten()
        if returns is not None:
            returns = returns.flatten()

    bsz = logprobs.shape[0]
    proximal_logprobs = kwargs.get("proximal_logprobs", None)
    if logprob_type == "token_level":
        # 保留每个 action 维度自己的 logprob，PPO clip 在最细粒度上执行。
        # logprobs, old_logprobs: [bsz, num_action_chunks, action_dim] -> [bsz, num_action_chunks, action_dim]
        logprobs = logprobs.reshape(bsz, -1, single_action_dim)
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim)
        if proximal_logprobs is not None:
            proximal_logprobs = proximal_logprobs.reshape(bsz, -1, single_action_dim)
        if versions is not None:
            versions = versions.reshape(bsz, -1, single_action_dim)
        advantages = advantages.unsqueeze(-1)
        if loss_mask is not None:
            loss_mask = loss_mask.unsqueeze(-1)
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.unsqueeze(-1)

    elif logprob_type == "action_level":
        # 对 action 维度求和，使每个 chunk step 对应一个 joint-action logprob。
        # logprobs, old_logprobs: [bsz, num_action_chunks, action_dim] -> [bsz, num_action_chunks]
        logprobs = logprobs.reshape(bsz, -1, single_action_dim).sum(dim=-1)
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim).sum(dim=-1)
        if proximal_logprobs is not None:
            proximal_logprobs = proximal_logprobs.reshape(
                bsz, -1, single_action_dim
            ).sum(dim=-1)
        if versions is not None:
            versions = versions.reshape(bsz, -1, single_action_dim)[..., 0]

    elif logprob_type == "chunk_level":
        # 对 chunk 内所有 step 和 action 维度一起求和，得到每条 chunk 轨迹
        # 一个标量 logprob，用于 PPO / GRPO 的 ratio 计算。
        # logprobs, old_logprobs: [bsz, num_action_chunks, action_dim] -> [bsz]
        logprobs = logprobs.reshape(bsz, -1, single_action_dim).sum(dim=[1, 2])
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim).sum(dim=[1, 2])
        if proximal_logprobs is not None:
            proximal_logprobs = proximal_logprobs.reshape(
                bsz, -1, single_action_dim
            ).sum(dim=[1, 2])
        if versions is not None:
            versions = versions.reshape(bsz, -1, single_action_dim)[:, 0, 0]

    target_shape = logprobs.shape
    # 将 step-level / scalar 级别的信息扩展到最终 loss 张量形状，方便广播计算。
    advantages = expand_to_target_dim(advantages, target_shape)
    loss_mask = expand_to_target_dim(loss_mask, target_shape)
    loss_mask_sum = expand_to_target_dim(loss_mask_sum, target_shape)
    values = expand_to_target_dim(values, target_shape)
    prev_values = expand_to_target_dim(prev_values, target_shape)
    returns = expand_to_target_dim(returns, target_shape)
    versions = expand_to_target_dim(versions, target_shape)

    kwargs.update(
        {
            "logprobs": logprobs,
            "old_logprobs": old_logprobs,
            "proximal_logprobs": proximal_logprobs,
            "versions": versions,
            "advantages": advantages,
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
            "values": values,
            "prev_values": prev_values,
            "returns": returns,
        }
    )

    return kwargs


def postprocess_loss_metric(metrics_data: dict) -> dict:
    """把 tensor metric detach 成 Python 标量，方便日志系统记录。"""
    for k, v in metrics_data.items():
        if isinstance(v, torch.Tensor):
            metrics_data[k] = v.detach().item()
        elif isinstance(v, (float, int)):
            metrics_data[k] = v
    return metrics_data


def expand_to_target_dim(tensor, target_shape):
    """通过补 trailing singleton 维度，让 tensor 可以广播到目标形状。"""
    if tensor is None:
        return None
    if tensor.shape != target_shape:
        while len(tensor.shape) < len(target_shape):
            tensor = tensor.unsqueeze(-1)
    return tensor


def safe_normalize(array, loss_mask):
    """只用有效 mask 位置统计均值方差，避免无效 padding 影响归一化。"""
    valid_array = array[loss_mask]
    if len(valid_array) > 0:
        mean = valid_array.mean()
        std = valid_array.std()
        array = (array - mean) / (std + 1e-5)

    return array

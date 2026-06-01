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

"""动态 rollout 场景下的 loss scale 规则。

这些函数用于多智能体 / 多 turn 的动态 batch，把 group、agent、turn 三个层级
的归一化因子折叠到 advantage 或 loss_scales 里，避免长轨迹或多 agent 轨迹
在总 loss 中占据过大权重。
"""

import torch

from rlinf.algorithms.registry import register_loss_scale


@register_loss_scale("group_level")
def group_scale(context, batch):
    """应用最外层 group-level 归一化因子。

    这一步对应 GRPO 里的顶层 `1 / G` 缩放。具体来说，它会根据本地动态
    turn 数、数据并行规模和配置的 actor global batch size，重新缩放当前
    advantages，使不同 worker 合并后的有效 batch 贡献保持一致。
    """
    folding_scale = context["folding_scale"]
    assert "group_level" not in folding_scale, (
        "`group_level` loss scaling can only be applied once. Apply the "
        "group-level factor before any `agent_level` or `turn_level` factor."
    )
    context["folding_scale"].append("group_level")

    num_sequence = len(batch["idx_to_traj"])
    dp_world_size = context.get("data_parallel_world_size", 1)
    # 将本地动态 turn 数换算回 actor update 使用的全局 batch 归一化尺度。
    group_scale = num_sequence * dp_world_size / context["actor_global_batch_size"]
    batch["advantages"] *= group_scale
    return batch


@register_loss_scale("agent_level")
def agent_scale(
    context: dict,
    batch: dict[str, torch.Tensor],
) -> dict:
    """在每条轨迹内部应用 agent-level 和临时 turn-level 缩放。

    这一层贡献两个因子：
    - `1 / A_i`：按轨迹 i 中的 agent 数量归一化。
    - 临时的 `1 / T_{i,a}`：让同一轨迹、同一 agent 的所有 turn 先均匀加权。

    这里一条 trajectory 对应采样出的 rollout i，一个 sub_traj 对应该轨迹中
    的 agent a。如果后面继续应用 `turn_level`，后者会把这里的均匀 turn 权重
    进一步改成 token-proportional 权重。
    """
    folding_scale = context["folding_scale"]
    assert "group_level" in folding_scale and "agent_level" not in folding_scale, (
        "`agent_level` loss scaling requires `group_level` to be applied first, "
        "and it can only be applied once."
    )
    context["folding_scale"].append("agent_level")

    idx_to_sub_traj = batch["extra:idx_to_sub_traj"].tolist()
    traj_to_idx = {}
    # 先按轨迹 i 聚合所有 flatten 后的 turn。
    for idx, traj in enumerate(batch["idx_to_traj"]):
        if traj not in traj_to_idx:
            traj_to_idx[traj] = []
        traj_to_idx[traj].append(idx)

    for traj, traj_idxes in traj_to_idx.items():
        sub_traj_to_idx = {}
        # 在一条轨迹内部，再按 sub-trajectory / agent a 分组。
        for idx in traj_idxes:
            sub_traj = idx_to_sub_traj[idx]
            if sub_traj not in sub_traj_to_idx:
                sub_traj_to_idx[sub_traj] = []
            sub_traj_to_idx[sub_traj].append(idx)

        for sub_traj_idxes in sub_traj_to_idx.values():
            for idx in sub_traj_idxes:
                # 对同一 agent 的所有 turn 先施加 1 / A_i 和均匀的 1 / T_{i,a}。
                batch["loss_scales"][idx] *= (
                    1 / len(sub_traj_to_idx) / len(sub_traj_idxes)
                )
    return batch


@register_loss_scale("turn_level")
def turn_scale(
    context: dict,
    batch: dict[str, torch.Tensor],
) -> dict:
    """把均匀 turn 权重细化为按 token 数占比的 turn 权重。

    该函数必须在 `agent_scale` 之后执行。此时 `loss_scales` 已经含有
    `1 / A_i` 和 `1 / T_{i,a}`。`turn_scale` 会把均匀 turn 权重转换为：

    `1 / A_i * 1 / T_{i,a}` ->
    `1 / A_i * |o_t^{i,a}| / sum_t |o_t^{i,a}|`

    因为 actor loss 后续还会在有效 token 上 reduce，这样可以实现按每个 agent
    的总 token 数 `sum_t |o_t^{i,a}|` 做归一化。
    """
    folding_scale = context["folding_scale"]
    assert (
        "group_level" in folding_scale
        and "agent_level" in folding_scale
        and "turn_level" not in folding_scale
    ), (
        "`turn_level` loss scaling requires both `group_level` and "
        "`agent_level` to be applied first, and it can only be applied once."
    )
    context["folding_scale"].append("turn_level")

    idx_to_sub_traj = batch["extra:idx_to_sub_traj"].tolist()
    traj_to_idx = {}
    # 先按轨迹 i 聚合所有 flatten 后的 turn。
    for idx, traj in enumerate(batch["idx_to_traj"]):
        if traj not in traj_to_idx:
            traj_to_idx[traj] = []
        traj_to_idx[traj].append(idx)

    for traj, traj_idxes in traj_to_idx.items():
        sub_traj_to_idx = {}
        # 在一条轨迹内部，再按 sub-trajectory / agent a 分组。
        for idx in traj_idxes:
            sub_traj = idx_to_sub_traj[idx]
            if sub_traj not in sub_traj_to_idx:
                sub_traj_to_idx[sub_traj] = []
            sub_traj_to_idx[sub_traj].append(idx)

        for sub_traj_idxes in sub_traj_to_idx.values():
            # 统计同一 agent 下每个 turn 的有效 response token 数，
            # 再除以该 agent 的总 token 数。
            masked_counts = [
                batch["response_mask"][idx].sum().item() for idx in sub_traj_idxes
            ]
            masked_count_all = sum(masked_counts)
            for i, idx in enumerate(sub_traj_idxes):
                # agent_scale 已经乘过 1 / T_{i,a}。这里再乘
                # T_{i,a} * |o_t| / sum_t |o_t|，合并后得到
                # |o_t| / sum_t |o_t|。
                batch["loss_scales"][idx] *= (
                    1 * len(sub_traj_idxes) * masked_counts[i] / masked_count_all
                )
    return batch

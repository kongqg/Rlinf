from __future__ import annotations

"""本地 RL 训练的优化器构造工具。"""

import torch


def build_actor_critic_optimizer(
    model: torch.nn.Module,
    *,
    actor_lr: float,
    value_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """构造 AdamW，并给主模型参数和 value head 参数设置不同学习率。"""
    actor_params = []
    critic_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # value head 一般需要单独学习率；这里按模型参数名进行分组。
        if "value_head" in name or "model.value_head" in name:
            critic_params.append(param)
        else:
            actor_params.append(param)

    betas = (0.9, 0.95)
    param_groups = []
    if actor_params:
        param_groups.append({"params": actor_params, "lr": actor_lr, "betas": betas})
    if critic_params:
        param_groups.append({"params": critic_params, "lr": value_lr, "betas": betas})
    if not param_groups:
        raise RuntimeError("No trainable parameters found for local RL optimizer.")
    # 每个 param group 已经写入实际学习率；这里的顶层 lr 只是 AdamW 的默认值。
    return torch.optim.AdamW(
        param_groups,
        lr=actor_lr,
        weight_decay=weight_decay,
        eps=1.0e-8,
    )

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

"""算法注册中心：把配置里的算法名映射到具体实现函数。

训练代码只需要传入配置中选择的名字，比如 ``adv_type`` 或 ``loss_type``。
本文件负责做一层很薄的分发，把高层配置连接到 ``advantages.py``、
``losses.py`` 和 ``loss_scales.py`` 中真正的算法实现。
"""

from functools import wraps
from typing import Callable, Optional

import torch

from rlinf.algorithms.utils import (
    calculate_scores,
    postprocess_embodied_advantages_outputs,
    postprocess_loss_metric,
    postprocess_reasoning_advantages_outputs,
    preprocess_embodied_advantages_inputs,
    preprocess_loss_inputs,
    preprocess_reasoning_advantages_inputs,
)

# 将配置里可见的 advantage 名字映射到具体函数。这样新增优势估计方法时，
# 只需要在函数上加装饰器，不需要改 runner 里的 if/else。
ADV_REGISTRY: dict[str, Callable] = {}


def register_advantage(name: str):
    """注册 advantage / return 计算函数的装饰器。"""

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        ADV_REGISTRY[name.lower()] = wrapper
        return wrapper

    return decorator


def get_adv_and_returns(name: str) -> Callable:
    """按名字取出已经注册的 advantage 函数。"""
    if name.lower() not in ADV_REGISTRY:
        raise ValueError(
            f"Advantage '{name}' not registered. Available: {list(ADV_REGISTRY.keys())}"
        )
    return ADV_REGISTRY[name.lower()]


# policy loss 的注册逻辑和 advantage 保持一致，runner 不需要硬编码 PPO、GRPO
# 或其他 loss 变体的分支。
LOSS_REGISTRY: dict[str, Callable] = {}


def register_policy_loss(name: str):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        LOSS_REGISTRY[name.lower()] = wrapper
        return wrapper

    return decorator


def get_policy_loss(name: str):
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Loss {name} not registered")
    return LOSS_REGISTRY[name]


def policy_loss(**kwargs) -> tuple[torch.Tensor, dict]:
    """统一的 actor loss 入口。"""
    loss_type = kwargs["loss_type"]
    loss_fn = get_policy_loss(loss_type)

    task_type = kwargs["task_type"]
    if task_type == "embodied":
        # 具身任务常见 action chunk / token-level 张量形状，先转换成通用
        # loss 函数期望的形状约定。
        kwargs = preprocess_loss_inputs(**kwargs)

    loss, metrics_data = loss_fn(**kwargs)

    if task_type == "embodied":
        # 将具身任务里的 tensor 指标转成训练日志系统可以直接记录的标量。
        metrics_data = postprocess_loss_metric(metrics_data)
    return loss, metrics_data


def calculate_adv_and_returns(**kwargs) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    统一的 advantage / return 计算入口。
    先根据任务类型做输入预处理，再根据注册表分发到具体算法。
    """
    adv_type = kwargs["adv_type"]
    fn = get_adv_and_returns(adv_type)

    task_type = kwargs["task_type"]
    if task_type == "embodied":
        kwargs = preprocess_embodied_advantages_inputs(**kwargs)
        if adv_type != "gae":
            # 非 GAE 的具身算法先把 dense reward 汇总成每条轨迹的最终 score；
            # GAE 则保留逐步 reward 和 critic value，用于时序 bootstrap。
            kwargs = calculate_scores(**kwargs)
        advantages, returns = fn(**kwargs)
        res = postprocess_embodied_advantages_outputs(
            advantages=advantages, returns=returns, **kwargs
        )
    else:
        # reasoning 任务使用 sequence-level reward 形状，需要经过另一套 shape
        # adapter，但后面仍然复用同一套 advantage 注册表。
        kwargs = preprocess_reasoning_advantages_inputs(**kwargs)
        advantages, returns = fn(**kwargs)
        res = postprocess_reasoning_advantages_outputs(advantages, returns)
    return res


# 可选的 loss 后处理 / 缩放函数，通常由配置选择。
LOSS_SCALE_REGISTRY: dict[str, Callable] = {}


def register_loss_scale(name: str):
    def decorator(fn):
        LOSS_SCALE_REGISTRY[name.lower()] = fn
        return fn

    return decorator


def get_loss_scales(names: list[str]) -> list[Callable]:
    loss_scales = []
    for name in names:
        if name not in LOSS_SCALE_REGISTRY:
            raise ValueError(f"Loss scale process {name} not registered")
        loss_scales.append(LOSS_SCALE_REGISTRY[name])
    return loss_scales


# tool-call parser 在这个最小 vendor 版本里是可选能力；保留注册表可以让调用方
# 得到明确错误，而不是因为缺少依赖而静默失败。
TOOLCALL_PARSER_REGISTRY: dict[str, Callable] = {}


def register_toolcall_parser(name: str):
    def decorator(cls):
        TOOLCALL_PARSER_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def get_toolcall_parser(name: str) -> Callable:
    if not TOOLCALL_PARSER_REGISTRY:
        raise NotImplementedError(
            "Tool-call parsers were not kept in the minimal local RLinf vendor."
        )

    if name not in TOOLCALL_PARSER_REGISTRY:
        raise ValueError(f"Toolcall parser {name} not registered")
    cls = TOOLCALL_PARSER_REGISTRY[name]
    return cls()

from __future__ import annotations

"""训练设备选择工具。"""

from typing import TYPE_CHECKING

from rlinf.training.common.distributed import DistributedContext

if TYPE_CHECKING:
    import torch


def device_of(device_name: str) -> "torch.device":
    """把设备名转成 torch.device，并在显式请求 CUDA 时检查可用性。"""
    import torch

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return device


def device_for_training(device_name: str, dist_ctx: DistributedContext) -> "torch.device":
    """根据分布式上下文选择当前进程实际使用的训练设备。"""
    import torch

    if device_name == "cuda" and not torch.cuda.is_available():
        # 本地环境没有 CUDA 时自动退回 CPU，方便做小规模 smoke test。
        return torch.device("cpu")
    if dist_ctx.is_distributed and device_name == "cuda":
        # 分布式训练时，每个进程绑定到自己的 local_rank 对应 GPU。
        return torch.device("cuda", dist_ctx.local_rank)
    return torch.device(device_name)

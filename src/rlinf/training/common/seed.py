from __future__ import annotations

"""训练复现实验用的随机种子工具。"""

import random

import numpy as np


def seed_everything(seed: int) -> None:
    """同时设置 Python、NumPy 和 PyTorch 的随机种子。"""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # 多卡训练时每张 CUDA 设备都要设置，避免不同 rank / device 行为不一致。
        torch.cuda.manual_seed_all(seed)


def _set_seed(seed: int) -> None:
    """兼容旧调用路径的内部别名。"""
    seed_everything(seed)

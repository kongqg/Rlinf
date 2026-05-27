from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_seed(seed: int) -> None:
    seed_everything(seed)

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch.distributed.fsdp import FSDP, FullStateDictConfig, StateDictType


@contextmanager
def full_state_dict_context(model: torch.nn.Module) -> Iterator[None]:
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            yield
    else:
        yield


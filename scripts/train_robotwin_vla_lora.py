#!/usr/bin/env python3
"""LoRA SFT entrypoint for RoboTwin place_phone_stand on pi0.5.

This is a thin preset over `train_robotwin_vla.py`: the data path, loss,
checkpoint format, W&B logging, and optional FSDP training loop stay the same,
while RLinf-style PEFT LoRA is enabled for the OpenPI VLM subtree.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import tyro

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from bentele.training.sft.args import Args as BaseArgs


@dataclass(frozen=True)
class Args(BaseArgs):
    is_lora: bool = True
    lora_rank: int = 32
    lora_path: str | None = None
    wandb_tags: tuple[str, ...] = ("lora",)


if __name__ == "__main__":
    args = tyro.cli(Args)
    from bentele.training.sft.trainer import run

    run(args)

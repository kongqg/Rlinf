from __future__ import annotations

from dataclasses import dataclass

import tyro

from rlinf.projects.robotwin.adapters.sft_task import RobotwinSFTTaskSpec
from rlinf.projects.robotwin.training.sft.args import Args as BaseArgs
from rlinf.training.sft.trainer import run_sft_training


@dataclass(frozen=True)
class Args(BaseArgs):
    is_lora: bool = True
    lora_rank: int = 32
    lora_path: str | None = None
    wandb_tags: tuple[str, ...] = ("lora",)


def main() -> None:
    args = tyro.cli(Args)
    run_sft_training(args, RobotwinSFTTaskSpec())


if __name__ == "__main__":
    main()

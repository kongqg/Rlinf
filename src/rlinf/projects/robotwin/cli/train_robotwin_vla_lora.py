from __future__ import annotations

from dataclasses import dataclass

import tyro

from rlinf.projects.robotwin.training.sft.args import Args as BaseArgs


@dataclass(frozen=True)
class Args(BaseArgs):
    is_lora: bool = True
    lora_rank: int = 32
    lora_path: str | None = None
    wandb_tags: tuple[str, ...] = ("lora",)


def main() -> None:
    args = tyro.cli(Args)
    from rlinf.projects.robotwin.training.sft.trainer import run

    run(args)


if __name__ == "__main__":
    main()


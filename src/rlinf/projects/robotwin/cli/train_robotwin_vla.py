from __future__ import annotations

import tyro

from rlinf.projects.robotwin.training.sft.args import Args


def main() -> None:
    args = tyro.cli(Args)
    from rlinf.projects.robotwin.training.sft.trainer import run_sft_training

    run_sft_training(args)


if __name__ == "__main__":
    main()

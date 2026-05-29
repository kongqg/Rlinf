from __future__ import annotations

import tyro

from rlinf.projects.robotwin.adapters.sft_task import RobotwinSFTTaskSpec
from rlinf.projects.robotwin.training.sft.args import Args
from rlinf.training.sft.trainer import run_sft_training


def main() -> None:
    args = tyro.cli(Args)
    run_sft_training(args, RobotwinSFTTaskSpec())


if __name__ == "__main__":
    main()

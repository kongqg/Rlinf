from __future__ import annotations

import tyro

from rlinf.projects.robotwin.adapters.rl_task import RobotwinRLTaskSpec
from rlinf.projects.robotwin.training.local_rl.args import Args
from rlinf.training.rl.runner import run_rl_training


def main() -> None:
    args = tyro.cli(Args)
    run_rl_training(args, RobotwinRLTaskSpec())


if __name__ == "__main__":
    main()

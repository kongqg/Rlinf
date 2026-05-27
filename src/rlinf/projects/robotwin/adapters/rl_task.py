from __future__ import annotations

from rlinf.projects.robotwin.training.local_rl.args import Args


class RobotwinRLTaskSpec:
    """Lightweight RoboTwin adapter for generic RL entrypoints."""

    def run_rl_training(self, args: Args) -> None:
        from rlinf.projects.robotwin.adapters._rl_impl import main

        main(args)


__all__ = ["RobotwinRLTaskSpec"]


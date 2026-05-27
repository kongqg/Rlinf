from __future__ import annotations

from rlinf.training.rl.interfaces import ArgsT, RLTaskSpec


def run_rl_training(args: ArgsT, task: RLTaskSpec[ArgsT]) -> None:
    """Run RL through a project adapter.

    This first-stage entrypoint enforces the dependency direction:
    `rlinf.training` knows only the adapter protocol, never RoboTwin.
    """

    return task.run_rl_training(args)


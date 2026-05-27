from __future__ import annotations

from typing import Protocol, TypeVar


ArgsT = TypeVar("ArgsT")


class RLTaskSpec(Protocol[ArgsT]):
    """Project adapter interface for the generic RL entrypoint."""

    def run_rl_training(self, args: ArgsT) -> None:
        """Run project-specific RL without leaking project imports into RLinf core."""


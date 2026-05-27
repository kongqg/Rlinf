from __future__ import annotations

from typing import Protocol, TypeVar


ArgsT = TypeVar("ArgsT")


class SFTTaskSpec(Protocol[ArgsT]):
    """Project adapter interface for the generic SFT entrypoint."""

    def run_sft_training(self, args: ArgsT) -> None:
        """Run project-specific SFT without leaking project imports into RLinf core."""


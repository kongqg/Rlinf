from __future__ import annotations

from rlinf.projects.robotwin.training.sft.args import Args


class RobotwinSFTTaskSpec:
    """Lightweight RoboTwin/pi0.5 adapter for generic SFT entrypoints."""

    def run_sft_training(self, args: Args) -> None:
        from rlinf.projects.robotwin.adapters._sft_impl import run

        run(args)


__all__ = ["RobotwinSFTTaskSpec"]


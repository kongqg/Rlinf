from __future__ import annotations

from rlinf.training.sft.interfaces import ArgsT, SFTTaskSpec


def run_sft_training(args: ArgsT, task: SFTTaskSpec[ArgsT]) -> None:
    """Run SFT through a project adapter.

    First-stage decoupling keeps project-specific data/model/eval code outside
    `rlinf.training`. The adapter owns the concrete recipe; this generic entry
    point fixes dependency direction and gives future common-loop extraction a
    stable API.
    """

    return task.run_sft_training(args)


from __future__ import annotations

import tyro

from bentele.training.local_rl.args import Args


def main() -> None:
    args = tyro.cli(Args)
    from bentele.training.local_rl.runner import run_local_rl_training

    run_local_rl_training(args)


if __name__ == "__main__":
    main()

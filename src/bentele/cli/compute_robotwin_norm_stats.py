from __future__ import annotations

from collections.abc import Sequence

import tyro


def _main(
    dataset_dirs: Sequence[str],
    output_dir: str,
    config_name: str = "pi05_aloha_robotwin",
    batch_size: int = 512,
    max_frames_per_dataset: int | None = None,
    use_quantile_norm: bool = False,
) -> None:
    from bentele.robotwin.norm_stats import main as norm_stats_main

    norm_stats_main(
        dataset_dirs=dataset_dirs,
        output_dir=output_dir,
        config_name=config_name,
        batch_size=batch_size,
        max_frames_per_dataset=max_frames_per_dataset,
        use_quantile_norm=use_quantile_norm,
    )


def main() -> None:
    tyro.cli(_main)


if __name__ == "__main__":
    main()

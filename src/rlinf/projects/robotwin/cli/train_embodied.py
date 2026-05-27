from __future__ import annotations

import hydra
import torch.multiprocessing as mp

from rlinf.projects.robotwin.integration.rlinf_embodied import run_sync_embodied_training

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.3",
    config_path="../configs/embodiment",
    config_name="robotwin_place_phone_stand_ppo_openpi_pi05",
)
def main(cfg) -> None:
    run_sync_embodied_training(cfg)


if __name__ == "__main__":
    main()

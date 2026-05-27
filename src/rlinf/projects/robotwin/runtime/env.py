from __future__ import annotations

import os


def ensure_torch_transformers_runtime() -> None:
    """Keep transformers/openpi on the torch path and away from TF/Flax."""

    os.environ["USE_TF"] = "0"
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    os.environ.setdefault("USE_TORCH", "1")
    os.environ.setdefault("USE_FLAX", "0")


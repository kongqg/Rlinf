from __future__ import annotations

from typing import Any


class HybridComponentPlacement:
    """Lightweight stand-in for RLinf placement logic."""

    def __init__(self, cfg: Any, cluster: Any) -> None:
        self.cfg = cfg
        self.cluster = cluster

    def get_strategy(self, component_name: str) -> Any:
        placement_cfg = self.cfg.cluster.component_placement
        if hasattr(placement_cfg, "get"):
            return placement_cfg.get(component_name)
        return getattr(placement_cfg, component_name, None)

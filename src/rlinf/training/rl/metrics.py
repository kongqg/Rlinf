from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.training.common.logging import json_ready


def summarize_env_metrics(env_metrics: dict[str, list[torch.Tensor]]) -> dict[str, Any]:
    summarized = {}
    for key, value in env_metrics.items():
        if value:
            summarized[key] = json_ready(torch.cat(value, dim=0))
    return summarized


def build_wandb_payload(
    *,
    step: int,
    summary: dict[str, Any],
    env_metrics_summary: dict[str, Any],
) -> dict[str, float]:
    payload: dict[str, float] = {
        "rollout/reward_sum": float(summary["reward_sum"]),
        "rollout/adv_mean": float(summary["adv_mean"]),
        "rollout/return_mean": float(summary["return_mean"]),
    }
    for key, value in summary["train_metrics"].items():
        payload[f"train/{key}"] = float(value)

    for key, value in env_metrics_summary.items():
        if isinstance(value, list):
            if not value:
                continue
            payload[f"env/{key}"] = float(np.mean(value))
        elif isinstance(value, bool):
            payload[f"env/{key}"] = float(value)
        else:
            payload[f"env/{key}"] = float(value)

    payload["step"] = float(step)
    return payload


def append_history(output_dir: Path, summary: dict[str, Any]) -> None:
    history_path = output_dir / "history.jsonl"
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(json_ready(summary), ensure_ascii=False) + "\n")

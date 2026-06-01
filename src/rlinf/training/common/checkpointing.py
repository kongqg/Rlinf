from __future__ import annotations

"""checkpoint / 训练状态落盘相关的简单写文件工具。"""

import json
from pathlib import Path
from typing import Any

from rlinf.training.common.logging import json_ready


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """把训练状态写成格式化 JSON，并自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        # json_ready 会把 Path、Tensor、ndarray 等对象转换成 JSON 可序列化类型。
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

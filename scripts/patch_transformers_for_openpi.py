#!/usr/bin/env python3
"""Install OpenPI's local transformers replacements into site-packages."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

EXPECTED_TRANSFORMERS_VERSION = "4.53.2"


def copy_tree(src: Path, dst: Path) -> None:
    for path in src.rglob("*"):
        relative = path.relative_to(src)
        target = dst / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src" / "openpi" / "models_pytorch" / "transformers_replace"
    if not src_dir.exists():
        print(f"OpenPI transformers replacements not found: {src_dir}", file=sys.stderr)
        return 1

    try:
        import transformers
    except ImportError:
        print("transformers is not installed. Run `pip install -r requirements.txt` first.", file=sys.stderr)
        return 1

    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        print(
            "Unexpected transformers version: "
            f"{transformers.__version__} (expected {EXPECTED_TRANSFORMERS_VERSION}).",
            file=sys.stderr,
        )
        return 1

    transformers_root = Path(transformers.__file__).resolve().parent
    copy_tree(src_dir, transformers_root)
    print(f"Patched transformers in {transformers_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

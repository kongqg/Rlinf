#!/usr/bin/env python3
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from bentele.cli.train_robotwin_vla import main


if __name__ == "__main__":
    main()

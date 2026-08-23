"""Compatibility shim. Prefer: python scripts/quality/run_r13_example_trace.py"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_TARGET = Path(__file__).resolve().parent.joinpath("quality", "run_r13_example_trace.py")

if __name__ == "__main__":
    sys.argv[0] = str(_TARGET)
    runpy.run_path(str(_TARGET), run_name="__main__")

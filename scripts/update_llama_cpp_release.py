"""Compatibility shim. Prefer: python scripts/ops/update_llama_cpp_release.py"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_TARGET = Path(__file__).resolve().parent.joinpath("ops", "update_llama_cpp_release.py")

if __name__ == "__main__":
    sys.argv[0] = str(_TARGET)
    runpy.run_path(str(_TARGET), run_name="__main__")

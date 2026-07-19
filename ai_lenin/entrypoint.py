"""Unified project entrypoint.

Run with:
    python -m ai_lenin.entrypoint
"""

from __future__ import annotations

import asyncio
import platform

from src.main import async_main


def main() -> int:
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        return 130
    except Exception:  # noqa: BLE001
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

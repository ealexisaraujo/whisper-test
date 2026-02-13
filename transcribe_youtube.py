#!/usr/bin/env python3
"""Thin shim — all logic lives in the ``transcribe`` package.

Preserves backward-compatible imports (``from transcribe_youtube import X``)
and the ``python transcribe_youtube.py`` CLI entry point.
"""

from transcribe import *  # noqa: F401,F403
from transcribe import main

if __name__ == "__main__":
    raise SystemExit(main())

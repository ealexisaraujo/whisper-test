#!/usr/bin/env python3
"""Thin shim — all logic lives in the ``transcribe`` package.

Preserves backward-compatible imports (``from transcribe_youtube import X``)
and the ``python transcribe_youtube.py`` CLI entry point.
"""

from transcribe import *  # noqa: F401,F403
from transcribe import main


def __getattr__(name):  # `transcribe_youtube.run_auto` stays importable without loading the auto pipeline eagerly
    if name == "run_auto":
        import transcribe

        return transcribe.run_auto
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    raise SystemExit(main())

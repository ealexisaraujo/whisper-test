"""Shared utility functions used by multiple package modules."""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Any, Optional


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sanitize_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = value.strip("._")
    return value or f"transcript_{int(time.time())}"


def _probe_duration_seconds(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.check_output(command, text=True).strip()
        return float(result)
    except Exception:
        return 0.0

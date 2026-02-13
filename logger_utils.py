"""Logging helpers for long-running transcription pipelines."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator


def configure_logging(log_level: str, log_file: str | None = None) -> None:
    """Configure stream + optional file logging with a consistent format."""
    root = logging.getLogger()
    root.handlers.clear()

    level = getattr(logging, str(log_level).upper(), logging.INFO)
    root.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def format_bytes(num_bytes: int) -> str:
    """Pretty byte formatter for log output."""
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


@contextmanager
def log_stage(logger: logging.Logger, label: str, **details: Any) -> Iterator[None]:
    """Log start/end/failure with elapsed time for pipeline stages."""
    suffix = ""
    if details:
        ordered = ", ".join(f"{key}={details[key]}" for key in sorted(details))
        suffix = f" ({ordered})"

    started_at = time.perf_counter()
    logger.info("Stage started: %s%s", label, suffix)
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - started_at
        logger.exception("Stage failed: %s (elapsed=%.2fs)", label, elapsed)
        raise
    else:
        elapsed = time.perf_counter() - started_at
        logger.info("Stage completed: %s (elapsed=%.2fs)", label, elapsed)

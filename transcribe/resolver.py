"""Input resolution: YouTube URL vs local file."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from logger_utils import format_bytes

from .models import SourceRequest, TranscriptionError

LOGGER = logging.getLogger("transcribe_youtube")


def _normalize_input_source(value: str) -> str:
    raw = value.strip()
    if not raw.lower().startswith(("http://", "https://")):
        return raw

    normalized = re.sub(r"\\([/?=&])", r"\1", raw)
    if normalized != raw:
        LOGGER.warning(
            "Detected escaped URL characters in input; normalized for downloader compatibility."
        )
    return normalized


class InputResolver:
    """Resolves whether the input is a YouTube URL or local media path."""

    YOUTUBE_HOSTS = ("youtube.com", "youtu.be")

    @classmethod
    def resolve(cls, input_value: str) -> SourceRequest:
        input_value = _normalize_input_source(input_value)
        if cls._is_youtube_url(input_value):
            LOGGER.info("Input resolved as YouTube URL.")
            return SourceRequest(source=input_value, is_youtube=True)

        local = Path(input_value).expanduser().resolve()
        if not local.exists():
            raise TranscriptionError(f"Input file not found: {local}")
        try:
            size_label = format_bytes(local.stat().st_size)
        except OSError:
            size_label = "unknown"
        LOGGER.info("Input resolved as local file: %s (size=%s)", local, size_label)
        return SourceRequest(source=str(local), is_youtube=False, local_path=local)

    @classmethod
    def _is_youtube_url(cls, value: str) -> bool:
        value_l = value.lower()
        if not value_l.startswith(("http://", "https://")):
            return False
        return any(host in value_l for host in cls.YOUTUBE_HOSTS)

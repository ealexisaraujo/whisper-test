"""Input resolution: remote URL (YouTube, X) vs local file."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

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
    """Resolves whether the input is a supported remote URL or local media path."""

    PLATFORM_HOSTS = {
        "youtube": ("youtube.com", "youtu.be"),
        "x": ("x.com", "twitter.com"),
    }
    PLATFORM_LABELS = {"youtube": "YouTube", "x": "X"}

    @classmethod
    def resolve(cls, input_value: str) -> SourceRequest:
        input_value = _normalize_input_source(input_value)
        platform = cls._detect_platform(input_value)
        if platform:
            LOGGER.info("Input resolved as %s URL.", cls.PLATFORM_LABELS[platform])
            return SourceRequest(
                source=input_value, is_youtube=platform == "youtube", platform=platform
            )

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
    def _detect_platform(cls, value: str) -> Optional[str]:
        if not value.lower().startswith(("http://", "https://")):
            return None
        host = (urlparse(value).hostname or "").lower()
        for platform, hosts in cls.PLATFORM_HOSTS.items():
            if any(host == h or host.endswith("." + h) for h in hosts):
                return platform
        return None

    @classmethod
    def _is_youtube_url(cls, value: str) -> bool:
        return cls._detect_platform(value) == "youtube"

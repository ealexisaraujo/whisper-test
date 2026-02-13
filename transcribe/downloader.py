"""YouTube audio downloader using yt-dlp."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from logger_utils import format_bytes, log_stage

from .models import SourceMedia, TranscriptionError

LOGGER = logging.getLogger("transcribe_youtube")


def _sanitize_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = value.strip("._")
    return value or f"transcript_{int(time.time())}"


def _extract_youtube_id(url: str) -> Optional[str]:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]

    if host == "youtu.be":
        candidate = parsed.path.strip("/").split("/", 1)[0]
        return candidate or None

    if host.endswith("youtube.com"):
        query = parse_qs(parsed.query)
        candidate = (query.get("v") or [None])[0]
        if candidate:
            return candidate
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"}:
            return parts[1]

    return None


class MediaDownloader:
    """Downloads YouTube audio using yt-dlp."""

    def download(
        self,
        url: str,
        output_dir: Path,
        cookies_file: Optional[str] = None,
        browser_cookies: Optional[str] = None,
    ) -> SourceMedia:
        try:
            import yt_dlp
        except ModuleNotFoundError as exc:
            raise TranscriptionError(
                "yt-dlp is required for YouTube inputs. Install requirements.txt first."
            ) from exc

        output_template = str(output_dir / "%(id)s.%(ext)s")
        ydl_opts: Dict[str, Any] = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "quiet": True,
            "noplaylist": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "m4a",
                }
            ],
        }

        if cookies_file:
            ydl_opts["cookiefile"] = cookies_file
        if browser_cookies:
            ydl_opts["cookiesfrombrowser"] = (browser_cookies,)

        if cookies_file or browser_cookies:
            LOGGER.info("YouTube auth options enabled (path details hidden for safety).")

        LOGGER.info("Starting YouTube download/extract with yt-dlp...")
        try:
            with log_stage(LOGGER, "youtube_download", source=url):
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
        except Exception as exc:
            raise TranscriptionError(f"YouTube download failed: {exc}") from exc

        video_id = info.get("id") or f"youtube_{int(time.time())}"
        title = info.get("title") or video_id
        LOGGER.info("YouTube metadata resolved: id=%s title=%s", video_id, title)
        requested_id = _extract_youtube_id(url)
        if requested_id and video_id and requested_id != video_id:
            raise TranscriptionError(
                "YouTube ID mismatch detected: requested "
                f"'{requested_id}' but downloader resolved '{video_id}'. "
                "This usually happens when the URL contains escaped characters. "
                f"Use this exact URL: https://www.youtube.com/watch?v={requested_id}"
            )

        preferred = output_dir / f"{video_id}.m4a"
        if preferred.exists():
            audio_path = preferred
        else:
            candidates = sorted(output_dir.glob(f"{video_id}.*"), key=lambda p: p.stat().st_mtime)
            if not candidates:
                raise TranscriptionError("YouTube download completed but audio file was not found.")
            audio_path = candidates[-1]

        try:
            audio_size = format_bytes(audio_path.stat().st_size)
        except OSError:
            audio_size = "unknown"
        LOGGER.info("Downloaded audio file: %s (size=%s)", audio_path, audio_size)

        return SourceMedia(
            source=url,
            local_path=audio_path,
            source_name=_sanitize_filename(title),
            metadata={"youtube_id": video_id, "youtube_title": title},
        )

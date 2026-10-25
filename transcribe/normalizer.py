"""Audio normalization to backend-appropriate WAV format."""

from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path
from typing import Optional

from logger_utils import log_stage

from .models import NormalizedMedia, TranscriptionError

LOGGER = logging.getLogger("transcribe_youtube")


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


def _probe_sample_rate(path: Path) -> Optional[int]:
    """Sample rate in Hz of the first audio stream, or None when it cannot be probed."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        return int(subprocess.check_output(command, text=True).strip().splitlines()[0])
    except Exception:
        return None


class AudioNormalizer:
    """Converts any media input into backend-appropriate WAV for predictable behavior."""

    SAMPLE_RATES = {
        "mlx-whisper": 16000,
        "openai": 16000,
        "eg2": 16000,
        "auto": 16000,
    }

    # EBU R128 loudness normalization (integrated -23 LUFS), applied only for the auto pipeline.
    AUTO_LOUDNORM = "loudnorm=I=-23:TP=-1:LRA=11"

    def normalize(self, media_path: Path, backend: str, output_dir: Path) -> NormalizedMedia:
        sample_rate = self.SAMPLE_RATES[backend]
        out = output_dir / f"normalized_{backend}.wav"
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(media_path),
            "-vn",
            "-ac",
            "1",
            *(["-af", self.AUTO_LOUDNORM] if backend == "auto" else []),
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(out),
        ]

        LOGGER.info(
            "Normalizing media to WAV: backend=%s sample_rate=%sHz input=%s output=%s",
            backend,
            sample_rate,
            media_path,
            out,
        )
        LOGGER.debug("Normalization command: %s", " ".join(shlex.quote(part) for part in command))
        try:
            with log_stage(LOGGER, "audio_normalization", backend=backend, sample_rate=sample_rate):
                subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise TranscriptionError("ffmpeg not found. Please install ffmpeg first.") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise TranscriptionError(f"Audio normalization failed: {stderr or exc}") from exc

        duration = _probe_duration_seconds(out)
        LOGGER.info("Normalized audio duration: %.2fs", duration)
        return NormalizedMedia(path=out, duration_sec=duration)

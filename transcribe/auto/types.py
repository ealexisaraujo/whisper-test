"""Plain, JSON-serialisable (via `dataclasses.asdict`) types shared by the auto pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Word:
    """One recognised word with absolute (whole-file) timestamps in seconds."""

    text: str
    start: float
    end: float
    prob: Optional[float] = None


@dataclass
class Chunk:
    """A VAD-derived slice of the audio, in seconds. Chunks never overlap."""

    index: int
    start: float
    end: float


@dataclass
class EngineOutput:
    """Words per chunk index for one ASR engine; `flags` holds hallucination flags per chunk."""

    engine: str
    language: Optional[str]
    chunks: Dict[int, List[Word]] = field(default_factory=dict)
    flags: Dict[int, List[str]] = field(default_factory=dict)


@dataclass
class AudioProfile:
    """Result of the probe stage; `key` routes the engine pair (e.g. `en-phone-noisy`)."""

    duration: float
    speech_seconds: float
    speech_ratio: float
    language: str
    language_windows: List[Tuple[float, str, float]]  # (start_seconds, language, confidence)
    code_switching: bool
    snr_db: Optional[float]
    phone_band: bool
    key: str


@dataclass
class StageStatus:
    """Outcome of one pipeline stage: `ok`, `skipped: <reason>` or `failed`."""

    name: str
    status: str
    detail: str = ""
    seconds: float = 0.0

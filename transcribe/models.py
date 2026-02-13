"""Data models and exception types for the transcription pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


class TranscriptionError(RuntimeError):
    """Base error for transcription failures."""


class CapabilityError(TranscriptionError):
    """Raised when the requested backend/model/output combination is unsupported."""


@dataclass
class SourceRequest:
    source: str
    is_youtube: bool
    local_path: Optional[Path] = None


@dataclass
class SourceMedia:
    source: str
    local_path: Path
    source_name: str
    metadata: Dict[str, Any]


@dataclass
class NormalizedMedia:
    path: Path
    duration_sec: float


@dataclass
class BackendResult:
    text: str
    segments: List[Dict[str, Any]]
    language: Optional[str]
    duration_sec: Optional[float]
    raw: Dict[str, Any]

"""On-disk stage cache for the auto pipeline (engine outputs, VAD spans, probe profile).

A cache key is the sha256 of the stage name plus the caller's JSON config (which carries the audio
sha256). Values are plain JSON (dataclasses are serialised with `asdict`); a missing, unreadable or
corrupt entry is simply a miss. Picker, verify and refine outputs are deliberately not cached.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from .types import AudioProfile, EngineOutput, Word

LOGGER = logging.getLogger("transcribe_youtube")

CACHE_VERSION = 1


def default_root() -> Path:
    override = os.getenv("TRANSCRIBE_AUTO_CACHE_DIR")
    return Path(override).expanduser() if override else Path.home() / ".cache" / "whisper_translate" / "auto"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class StageCache:
    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root is not None else default_root()

    def _path(self, stage: str, key: Dict[str, Any]) -> Path:
        blob = json.dumps({"v": CACHE_VERSION, "stage": stage, "key": key}, sort_keys=True, default=str)
        return self.root / stage / f"{hashlib.sha256(blob.encode('utf-8')).hexdigest()}.json"

    def get(self, stage: str, key: Dict[str, Any]) -> Optional[Any]:
        path = self._path(stage, key)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            LOGGER.warning("auto cache: ignoring unreadable entry %s (%s)", path, exc)
            return None

    def put(self, stage: str, key: Dict[str, Any], value: Any) -> None:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            value = dataclasses.asdict(value)
        path = self._path(stage, key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False)
            os.replace(tmp, path)
        except OSError as exc:  # a cache that cannot write must never fail the run
            LOGGER.warning("auto cache: could not write %s (%s)", path, exc)


# ----------------------------------------------------------------------------- rebuilders


def engine_output_from_dict(data: Dict[str, Any]) -> EngineOutput:
    """Inverse of `asdict(EngineOutput)` after a JSON round trip (chunk keys come back as strings)."""
    return EngineOutput(
        engine=data["engine"],
        language=data.get("language"),
        chunks={
            int(index): [Word(w["text"], float(w["start"]), float(w["end"]), w.get("prob")) for w in words]
            for index, words in (data.get("chunks") or {}).items()
        },
        flags={int(index): list(reasons) for index, reasons in (data.get("flags") or {}).items()},
    )


def profile_from_dict(data: Dict[str, Any]) -> AudioProfile:
    """Inverse of `asdict(AudioProfile)` after a JSON round trip (window tuples come back as lists)."""
    return AudioProfile(
        duration=float(data["duration"]),
        speech_seconds=float(data["speech_seconds"]),
        speech_ratio=float(data["speech_ratio"]),
        language=data["language"],
        language_windows=[(float(s), str(lang), float(conf)) for s, lang, conf in data["language_windows"]],
        code_switching=bool(data["code_switching"]),
        snr_db=data.get("snr_db"),
        phone_band=bool(data["phone_band"]),
        key=data["key"],
    )

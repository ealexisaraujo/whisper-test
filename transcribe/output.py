"""Output writing: TXT, SRT, JSON formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .models import CapabilityError, TranscriptionError


class OutputWriter:
    """Writes TXT/SRT/JSON outputs from the normalized schema."""

    @staticmethod
    def write(
        payload: Dict[str, Any],
        output_formats: Sequence[str],
        output_dir: Path,
        output_basename: str,
    ) -> Dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        written: Dict[str, Path] = {}

        for fmt in output_formats:
            path = output_dir / f"{output_basename}.{fmt}"
            if fmt == "json":
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            elif fmt == "txt":
                path.write_text(_render_txt(payload), encoding="utf-8")
            elif fmt == "srt":
                path.write_text(_render_srt(payload), encoding="utf-8")
            else:
                raise TranscriptionError(f"Unsupported output format: {fmt}")
            written[fmt] = path

        return written


def format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _render_txt(payload: Mapping[str, Any]) -> str:
    segments = payload.get("segments") or []
    if not segments:
        return str(payload.get("text") or "").strip() + "\n"

    lines: List[str] = []
    for seg in segments:
        start = seg.get("start")
        end = seg.get("end")
        speaker = seg.get("speaker")
        text = str(seg.get("text") or "").strip()
        if start is not None and end is not None:
            head = f"[{float(start):.2f} --> {float(end):.2f}]"
        else:
            head = "[N/A]"
        if speaker:
            lines.append(f"{head} Speaker {speaker}: {text}")
        else:
            lines.append(f"{head} {text}")
    return "\n".join(lines) + "\n"


def _render_srt(payload: Mapping[str, Any]) -> str:
    segments = payload.get("segments") or []
    if not segments:
        raise CapabilityError(
            "SRT output requires timestamped segments, but none were returned by this backend/model."
        )

    lines: List[str] = []
    idx = 1
    for seg in segments:
        start = seg.get("start")
        end = seg.get("end")
        text = str(seg.get("text") or "").strip()
        if start is None or end is None:
            raise CapabilityError(
                "SRT output requires start/end timestamps for every segment."
            )

        speaker = seg.get("speaker")
        if speaker:
            text = f"Speaker {speaker}: {text}"

        lines.append(str(idx))
        lines.append(f"{format_srt_timestamp(float(start))} --> {format_srt_timestamp(float(end))}")
        lines.append(text)
        lines.append("")
        idx += 1

    return "\n".join(lines)

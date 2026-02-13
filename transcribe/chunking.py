"""Audio chunking, offset logic, and backend dispatch."""

from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from logger_utils import format_bytes, log_stage

from .backends import TranscriptionBackend, combine_text_from_segments
from .models import BackendResult, NormalizedMedia, TranscriptionError
from .utils import _probe_duration_seconds, _to_float

LOGGER = logging.getLogger("transcribe_youtube")


def run_selected_backend(
    backend_name: str,
    backends: Mapping[str, TranscriptionBackend],
    *,
    audio_path: Path,
    language: Optional[str],
    context_info: Optional[str],
    timeout_seconds: int,
    generation: Mapping[str, Any],
) -> BackendResult:
    try:
        audio_size = format_bytes(audio_path.stat().st_size)
    except OSError:
        audio_size = "unknown"
    LOGGER.info(
        "Dispatching backend=%s audio=%s size=%s language=%s timeout=%ss",
        backend_name,
        audio_path,
        audio_size,
        language or "auto",
        timeout_seconds,
    )
    backend = backends[backend_name]
    return backend.transcribe(
        audio_path=audio_path,
        language=language,
        context_info=context_info,
        timeout_seconds=timeout_seconds,
        generation=generation,
    )


def offset_segments(
    segments: Sequence[Mapping[str, Any]],
    offset_seconds: float,
) -> List[Dict[str, Any]]:
    if not segments or offset_seconds <= 0:
        return [dict(seg) for seg in segments]

    adjusted: List[Dict[str, Any]] = []
    for seg in segments:
        item = dict(seg)
        start = _to_float(item.get("start"))
        end = _to_float(item.get("end"))
        if start is not None:
            item["start"] = start + offset_seconds
        if end is not None:
            item["end"] = end + offset_seconds
        adjusted.append(item)
    return adjusted


def should_chunk_audio(
    *,
    chunk_mode: str,
    backend: str,
    duration_sec: float,
    audio_path: Path,
    chunk_threshold_minutes: int,
    openai_max_file_mb: int,
) -> bool:
    if chunk_mode == "off":
        return False
    if chunk_mode == "force":
        return True

    if duration_sec > 0 and duration_sec >= (chunk_threshold_minutes * 60):
        return True

    if backend == "openai":
        size_mb = audio_path.stat().st_size / (1024 * 1024)
        if size_mb >= openai_max_file_mb:
            return True

    return False


def compute_effective_chunk_seconds(
    *,
    chunk_seconds: int,
) -> int:
    return max(1, chunk_seconds)


def split_audio_into_chunks(
    *,
    input_audio: Path,
    chunk_seconds: int,
    chunks_dir: Path,
) -> List[Path]:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_pattern = chunks_dir / "chunk_%03d.wav"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_audio),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-c",
        "copy",
        str(chunk_pattern),
    ]
    LOGGER.info(
        "Chunking audio: input=%s chunk_seconds=%s output_pattern=%s",
        input_audio,
        chunk_seconds,
        chunk_pattern,
    )
    LOGGER.debug("Chunking command: %s", " ".join(shlex.quote(part) for part in command))
    try:
        with log_stage(LOGGER, "audio_chunking", chunk_seconds=chunk_seconds):
            subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise TranscriptionError(
            f"Audio chunking failed: {stderr or exc}"
        ) from exc

    chunk_paths = sorted(chunks_dir.glob("chunk_*.wav"))
    if not chunk_paths:
        raise TranscriptionError("Chunking produced no files.")
    LOGGER.info("Chunking completed: %s chunks generated.", len(chunk_paths))
    return chunk_paths


def transcribe_with_chunking(
    *,
    args: Any,
    backend: TranscriptionBackend,
    normalized: NormalizedMedia,
    context_info: Optional[str],
    generation: Mapping[str, Any],
    temp_dir: Path,
) -> BackendResult:
    effective_chunk_seconds = compute_effective_chunk_seconds(
        chunk_seconds=args.chunk_seconds,
    )
    should_chunk = should_chunk_audio(
        chunk_mode=args.chunk_mode,
        backend=args.backend,
        duration_sec=normalized.duration_sec,
        audio_path=normalized.path,
        chunk_threshold_minutes=args.chunk_threshold_minutes,
        openai_max_file_mb=args.openai_max_file_mb,
    )

    try:
        normalized_size = format_bytes(normalized.path.stat().st_size)
    except OSError:
        normalized_size = "unknown"
    LOGGER.info(
        "Chunk decision: enabled=%s mode=%s duration_sec=%.2f threshold_min=%s effective_chunk_seconds=%s file_size=%s openai_max_file_mb=%s",
        should_chunk,
        args.chunk_mode,
        normalized.duration_sec,
        args.chunk_threshold_minutes,
        effective_chunk_seconds,
        normalized_size,
        args.openai_max_file_mb,
    )

    if not should_chunk:
        return run_selected_backend(
            args.backend,
            {args.backend: backend},
            audio_path=normalized.path,
            language=args.language,
            context_info=context_info,
            timeout_seconds=max(0, args.timeout_seconds),
            generation=generation,
        )

    LOGGER.info(
        "Long audio detected (%.2f min). Auto-chunking into %s-second segments.",
        normalized.duration_sec / 60 if normalized.duration_sec > 0 else 0.0,
        effective_chunk_seconds,
    )

    chunks_dir = temp_dir / "chunks"
    chunk_paths = split_audio_into_chunks(
        input_audio=normalized.path,
        chunk_seconds=effective_chunk_seconds,
        chunks_dir=chunks_dir,
    )

    if args.save_chunks_dir:
        save_dir = Path(args.save_chunks_dir).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        for chunk_path in chunk_paths:
            shutil.copy2(chunk_path, save_dir / chunk_path.name)
        LOGGER.info("Saved chunk files to %s", save_dir)

    if len(chunk_paths) == 1:
        LOGGER.info("Chunking generated one segment; running single pass.")
        return run_selected_backend(
            args.backend,
            {args.backend: backend},
            audio_path=chunk_paths[0],
            language=args.language,
            context_info=context_info,
            timeout_seconds=max(0, args.timeout_seconds),
            generation=generation,
        )

    all_segments: List[Dict[str, Any]] = []
    text_parts: List[str] = []
    chunk_meta: List[Dict[str, Any]] = []
    running_offset = 0.0
    final_language = None

    for idx, chunk_path in enumerate(chunk_paths, start=1):
        LOGGER.info("Transcribing chunk %s/%s (%s)", idx, len(chunk_paths), chunk_path.name)
        chunk_duration = _probe_duration_seconds(chunk_path)
        chunk_result = run_selected_backend(
            args.backend,
            {args.backend: backend},
            audio_path=chunk_path,
            language=args.language,
            context_info=context_info,
            timeout_seconds=max(0, args.timeout_seconds),
            generation=generation,
        )

        adjusted_segments = offset_segments(chunk_result.segments, running_offset)
        all_segments.extend(adjusted_segments)

        if chunk_result.text.strip():
            text_parts.append(chunk_result.text.strip())

        if final_language is None and chunk_result.language:
            final_language = chunk_result.language

        chunk_meta.append(
            {
                "chunk_index": idx,
                "chunk_file": chunk_path.name,
                "offset_sec": running_offset,
                "duration_sec": chunk_duration,
                "segments": len(chunk_result.segments),
            }
        )

        if chunk_duration > 0:
            running_offset += chunk_duration
        elif chunk_result.duration_sec and chunk_result.duration_sec > 0:
            running_offset += chunk_result.duration_sec

    merged_text = "\n\n".join(text_parts).strip()
    if not merged_text:
        merged_text = combine_text_from_segments(all_segments)

    return BackendResult(
        text=merged_text,
        segments=all_segments,
        language=final_language,
        duration_sec=running_offset if running_offset > 0 else normalized.duration_sec,
        raw={
            "chunking": {
                "enabled": True,
                "chunk_count": len(chunk_paths),
                "chunk_seconds": effective_chunk_seconds,
                "chunks": chunk_meta,
            }
        },
    )

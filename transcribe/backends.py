"""Transcription backend implementations and segment normalizers."""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from logger_utils import format_bytes, log_stage

from .constants import (
    OPENAI_PROMPT_CAPABLE_MODELS,
    OPENAI_TRANSCRIPTION_MODELS,
)
from .models import BackendResult, CapabilityError, TranscriptionError
from .utils import _probe_duration_seconds, _sanitize_filename, _to_float

LOGGER = logging.getLogger("transcribe_youtube")


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _run_with_progress(
    fn: Callable[[], Any],
    *,
    activity: str,
    timeout_seconds: int,
    progress_log_seconds: int,
) -> Any:
    result_holder: Dict[str, Any] = {}
    error_holder: Dict[str, BaseException] = {}
    done = threading.Event()

    def _target() -> None:
        try:
            result_holder["value"] = fn()
        except BaseException as exc:  # pragma: no cover - defensive passthrough
            error_holder["error"] = exc
        finally:
            done.set()

    worker = threading.Thread(
        target=_target,
        name=f"{_sanitize_filename(activity)}_worker",
        daemon=True,
    )
    worker.start()

    started_at = time.perf_counter()
    heartbeat = max(1, int(progress_log_seconds))
    timeout_limit = max(0, int(timeout_seconds))

    while True:
        elapsed = time.perf_counter() - started_at
        wait_for = float(heartbeat)
        if timeout_limit > 0:
            remaining = timeout_limit - elapsed
            if remaining <= 0:
                raise FutureTimeoutError()
            wait_for = min(wait_for, remaining)

        if done.wait(timeout=wait_for):
            break

        elapsed = time.perf_counter() - started_at
        if timeout_limit > 0:
            LOGGER.info(
                "%s still running... elapsed=%.1fs timeout=%ss remaining=%.1fs",
                activity,
                elapsed,
                timeout_limit,
                max(0.0, timeout_limit - elapsed),
            )
        else:
            LOGGER.info("%s still running... elapsed=%.1fs", activity, elapsed)

    if "error" in error_holder:
        raise error_holder["error"]
    return result_holder.get("value")


def _coerce_openai_response(response: Any) -> Dict[str, Any]:
    if isinstance(response, dict):
        return response
    if isinstance(response, str):
        return {"text": response}
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "to_dict"):
        return response.to_dict()

    text = getattr(response, "text", None)
    if text is not None:
        return {"text": text}

    return {"text": str(response)}


class TranscriptionBackend:
    """Backend interface."""

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str],
        context_info: Optional[str],
        timeout_seconds: int,
        generation: Mapping[str, Any],
    ) -> BackendResult:
        raise NotImplementedError


class OpenAIBackend(TranscriptionBackend):
    """OpenAI transcription backend."""

    def __init__(self, model_name: str, base_url: Optional[str] = None):
        self.model_name = model_name
        self.base_url = base_url

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str],
        context_info: Optional[str],
        timeout_seconds: int,
        generation: Mapping[str, Any],
    ) -> BackendResult:
        progress_log_seconds = max(1, int(generation.get("progress_log_seconds", 20)))

        LOGGER.info("OpenAI transcription started: model=%s audio=%s", self.model_name, audio_path)
        if self.model_name not in OPENAI_TRANSCRIPTION_MODELS:
            raise CapabilityError(
                f"Unsupported OpenAI model '{self.model_name}'. Supported: {sorted(OPENAI_TRANSCRIPTION_MODELS)}"
            )

        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise TranscriptionError(
                "OpenAI SDK not installed. Install requirements.txt first."
            ) from exc

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise TranscriptionError(
                "OPENAI_API_KEY is missing. Set it in your environment or .env file."
            )

        client = OpenAI(api_key=api_key, base_url=self.base_url or None)

        if self.model_name == "whisper-1":
            response_format = "verbose_json"
        elif self.model_name == "gpt-4o-transcribe-diarize":
            response_format = "diarized_json"
        else:
            response_format = "json"

        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "response_format": response_format,
        }
        if language:
            kwargs["language"] = language
        if context_info and self.model_name in OPENAI_PROMPT_CAPABLE_MODELS:
            kwargs["prompt"] = context_info

        try:
            input_size = format_bytes(audio_path.stat().st_size)
        except OSError:
            input_size = "unknown"
        LOGGER.info(
            "OpenAI request config: response_format=%s language=%s prompt=%s input_size=%s timeout=%ss progress_log_seconds=%s",
            response_format,
            language or "auto",
            "yes" if "prompt" in kwargs else "no",
            input_size,
            timeout_seconds,
            progress_log_seconds,
        )

        def _request() -> Any:
            with audio_path.open("rb") as handle:
                return client.audio.transcriptions.create(file=handle, **kwargs)

        try:
            with log_stage(LOGGER, "openai_transcription_request", model=self.model_name):
                response = _run_with_progress(
                    _request,
                    activity="OpenAI transcription request",
                    timeout_seconds=timeout_seconds,
                    progress_log_seconds=progress_log_seconds,
                )
        except FutureTimeoutError as exc:
            raise TranscriptionError(
                "OpenAI transcription timed out. Increase --timeout-seconds or retry the request."
            ) from exc
        except Exception as exc:
            raise TranscriptionError(f"OpenAI transcription failed: {exc}") from exc

        data = _coerce_openai_response(response)
        text = str(data.get("text") or "").strip()

        raw_segments = data.get("segments")
        segments = normalize_openai_segments(raw_segments if isinstance(raw_segments, list) else [])

        if not text:
            text = combine_text_from_segments(segments)
        LOGGER.info(
            "OpenAI transcription finished: text_chars=%s segments=%s language=%s",
            len(text),
            len(segments),
            data.get("language") or language or "unknown",
        )

        return BackendResult(
            text=text,
            segments=segments,
            language=(data.get("language") or language),
            duration_sec=float(data.get("duration", 0.0)) or _probe_duration_seconds(audio_path),
            raw=data,
        )


class MLXWhisperBackend(TranscriptionBackend):
    """MLX-optimized Whisper backend for Apple Silicon."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str],
        context_info: Optional[str],
        timeout_seconds: int,
        generation: Mapping[str, Any],
    ) -> BackendResult:
        try:
            import mlx_whisper
        except ModuleNotFoundError as exc:
            raise TranscriptionError(
                "mlx-whisper backend dependencies missing. "
                "Install with: pip install mlx-whisper"
            ) from exc

        progress_log_seconds = max(1, int(generation.get("progress_log_seconds", 20)))
        LOGGER.info(
            "MLX Whisper transcription started: model=%s audio=%s",
            self.model_name,
            audio_path,
        )

        kwargs: Dict[str, Any] = {"path_or_hf_repo": self.model_name}
        if language:
            kwargs["language"] = language
        if context_info:
            kwargs["initial_prompt"] = context_info

        def _transcribe() -> Any:
            return mlx_whisper.transcribe(str(audio_path), **kwargs)

        try:
            result = _run_with_progress(
                _transcribe,
                activity="MLX Whisper transcription",
                timeout_seconds=timeout_seconds,
                progress_log_seconds=progress_log_seconds,
            )
        except FutureTimeoutError as exc:
            raise TranscriptionError(
                "MLX Whisper transcription timed out. Increase --timeout-seconds or use a smaller model."
            ) from exc
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"MLX Whisper transcription failed: {exc}") from exc

        segments = normalize_whisper_segments(result.get("segments") or [])
        text = str(result.get("text") or "").strip() or combine_text_from_segments(segments)
        LOGGER.info(
            "MLX Whisper transcription finished: text_chars=%s segments=%s language=%s",
            len(text),
            len(segments),
            result.get("language") or language or "unknown",
        )

        return BackendResult(
            text=text,
            segments=segments,
            language=result.get("language"),
            duration_sec=_probe_duration_seconds(audio_path),
            raw=result,
        )


class BackendFactory:
    """Builds backend instances from CLI args."""

    @staticmethod
    def create(args: argparse.Namespace) -> TranscriptionBackend:
        if args.backend == "mlx-whisper":
            return MLXWhisperBackend(model_name=args.model)
        if args.backend == "openai":
            return OpenAIBackend(model_name=args.model, base_url=args.openai_base_url)
        raise TranscriptionError(f"Unsupported backend: {args.backend}")


def normalize_openai_segments(raw_segments: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for seg in raw_segments:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        start = _to_float(_first_not_none(seg.get("start"), seg.get("start_time")))
        end = _to_float(_first_not_none(seg.get("end"), seg.get("end_time")))
        speaker = _first_not_none(seg.get("speaker"), seg.get("speaker_id"))
        normalized.append(
            {
                "start": start,
                "end": end,
                "speaker": str(speaker) if speaker is not None else None,
                "text": text,
            }
        )
    return normalized


def normalize_whisper_segments(raw_segments: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for seg in raw_segments:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        normalized.append(
            {
                "start": _to_float(seg.get("start")),
                "end": _to_float(seg.get("end")),
                "speaker": None,
                "text": text,
            }
        )
    return normalized


def combine_text_from_segments(segments: Sequence[Mapping[str, Any]]) -> str:
    parts = [str(seg.get("text") or "").strip() for seg in segments]
    return " ".join([part for part in parts if part]).strip()

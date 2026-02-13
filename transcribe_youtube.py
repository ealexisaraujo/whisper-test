#!/usr/bin/env python3
"""Backend-driven transcription CLI for YouTube and local media files.

Primary backend is VibeVoice-ASR (best-effort on macOS), with optional OpenAI
and local Whisper backends. Backend switching is manual by design.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import parse_qs, urlparse

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal test envs
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False

LOGGER = logging.getLogger("transcribe_youtube")

SUPPORTED_BACKENDS = ("vibevoice", "openai", "whisper")
SUPPORTED_OUTPUT_FORMATS = ("txt", "srt", "json")

DEFAULT_MODELS = {
    "vibevoice": "microsoft/VibeVoice-ASR",
    "openai": "gpt-4o-transcribe",
    "whisper": "medium",
}

OPENAI_TRANSCRIPTION_MODELS = {
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
    "gpt-4o-mini-transcribe-2025-12-15",
    "gpt-4o-transcribe-diarize",
    "whisper-1",
}

OPENAI_PROMPT_CAPABLE_MODELS = {
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
    "gpt-4o-mini-transcribe-2025-12-15",
    "whisper-1",
}

OPENAI_TIMESTAMP_MODELS = {
    "whisper-1",
    "gpt-4o-transcribe-diarize",
}


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


class InputResolver:
    """Resolves whether the input is a YouTube URL or local media path."""

    YOUTUBE_HOSTS = ("youtube.com", "youtu.be")

    @classmethod
    def resolve(cls, input_value: str) -> SourceRequest:
        input_value = _normalize_input_source(input_value)
        if cls._is_youtube_url(input_value):
            return SourceRequest(source=input_value, is_youtube=True)

        local = Path(input_value).expanduser().resolve()
        if not local.exists():
            raise TranscriptionError(f"Input file not found: {local}")
        return SourceRequest(source=str(local), is_youtube=False, local_path=local)

    @classmethod
    def _is_youtube_url(cls, value: str) -> bool:
        value_l = value.lower()
        if not value_l.startswith(("http://", "https://")):
            return False
        return any(host in value_l for host in cls.YOUTUBE_HOSTS)


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

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
        except Exception as exc:
            raise TranscriptionError(f"YouTube download failed: {exc}") from exc

        video_id = info.get("id") or f"youtube_{int(time.time())}"
        title = info.get("title") or video_id
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

        return SourceMedia(
            source=url,
            local_path=audio_path,
            source_name=_sanitize_filename(title),
            metadata={"youtube_id": video_id, "youtube_title": title},
        )


class AudioNormalizer:
    """Converts any media input into backend-appropriate WAV for predictable behavior."""

    SAMPLE_RATES = {
        "vibevoice": 24000,
        "openai": 16000,
        "whisper": 16000,
    }

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
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(out),
        ]

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise TranscriptionError("ffmpeg not found. Please install ffmpeg first.") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise TranscriptionError(f"Audio normalization failed: {stderr or exc}") from exc

        duration = _probe_duration_seconds(out)
        return NormalizedMedia(path=out, duration_sec=duration)


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


class VibeVoiceBackend(TranscriptionBackend):
    """Local VibeVoice backend (best-effort on Mac)."""

    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.requested_device = device

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str],
        context_info: Optional[str],
        timeout_seconds: int,
        generation: Mapping[str, Any],
    ) -> BackendResult:
        del language  # VibeVoice handles multilingual input without explicit language flag.

        try:
            import torch
            from vibevoice.modular.modeling_vibevoice_asr import (
                VibeVoiceASRForConditionalGeneration,
            )
            from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
        except ModuleNotFoundError as exc:
            raise TranscriptionError(
                "VibeVoice dependencies are missing. Install optional stack with: "
                "pip install -r requirements-vibevoice.txt"
            ) from exc

        device = _resolve_torch_device(torch, self.requested_device)
        if platform.system() == "Darwin" and device in {"cpu", "mps"}:
            LOGGER.warning(
                "Running VibeVoice on macOS (%s): best-effort mode; this may be slow or OOM.",
                device,
            )

        dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32

        try:
            processor = VibeVoiceASRProcessor.from_pretrained(
                self.model_name,
                language_model_pretrained_name="Qwen/Qwen2.5-7B",
            )
            model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                self.model_name,
                dtype=dtype,
                attn_implementation="sdpa",
                trust_remote_code=True,
            )
            model = model.to(device)
            model.eval()
        except Exception as exc:
            raise TranscriptionError(
                f"Failed to load VibeVoice model '{self.model_name}' on {device}: {exc}"
            ) from exc

        inputs = processor(
            audio=str(audio_path),
            sampling_rate=None,
            return_tensors="pt",
            add_generation_prompt=True,
            context_info=context_info,
        )
        inputs = {
            k: v.to(device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        max_new_tokens = int(generation.get("max_new_tokens", 512))
        temperature = float(generation.get("temperature", 0.0))
        top_p = float(generation.get("top_p", 1.0))
        do_sample = bool(generation.get("do_sample", False))
        num_beams = int(generation.get("num_beams", 1))
        repetition_penalty = float(generation.get("repetition_penalty", 1.0))

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": processor.pad_id,
            "eos_token_id": processor.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        def _generate() -> Any:
            with torch.no_grad():
                return model.generate(**inputs, **gen_kwargs)

        try:
            if timeout_seconds > 0:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_generate)
                    output_ids = future.result(timeout=timeout_seconds)
            else:
                output_ids = _generate()
        except FutureTimeoutError as exc:
            raise TranscriptionError(
                "VibeVoice inference timed out. Increase --timeout-seconds or lower generation settings."
            ) from exc
        except RuntimeError as exc:
            msg = str(exc)
            if "out of memory" in msg.lower():
                raise TranscriptionError(
                    "VibeVoice ran out of memory. Try --device cpu or use --backend openai."
                ) from exc
            raise TranscriptionError(f"VibeVoice inference failed: {exc}") from exc

        input_len = int(inputs["input_ids"].shape[1])
        generated_ids = output_ids[0, input_len:]

        eos_id = processor.tokenizer.eos_token_id
        if eos_id is not None:
            eos_positions = (generated_ids == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                generated_ids = generated_ids[: int(eos_positions[0])]

        raw_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
        try:
            raw_segments = processor.post_process_transcription(raw_text)
        except Exception:
            raw_segments = []

        segments = normalize_vibevoice_segments(raw_segments)
        text = raw_text or combine_text_from_segments(segments)

        return BackendResult(
            text=text,
            segments=segments,
            language=None,
            duration_sec=_probe_duration_seconds(audio_path),
            raw={"raw_text": raw_text, "segments": raw_segments},
        )


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
        del timeout_seconds
        del generation

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
            with audio_path.open("rb") as handle:
                response = client.audio.transcriptions.create(file=handle, **kwargs)
        except Exception as exc:
            raise TranscriptionError(f"OpenAI transcription failed: {exc}") from exc

        data = _coerce_openai_response(response)
        text = str(data.get("text") or "").strip()

        raw_segments = data.get("segments")
        segments = normalize_openai_segments(raw_segments if isinstance(raw_segments, list) else [])

        if not text:
            text = combine_text_from_segments(segments)

        return BackendResult(
            text=text,
            segments=segments,
            language=(data.get("language") or language),
            duration_sec=float(data.get("duration", 0.0)) or _probe_duration_seconds(audio_path),
            raw=data,
        )


class WhisperBackend(TranscriptionBackend):
    """Local open-source Whisper backend."""

    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.requested_device = device

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str],
        context_info: Optional[str],
        timeout_seconds: int,
        generation: Mapping[str, Any],
    ) -> BackendResult:
        del timeout_seconds
        del generation

        try:
            import torch
            import whisper
        except ModuleNotFoundError as exc:
            raise TranscriptionError(
                "Whisper backend dependencies missing. Install requirements.txt first."
            ) from exc

        device = _resolve_torch_device(torch, self.requested_device)

        try:
            model = whisper.load_model(self.model_name, device=device)
        except Exception as exc:
            raise TranscriptionError(
                f"Failed to load Whisper model '{self.model_name}' on {device}: {exc}"
            ) from exc

        kwargs: Dict[str, Any] = {
            "fp16": device == "cuda",
            "verbose": False,
        }
        if language:
            kwargs["language"] = language
        if context_info:
            kwargs["initial_prompt"] = context_info

        try:
            result = model.transcribe(str(audio_path), **kwargs)
        except Exception as exc:
            raise TranscriptionError(f"Whisper transcription failed: {exc}") from exc

        segments = normalize_whisper_segments(result.get("segments") or [])
        text = str(result.get("text") or "").strip() or combine_text_from_segments(segments)

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
        if args.backend == "vibevoice":
            return VibeVoiceBackend(model_name=args.model, device=args.device)
        if args.backend == "openai":
            return OpenAIBackend(model_name=args.model, base_url=args.openai_base_url)
        if args.backend == "whisper":
            return WhisperBackend(model_name=args.model, device=args.device)
        raise TranscriptionError(f"Unsupported backend: {args.backend}")


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


def parse_output_formats(raw: str) -> List[str]:
    values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not values:
        raise TranscriptionError("--output-formats cannot be empty.")

    invalid = [v for v in values if v not in SUPPORTED_OUTPUT_FORMATS]
    if invalid:
        raise TranscriptionError(
            f"Unsupported output formats: {invalid}. Supported: {SUPPORTED_OUTPUT_FORMATS}"
        )

    # De-duplicate while preserving order.
    deduped: List[str] = []
    for item in values:
        if item not in deduped:
            deduped.append(item)
    return deduped


def build_context_info(context_info: Optional[str], hotwords: Optional[str]) -> Optional[str]:
    ctx = (context_info or "").strip()
    words = [w.strip() for w in (hotwords or "").split(",") if w.strip()]

    if words:
        hotword_line = "Hotwords: " + ", ".join(words)
        if ctx:
            return f"{ctx}\n{hotword_line}"
        return hotword_line

    return ctx or None


def validate_openai_output_constraints(model: str, output_formats: Sequence[str]) -> None:
    if model.startswith("gpt-4o-transcribe") and model != "gpt-4o-transcribe-diarize":
        if "srt" in output_formats:
            raise CapabilityError(
                "SRT output requires timestamps, but this OpenAI model does not return segment timestamps. "
                "Use --model whisper-1 or --model gpt-4o-transcribe-diarize, or switch backend."
            )


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
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise TranscriptionError(
            f"Audio chunking failed: {stderr or exc}"
        ) from exc

    chunk_paths = sorted(chunks_dir.glob("chunk_*.wav"))
    if not chunk_paths:
        raise TranscriptionError("Chunking produced no files.")
    return chunk_paths


def transcribe_with_chunking(
    *,
    args: argparse.Namespace,
    backend: TranscriptionBackend,
    normalized: NormalizedMedia,
    context_info: Optional[str],
    generation: Mapping[str, Any],
    temp_dir: Path,
) -> BackendResult:
    if not should_chunk_audio(
        chunk_mode=args.chunk_mode,
        backend=args.backend,
        duration_sec=normalized.duration_sec,
        audio_path=normalized.path,
        chunk_threshold_minutes=args.chunk_threshold_minutes,
        openai_max_file_mb=args.openai_max_file_mb,
    ):
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
        args.chunk_seconds,
    )

    chunks_dir = temp_dir / "chunks"
    chunk_paths = split_audio_into_chunks(
        input_audio=normalized.path,
        chunk_seconds=args.chunk_seconds,
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
                "chunk_seconds": args.chunk_seconds,
                "chunks": chunk_meta,
            }
        },
    )


def normalize_vibevoice_segments(raw_segments: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for seg in raw_segments:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        start = _to_float(seg.get("start") or seg.get("start_time"))
        end = _to_float(seg.get("end") or seg.get("end_time"))
        speaker = seg.get("speaker") or seg.get("speaker_id")
        normalized.append(
            {
                "start": start,
                "end": end,
                "speaker": str(speaker) if speaker is not None else None,
                "text": text,
            }
        )
    return normalized


def normalize_openai_segments(raw_segments: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for seg in raw_segments:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        start = _to_float(seg.get("start") or seg.get("start_time"))
        end = _to_float(seg.get("end") or seg.get("end_time"))
        speaker = seg.get("speaker") or seg.get("speaker_id")
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


def format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


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


def _build_schema(
    *,
    backend: str,
    model: str,
    source: str,
    language: Optional[str],
    duration_sec: float,
    context_info: Optional[str],
    segments: Sequence[Mapping[str, Any]],
    text: str,
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "backend": backend,
        "model": model,
        "source": source,
        "language": language,
        "duration_sec": duration_sec,
        "context_info": context_info,
        "segments": list(segments),
        "text": text,
        "created_at": _now_utc_iso(),
        "metadata": dict(metadata),
    }


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


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_torch_device(torch_module: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch_module.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def _sanitize_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = value.strip("._")
    return value or f"transcript_{int(time.time())}"


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


def has_transcription_content(result: BackendResult) -> bool:
    if (result.text or "").strip():
        return True

    for seg in result.segments:
        if str(seg.get("text") or "").strip():
            return True

    return False


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


def _build_switch_hints(args: argparse.Namespace) -> List[str]:
    hints: List[str] = []
    for backend in SUPPORTED_BACKENDS:
        if backend == args.backend:
            continue
        cmd = [
            "python",
            str(Path(__file__).resolve()),
            args.input,
            "--backend",
            backend,
        ]
        hints.append(" ".join(shlex.quote(part) for part in cmd))
    return hints


def _default_basename(source_name: str) -> str:
    base = _sanitize_filename(source_name)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{stamp}"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe YouTube/local media using VibeVoice/OpenAI/Whisper backends"
    )
    parser.add_argument("input", help="YouTube URL or local media file path")

    parser.add_argument(
        "--backend",
        choices=SUPPORTED_BACKENDS,
        default="vibevoice",
        help="Transcription backend (default: vibevoice)",
    )
    parser.add_argument("--model", help="Model name override for the selected backend")
    parser.add_argument(
        "--language",
        help="Optional language hint (e.g., en, es). Omit for auto-detection.",
    )

    parser.add_argument(
        "--output-formats",
        default="txt,srt,json",
        help="Comma-separated list: txt,srt,json (default: txt,srt,json)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for generated transcript files (default: current directory)",
    )
    parser.add_argument(
        "--output-basename",
        help="Base filename without extension. Default derives from source name + timestamp.",
    )

    parser.add_argument(
        "--context-info",
        help="Optional free-form context to improve transcription accuracy.",
    )
    parser.add_argument(
        "--hotwords",
        help="Optional comma-separated hotwords (merged into context info).",
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Device hint for local backends (default: auto)",
    )

    parser.add_argument("--cookies", help="Path to YouTube cookies file")
    parser.add_argument(
        "--browser-cookies",
        choices=["chrome", "firefox", "opera", "edge", "safari"],
        help="Browser name to read cookies from for yt-dlp",
    )

    parser.add_argument(
        "--openai-base-url",
        default=os.getenv("OPENAI_BASE_URL"),
        help="Optional OpenAI-compatible base URL override",
    )

    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=int(os.getenv("TRANSCRIBE_TIMEOUT_SECONDS", "0") or "0"),
        help="Inference timeout in seconds (0 disables timeout)",
    )
    parser.add_argument(
        "--chunk-mode",
        choices=("auto", "off", "force"),
        default=os.getenv("TRANSCRIBE_CHUNK_MODE", "auto"),
        help="Long-audio chunking mode (default: auto)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=int(os.getenv("TRANSCRIBE_CHUNK_SECONDS", "1800")),
        help="Chunk length in seconds when chunking is enabled (default: 1800)",
    )
    parser.add_argument(
        "--chunk-threshold-minutes",
        type=int,
        default=int(os.getenv("TRANSCRIBE_CHUNK_THRESHOLD_MINUTES", "45")),
        help="Auto-chunk threshold in minutes (default: 45)",
    )
    parser.add_argument(
        "--openai-max-file-mb",
        type=int,
        default=int(os.getenv("OPENAI_MAX_FILE_MB", "24")),
        help="Auto-chunk when OpenAI input file reaches this size in MB (default: 24)",
    )
    parser.add_argument(
        "--save-chunks-dir",
        help="Optional directory to persist generated chunk WAV files.",
    )

    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)

    parser.add_argument(
        "--log-level",
        default=os.getenv("TRANSCRIBE_LOG_LEVEL", "INFO"),
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_dotenv()
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    try:
        args.model = args.model or DEFAULT_MODELS[args.backend]
        output_formats = parse_output_formats(args.output_formats)

        if args.backend == "openai":
            validate_openai_output_constraints(args.model, output_formats)

        context_info = build_context_info(args.context_info, args.hotwords)

        generation = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "do_sample": args.do_sample,
            "num_beams": args.num_beams,
            "repetition_penalty": args.repetition_penalty,
        }
        if args.chunk_seconds <= 0:
            raise TranscriptionError("--chunk-seconds must be greater than 0.")
        if args.chunk_threshold_minutes <= 0:
            raise TranscriptionError("--chunk-threshold-minutes must be greater than 0.")
        if args.openai_max_file_mb <= 0:
            raise TranscriptionError("--openai-max-file-mb must be greater than 0.")

        request = InputResolver.resolve(args.input)

        with tempfile.TemporaryDirectory(prefix="transcribe_pipeline_") as temp_dir_name:
            temp_dir = Path(temp_dir_name)

            if request.is_youtube:
                LOGGER.info("Resolving YouTube input...")
                media = MediaDownloader().download(
                    request.source,
                    output_dir=temp_dir,
                    cookies_file=args.cookies,
                    browser_cookies=args.browser_cookies,
                )
            else:
                local_path = request.local_path
                assert local_path is not None
                media = SourceMedia(
                    source=request.source,
                    local_path=local_path,
                    source_name=_sanitize_filename(local_path.stem),
                    metadata={},
                )

            LOGGER.info("Normalizing media for backend '%s'...", args.backend)
            normalized = AudioNormalizer().normalize(media.local_path, args.backend, temp_dir)

            backend = BackendFactory.create(args)
            backend_result = transcribe_with_chunking(
                args=args,
                backend=backend,
                normalized=normalized,
                context_info=context_info,
                generation=generation,
                temp_dir=temp_dir,
            )
            if not has_transcription_content(backend_result):
                raise TranscriptionError(
                    "Transcription produced no text content. "
                    "Verify the input URL/media, backend/model, and language hint."
                )

            duration_sec = (
                backend_result.duration_sec
                if backend_result.duration_sec is not None and backend_result.duration_sec > 0
                else normalized.duration_sec
            )
            metadata: Dict[str, Any] = dict(media.metadata)
            if isinstance(backend_result.raw, dict):
                chunking = backend_result.raw.get("chunking")
                if isinstance(chunking, dict):
                    metadata["chunking"] = chunking

            schema = _build_schema(
                backend=args.backend,
                model=args.model,
                source=media.source,
                language=backend_result.language or args.language,
                duration_sec=duration_sec,
                context_info=context_info,
                segments=backend_result.segments,
                text=backend_result.text,
                metadata=metadata,
            )

            output_dir = Path(args.output_dir).expanduser().resolve()
            basename = _sanitize_filename(args.output_basename) if args.output_basename else _default_basename(media.source_name)

            written = OutputWriter.write(
                payload=schema,
                output_formats=output_formats,
                output_dir=output_dir,
                output_basename=basename,
            )

            LOGGER.info("Transcription completed successfully.")
            for fmt, path in written.items():
                LOGGER.info("%s -> %s", fmt.upper(), path)

        return 0

    except TranscriptionError as exc:
        LOGGER.error(str(exc))
        LOGGER.error("No automatic fallback was applied (manual switch policy).")
        for hint in _build_switch_hints(args):
            LOGGER.error("Try: %s", hint)
        return 1
    except Exception as exc:
        LOGGER.error("Unexpected error: %s", exc)
        LOGGER.debug("Unhandled exception details", exc_info=True)
        LOGGER.error("No automatic fallback was applied (manual switch policy).")
        for hint in _build_switch_hints(args):
            LOGGER.error("Try: %s", hint)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

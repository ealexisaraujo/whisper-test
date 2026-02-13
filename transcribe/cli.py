"""CLI entry point: argument parsing, main pipeline orchestration."""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from logger_utils import configure_logging, log_stage

from .backends import BackendFactory
from .chunking import transcribe_with_chunking
from .constants import DEFAULT_MODELS, SUPPORTED_BACKENDS, SUPPORTED_OUTPUT_FORMATS
from .downloader import MediaDownloader
from .models import (
    BackendResult,
    CapabilityError,
    SourceMedia,
    TranscriptionError,
)
from .normalizer import AudioNormalizer
from .output import OutputWriter
from .resolver import InputResolver
from .utils import _sanitize_filename

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal test envs
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False

LOGGER = logging.getLogger("transcribe_youtube")

# Resolve the root-level shim path for switch hints.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SHIM_PATH = _PROJECT_ROOT / "transcribe_youtube.py"


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


def has_transcription_content(result: BackendResult) -> bool:
    if (result.text or "").strip():
        return True

    for seg in result.segments:
        if str(seg.get("text") or "").strip():
            return True

    return False


def _build_switch_hints(args: argparse.Namespace) -> List[str]:
    hints: List[str] = []
    for backend in SUPPORTED_BACKENDS:
        if backend == args.backend:
            continue
        cmd = [
            "python",
            str(_SHIM_PATH),
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
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": dict(metadata),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe YouTube/local media using MLX-Whisper or OpenAI backends"
    )
    parser.add_argument("input", help="YouTube URL or local media file path")

    parser.add_argument(
        "--backend",
        choices=SUPPORTED_BACKENDS,
        default="mlx-whisper",
        help="Transcription backend (default: mlx-whisper)",
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

    parser.add_argument(
        "--log-level",
        default=os.getenv("TRANSCRIBE_LOG_LEVEL", "INFO"),
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    parser.add_argument(
        "--log-file",
        default=os.getenv("TRANSCRIBE_LOG_FILE"),
        help="Optional file path for persistent logs.",
    )
    parser.add_argument(
        "--progress-log-seconds",
        type=int,
        default=int(os.getenv("TRANSCRIBE_PROGRESS_LOG_SECONDS", "20")),
        help="Emit periodic progress logs every N seconds during long backend operations.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_dotenv()
    args = parse_args(argv)

    configure_logging(args.log_level, args.log_file)

    try:
        if args.progress_log_seconds <= 0:
            raise TranscriptionError("--progress-log-seconds must be greater than 0.")

        args.model = args.model or DEFAULT_MODELS[args.backend]
        output_formats = parse_output_formats(args.output_formats)

        LOGGER.info(
            "Run configuration: backend=%s model=%s input=%s language=%s outputs=%s chunk_mode=%s timeout=%ss progress_log_seconds=%s",
            args.backend,
            args.model,
            args.input,
            args.language or "auto",
            ",".join(output_formats),
            args.chunk_mode,
            args.timeout_seconds,
            args.progress_log_seconds,
        )
        if args.log_file:
            LOGGER.info("Persistent log file enabled: %s", args.log_file)

        if args.backend == "openai":
            validate_openai_output_constraints(args.model, output_formats)

        context_info = build_context_info(args.context_info, args.hotwords)

        generation: Mapping[str, Any] = {
            "progress_log_seconds": args.progress_log_seconds,
        }
        if args.chunk_seconds <= 0:
            raise TranscriptionError("--chunk-seconds must be greater than 0.")
        if args.chunk_threshold_minutes <= 0:
            raise TranscriptionError("--chunk-threshold-minutes must be greater than 0.")
        if args.openai_max_file_mb <= 0:
            raise TranscriptionError("--openai-max-file-mb must be greater than 0.")

        with log_stage(LOGGER, "resolve_input"):
            request = InputResolver.resolve(args.input)

        with tempfile.TemporaryDirectory(prefix="transcribe_pipeline_") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            LOGGER.info("Temporary workspace created: %s", temp_dir)

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
            LOGGER.info("Media summary: source=%s duration_sec=%.2f", media.source, normalized.duration_sec)

            backend = BackendFactory.create(args)
            with log_stage(LOGGER, "backend_transcription", backend=args.backend, model=args.model):
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

            with log_stage(LOGGER, "write_outputs", output_dir=output_dir, basename=basename):
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

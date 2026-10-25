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
from .constants import (
    DEFAULT_MODELS,
    DEFAULT_REFINE_MODEL,
    OPENAI_TIMESTAMP_MODELS,
    OPENAI_TRANSCRIPTION_MODELS,
    SUPPORTED_BACKENDS,
    SUPPORTED_OUTPUT_FORMATS,
)
from .downloader import MediaDownloader
from .models import (
    BackendResult,
    CapabilityError,
    SourceMedia,
    TranscriptionError,
)
from .normalizer import AudioNormalizer, _probe_sample_rate
from .downloader import _sanitize_filename
from .output import OutputWriter
from .refine import ClaudeRefiner, build_refine_context
from .resolver import InputResolver

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal test envs
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False

LOGGER = logging.getLogger("transcribe_youtube")

# Resolve the root-level shim path for switch hints.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SHIM_PATH = _PROJECT_ROOT / "transcribe_youtube.py"


def run_auto(*args: Any, **kwargs: Any):
    """`--backend auto` entry point, imported lazily: the other backends never load `transcribe.auto`."""
    from .auto.pipeline import run_auto as _run_auto

    return _run_auto(*args, **kwargs)


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


def validate_refine_constraints(backend: str, model: str) -> None:
    if backend == "openai" and model in OPENAI_TRANSCRIPTION_MODELS and model not in OPENAI_TIMESTAMP_MODELS:
        raise CapabilityError(
            "--refine needs segmented output, but this OpenAI model returns text only. "
            "Use --model whisper-1 or --model gpt-4o-transcribe-diarize, or switch backend."
        )


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_tristate(name: str) -> Optional[bool]:
    """True / False for an explicit on / off value, None when unset (or unrecognised)."""
    value = os.getenv(name, "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def has_transcription_content(result: BackendResult) -> bool:
    if (result.text or "").strip():
        return True

    for seg in result.segments:
        if str(seg.get("text") or "").strip():
            return True

    return False


def _log_auto_warnings(meta: Mapping[str, Any]) -> None:
    """One WARNING per degraded stage (skipped/partial/failed), per note, and for low trust."""
    for stage in meta.get("stages", []):
        status = str(stage.get("status", ""))
        if status not in ("ok", "off"):
            LOGGER.warning("auto stage '%s': %s", stage.get("name"), status)
    for note in meta.get("notes", []):
        LOGGER.warning("auto: %s", note)
    trust = meta.get("trust") or {}
    if trust.get("level") == "uncalibrated":
        LOGGER.info("auto: trust score not calibrated for this profile (%s)", trust.get("reason") or "no validated fit")
    elif trust.get("level") == "low":
        LOGGER.warning(
            "auto: low trust in this transcript (estimated WER %s); review the [unclear: ...] spans",
            "n/a" if trust.get("estimated_wer") is None else f"{trust['estimated_wer']:.1%}",
        )


def _log_auto_refine(meta: Mapping[str, Any], requested: Optional[bool]) -> None:
    """Refine outcome of an auto run: the profile is only known after the probe, so this reports it afterwards."""
    entry = meta.get("profile_entry") or {}
    if requested is False or entry.get("refine") is not False:
        return
    key = meta.get("profile_key")
    LOGGER.info("refine: off (profile %s)", key)
    if requested is True:
        LOGGER.info("refine is off for profile %s; --refine ignored", key)


def _split_csv(raw: Optional[str]) -> List[str]:
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


# CLI options the auto pipeline does not use (it has its own VAD chunking and local engines).
_AUTO_IGNORED_OPTIONS = (
    ("--model", "model"),
    ("--chunk-mode", "chunk_mode"),
    ("--chunk-seconds", "chunk_seconds"),
    ("--chunk-threshold-minutes", "chunk_threshold_minutes"),
    ("--openai-max-file-mb", "openai_max_file_mb"),
    ("--openai-base-url", "openai_base_url"),
    ("--save-chunks-dir", "save_chunks_dir"),
    ("--timeout-seconds", "timeout_seconds"),
    ("--context-info", "context_info"),
)


def _ignored_auto_options(args: argparse.Namespace, user_model: Optional[str]) -> List[str]:
    """Names of options the user set (differently from their defaults) that `--backend auto` ignores."""
    defaults = parse_args(["-"])  # "-" is a plain positional for argparse
    ignored = []
    for flag, dest in _AUTO_IGNORED_OPTIONS:
        value = user_model if dest == "model" else getattr(args, dest, None)
        if value != getattr(defaults, dest, None):
            ignored.append(flag)
    return ignored


def _no_fallback_message(args: argparse.Namespace) -> str:
    if getattr(args, "backend", None) == "auto":
        return "auto pipeline: core stage failed (no engine switching)"
    return "No automatic fallback was applied (manual switch policy)."


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
        description="Transcribe YouTube/X/local media using MLX-Whisper or OpenAI backends"
    )
    parser.add_argument("input", help="YouTube or X (Twitter) URL, or local media file path")

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

    parser.add_argument("--cookies", help="Path to cookies file for yt-dlp (YouTube/X)")
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
        "--refine",
        action=argparse.BooleanOptionalAction,
        default=_env_tristate("TRANSCRIBE_REFINE"),
        help="Fix ASR errors in segment texts with Claude via the logged-in Claude Code CLI "
        "(timestamps unchanged). Unset: off for the existing backends (TRANSCRIBE_REFINE=1 turns it on), "
        "per profile for --backend auto (gated refine; TRANSCRIBE_REFINE=0 or --no-refine turns it off).",
    )
    parser.add_argument(
        "--refine-model",
        default=os.getenv("TRANSCRIBE_REFINE_MODEL") or DEFAULT_REFINE_MODEL,
        help=f"Claude model alias for --refine (default: {DEFAULT_REFINE_MODEL})",
    )

    parser.add_argument(
        "--terms",
        help="Comma-separated names/terms the speakers likely say (auto backend: bias hints for the refine stage).",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        help="Exact number of speakers (auto backend): 1 skips diarization, N>1 passes it to the diarizer.",
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

        user_model = args.model
        args.model = args.model or DEFAULT_MODELS[args.backend]
        output_formats = parse_output_formats(args.output_formats)
        is_auto = args.backend == "auto"
        refine_on = True if (is_auto and args.refine is None) else bool(args.refine)

        LOGGER.info(
            "Run configuration: backend=%s model=%s input=%s language=%s outputs=%s chunk_mode=%s timeout=%ss progress_log_seconds=%s refine=%s",
            args.backend,
            args.model,
            args.input,
            args.language or "auto",
            ",".join(output_formats),
            args.chunk_mode,
            args.timeout_seconds,
            args.progress_log_seconds,
            (f"{args.refine_model} (only if the selected profile enables it)" if is_auto else args.refine_model)
            if refine_on
            else "off",
        )
        if args.log_file:
            LOGGER.info("Persistent log file enabled: %s", args.log_file)

        if args.backend == "openai":
            validate_openai_output_constraints(args.model, output_formats)

        refiner: Optional[ClaudeRefiner] = None
        if refine_on and not is_auto:  # auto runs its own gated refine (missing CLI skips that stage)
            validate_refine_constraints(args.backend, args.model)
            refiner = ClaudeRefiner(model=args.refine_model)  # fails fast if the claude CLI is missing

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

            if request.is_remote:
                LOGGER.info("Resolving remote input (%s)...", request.platform)
                media = MediaDownloader().download(
                    request.source,
                    output_dir=temp_dir,
                    cookies_file=args.cookies,
                    browser_cookies=args.browser_cookies,
                    platform=request.platform,
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

            auto_meta: Optional[Dict[str, Any]] = None
            if is_auto:
                terms = _split_csv(args.terms) + [
                    w for w in _split_csv(args.hotwords) if w not in _split_csv(args.terms)
                ]
                ignored = _ignored_auto_options(args, user_model)
                if ignored:
                    LOGGER.info("auto backend ignores these options: %s", ", ".join(ignored))
                if args.num_speakers is not None and args.num_speakers <= 0:
                    raise TranscriptionError("--num-speakers must be greater than 0.")
                with log_stage(LOGGER, "auto_pipeline"):
                    backend_result, auto_meta = run_auto(
                        normalized.path,
                        dict(media.metadata),
                        language=args.language,
                        terms=terms,
                        num_speakers=args.num_speakers,
                        refine=refine_on,
                        refine_model=args.refine_model,
                        original_sample_rate=_probe_sample_rate(media.local_path),
                    )
                _log_auto_warnings(auto_meta)
                _log_auto_refine(auto_meta, args.refine)
            else:
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

            if refiner is not None:
                refine_context = build_refine_context(
                    media.metadata,
                    title=None if request.is_remote else media.local_path.stem,
                    language=backend_result.language or args.language,
                    notes=context_info,
                )
                with log_stage(LOGGER, "refine", model=refiner.model, segments=len(backend_result.segments)):
                    backend_result = refiner.refine(backend_result, refine_context)

            duration_sec = (
                backend_result.duration_sec
                if backend_result.duration_sec is not None and backend_result.duration_sec > 0
                else normalized.duration_sec
            )
            metadata: Dict[str, Any] = dict(media.metadata)
            if auto_meta is not None:
                metadata["auto"] = auto_meta
                if auto_meta.get("no_speech"):
                    metadata["no_speech"] = True
            elif isinstance(backend_result.raw, dict):
                chunking = backend_result.raw.get("chunking")
                if isinstance(chunking, dict):
                    metadata["chunking"] = chunking
                refine = backend_result.raw.get("refine")
                if isinstance(refine, dict):
                    metadata["refine"] = refine

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
        LOGGER.error(_no_fallback_message(args))
        for hint in _build_switch_hints(args):
            LOGGER.error("Try: %s", hint)
        return 1
    except Exception as exc:
        LOGGER.error("Unexpected error: %s", exc)
        LOGGER.debug("Unhandled exception details", exc_info=True)
        LOGGER.error(_no_fallback_message(args))
        for hint in _build_switch_hints(args):
            LOGGER.error("Try: %s", hint)
        return 1

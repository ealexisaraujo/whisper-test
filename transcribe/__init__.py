"""Transcription pipeline package — re-exports public API for backward compatibility."""

from .backends import (
    BackendFactory,
    MLXWhisperBackend,
    OpenAIBackend,
    TranscriptionBackend,
    _run_with_progress,
    combine_text_from_segments,
    normalize_openai_segments,
    normalize_whisper_segments,
)
from .chunking import (
    compute_effective_chunk_seconds,
    offset_segments,
    run_selected_backend,
    should_chunk_audio,
    split_audio_into_chunks,
    transcribe_with_chunking,
)
from .cli import (
    _build_schema,
    build_context_info,
    has_transcription_content,
    main,
    parse_args,
    parse_output_formats,
    validate_openai_output_constraints,
)
from .constants import (
    DEFAULT_MODELS,
    OPENAI_PROMPT_CAPABLE_MODELS,
    OPENAI_TIMESTAMP_MODELS,
    OPENAI_TRANSCRIPTION_MODELS,
    SUPPORTED_BACKENDS,
    SUPPORTED_OUTPUT_FORMATS,
)
from .downloader import MediaDownloader
from .models import (
    BackendResult,
    CapabilityError,
    NormalizedMedia,
    SourceMedia,
    SourceRequest,
    TranscriptionError,
)
from .normalizer import AudioNormalizer
from .output import OutputWriter, format_srt_timestamp
from .resolver import InputResolver

__all__ = [
    # Models & exceptions
    "TranscriptionError",
    "CapabilityError",
    "SourceRequest",
    "SourceMedia",
    "NormalizedMedia",
    "BackendResult",
    # Constants
    "SUPPORTED_BACKENDS",
    "SUPPORTED_OUTPUT_FORMATS",
    "DEFAULT_MODELS",
    "OPENAI_TRANSCRIPTION_MODELS",
    "OPENAI_PROMPT_CAPABLE_MODELS",
    "OPENAI_TIMESTAMP_MODELS",
    # Pipeline classes
    "InputResolver",
    "MediaDownloader",
    "AudioNormalizer",
    "TranscriptionBackend",
    "OpenAIBackend",
    "MLXWhisperBackend",
    "BackendFactory",
    "OutputWriter",
    # Functions
    "parse_args",
    "main",
    "build_context_info",
    "validate_openai_output_constraints",
    "parse_output_formats",
    "has_transcription_content",
    "run_selected_backend",
    "offset_segments",
    "should_chunk_audio",
    "compute_effective_chunk_seconds",
    "split_audio_into_chunks",
    "transcribe_with_chunking",
    "normalize_openai_segments",
    "normalize_whisper_segments",
    "combine_text_from_segments",
    "format_srt_timestamp",
    "_build_schema",
    "_run_with_progress",
]

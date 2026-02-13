"""Shared constants for backends, models, and output formats."""

SUPPORTED_BACKENDS = ("mlx-whisper", "openai")
SUPPORTED_OUTPUT_FORMATS = ("txt", "srt", "json")

DEFAULT_MODELS = {
    "mlx-whisper": "mlx-community/whisper-turbo",
    "openai": "gpt-4o-transcribe",
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

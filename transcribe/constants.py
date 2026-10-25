"""Shared constants for backends, models, and output formats."""

SUPPORTED_BACKENDS = ("mlx-whisper", "openai", "eg2")
SUPPORTED_OUTPUT_FORMATS = ("txt", "srt", "json")

DEFAULT_MODELS = {
    "mlx-whisper": "mlx-community/whisper-turbo",
    "openai": "gpt-4o-transcribe",
    "eg2": "eg2demo-v5",  # EmbeddingGemma 2 Core ML models; cache dir overridable via EG2_MODEL_DIR
}

EG2_LANGUAGES = {"en"}  # closed lowercase-ASCII English vocabulary (reader_words.json)

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

# Refine stage (opt-in): Claude model alias passed to `claude --model`; never pin a version id here.
DEFAULT_REFINE_MODEL = "sonnet"

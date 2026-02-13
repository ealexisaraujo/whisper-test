# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python 3.11 CLI tool for transcribing YouTube videos and local media files using two manually-selected ASR backends: **MLX-Whisper** (Apple Silicon) and **OpenAI** (cloud). There is **no automatic fallback** between backends — exactly one backend runs per execution, selected via `--backend`.

## Commands

```bash
# Run all tests
python -m pytest -q

# Run a single test file
python -m pytest tests/test_transcribe_youtube.py -q

# Run a single test by name
python -m pytest tests/test_transcribe_youtube.py -k "test_name" -q

# Syntax check
python -m py_compile transcribe_youtube.py

# Run the CLI
python transcribe_youtube.py <youtube-url-or-local-file> --backend {mlx-whisper,openai}

# Install dependencies
pip install -r requirements.txt
```

Requires `ffmpeg` and `ffprobe` in PATH.

## Architecture

Production code lives in the `transcribe/` package with a thin shim at the root:

- **`transcribe_youtube.py`** — thin shim (`from transcribe import *`) that preserves backward-compatible imports and the `python transcribe_youtube.py` CLI entry point.
- **`logger_utils.py`** — centralized logging config, `log_stage()` context manager for timing pipeline stages, `format_bytes()` helper.

### Package layout (`transcribe/`)

```
transcribe/
├── __init__.py      # Re-exports full public API for backward compat
├── models.py        # Dataclasses (SourceRequest, SourceMedia, NormalizedMedia, BackendResult) + exceptions
├── constants.py     # SUPPORTED_BACKENDS, DEFAULT_MODELS, OPENAI_* sets
├── utils.py         # _to_float, _first_not_none, _sanitize_filename, _run_with_progress, _probe_duration_seconds, etc.
├── resolver.py      # InputResolver
├── downloader.py    # MediaDownloader
├── normalizer.py    # AudioNormalizer
├── backends.py      # TranscriptionBackend, OpenAIBackend, MLXWhisperBackend, BackendFactory + segment normalizers
├── chunking.py      # should_chunk_audio, split_audio_into_chunks, offset_segments, transcribe_with_chunking, run_selected_backend
├── output.py        # OutputWriter, format_srt_timestamp, _render_txt, _render_srt, _build_schema
└── cli.py           # parse_args, main, build_context_info, validate_openai_output_constraints, parse_output_formats, has_transcription_content
```

Key classes follow a linear pipeline: `InputResolver` -> `MediaDownloader` -> `AudioNormalizer` -> `BackendFactory`/`TranscriptionBackend` -> `OutputWriter`. Data flows through dataclasses: `SourceRequest` -> `SourceMedia` -> `NormalizedMedia` -> `BackendResult`.

Dependency graph (no cycles): `models/constants` <- `utils` <- `resolver/downloader/normalizer` <- `backends` <- `chunking` <- `output` <- `cli`.

### Backend Design

Each backend is a `TranscriptionBackend` subclass (`MLXWhisperBackend`, `OpenAIBackend`). They share a unified output schema (segments with `start`/`end`/`speaker`/`text`) and produce TXT, SRT, and JSON outputs.

Key constraints to preserve:
- **No automatic backend fallback** — if one fails, raise `TranscriptionError`, don't try another
- **`CapabilityError`** — raised when a backend can't fulfill a request (e.g., SRT from a non-timestamp OpenAI model)
- Audio normalization: 16kHz for both backends

### Auto-chunking

Long audio is automatically split and merged. Chunking triggers by duration threshold or file size limits (OpenAI). The `transcribe_with_chunking()` function handles split -> transcribe each -> `offset_segments()` -> merge.

## Environment Configuration

Copy `.env.example` to `.env`. Key variables:
- `OPENAI_API_KEY` / `OPENAI_BASE_URL` — for the OpenAI backend
- `TRANSCRIBE_CHUNK_MODE` — `auto`/`off`/`force`
- `TRANSCRIBE_LOG_LEVEL` — `DEBUG`/`INFO`/`WARNING`/`ERROR`

## Extended Documentation

Detailed agent guidance lives in `docs/agents/`:
- `transcription-workflow.md` — pipeline architecture
- `backend-capabilities.md` — backend comparison matrix
- `command-recipes.md` — copy-paste command examples
- `testing-and-release.md` — test/release checklist
- `troubleshooting.md` — common issues and fixes

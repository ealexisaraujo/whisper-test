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

All production code lives in two files at the project root:

- **`transcribe_youtube.py`** — the entire pipeline: CLI parsing, input resolution, download, audio normalization, backend transcription, chunking, and output writing. Key classes follow a linear pipeline: `InputResolver` -> `MediaDownloader` -> `AudioNormalizer` -> `BackendFactory`/`TranscriptionBackend` -> `OutputWriter`. Data flows through dataclasses: `SourceRequest` -> `SourceMedia` -> `NormalizedMedia` -> `BackendResult`.
- **`logger_utils.py`** — centralized logging config, `log_stage()` context manager for timing pipeline stages, `format_bytes()` helper.

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

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python 3.11 CLI tool for transcribing YouTube videos and local media files using three manually-selected ASR backends: **MLX-Whisper** (Apple Silicon), **OpenAI** (cloud), and **EG2** (EmbeddingGemma 2 on Core ML/ANE, English only). There is **no automatic fallback** between backends — exactly one backend runs per execution, selected via `--backend`.

## Commands

```bash
# Run all tests (always use the repo venv: .venv/bin/python)
.venv/bin/python -m pytest -q

# Run a single test file
.venv/bin/python -m pytest tests/test_transcribe_youtube.py -q

# Run a single test by name
.venv/bin/python -m pytest tests/test_transcribe_youtube.py -k "test_name" -q

# Syntax check
.venv/bin/python -m py_compile transcribe_youtube.py

# Run the CLI
.venv/bin/python transcribe_youtube.py <youtube-url-or-local-file> --backend {mlx-whisper,openai,eg2}

# Install dependencies
.venv/bin/pip install -r requirements.txt

# Benchmark WER/speed on LibriSpeech long-form chapters (add --refine to score the refine stage)
.venv/bin/python benchmarks/bench_asr.py --librispeech <path>/LibriSpeech/test-clean --engines eg2 --limit 12

# Check video language before picking a backend (eg2 is English-only)
.venv/bin/yt-dlp --skip-download --print "%(title)s | %(duration_string)s | %(language)s" <url>
```

Requires `ffmpeg` and `ffprobe` in PATH.

Outputs are written to the repo root as `<title>_<timestamp>.{txt,srt,json}`. They are user deliverables: don't commit them.

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
├── resolver.py      # InputResolver + _normalize_input_source
├── downloader.py    # MediaDownloader + _sanitize_filename, _extract_youtube_id
├── normalizer.py    # AudioNormalizer + _probe_duration_seconds
├── eg2.py           # EmbeddingGemma 2 engine: model download, LogMel, Core ML encoder/reader/scorer, CTC collapse, punctuation
├── backends.py      # TranscriptionBackend, OpenAIBackend, MLXWhisperBackend, EG2Backend, BackendFactory + segment normalizers + _run_with_progress, _to_float
├── chunking.py      # should_chunk_audio, split_audio_into_chunks, offset_segments, transcribe_with_chunking, run_selected_backend
├── refine.py        # Opt-in refine stage: ClaudeRefiner (Claude Code CLI headless), build_refine_context, map_uncertain_words, refine_diverges
├── output.py        # OutputWriter, format_srt_timestamp, _render_txt, _render_srt
└── cli.py           # parse_args, main, build_context_info, validate_openai_output_constraints, validate_refine_constraints, parse_output_formats, has_transcription_content, _build_schema
```

Key classes follow a linear pipeline: `InputResolver` -> `MediaDownloader` -> `AudioNormalizer` -> `BackendFactory`/`TranscriptionBackend` -> (optional `ClaudeRefiner`) -> `OutputWriter`. Data flows through dataclasses: `SourceRequest` -> `SourceMedia` -> `NormalizedMedia` -> `BackendResult`.

Dependency graph (no cycles): `models/constants` <- `resolver/downloader/normalizer` <- `backends` <- `chunking` <- `refine` <- `output` <- `cli`.

### Backend Design

Each backend is a `TranscriptionBackend` subclass (`MLXWhisperBackend`, `OpenAIBackend`, `EG2Backend`). They share a unified output schema (segments with `start`/`end`/`speaker`/`text`) and produce TXT, SRT, and JSON outputs.

Key constraints to preserve:
- **No automatic backend fallback** — if one fails, raise `TranscriptionError`, don't try another
- **`CapabilityError`** — raised when a backend can't fulfill a request (e.g., SRT from a non-timestamp OpenAI model)
- Audio normalization: 16kHz for all backends
- `_build_schema` does not serialize `BackendResult.raw`; EG2 per-word scores exist only in memory (the refine stage reads them there)

### EG2 gotchas

- `reader_P6` returns NaN on CPU/GPU (fp16): keep it on `CPU_AND_NE`. After touching compute units, re-run the benchmark with `--limit 4` and confirm WER stays ≈4 %.
- Models are cached in `~/.cache/whisper_translate/eg2/v5` (`EG2_MODEL_DIR`). The model host returns 403 to Python's default User-Agent, so downloads send a curl UA. The first ANE load compiles for about 110 s.
- English-only closed vocabulary: a non-English `--language` raises `CapabilityError`. EG2 handles long audio itself, so auto-chunking is skipped.

### Auto-chunking

Long audio is automatically split and merged. Chunking triggers by duration threshold or file size limits (OpenAI). The `transcribe_with_chunking()` function handles split -> transcribe each -> `offset_segments()` -> merge.

### Refine stage

Opt-in (`--refine` / `TRANSCRIBE_REFINE=1`, off by default). After the backend and chunk merge, `ClaudeRefiner` sends segment texts in batches to the locally installed, logged-in Claude Code CLI (`claude -p --safe-mode --strict-mcp-config --disable-slash-commands --tools "" ...`, structured JSON output) to fix ASR errors. Model via `--refine-model` / `TRANSCRIBE_REFINE_MODEL` (alias, default `sonnet`); binary via `CLAUDE_BIN` or PATH (missing -> `CapabilityError`). Timestamps and segment count are preserved; diverging/empty rewrites keep the original; a batch failing twice raises `TranscriptionError`. Stats and original texts go to JSON `metadata.refine`. Tests mock `subprocess.run`; never call the real CLI in tests.

- Never drop `--safe-mode --strict-mcp-config --disable-slash-commands --tools ""`: without them user plugins/MCP load (a tiny call: $1.84 vs $0.006). `--bare` refuses OAuth.
- LibriSpeech overstates refine gains (public-domain books the model has memorized). Judge it on a real talk against Whisper. Context (title/description) can plant wrong names.

## Environment Configuration

Copy `.env.example` to `.env`. Key variables:
- `OPENAI_API_KEY` / `OPENAI_BASE_URL` — for the OpenAI backend
- `TRANSCRIBE_CHUNK_MODE` — `auto`/`off`/`force`
- `TRANSCRIBE_LOG_LEVEL` — `DEBUG`/`INFO`/`WARNING`/`ERROR`
- `TRANSCRIBE_REFINE` / `TRANSCRIBE_REFINE_MODEL` / `CLAUDE_BIN` — refine stage

## Extended Documentation

Detailed agent guidance lives in `docs/agents/`:
- `transcription-workflow.md` — pipeline architecture
- `backend-capabilities.md` — backend comparison matrix
- `command-recipes.md` — copy-paste command examples
- `testing-and-release.md` — test/release checklist
- `troubleshooting.md` — common issues and fixes
- `eg2-engine-analysis.md` — reverse-engineered EG2 pipeline with evidence
- `eg2-benchmark.md` — EG2 vs mlx-whisper speed/WER benchmark (`benchmarks/bench_asr.py`)

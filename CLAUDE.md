# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python 3.11 CLI tool for transcribing YouTube videos and local media files using four manually-selected backends: **MLX-Whisper** (Apple Silicon), **OpenAI** (cloud), **EG2** (EmbeddingGemma 2 on Core ML/ANE, English only), and **auto** (local pipeline: probe, VAD chunks, then per-profile engines). The auto architecture supports two-engine fusion, a local LLM pick, an audio verify and a gated Claude refine, but **the shipped, calibrated profiles switch all of that off per the measured data**: English = Qwen3-ASR alone, Spanish = Whisper large-v3 alone, other languages = Whisper large-v3 alone (unvalidated), and no correction stage runs (see `docs/agents/auto-pipeline-results.md`). The backend is selected via `--backend`. There is **no automatic fallback**: the auto pipeline uses deterministic routing and raises when a chosen engine fails.

## Which engine to use (daily)

Pick by the audio, then run `.venv/bin/python transcribe_youtube.py <url-or-file> --backend <engine>`. Numbers come from the eval suite (`docs/agents/auto-pipeline-results.md`).

| Audio | Engine | Why |
|---|---|---|
| Talk, interview or podcast with one main speaker, English or Spanish | `auto` | Most accurate overall (9.0 % WER vs 9.7 % for `mlx-whisper` turbo); invents nothing on silence (turbo: ~15 words/min) |
| Meeting, or several people talking over each other | `mlx-whisper` (default) | 21.9 % WER vs 30.7 % for `auto` |
| English subtitles where SRT timing matters | `mlx-whisper` | `auto` English word times are approximate (Qwen3-ASR has none) |
| Quick English draft | `eg2` | ~100× real time, about twice the errors |
| Any other language | `mlx-whisper` | `auto` uses Whisper large-v3 there too, but unvalidated |
| No local Mac / paid cloud is fine | `openai` | Needs `OPENAI_API_KEY` |

Check the language first when unsure (`yt-dlp --skip-download --print "%(language)s" <url>`): `eg2` is English-only.

Models kept locally for daily use (~9.5 GB): `mlx-community/whisper-turbo` (`mlx-whisper`), EG2 (`~/.cache/whisper_translate/eg2`), `mlx-community/Qwen3-ASR-1.7B-bf16` and `mlx-community/whisper-large-v3-mlx` (`auto`), plus Silero VAD. The picker (Gemma 4 26B, ~15 GB), Parakeet and the forced aligner were removed from the cache because no shipped profile uses them; they download again automatically if a profile turns them on.

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
.venv/bin/python transcribe_youtube.py <youtube-url-or-local-file> --backend {mlx-whisper,openai,eg2,auto}

# Install dependencies
.venv/bin/pip install -r requirements.txt

# Benchmark WER/speed on LibriSpeech long-form chapters (add --refine to score the refine stage)
.venv/bin/python benchmarks/bench_asr.py --librispeech <path>/LibriSpeech/test-clean --engines eg2 --limit 12

# Eval suite (auto pipeline): score systems per slice, compare against a base, regenerate profiles from dev results
.venv/bin/python -m benchmarks.suite.run_suite --systems mlx-whisper-turbo,auto:+verify --manifest smoke|full --split dev|test|all --out <json>
.venv/bin/python -m benchmarks.suite.gate --base <base.json> --cand <cand.json> --split test
.venv/bin/python -m benchmarks.suite.make_profiles --results <full.json> [--write transcribe/profiles.json]

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
├── profiles.json    # Per-audio-profile engines/stages for `auto`, generated by benchmarks/suite/make_profiles.py
├── auto/            # `--backend auto`: types, probe, chunks, engines, guards, align, picker, verify, resolve, refine_gated, terms, phonetics, diarize, assemble, cache, profiles, pipeline
└── cli.py           # parse_args, main, build_context_info, validate_openai_output_constraints, validate_refine_constraints, parse_output_formats, has_transcription_content, _build_schema
```

Key classes follow a linear pipeline: `InputResolver` -> `MediaDownloader` -> `AudioNormalizer` -> `BackendFactory`/`TranscriptionBackend` -> (optional `ClaudeRefiner`) -> `OutputWriter`. Data flows through dataclasses: `SourceRequest` -> `SourceMedia` -> `NormalizedMedia` -> `BackendResult`.

Dependency graph (no cycles): `models/constants` <- `resolver/downloader/normalizer` <- `backends` <- `chunking` <- `refine` <- `output` <- `cli`. `transcribe/auto/*` imports only `models`, `constants`, `refine` (`resolve_claude_bin`) and `eg2` (lazily, in the EG2 engine); `cli.py` and `transcribe/__init__.py` import `auto/` lazily (inside the auto branch / `transcribe.run_auto` on first access), so the other backends never load `transcribe.auto`, `jellyfish` or `pronouncing` (tested).

### Backend Design

Each backend is a `TranscriptionBackend` subclass (`MLXWhisperBackend`, `OpenAIBackend`, `EG2Backend`). They share a unified output schema (segments with `start`/`end`/`speaker`/`text`) and produce TXT, SRT, and JSON outputs.

Key constraints to preserve:
- **No automatic fallback; the auto pipeline uses deterministic routing + fusion and raises when a chosen engine fails** — a failing backend or core auto stage raises `TranscriptionError`, never another engine. Auto enhancement stages (diarize, pick, verify, refine) skip and are flagged `skipped: <reason>` in the JSON
- **`CapabilityError`** — raised when a backend can't fulfill a request (e.g., SRT from a non-timestamp OpenAI model)
- Audio normalization: 16kHz for all backends
- `_build_schema` does not serialize `BackendResult.raw`; EG2 per-word scores exist only in memory (the refine stage reads them there)

### EG2 gotchas

- `reader_P6` returns NaN on CPU/GPU (fp16): keep it on `CPU_AND_NE`. After touching compute units, re-run the benchmark with `--limit 4` and confirm WER stays ≈4 %.
- Models are cached in `~/.cache/whisper_translate/eg2/v5` (`EG2_MODEL_DIR`). The model host returns 403 to Python's default User-Agent, so downloads send a curl UA. The first ANE load compiles for about 110 s.
- English-only closed vocabulary: a non-English `--language` raises `CapabilityError`. EG2 handles long audio itself, so auto-chunking is skipped.

### Auto pipeline (`--backend auto`)

Ingest (EBU R128 loudness) -> Silero VAD chunks (<=30 s) -> probe (language windows, phone band, SNR) -> profile key (`en`, `es`, `other`, `-phone`/`-noisy` suffixes) -> entry in `transcribe/profiles.json` -> engine A + engine B (Qwen3-ASR 1.7B, Whisper large-v3; one at a time, released after) -> hallucination guards -> word alignment into solid/disputed/missing spans -> Gemma 4 26B-A4B picker (A/B/unclear only) -> Whisper teacher-forced verify with a margin -> gated Claude refine (term list, phonetic and verbatim gates, audio re-check) -> speakers (pyannote, needs `HF_TOKEN`) -> segments + trust score. Unresolved spans render as `[unclear: best | other]` (best guess first) when two engines ran; under the shipped single-engine profiles `[unclear: word]` appears only from Whisper low-probability words (Spanish/other), with no alternative. A profile may be single-engine with stages off: `make_profiles` switches off anything that doesn't beat the best single engine on dev (current calibration: Qwen3-ASR alone for English, Whisper large-v3 alone for Spanish). Results and the evidence: `docs/agents/auto-pipeline-results.md`.

- Engine, VAD and probe outputs are cached in `~/.cache/whisper_translate/auto` (keyed by audio hash + config); bump `cache.CACHE_VERSION` after changing stage logic. The suite caches its own per-item outputs in `~/.cache/whisper_translate/suite/cache` and memoizes picker answers and verify scores in `.../suite/picker` and `.../suite/scorer` (keyed by memo versions in `run_suite.py`; `--refresh` bypasses them).
- Peak memory: the picker is ~13.4 GB; stages load one model set at a time. Never run two picker processes at once on a 32 GB machine.
- Eval data lives in `~/.cache/whisper_translate/suite/` and is never committed. Per-item manifests (`benchmarks/suite/manifests/*.jsonl`) and results (`benchmarks/suite/results/*.json`) are **local-only and git-ignored** (they embed TED-LIUM reference text, CC BY-NC-ND, and local paths); regenerate them with `benchmarks.suite.datasets prepare` / `run_suite`, and commit only the aggregates written by `run_suite --summarize <results.json>` (`<name>.summary.json`).
- Known limitations (see the results doc): Qwen3-ASR has no word times, so English SRT/word times are approximate (`metadata.auto.timestamps = "approximate"`; pending: wire Qwen3-ForcedAligner); the trust score is `"uncalibrated"` (no estimated WER) for every shipped profile because its measured Spearman is below 0.7 (en 0.24, es 0.63) or its features are constant (single Qwen3/Parakeet/EG2); the verify audio window was fixed to cover the context words and verify needs re-measuring.

### Auto-chunking

Long audio is automatically split and merged. Chunking triggers by duration threshold or file size limits (OpenAI). The `transcribe_with_chunking()` function handles split -> transcribe each -> `offset_segments()` -> merge.

### Refine stage

Opt-in for the existing backends (`--refine` / `TRANSCRIBE_REFINE=1`, off by default); `--backend auto` runs its own gated refine (`transcribe/auto/refine_gated.py`) unless `--no-refine`, `TRANSCRIBE_REFINE=0` or the profile turns it off (an explicit `--refine` on a profile with refine off is ignored with an INFO note; the shipped en/es/other profiles have refine off). After the backend and chunk merge, `ClaudeRefiner` sends segment texts in batches to the locally installed, logged-in Claude Code CLI (`claude -p --safe-mode --strict-mcp-config --disable-slash-commands --tools "" ...`, structured JSON output) to fix ASR errors. Model via `--refine-model` / `TRANSCRIBE_REFINE_MODEL` (alias, default `sonnet`); binary via `CLAUDE_BIN` or PATH (missing -> `CapabilityError`). Timestamps and segment count are preserved; diverging/empty rewrites keep the original; a batch failing twice raises `TranscriptionError`. Stats and original texts go to JSON `metadata.refine`. Tests mock `subprocess.run`; never call the real CLI in tests.

- Never drop `--safe-mode --strict-mcp-config --disable-slash-commands --tools ""`: without them user plugins/MCP load (a tiny call: $1.84 vs $0.006). `--bare` refuses OAuth.
- LibriSpeech overstates refine gains (public-domain books the model has memorized). Judge it on a real talk against Whisper. Context (title/description) can plant wrong names.

## Environment Configuration

Copy `.env.example` to `.env`. Key variables:
- `OPENAI_API_KEY` / `OPENAI_BASE_URL` — for the OpenAI backend
- `TRANSCRIBE_CHUNK_MODE` — `auto`/`off`/`force`
- `TRANSCRIBE_LOG_LEVEL` — `DEBUG`/`INFO`/`WARNING`/`ERROR`
- `TRANSCRIBE_REFINE` / `TRANSCRIBE_REFINE_MODEL` / `CLAUDE_BIN` — refine stage
- `HF_TOKEN` — pyannote diarization for `auto` (without it diarization is skipped and everything is `Speaker 1`)
- `TRANSCRIBE_AUTO_CACHE_DIR` — auto pipeline stage cache location

## Extended Documentation

Detailed agent guidance lives in `docs/agents/`:
- `transcription-workflow.md` — pipeline architecture
- `backend-capabilities.md` — backend comparison matrix
- `command-recipes.md` — copy-paste command examples
- `testing-and-release.md` — test/release checklist
- `troubleshooting.md` — common issues and fixes
- `eg2-engine-analysis.md` — reverse-engineered EG2 pipeline with evidence
- `eg2-benchmark.md` — EG2 vs mlx-whisper speed/WER benchmark (`benchmarks/bench_asr.py`)
- `auto-pipeline-spike.md` — model choices for `auto` (engines, picker, verifier) with the spike evidence
- `auto-pipeline-results.md` — eval suite results per slice, success criteria, profile decisions, real-audio runs
- `refine-evals.md` — earlier refine audit work (superseded by the graded suite)

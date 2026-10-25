# Any-Audio Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--backend auto`. It probes the audio, runs two local ASR engines on the same VAD chunks, aligns
them and locks agreed words, resolves disputes with a local picker LLM plus Whisper audio scoring, applies
gated Claude refine, labels speakers, and writes transcripts with visible uncertainty. Every stage is graded by
a self-scoring eval suite built on public datasets with human references.

**Architecture:** A new subpackage `transcribe/auto/` holds one module per stage, connected by dataclasses in
`transcribe/auto/types.py`. `transcribe/auto/pipeline.py` orchestrates the stages, caches each one, and records
stage statuses. Per-profile switches (which engines and stages are on) come from `transcribe/profiles.json`,
which the suite generates (`benchmarks/suite/`). The existing single-backend modes are untouched.

**Tech Stack:** Python 3.11 (`.venv`), MLX (mlx-whisper, mlx-lm, mlx-audio / parakeet-mlx), silero-vad
(torch), pyannote.audio (community-1), jiwer + whisper_normalizer, pronouncing (CMUdict) + jellyfish,
huggingface_hub / datasets / soundfile for eval data, pytest.

**Spec:** `docs/superpowers/specs/2026-10-25-any-audio-pipeline-design.md` (read it before any task).

## Global Constraints

- Local-only speech stack: no paid APIs. The OpenAI backend stays as-is and is not used by `auto`.
- Claude only through the existing headless CLI flags: `-p --safe-mode --strict-mcp-config --disable-slash-commands --tools "" --no-session-persistence` (reuse `ClaudeRefiner.build_command` style; never drop these flags).
- M1 Pro, 32 GB: load one stage's models at a time, release them before the next stage, and keep peak resident memory under about 24 GB.
- Accuracy over speed: 20 min per audio hour is a soft target (reported, never blocks).
- Languages: English and Spanish fully supported. Other languages get best effort: no refine and no phonetic gates, flagged.
- Speakers labeled `Speaker 1`, `Speaker 2`, …; never names.
- Verbatim policy: never remove fillers, false starts or repeats that both engines heard.
- No denoising stage. EBU R128 loudness normalization only.
- Failure policy (spec §8a):
  - Core stages (ingest, probe, chosen ASR engines) raise `TranscriptionError` / `CapabilityError`.
  - Enhancement stages (diarize, pick, verify, refine) skip and flag `"skipped: <reason>"`.
  - A missing `HF_TOKEN` skips diarization; it does not abort.
- `[unclear: best | other]` markers always carry the best guess first. Metrics score the best guess. Target density is ≤ 3 % of words.
- Tests never load real models or call the real Claude CLI. Real-model tests are marked `@pytest.mark.slow` and run only when `RUN_SLOW=1`.
- Keep the existing dependency direction: `transcribe/auto/*` may import from `models`, `normalizer`, `refine` and `output`; nothing outside `auto/` and `cli.py` imports `auto/`.
- **Do not git commit** (the user commits only when they ask). Each task's last step records its changed files in `docs/superpowers/plans/2026-10-25-any-audio-pipeline.progress.md`.
- Use `.venv/bin/python`. The full test suite (`.venv/bin/python -m pytest -q`) must stay green after every task.
- Eval datasets and audio live under `~/.cache/whisper_translate/suite/` and are never committed. Manifests (JSON) and result JSONs live under `benchmarks/suite/` and can be committed later.

## Review Focus

1. **Speech-free or mostly silent/music input.** Expect a near-empty transcript, no invented text, and no crash on zero chunks. Tests in Tasks 4, 5 and 13.
2. **Long input (≥ 1 h, hundreds of chunks).** Memory stays bounded because models are released between stages, and the cache keeps a re-run from redoing ASR. Tests in Task 13 (release called; cache hit).
3. **One engine returns empty text for some chunks.** Treated as one-sided slots, not as an engine failure. An exception from an engine is a core failure. Tests in Tasks 6 and 13.
4. **Spanish text with accents, ñ and ¿¡ punctuation.** Normalization for alignment is accent-insensitive, but outputs keep the original characters. Tests in Tasks 6 and 12.
5. **Claude CLI missing or failing, picker model missing, no `HF_TOKEN`.** Outputs are still written, the stage is flagged and the trust score is lowered. Tests in Task 13.

---

### Task 0: Environment spike: install and prove every model on this Mac

**Files:**
- Modify: `requirements.txt` (add the deps that work, pinned to minimum versions, each with a one-line comment)
- Create: `docs/agents/auto-pipeline-spike.md`
- Create: `benchmarks/suite/spike.py` (throwaway measurement script; keep it for reproducibility)

**Interfaces:**
- Produces: the decisions later tasks read from the spike doc:
  - `ENGINE_A` / `ENGINE_B` ids per language from {`whisper-large-v3`, `qwen3-asr-1.7b`, `parakeet-v3`, `eg2`}
  - `PICKER_MODEL` (an mlx-lm repo id)
  - the verifier (mlx-whisper large-v3)
  - the VAD (silero-vad)
  - diarization availability (needs `HF_TOKEN`)

- [ ] **Step 1:** Install candidates into `.venv`: `silero-vad`, `mlx-lm`, `mlx-audio`, `parakeet-mlx`, `pyannote.audio`, `pronouncing`, `jellyfish`, `datasets`, `huggingface_hub`.
  - If any install would downgrade `torch`/`transformers`/`numpy` and break the existing tests, stop. Record it and try the alternative (e.g. Qwen3-ASR via `mlx-audio` instead of transformers).
  - Run `.venv/bin/python -m pytest -q`. Expected: all existing tests pass.
- [ ] **Step 2:** Make a 30 s English clip and a 30 s Spanish clip with known references, from LibriSpeech test-clean and FLEURS es via `huggingface_hub`/`datasets`. Write them to `~/.cache/whisper_translate/suite/spike/`.
- [ ] **Step 3:** In `benchmarks/suite/spike.py`, run each engine on both clips and record WER, ×RT, peak RSS, and whether word timestamps / word probabilities are available:
  - mlx-whisper `mlx-community/whisper-large-v3-mlx` with `word_timestamps=True`
  - Qwen3-ASR 1.7B (mlx-audio)
  - parakeet-mlx `mlx-community/parakeet-tdt-0.6b-v3`
  - EG2 (English only)
- [ ] **Step 4:** Load the picker candidates with mlx-lm: `mlx-community/Qwen3-30B-A3B-4bit`, then a Gemma 4 26B-A4B MLX 4-bit build if one exists.
  - Run one multiple-choice prompt (two hypotheses, return JSON `{"choice": "A"|"B"|"unclear"}`) and record load time, tokens/s, peak RSS, and whether the JSON is valid.
  - If neither fits in about 20 GB, try `mlx-community/Qwen3-14B-4bit`.
- [ ] **Step 5:** Verify Whisper teacher-forced scoring works. For the English clip, the mean token log-prob of the reference text must be higher than that of a corrupted text (3 words swapped). Use `mlx_whisper.load_models.load_model`, `log_mel_spectrogram`, `model.embed_audio`, `model.logits`.
- [ ] **Step 6:** Check silero-vad on the clip plus 10 s of silence (the speech spans must exclude the silence). Check pyannote import and report whether `HF_TOKEN` is set.
- [ ] **Step 7:** Write `docs/agents/auto-pipeline-spike.md` with:
  - a results table (engine × clip: WER, ×RT, RSS, timestamps, probs)
  - the chosen `ENGINE_A`/`ENGINE_B` per language. Rule: the two most accurate engines that both run, preferring a pair whose errors differ by architecture (Whisper + non-Whisper).
  - `PICKER_MODEL`
  - every fallback taken, with the reason
- [ ] **Step 8:** Record the changed files in the progress file.

### Task 1: Eval datasets and slice manifests

**Files:**
- Create: `benchmarks/suite/__init__.py`, `benchmarks/suite/datasets.py`, `benchmarks/suite/slices.json`
- Test: `tests/test_suite_datasets.py`

**Interfaces:**
- Produces:
  - `benchmarks/suite/manifests/<slice>.jsonl`, one JSON per item: `{"id", "slice", "split": "dev"|"test", "lang": "en"|"es"|"none", "audio": abs path (16 kHz mono wav), "duration", "reference": str, "speakers": [{"start","end","speaker"}] | null, "condition": "clean"|"noise"|"phone"|"nospeech"|"meeting"|"accent"}`
  - `manifests/smoke.jsonl`: about 30 min total, stratified across all slices
  - `def build_longform(utterances: list[dict], target_seconds: float, gap_seconds: float = 0.6) -> tuple[np.ndarray, str, list[dict]]`: concatenates utterances into one long file; returns audio, reference, and speaker turns
  - `def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray`
  - `def phone_band(audio: np.ndarray, sr: int = 16000) -> np.ndarray`: 8 kHz μ-law round trip via ffmpeg, back at 16 kHz
  - `def assign_split(item_id: str) -> str`: deterministic hash, 50 % dev / 50 % test
- CLI: `python -m benchmarks.suite.datasets prepare [--smoke-only]`

- [ ] **Step 1: Write failing tests**
  - `test_build_longform_concatenates_with_gaps`: two 1 s utterances + 0.6 s gap → length 2.6 s × 16000 ± 1 sample; reference `"a b"`; speakers `[{"start":0.0,"end":1.0,...},{"start":1.6,"end":2.6,...}]`.
  - `test_mix_at_snr_hits_target`: measured SNR within 0.1 dB of 5.0.
  - `test_assign_split_is_deterministic_and_balanced`: same id gives the same split; 1000 ids give 45–55 % dev.
  - `test_phone_band_removes_energy_above_4khz`: a 6 kHz sine loses ≥ 30 dB.
- [ ] **Step 2:** Run `.venv/bin/python -m pytest tests/test_suite_datasets.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement the functions plus `prepare` for these slices, defined in `slices.json` with source, config, minutes and condition:

  | Slice | Source | Minutes | Condition |
  |---|---|---|---|
  | `en_clean` | LibriSpeech test-clean/other | 40 | long-form, multi-speaker concatenation |
  | `en_talks` | TED-LIUM long-form (`distil-whisper/tedlium-long-form`) | 40 | |
  | `en_accents` | `edinburghcstr/edacc` | 40 | |
  | `en_business` | Earnings-22 (`distil-whisper/earnings22` or the ESB test config) | 40 | |
  | `en_meeting` | AMI (`diarizers-community/ami`, sdm, with speaker turns) | 40 | |
  | `es_fleurs` | FLEURS es_419 | 40 | |
  | `es_mls` | Multilingual LibriSpeech es test | 40 | |
  | `es_dialects` | crowdsourced Latin-American Spanish (openslr 71–76 mirrors on HF) | 30 | |
  | `en_noise` | `en_clean` mixed with white/pink/babble noise at 0, 5, 10 dB | 30 | |
  | `es_noise` | as above, Spanish | 20 | |
  | `en_phone`, `es_phone` | `phone_band` of clean items | 20 each | |
  | `nospeech` | 30 × 20 s clips: silence, white noise, babble-free ambient/music (ESC-50 `ashraq/esc50`) | 10 | reference `""` |

  - Total target 6–8 h; cap per slice by minutes.
  - When a source is unavailable (gated, removed, script-only), skip it, log it, and record the gap in `slices.json` under `"gaps"`. Never stop the whole prepare.
  - Code-switching: record it as a gap unless an ungated es-en corpus is found.
- [ ] **Step 4:** Run the tests. Expected: PASS. Then run `.venv/bin/python -m benchmarks.suite.datasets prepare`. Expected: manifests written, and a summary table of minutes per slice printed.
- [ ] **Step 5:** Record the changed files and the per-slice minutes in the progress file.

### Task 2: Metrics

**Files:**
- Create: `benchmarks/suite/metrics.py`
- Test: `tests/test_suite_metrics.py`

**Interfaces:**
- Produces:
  - `def normalize(text: str, lang: str) -> str`
    - en: `whisper_normalizer.english.EnglishTextNormalizer`
    - es: `BasicTextNormalizer` + lowercase
    - `normalize(text, lang, strip_accents=True)` is the accent-insensitive variant
  - `def wer_cer(refs: list[str], hyps: list[str], lang: str) -> dict`: keys `wer, cer, sub, del, ins, words`
  - `def entity_number_error(ref: str, hyp: str, lang: str) -> dict`: tokens with digits/number words plus capitalized non-sentence-initial tokens in the *raw* ref; reports `{"total", "missed"}`
  - `def insertions_per_minute(hyp: str, duration: float) -> float`
  - `def edit_effects(ref: str, before: str, after: str, lang: str) -> dict`: aligns ref↔before and ref↔after; returns `{"help": n, "harm": n, "neutral": n}`, counted per changed region by whether the region's errors went down, up or stayed
  - `def unclear_quality(ref: str, words: list[dict], lang: str) -> dict`: `words` items `{"text", "unclear": bool}`; returns `precision, recall, density`
  - `def spearman(xs: list[float], ys: list[float]) -> float`
  - `def der(ref_turns: list[dict], hyp_turns: list[dict], collar: float = 0.25) -> float`: simple frame-based (10 ms) DER with optimal speaker mapping (Hungarian via `scipy.optimize.linear_sum_assignment` if scipy is present, else greedy)
  - `def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]`

- [ ] **Step 1: Write failing tests**
  - `test_wer_identical_is_zero`
  - `test_wer_spanish_accents_and_punctuation`: `normalize("¿Qué tal, Señor?", "es") == "qué tal señor"`; with `strip_accents=True`, `"que tal senor"`
  - `test_edit_effects_counts_help_and_harm`: ref "the cat sat", before "the hat sat", after "the cat sat" → help 1; ref "the cat sat", before "the cat sat", after "the cat sad" → harm 1
  - `test_unclear_quality`: ref "a b c d", words a,X(unclear),c,d(unclear) → precision 0.5, recall 1.0, density 0.5
  - `test_insertions_per_minute_empty_ref`: hyp "thanks for watching", 30 s → 6.0
  - `test_der_perfect_and_swapped_labels_zero`
  - `test_bootstrap_ci_contains_mean`
- [ ] **Step 2:** Run `.venv/bin/python -m pytest tests/test_suite_metrics.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement. Use `jiwer.process_words` / `process_characters` for alignment.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Record changed files.

### Task 3: Suite runner and baseline for existing backends

**Files:**
- Create: `benchmarks/suite/run_suite.py`, `benchmarks/suite/results/` (dir)
- Test: `tests/test_suite_runner.py`

**Interfaces:**
- Consumes: Task 1 manifests, Task 2 metrics.
- Produces:
  - `class SystemUnderTest(Protocol): name: str; def run(self, item: dict) -> dict`. The returned dict has `{"text": str, "words": list[dict] | None, "segments": list[dict], "meta": dict}`.
  - `def run_suite(systems: list[SystemUnderTest], manifest: list[dict], out_path: Path, cache_dir: Path) -> dict`:
    - caches each system's output per item at `cache_dir/<system>/<item_id>.json`
    - writes `{"config", "created_at", "machine", "per_item": [...], "per_slice": {slice: {system: metrics}}}`
    - `per_slice` metrics: `wer, cer, entity_err, ins_per_min (nospeech only), xrt, edits (help/harm), unclear (precision/recall/density), der (if speakers)`
  - Built-in systems: `mlx-whisper-turbo` (existing `MLXWhisperBackend`) and `eg2` (`EG2Backend`, en items only), wrapped by an adapter calling `transcribe_with_chunking`-free direct `backend.transcribe(path, lang, None, 0, {"progress_log_seconds": 20})`.
- CLI: `python -m benchmarks.suite.run_suite --systems mlx-whisper-turbo,eg2 --manifest smoke|full --split dev|test|all --out benchmarks/suite/results/<name>.json`

- [ ] **Step 1: Write failing tests** with a fake system (returns the reference with one word dropped) and a 2-item manifest:
  - `test_run_suite_writes_per_slice_metrics`: per_slice WER > 0, and the result file has all keys
  - `test_run_suite_uses_cache`: a second run doesn't call `run` again
  - `test_language_filter_skips_eg2_for_es`
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS. Then run the baseline on smoke: `.venv/bin/python -m benchmarks.suite.run_suite --systems mlx-whisper-turbo,eg2 --manifest smoke --out benchmarks/suite/results/baseline-smoke.json`. Expected: a per-slice table printed. Then run the full manifest into `baseline-full.json` (long; run it in the background).
- [ ] **Step 5:** Record changed files and the baseline headline numbers in the progress file.

### Task 4: Core types, VAD chunking, probe

**Files:**
- Create: `transcribe/auto/__init__.py`, `transcribe/auto/types.py`, `transcribe/auto/chunks.py`, `transcribe/auto/probe.py`
- Test: `tests/test_auto_chunks_probe.py`

**Interfaces:**
- Produces in `types.py` (dataclasses):
  - `Word(text: str, start: float, end: float, prob: float | None = None)`
  - `Chunk(index: int, start: float, end: float)`
  - `EngineOutput(engine: str, language: str | None, chunks: dict[int, list[Word]], flags: dict[int, list[str]])`, where `flags` holds hallucination flags per chunk
  - `AudioProfile(duration: float, speech_seconds: float, speech_ratio: float, language: str, language_windows: list[tuple[float, str, float]], code_switching: bool, snr_db: float | None, phone_band: bool, key: str)`
  - `StageStatus(name: str, status: str, detail: str = "", seconds: float = 0.0)`, where `status` is one of `"ok"`, `"skipped: <reason>"`, `"failed"`
- `chunks.py`:
  - `def speech_spans(samples: np.ndarray, sr: int = 16000) -> list[tuple[float, float]]` (silero-vad)
  - `def make_chunks(spans: list[tuple[float, float]], samples: np.ndarray, sr: int = 16000, max_seconds: float = 30.0, merge_gap: float = 0.5) -> list[Chunk]`: merges spans closer than `merge_gap`. Splits anything longer than `max_seconds` at the lowest-energy 100 ms frame inside the last 10 s of the window. Chunks never overlap.
- `probe.py`:
  - `def build_profile(samples: np.ndarray, sr: int, spans: list[tuple[float, float]], lid: Callable[[np.ndarray], tuple[str, float]], original_sample_rate: int | None) -> AudioProfile`
    - Runs LID on up to 5 speech windows of 30 s spread evenly.
    - `code_switching` = windows disagree with confidence > 0.6.
    - `language` = the majority language.
    - `snr_db` = 10·log10(speech energy / non-speech energy), `None` if there is no non-speech.
    - `phone_band` = original_sample_rate ≤ 8000, or < 1 % of energy above 4 kHz.
    - `key` = `"{lang}"` + (`"-phone"` if phone_band) + (`"-noisy"` if snr_db < 10). `lang` is `en`/`es`/`other`, and code-switching forces `other`, which routes to the multilingual pair with no forced language.
  - `def whisper_lid(model_repo: str) -> Callable[[np.ndarray], tuple[str, float]]`: mlx-whisper `detect_language` on the 30 s mel window

- [ ] **Step 1: Write failing tests** (VAD and LID are injected or mocked):
  - `test_make_chunks_merges_and_caps`: spans [(0,10),(10.3,25),(26,70)] → no chunk > 30 s, no overlaps, coverage of all speech
  - `test_make_chunks_empty_spans_returns_empty`
  - `test_build_profile_majority_and_code_switching`: LID returns en, en, es (conf 0.9) → language "en", code_switching True
  - `test_build_profile_key_phone_noisy`: original_sample_rate 8000, snr 5 → key "en-phone-noisy"
  - `test_build_profile_no_speech`: spans [] → speech_ratio 0, language "other", key "other"
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. silero-vad is loaded lazily inside `speech_spans`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Record changed files.

### Task 5: Local engines and hallucination guards

**Files:**
- Create: `transcribe/auto/engines.py`, `transcribe/auto/guards.py`
- Test: `tests/test_auto_engines_guards.py`

**Interfaces:**
- Consumes: `Chunk`, `Word`, `EngineOutput`; the engine ids chosen in Task 0.
- Produces:
  - `class LocalEngine(Protocol): name: str; def transcribe_chunks(self, samples: np.ndarray, chunks: list[Chunk], language: str | None) -> EngineOutput; def release(self) -> None`
  - Implementations for the engines Task 0 kept: `WhisperEngine(repo: str = "mlx-community/whisper-large-v3-mlx")`, `Qwen3AsrEngine`, `ParakeetEngine`, `EG2LocalEngine`.
    - `WhisperEngine` uses `word_timestamps=True`, `condition_on_previous_text=False`, `compression_ratio_threshold=2.4`, `logprob_threshold=-1.0`, `no_speech_threshold=0.6`, and `language` forced per chunk only if given.
    - Engines without word timestamps spread each chunk's words across the chunk by character length.
    - Engines without probabilities set `prob=None`.
  - `def make_engine(engine_id: str) -> LocalEngine` (raises `CapabilityError` for an unknown or uninstalled id)
  - `guards.py`:
    - `HALLUCINATION_PHRASES: tuple[str, ...]`: at least "thanks for watching", "thank you for watching", "subscribe", "subtitles by", "gracias por ver", "suscríbete", "amara.org"
    - `def apply_guards(out: EngineOutput, chunk_audio_rms: dict[int, float]) -> EngineOutput`:
      - drops a chunk's words when a known phrase is the whole chunk text, when a token 4-gram repeats ≥ 3 times (keeping the first occurrence), or when the chunk RMS is below −50 dBFS
      - appends the reason to `flags[chunk]`

- [ ] **Step 1: Write failing tests:**
  - `test_guard_drops_known_phrase_only_chunk`
  - `test_guard_collapses_repetition_loop`: "we go we go we go we go we go" → keeps one "we go", flag "repetition"
  - `test_guard_keeps_normal_text`
  - `test_guard_spanish_phrase`: "Gracias por ver el video" whole chunk → dropped
  - `test_spread_words_by_chars`: "hi there" in chunk 10–12 s → words have start/end inside [10, 12], monotonic
  - `test_make_engine_unknown_raises`
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. Real model calls live only in the engine classes. Add `@pytest.mark.slow` test `test_whisper_engine_real_clip` using the Task 0 spike clip: WER < 0.15.
- [ ] **Step 4:** Run `.venv/bin/python -m pytest tests/test_auto_engines_guards.py -q`. Expected: PASS (slow test skipped). Then run `RUN_SLOW=1 .venv/bin/python -m pytest tests/test_auto_engines_guards.py -q -m slow`. Expected: PASS.
- [ ] **Step 5:** Record changed files.

### Task 6: Alignment, confidence and tiers

**Files:**
- Create: `transcribe/auto/align.py`
- Test: `tests/test_auto_align.py`

**Interfaces:**
- Consumes: two `EngineOutput`s over the same chunks.
- Produces:
  - `Slot(chunk: int, kind: str, a: Word | None, b: Word | None)`, with `kind` in `"agree"`, `"substitute"`, `"a_only"`, `"b_only"`
  - `Span(id: int, chunk: int, slots: list[Slot], tier: str, left_context: str, right_context: str)`, with `tier` in `"solid"`, `"disputed"`, `"missing"`. Consecutive non-solid slots in a chunk merge into one span; each solid word is its own solid span.
  - `def norm_token(text: str, lang: str) -> str`: lowercase, strip punctuation incl. ¿¡, strip accents, numbers through the same normalizer as metrics
  - `def align_chunk(a: list[Word], b: list[Word], lang: str) -> list[Slot]`: Levenshtein over `norm_token`; ties prefer substitution over insert+delete
  - `def build_spans(slots_by_chunk: dict[int, list[Slot]], solid_min_prob: float = 0.5, context_words: int = 8) -> list[Span]`
    - `agree` is solid when every available prob ≥ `solid_min_prob`; otherwise a disputed span that is a candidate for refine
    - `substitute` is disputed; `a_only`/`b_only` is missing
  - `def word_confidence(slot: Slot) -> float`: agree → mean of available probs (default 0.9); substitute → 0.3 · max prob; one-sided → 0.2
  - `def single_engine_spans(out: EngineOutput, lang: str, solid_min_prob: float = 0.5) -> list[Span]`: for profiles running one engine; words with prob < threshold become disputed with the only candidate

- [ ] **Step 1: Write failing tests:**
  - `test_align_identical_all_agree`
  - `test_align_substitution_and_insertion`: a "the cat sat down", b "the hat sat" → kinds [agree, substitute, agree, a_only]
  - `test_align_ignores_case_punct_accents`: a "¿Qué tal?" b "que tal" → all agree
  - `test_align_one_engine_empty_chunk`: b [] → all a_only, no exception
  - `test_build_spans_merges_consecutive_disputes_and_contexts`
  - `test_agree_low_prob_is_not_solid`
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement (pure Python DP; chunks are ≤ 30 s, so O(n·m) is fine).
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Record changed files.

### Task 7: Audio verifier (Whisper teacher-forced scoring)

**Files:**
- Create: `transcribe/auto/verify.py`
- Test: `tests/test_auto_verify.py`

**Interfaces:**
- Produces:
  - `class AudioScorer(Protocol): def score(self, samples: np.ndarray, start: float, end: float, text: str, lang: str | None) -> float; def release(self) -> None`
  - `class WhisperScorer(repo: str = "mlx-community/whisper-large-v3-mlx")`:
    - Pads the window to [start − 0.3, end + 0.3] s and caps it at 30 s.
    - Mean log-prob per text token, after the SOT sequence (`<|sot|><|lang|><|transcribe|><|notimestamps|>`).
    - Caches audio features per window.
  - `def compare(scorer: AudioScorer, samples, start, end, candidates: list[str], lang) -> list[float]`

- [ ] **Step 1: Write failing tests** with a fake model whose `logits` returns a fixed distribution:
  - `test_compare_returns_score_per_candidate`
  - `test_window_padding_and_cap`: start 0.1 → window starts at 0.0; 40 s span → 30 s window
  - slow `test_reference_beats_corruption_real`: Task 0 English clip; reference score > 3-word-swap score
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement with mlx-whisper internals (`load_model`, `log_mel_spectrogram`, `pad_or_trim`, `get_tokenizer`).
- [ ] **Step 4:** Run (and with `RUN_SLOW=1`). Expected: PASS.
- [ ] **Step 5:** Record changed files.

### Task 8: Local picker

**Files:**
- Create: `transcribe/auto/picker.py`
- Test: `tests/test_auto_picker.py`

**Interfaces:**
- Consumes: `Span` (Task 6).
- Produces:
  - `class Picker(Protocol): def pick(self, spans: list[Span], lang: str, terms: list[str]) -> dict[int, str]; def release(self) -> None`. Values are `"a"`, `"b"` or `"unclear"`, keyed by `Span.id`.
  - `class MlxPicker(repo: str, batch_size: int = 15, seed: int = 0)`:
    - The prompt shows each span as `id`, left context, `[option 1 | option 2]`, right context, plus the term list.
    - **Option order is shuffled per span with `random.Random(seed + span.id)`** and mapped back. This avoids position bias.
    - Asks for JSON `{"choices": [{"id": int, "choice": "1"|"2"|"unclear"}]}`.
    - Missing, invalid or duplicate ids → `"unclear"`.
    - Generation uses temperature 0.
  - `def span_text(span: Span, side: str) -> str`: the candidate text for side `"a"`/`"b"`; a one-sided span gives `""` for the missing side

- [ ] **Step 1: Write failing tests** with a fake `generate`:
  - `test_picker_maps_shuffled_options_back`: the fake always answers "1"; the result must equal the side that was shown first for each id
  - `test_picker_invalid_json_gives_unclear`
  - `test_picker_batches`: 32 spans, batch 15 → 3 calls
  - `test_span_text_one_sided`
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement with `mlx_lm.load` / `mlx_lm.generate`, lazy-loaded.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Record changed files.

### Task 9: Resolution (pick + verify → fused words)

**Files:**
- Create: `transcribe/auto/resolve.py`
- Test: `tests/test_auto_resolve.py`

**Interfaces:**
- Consumes: `Span`, the picker result, `AudioScorer`.
- Produces:
  - `FusedWord(text: str, start: float, end: float, tier: str, confidence: float, source: str, alternatives: list[str], unclear: bool, speaker: str | None = None)`
    - `source` is one of `"agree"`, `"a"`, `"b"`, `"pick"`, `"verify"`, `"refine"`
    - `tier` is one of `"solid"`, `"resolved"`, `"unclear"`
  - `def resolve(spans: list[Span], picks: dict[int, str] | None, scorer: AudioScorer | None, samples, lang, margin: float = 0.05, default_side: str = "a") -> list[FusedWord]`
- Rules (per non-solid span; `sa`, `sb` = scores of the full window text with side a / side b):

  | Picker | Verifier | Result |
  |---|---|---|
  | X | `s_X ≥ s_other − margin` | X, tier `resolved`, source `pick` |
  | X | `s_other > s_X + margin` | other, tier `resolved`, source `verify` |
  | unclear or `None` | \|sa − sb\| > margin | higher-scoring side, `resolved`, source `verify` |
  | unclear or `None` | \|sa − sb\| ≤ margin | higher side as best guess, `unclear=True`, `alternatives=[other]` |
  | any | scorer `None` (skipped) | picker choice (or `default_side`) with `unclear=True` |

  - A missing span whose winner is the empty side produces no words.
  - Solid spans pass through as `source="agree"`.

- [ ] **Step 1: Write failing tests:** one per table row, plus `test_missing_span_empty_winner_drops_words` and `test_solid_passthrough`.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. Window text = left context + candidate + right context. Window time = the span's word times ± 0.3 s.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Record changed files.

### Task 10: Phonetics and term list

**Files:**
- Create: `transcribe/auto/phonetics.py`, `transcribe/auto/terms.py`
- Test: `tests/test_auto_phonetics_terms.py`

**Interfaces:**
- Produces:
  - `def phonemes(text: str, lang: str) -> str`
    - en: CMUdict via `pronouncing`, stress digits stripped; out-of-vocabulary words → `jellyfish.metaphone`
    - es: rule-based: c/z/s → s (seseo), b/v → b, ll/y → y, silent h, qu/k/c(a,o,u) → k, g(e,i)/j → x, ñ, rr → r
  - `def phonetic_distance(a: str, b: str, lang: str) -> float`: normalized Levenshtein over phoneme strings, 0..1
  - `def build_terms(metadata: dict, extra: list[str] | None = None) -> list[str]`:
    - From title, description and chapters: CamelCase tokens, ALL-CAPS acronyms (≥ 2 letters), tokens with digits, capitalized multi-word names (2–3 consecutive capitalized words not sentence-initial), and `--terms` extras.
    - Deduplicated case-insensitively, capped at 200.
    - URLs, hashtags and @handles are excluded.

- [ ] **Step 1: Write failing tests:**
  - `test_phonetic_distance_en_close`: "self tension" vs "self attention" < 0.35; "attention" vs "banana" > 0.6
  - `test_phonetic_distance_es_seseo`: "casa" vs "caza" == 0.0
  - `test_build_terms_extracts_names_products`: title "Andrej Karpathy: Software Is Changing (Again)", description containing "MenuGen", "LLMs", "GPT-4" and a URL → includes "Andrej Karpathy", "MenuGen", "LLMs", "GPT-4"; excludes the URL
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Record changed files.

### Task 11: Gated Claude refine

**Files:**
- Create: `transcribe/auto/refine_gated.py`
- Test: `tests/test_auto_refine_gated.py`

**Interfaces:**
- Consumes: `FusedWord` list (Task 9), terms (Task 10), `AudioScorer` (Task 7). Reuses `transcribe.refine.resolve_claude_bin` and the headless command flags.
- Produces:
  - `def refine_targets(words: list[FusedWord], spans_meta: list[dict], terms: list[str], lang: str) -> list[dict]`. Each target is `{"id", "start_idx", "end_idx", "text", "alternatives", "left", "right", "reason": "unclear"|"low_conf_agree"|"term_near"}`.
    - `low_conf_agree` = agreed words whose probs were < 0.5 for both engines.
    - `term_near` = a word n-gram (1–3) with `phonetic_distance` ≤ 0.35 to a term, but not equal to it.
  - `class GatedRefiner(model: str | None = None, claude_bin: str | None = None, batch_size: int = 40, timeout_seconds: int = 300)`, with `def refine(self, words, targets, terms, lang, scorer, samples) -> tuple[list[FusedWord], dict]`
    - Calls Claude with JSON schema `{"edits": [{"id": int, "replacement": str}]}`. Only ids it wants to change.
    - The returned stats dict holds `{"targets", "proposed", "accepted", "rejected_by_gate": {gate: n}, "rejected_by_audio", "status"}`.
  - `def gate_edit(target: dict, replacement: str, terms: list[str], lang: str) -> str | None`: returns the failing gate name, or `None` if it passes. Gates are checked in this order:
    1. `"empty"`: the replacement is empty.
    2. `"length"`: word count outside ±20 % (minimum ±1 word).
    3. `"phonetic"`: distance > 0.4 to *every* alternative/original **and** not a term within distance 0.35.
    4. `"filler_removed"`: removes a token in FILLERS (en: um, uh, like, you know, so, right, i mean; es: eh, este, o sea, pues, bueno, ¿no?) that the original has.
    5. `"pronoun_swap"`: replaces a non-pronoun content word with a pronoun (en: it, they, them, this, that, he, she; es: lo, la, eso, esto, él, ella, ellos).
    6. `"locked"`: `reason == "low_conf_agree"` and the replacement isn't a term match.
  - After the gates: audio check with `compare([original, replacement])`. Accept if `s_repl ≥ s_orig + 0.05`, or for term matches if `s_repl ≥ s_orig − 0.10`. Accepted words get `source="refine"`, `tier="resolved"`, `unclear=False`. Rejected → unchanged, and an `unclear` target stays unclear.
  - On a Claude CLI failure (missing binary, non-zero exit, timeout after 2 attempts): return the words unchanged with `status="skipped: <reason>"`. Never raise.
  - Language not in {en, es}: `status="skipped: unsupported language"`.

- [ ] **Step 1: Write failing tests** (mock `subprocess.run` exactly like `tests/test_refine.py` does):
  - `test_gate_rejects_filler_removal`: "so um we built it" → "we built it" → "filler_removed"
  - `test_gate_rejects_pronoun_swap`: "prompt an LLM" alternatives → "prompt it" → "pronoun_swap"
  - `test_gate_allows_term_fix`: "self tension" → "self attention" with terms ["self attention"] → None
  - `test_gate_phonetic_rejects_unrelated`: "AI" for "LLMs", no term → "phonetic"
  - `test_refine_skips_when_cli_missing`: status starts with "skipped"; words unchanged
  - `test_refine_audio_check_rejects`: the fake scorer prefers the original → rejected_by_audio 1
  - `test_command_keeps_safe_flags`: the built command contains `--safe-mode`, `--strict-mcp-config`, `--disable-slash-commands`, `--tools`, `""`
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. The system prompt states the verbatim policy, the no-synonyms rule, "return only edits you are sure of", and that terms come from metadata and may be wrong.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Record changed files.

### Task 12: Diarization, assembly, trust score, outputs

**Files:**
- Create: `transcribe/auto/diarize.py`, `transcribe/auto/assemble.py`
- Modify: `transcribe/output.py` (TXT/SRT only if needed: markers already live inside the text; check that `Speaker {speaker}` rendering works with `"1"`, `"2"`)
- Test: `tests/test_auto_assemble.py`

**Interfaces:**
- Produces:
  - `def diarize(audio_path: Path, num_speakers: int | None = None) -> tuple[list[dict], str]`: pyannote `pyannote/speaker-diarization-community-1`. Returns `(turns [{"start","end","speaker"}], status)`. With no `HF_TOKEN`, or on a load/run failure: `([], "skipped: <reason>")`. Speakers are renumbered `"1"`, `"2"`, … by first appearance.
  - `def assign_speakers(words: list[FusedWord], turns: list[dict]) -> None`: each word gets the speaker of the turn with the largest time overlap; with no turns, `"1"`.
  - `def build_segments(words: list[FusedWord], max_seconds: float = 12.0, pause: float = 0.8) -> list[dict]`
    - Splits on speaker change, pause ≥ `pause`, sentence-final punctuation after ≥ 3 s, or `max_seconds`.
    - Segment text renders unclear words as `[unclear: best | alt]`, grouping consecutive unclear words into one marker.
    - Each segment dict is `{"start","end","speaker","text","words":[...]}`.
  - `def trust_score(words: list[FusedWord], stage_statuses: list[StageStatus], coef: dict | None) -> dict`
    - Features: agreement_rate, unclear_rate, mean_confidence, hallucination_flag_rate, skipped_stages.
    - With `coef` present (from profiles.json), `estimated_wer` = linear model; otherwise `estimated_wer = 0.5*(1-agreement_rate) + unclear_rate`.
    - Returns `{"estimated_wer", "features", "level": "high"|"medium"|"low"}`, thresholds 0.08 / 0.2.

- [ ] **Step 1: Write failing tests:**
  - `test_build_segments_speaker_change_splits`
  - `test_unclear_marker_groups_consecutive`: words a, b(unclear alt x), c(unclear alt y), d → "a [unclear: b c | x y] d"
  - `test_spanish_text_preserved`: "¿Qué tal, señor?" words come out unchanged in the segment text
  - `test_assign_speakers_overlap`
  - `test_diarize_without_token_skips` (monkeypatch env without HF_TOKEN)
  - `test_trust_score_levels`
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Record changed files.

### Task 13: Pipeline orchestration, cache, profiles, CLI `--backend auto`

**Files:**
- Create: `transcribe/auto/pipeline.py`, `transcribe/auto/cache.py`, `transcribe/auto/profiles.py`, `transcribe/profiles.json`
- Modify:
  - `transcribe/constants.py`: add `"auto"` to `SUPPORTED_BACKENDS`; `DEFAULT_MODELS["auto"] = "auto"`
  - `transcribe/cli.py`: add an `auto` branch, plus `--terms` (comma list) and `--num-speakers` (int). For auto, `--refine` default None means on. Skip `validate_refine_constraints` for auto.
  - `transcribe/normalizer.py`: add an `"auto": 16000` rate, plus EBU R128 `loudnorm` for auto only
  - `transcribe/__init__.py`: export `run_auto`
- Test: `tests/test_auto_pipeline.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `profiles.py`: `def load_profiles(path: Path | None = None) -> dict` and `def select(profiles: dict, key: str) -> dict`. `select` falls back from `key` to the language-only key, then to `"default"`, and sets `"unvalidated": True` when the key isn't present.
    - Profile entry: `{"engines": [ids], "pick": bool, "verify": bool, "refine": bool, "margin": float, "solid_min_prob": float, "evidence": str | null}`
    - Top level: `{"version": 1, "generated_from": str | null, "profiles": {...}, "default": {...}, "trust": {"coef": {...}} | null}`
  - The initial `transcribe/profiles.json` uses the Task 0 engine pair for `en`/`es`; `other` → `["whisper-large-v3", <multilingual B>]`; all stages true; `generated_from: null`.
  - `cache.py`: `class StageCache(root: Path = ~/.cache/whisper_translate/auto)`, with `def get(self, stage: str, key: dict) -> Any | None` and `def put(self, stage: str, key: dict, value: Any) -> None`. Keys are a sha256 of the audio sha + stage + JSON config. JSON values are serialized from the dataclasses.
  - `pipeline.py`: `def run_auto(audio_path: Path, metadata: dict, *, language: str | None = None, terms: list[str] | None = None, num_speakers: int | None = None, refine: bool = True, profiles: dict | None = None, factories: dict | None = None) -> tuple[BackendResult, dict]`
    - Returns a `BackendResult` with `segments` from `build_segments` and a metadata dict: `{"profile": AudioProfile as dict, "profile_entry", "engines", "stages": [StageStatus...], "trust": {...}, "words": [FusedWord dicts]}`.
    - `factories` injects engines, scorer, picker, refiner, diarizer and lid, for tests.
  - Order: load audio → VAD/chunks → probe → (diarize) → engine A (release) → engine B (release) → guards → align → pick (release) → resolve with verify (release scorer after) → refine (gated; the scorer is re-loaded only if needed) → assign speakers → segments → trust.
  - Every stage is wrapped in a timer that appends a `StageStatus`. Enhancement-stage exceptions → `"skipped: <exc>"`. Core-stage exceptions → `TranscriptionError`.
  - Zero speech chunks → empty segments, metadata says `"no speech detected"`, no engine is called.
  - The CLI writes outputs through the existing `OutputWriter`. JSON `metadata["auto"]` = the metadata dict. The CLI logs a WARNING for each skipped stage and when the trust level is `low`.

- [ ] **Step 1: Write failing tests** with fake factories:
  - `test_run_auto_end_to_end_fakes`: two fake engines that differ on one word, a fake picker and a fake scorer → segment text has the picked word; stages all "ok"
  - `test_engine_exception_is_core_failure`: raises `TranscriptionError`
  - `test_picker_failure_skips_and_marks_unclear`: picker raises → stage "skipped", disputed words unclear, trust features show skipped_stages 1
  - `test_refine_cli_missing_still_writes`: through `cli.main([... "--backend", "auto"])` with fakes patched in; output files exist; JSON stages include refine skipped
  - `test_no_speech_no_engine_calls`
  - `test_releases_called_between_stages`: each fake records `release()`
  - `test_cache_hit_skips_engine_second_run`
  - `test_profiles_select_fallbacks_and_unvalidated`
  - `test_single_engine_profile`: profile engines length 1 → no picker call; low-prob words unclear
  - `test_existing_backends_unchanged`: `parse_args(["x.wav"]).backend == "mlx-whisper"`
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `.venv/bin/python -m pytest -q` (the whole suite). Expected: PASS.
- [ ] **Step 5:** Real smoke: `.venv/bin/python transcribe_youtube.py ~/.cache/whisper_translate/suite/spike/en_30s.wav --backend auto --output-dir /tmp/auto_smoke` (and the es clip). Expected: outputs written, stages listed, no core failure.
- [ ] **Step 6:** Record changed files.

### Task 14: Suite integration of `auto`, ablations, regression gate, profile generation

**Files:**
- Modify: `benchmarks/suite/run_suite.py` (add auto systems)
- Create: `benchmarks/suite/gate.py`, `benchmarks/suite/make_profiles.py`
- Test: `tests/test_suite_gate_profiles.py`

**Interfaces:**
- Consumes: `run_auto` with `factories` / profile overrides; Task 2 metrics; Task 3 runner.
- Produces:
  - Systems:
    - `auto:single-a`, `auto:single-b`
    - `auto:fused` (align + tie-break only: no picker, no verify; disputed spans → engine A, unclear)
    - `auto:+pick`, `auto:+verify` (pick + verify), `auto:+refine` (full)
  - All of them share the cached engine outputs (one ASR run per engine per item). Each item's metrics also include `edit_effects` for the pick, verify and refine stages (before = the text before the stage, after = the text after it).
  - `gate.py`: `def compare(base: dict, cand: dict, split: str = "test") -> dict`. A paired bootstrap per slice on per-item WER difference gives `{slice: {"delta", "ci", "worse": bool}}`, plus harm-rate deltas and nospeech insertions. `def passes(report: dict) -> bool` is false if any slice is `worse` (CI lower bound > 0), any stage's harm rate rises by more than 1 pp, or nospeech insertions rise.
  - `make_profiles.py`: `def make_profiles(results: dict, slice_to_profile: dict[str, str]) -> dict`
    - Each profile key gets the smallest configuration whose dev WER is not worse than any larger configuration (bootstrap CI) and whose stage harm rate is ≤ 2 %. A stage that raises WER or harm is switched off. A second engine that doesn't beat the best single engine leaves the profile single-engine.
    - Fits trust `coef` by least squares of per-file WER on the features (dev split).
    - Writes `transcribe/profiles.json` with `evidence` pointing to the results file.
    - `slice_to_profile`: `en_*` → `en`, `es_*` → `es`, `*_phone` → `{lang}-phone`, `*_noise` → `{lang}-noisy`, `nospeech` → all keys (insertions only).

- [ ] **Step 1: Write failing tests:**
  - `test_gate_flags_worse_slice` (synthetic per-item WERs)
  - `test_gate_passes_identical`
  - `test_make_profiles_turns_off_harmful_refine`: synthetic results where +refine raises es WER → `profiles["es"]["refine"] is False`
  - `test_make_profiles_single_engine_when_fusion_loses`
  - `test_trust_coef_fit_recovers_linear_relation`
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Record changed files.

### Task 15: Measurement run, tuning, profiles, docs

**Files:**
- Create: `benchmarks/suite/results/auto-smoke.json`, `benchmarks/suite/results/auto-full.json`, `docs/agents/auto-pipeline-results.md`
- Modify: `transcribe/profiles.json` (generated), `CLAUDE.md` (backend list, auto pipeline section, revised no-fallback rule, eval suite commands), `docs/agents/backend-capabilities.md`, `docs/agents/command-recipes.md`, `docs/agents/testing-and-release.md`, `docs/agents/refine-evals.md` (pointer: graded eval replaces human audit)

- [ ] **Step 1:** Run all systems on the smoke manifest: `.venv/bin/python -m benchmarks.suite.run_suite --systems mlx-whisper-turbo,eg2,auto:single-a,auto:single-b,auto:fused,auto:+pick,auto:+verify,auto:+refine --manifest smoke --out benchmarks/suite/results/auto-smoke.json`. Fix crashes before going on.
- [ ] **Step 2:** Tune on the **dev** split only. Grid over `margin ∈ {0.0, 0.05, 0.1, 0.2}` and `solid_min_prob ∈ {0.3, 0.5, 0.7}`, using cached engine outputs (no new ASR). Pick per language by the lowest dev WER, subject to unclear density ≤ 3 %.
- [ ] **Step 3:** Run the full manifest with the tuned settings (background; it takes hours) into `auto-full.json`. Then run `make_profiles` and write `transcribe/profiles.json`.
- [ ] **Step 4:** Report on the **test** split. Compare the profiled `auto` against the best single engine with `gate.compare`. Write `docs/agents/auto-pipeline-results.md` with:
  - a per-slice table, worst slice first: WER of every system, entity error, nospeech insertions, ×RT, peak memory
  - stage help/harm rates
  - unclear precision/recall/density
  - trust-score Spearman
  - which switches `make_profiles` turned off and why
  - spec §7 success criteria: met / not met, each with numbers
  - dataset gaps (code-switching; diarization if no HF_TOKEN)
- [ ] **Step 5:** Real-audio run: `--backend auto` on the 4 talks already transcribed (the URLs are in the existing JSONs' `source`) and one Spanish talk, picked from YouTube with `yt-dlp` search `ytsearch1:charla tecnología español conferencia` (prefer ≥ 15 min). Record agreement, unclear count, trust and stage statuses in the results doc. Don't upload anything.
- [ ] **Step 6:** Update `CLAUDE.md` and `docs/agents/*` as listed. Run `.venv/bin/python -m pytest -q`. Expected: PASS.
- [ ] **Step 7:** Record changed files and the headline results in the progress file.

### Task 16: Whole-branch review

- [ ] **Step 1:** A fresh reviewer reads the spec, this plan, the progress file and every changed file. It checks spec coverage, the global constraints (especially the Claude flags, no commits, verbatim gates, the failure policy and memory release) and the Review Focus tests.
- [ ] **Step 2:** Fix the findings. Re-run the full tests and the smoke suite.
- [ ] **Step 3:** Write the morning summary in the progress file: what was built, measured results against the success criteria, what's pending (HF token, dataset gaps), and how to run it.

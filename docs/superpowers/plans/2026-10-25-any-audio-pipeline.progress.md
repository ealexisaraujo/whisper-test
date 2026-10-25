# Any-audio pipeline: progress log

Plan: `docs/superpowers/plans/2026-10-25-any-audio-pipeline.md` · Spec: `docs/superpowers/specs/2026-10-25-any-audio-pipeline-design.md`
Started 2026-10-25 (user approved the design and asked for autonomous implementation overnight). Nothing is committed (the user commits on request).

| Task | Status | Changed files | Notes |
|---|---|---|---|
| 0 | done (not committed) | `requirements.txt` (modified); `benchmarks/suite/__init__.py`, `benchmarks/suite/spike.py`, `benchmarks/suite/spike-results.json`, `docs/agents/auto-pipeline-spike.md`, `docs/agents/auto-pipeline-spike-prefreeze.txt` (new) | ENGINE_A=qwen3-asr-1.7b, ENGINE_B=whisper-large-v3 (en, es); PICKER_MODEL=mlx-community/gemma-4-26b-a4b-it-4bit; transformers upgraded to 5.20.2 and huggingface_hub to 2.3.0 (47 tests pass; vibevoice/diffusers/gradio in the shared venv now fail `pip check`). Details: `docs/agents/auto-pipeline-spike.md` |

## Task 1: eval datasets and slice manifests (done, not committed)

Changed files: `benchmarks/suite/datasets.py`, `benchmarks/suite/slices.json`, `benchmarks/suite/manifests/*.jsonl` (13 slices + `smoke.jsonl` + `summary.json`), `tests/test_suite_datasets.py` (all new). Audio (760 MB, 16 kHz mono PCM WAV) lives in `~/.cache/whisper_translate/suite/audio/<slice>/`. Tests: 8 new, full suite 55 passed. Run: `.venv/bin/python -m benchmarks.suite.datasets prepare [--smoke-only] [--slices a,b] [--force]` (reuses existing manifests unless `--force`).

| Slice | Items | Minutes | Dev min | Test min | Smoke min |
|---|---:|---:|---:|---:|---:|
| en_clean | 10 | 40.9 | 11.4 | 29.5 | 2.2 |
| en_talks | 7 | 39.0 | 5.6 | 33.4 | 3.1 |
| en_accents | 9 | 43.9 | 25.8 | 18.1 | 2.0 |
| en_business | 9 | 38.0 | 12.0 | 26.1 | 1.9 |
| en_meeting | 9 | 35.7 | 23.6 | 12.2 | 1.9 |
| es_fleurs | 8 | 40.9 | 17.5 | 23.4 | 2.2 |
| es_mls | 8 | 41.0 | 18.6 | 22.4 | 2.2 |
| es_dialects | 6 | 30.3 | 18.2 | 12.1 | 2.1 |
| en_noise | 7 | 30.7 | 11.4 | 19.3 | 2.3 |
| es_noise | 4 | 21.0 | 2.2 | 18.8 | 2.2 |
| en_phone | 5 | 20.9 | 6.2 | 14.7 | 2.3 |
| es_phone | 4 | 21.0 | 2.2 | 18.8 | 2.2 |
| nospeech | 30 | 10.0 | 4.7 | 5.3 | 2.7 |
| **Total** | 116 | 413.3 (6.89 h) | 159.2 | 254.0 | 29.1 |

Gaps (also in `slices.json` under `gaps`): code-switching (Bangor-Miami is whole 30 min CHAT-annotated conversations without timestamps); no music in nospeech (ESC-50 has none); Peruvian/Puerto Rican/Venezuelan dialects (OpenSLR 73/74/75 mirrors are gated, no HF_TOKEN). Notes: noise/phone slices are paired with en_clean (test-clean) and es_fleurs items (same reference, split inherited), so dev/test minutes are lumpy for the small derived slices (es_noise 2.2 dev vs 18.8 test); en_meeting uses edinburghcstr/ami sdm because diarizers-community/ami has no transcript.

## Task 2: Metrics (done, not committed)
- benchmarks/suite/metrics.py (new)
- tests/test_suite_metrics.py (new; 29 tests after fix round 1, full suite 94 passed)

## Task 3: Suite runner and baseline (done, not committed)
- benchmarks/suite/run_suite.py (new), benchmarks/suite/results/ (new dir; baseline-smoke.json written, baseline-full.json in progress)
- tests/test_suite_runner.py (new; 9 tests, full suite 103 passed)
- Smoke baseline (20 items, dev+test): mlx-whisper-turbo mean per-slice WER 8.5 % (worst en_meeting 38.5 %, pooled WER 9.5 %, xrt 11.5, nospeech 14.2 ins/min);
  eg2 mean per-slice WER 18.8 % over 7 en slices (worst en_meeting 49.3 %, pooled 18.8 %, xrt 119, nospeech 0.0 ins/min, es items skipped).
- Full baseline running in background: log scratchpad/task3/full.log, out benchmarks/suite/results/baseline-full.json.

## Task 4: Core types, VAD chunking, probe (done, not committed)
- transcribe/auto/__init__.py, types.py, chunks.py, probe.py (new)
- tests/test_auto_chunks_probe.py (new; 11 fast tests + 2 slow), tests/conftest.py (new; registers `slow` marker, skips unless RUN_SLOW=1)
- Full suite 127 passed, 2 skipped; RUN_SLOW=1 real silero-vad + mlx-whisper LID on en_30s/es_30s passes.

## Task 5: Local engines and hallucination guards (done, not committed)
- transcribe/auto/engines.py, transcribe/auto/guards.py (new)
- tests/test_auto_engines_guards.py (new; 53 fast tests + 2 slow after fix round 1), full suite 184 passed, 6 skipped
- RUN_SLOW=1 real clip en_30s: whisper-large-v3 WER 0.0 %, qwen3-asr-1.7b WER 0.0 %.

## Task 6: Alignment, confidence and tiers (done, not committed)
- transcribe/auto/align.py (new)
- tests/test_auto_align.py (new; 12 tests)

## Task 7: Audio verifier (done, not committed)
- transcribe/auto/verify.py (new)
- tests/test_auto_verify.py (new; 10 fast tests + 4 slow), full suite 209 passed, 10 skipped
- RUN_SLOW=1 real large-v3: en_30s reference -0.138 vs 3-swap -0.876; es_30s -0.133 vs -1.588; one-word substitution (3-word span) en -3.093 vs -7.186, es -3.703 vs -6.960.

## Task 8: Local picker (done, not committed)
- transcribe/auto/picker.py (new)
- tests/test_auto_picker.py (new; 22 fast tests + 1 slow), full suite 233 passed, 13 skipped
- RUN_SLOW=1 real gemma-4-26b-a4b-it-4bit: load+first 13-14 s, single span 1.1 s, batch of 3 1.7 s; eat/heat -> eat, es aplazar -> aplazar, nonsense -> unclear.

## Task 9: Resolution (done, not committed)
- transcribe/auto/resolve.py (new)
- tests/test_auto_resolve.py (new; 24 tests), full suite green

## Task 10: Phonetics and term list (done, not committed)
- transcribe/auto/phonetics.py (new)
- transcribe/auto/terms.py (new)
- tests/test_auto_phonetics_terms.py (new; 35 tests)

## Task 11: Gated Claude refine (done, not committed)
- transcribe/auto/refine_gated.py (new)
- tests/test_auto_refine_gated.py (new; 46 tests after fix round 1), full suite 356 passed, 13 skipped

## Task 12: Diarization, assembly, trust score, outputs (done, not committed)
- transcribe/auto/diarize.py (new)
- transcribe/auto/assemble.py (new)
- tests/test_auto_assemble.py (new; 32 tests), transcribe/output.py unchanged (verified via OutputWriter test)

## Task 13: Pipeline orchestration, cache, profiles, CLI `--backend auto` (done, not committed)
- transcribe/auto/pipeline.py (new), transcribe/auto/cache.py (new), transcribe/auto/profiles.py (new), transcribe/profiles.json (new)
- transcribe/constants.py ("auto" backend + default model), transcribe/cli.py (auto branch, `--terms`, `--num-speakers`, `--refine` default None), transcribe/normalizer.py (auto rate, loudnorm for auto only, `_probe_sample_rate`), transcribe/output.py (empty SRT when `metadata.no_speech`), transcribe/__init__.py (export `run_auto`)
- tests/test_auto_pipeline.py (new; 27 tests), tests/test_refine.py (one assertion: `--refine` default is now None)

## Task 14: Suite integration of auto, ablations, regression gate, profile generation (done, not committed)
- benchmarks/suite/run_suite.py (AutoSystem + AUTO_SYSTEMS, `build_system("auto:*")`, `trust_features` passthrough in `score_item`)
- benchmarks/suite/gate.py (new), benchmarks/suite/make_profiles.py (new)
- transcribe/auto/pipeline.py (meta `stage_texts`: fused/pick/verify/refine best-guess texts)
- tests/test_suite_gate_profiles.py (new; 28 tests), full suite 459 passed, 13 skipped

## Task 15: Measurement run, tuning, profiles, docs (done, not committed)
- benchmarks/suite/results/auto-smoke.json, auto-full.json, auto-configured-full.json (new; configured = per-item composition from the ladder rungs using the calibrated profiles and each item's probed profile key)
- transcribe/profiles.json (generated by make_profiles from auto-full.json dev split: every en/es key single-engine; trust coefficients fitted)
- docs/agents/auto-pipeline-results.md (new); CLAUDE.md, docs/agents/backend-capabilities.md, command-recipes.md, testing-and-release.md, refine-evals.md (pointer: graded suite replaces the human audit)
- benchmarks/suite/run_suite.py: picker-answer and verify-score memos for the auto systems (`cached_picker_factory`, `cached_scorer_factory`); bug fix: `AutoSystem._engines` keeps the engine pair (from `default`) when a calibrated profile is single-engine, so the ablation ladder survives calibration
- transcribe/auto/engines.py: bug fix: `_transcribe_each` clears the MLX buffer cache after every chunk (Whisper large-v3 alone on the 30-min Karpathy audio: 25.7 GB peak without the clear, 6.3 GB with it, same output)
- tests: tests/conftest.py (autouse fixture: tests read the frozen tests/data/profiles_uncalibrated.json, not the generated live table), tests/data/profiles_uncalibrated.json (new), test_auto_pipeline.py (live profiles file validates), test_auto_engines_guards.py (cache cleared per chunk), test_suite_gate_profiles.py (memos, ladder pair after calibration, cache-key test uses a swapped pair); full suite 474 passed, 13 skipped
- Headline (test split): configured auto pooled WER 9.0 vs Qwen3 alone 9.2, Whisper large-v3 alone 10.6, turbo 9.7; gate vs single-a and single-b PASS, vs turbo FAIL (en_meeting +11.7 pts, es_mls +1.7, en_phone +0.2). Speech-free insertions 0.0/min (turbo 15). Pick harm 39 %, verify harm 54 %, refine 1 edit in the suite. Trust Spearman 0.77-0.81 on dual-engine rungs, 0.44 on the single-engine configured profiles. Margin grid {0, 0.05, 0.2} noise-level: kept 0.05. DER pending (no HF_TOKEN).
- Task 15 fix round 1: benchmarks/suite/make_profiles.py (`_decide_key` takes the ladder pair via `run_suite.ladder_pair`), benchmarks/suite/run_suite.py (`FROZEN_PAIRS`, `lang_key_of`, `ladder_pair`; memo keys carry PICKER/SCORER_MEMO_VERSION, CACHE_VERSION, picker decoding and scorer constants; `build_system(refresh=)` threads `--refresh` into the memos), transcribe/auto/engines.py (`_free_memory` reuses `_clear_mlx_cache`), tests/test_suite_gate_profiles.py (5 new tests), docs (results doc, backend-capabilities, command-recipes, CLAUDE.md); full suite 479 passed, 13 skipped

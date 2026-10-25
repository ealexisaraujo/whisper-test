# EG2 vs mlx-whisper — benchmark and claim review (2026-10-24)

How the engine works: [eg2-engine-analysis.md](eg2-engine-analysis.md). Raw numbers:
`benchmarks/results-librispeech-test-clean.json`. Re-run with `benchmarks/bench_asr.py`.

## Setup

- Machine: **Apple M1 Pro, 32 GB, macOS 26.6.2**, Python 3.11.1, coremltools 9.0, mlx-whisper 0.4.3.
  This is **not** the base M3 MacBook Air (8 GB) used in the vendor's chart.
- EG2: our Python port (`transcribe/eg2.py`) of the v5 models. Reader on the ANE, scorer on the GPU,
  punctuation on. The models are the vendor's files, byte-identical (SHA-256 checked against the manifest).
- Whisper: `mlx-community/whisper-turbo`, the repo default, `language="en"`.
- Accuracy: WER after Whisper's `EnglishTextNormalizer`, applied to both engines.
- Speed: wall time of transcription only. Audio is pre-decoded to 16 kHz arrays. Model load and one warm-up file are excluded.

## Results

### LibriSpeech test-clean: 87 chapters as long-form files, 5.40 h, human references

| Engine | × real time | s per audio hour | WER | Sub / Del / Ins |
|---|---:|---:|---:|---|
| EG2, 3 ANE workers | **107.8×** | 33.4 | 5.26 % | 1867 / 625 / 296 |
| EG2, 2 ANE workers (default) | 103.4× | 34.8 | 5.26 % | 1867 / 625 / 296 |
| EG2, 1 worker | 100.8× | 35.7 | 5.26 % | 1867 / 625 / 296 |
| EG2, app layout (2nd encoder on GPU) | 102.9× | 35.0 | 5.25 % | 1867 / 625 / 294 |
| mlx-whisper turbo | 16.5× | 218.7 | **1.93 %** | 711 / 152 / 161 |

Median chapter WER: EG2 4.82 %, Whisper 1.63 %. Worst chapter: EG2 12.9 %, Whisper 5.9 %.
A single 53.7-minute file ran at 121× (26.7 s), because fewer files means less per-file overhead.

### Real-world: "Data Engineering Resume Workshop with OpenAI Recruiter" (YouTube `zA4aBDzj_60`, 22.4 min)

Run end to end through the CLI (`--backend eg2` vs `--backend mlx-whisper`). There is no human
reference, so Whisper's output is the reference. These are **disagreement** rates, not true WER.

| | Time | EG2 vs Whisper |
|---|---|---|
| EG2 | 12 s transcribe + 4 s model load | 10.0 % word disagreement (S 245 / D 141 / I 53) |
| mlx-whisper | 70.4 s | — |

Typical EG2 misses: dropped suffixes (`worked→work`, `impactful→impact`, `improved→improve`),
function words (`and→in`, `their→the`), and names (`OpenAI→AI/opening/eye`, `Omari Sealy→Omar`).

## Claim review

| Vendor claim | Verdict | Evidence |
|---|---|---|
| "Read a transcript straight out of the embedding model": spoken word lands next to the written word | **Confirmed in design** | The reader outputs one 768-d vector per 40 ms frame (EG2's text-embedding size). Words come from a nearest-neighbour lookup against 229,457 unit-norm 768-d word vectors. There is no token decoder. |
| 288× real time (1 h in 12.6 s) on a base M3 Air | **Not reproduced; plausible only on newer hardware** | 103–121× on M1 Pro (30–35 s/h). The run is ANE-bound: encoder 73 ms + reader 27 ms per 18 s of new audio. That caps an M1 Pro near 180×. M3's faster ANE plus native Swift could get closer, but we have not measured it. |
| Faster than Parakeet/Granola-class models | Partly | Versus Whisper turbo on the same Mac: **6.3–6.5× faster**. Parakeet and Granite were not run here. |
| Accuracy | **Not stated by the vendor** | Measured: 5.26 % WER on clean read speech (2.7× Whisper turbo's errors). About 10 % disagreement with Whisper on a real tech talk. |
| "Nothing leaves the device" | Consistent with static strings; not exhaustively verified | The only hosts in the binary are the model host (`haku-models.tslfiles.org`: manifest, models, Sparkle appcast) and Google/HF legal links. Runtime traffic was not observed. |

## What the numbers mean

- EG2 trades accuracy for speed: about 6× faster, about 2.7× the errors on clean speech.
  Out-of-vocabulary words are **not** the main cause: only 0.24 % of LibriSpeech reference tokens and
  0.31 % of the YouTube talk's words are missing from its vocabulary. Most errors are substitutions,
  which is what you get from greedy per-frame nearest-word matching with no language model.
- The official model card says: "Do not use `float16`." The vendor's Core ML conversion is fp16 and
  palettized. Its reader returns NaN on CPU and GPU, so it only works on the ANE.
- English only. A non-English `--language` is rejected (no fallback).

## When to use which backend

| Need | Backend |
|---|---|
| Fast English draft, search/indexing, triage of many hours | `--backend eg2` |
| Publishable transcript, names/brands, summaries you will quote | `--backend mlx-whisper` |
| Spanish or any non-English audio | `--backend mlx-whisper` |

## Refine stage (`--refine`, Claude via headless `claude -p`) — measured 2026-10-24

| Test | EG2 | EG2 + refine | Whisper turbo | Refine cost / time |
|---|---:|---:|---:|---|
| LibriSpeech test-clean, 12 chapters (0.67 h), WER | 3.95 % | **1.44 %** | ≈1.23 % (same chapters) | $0.63 ($0.93 per audio hour), 239 s |
| YouTube `zA4aBDzj_60` (22.4 min), disagreement vs Whisper | 10.0 % | **6.8 %** | reference | $0.33, 47 s, 172/448 segments changed |

Run with `benchmarks/bench_asr.py --engines eg2 --limit 12 --refine` (model alias `sonnet`).

How to read it:
- **The LibriSpeech gain is inflated.** LibriSpeech is public-domain books that the model probably saw in training. It restored
  wording that can't be inferred from the audio errors (`flour fat sauce` → `flour fattened sauce`, exactly as in the reference).
  The YouTube talk is the more honest number: about a third fewer disagreements.
- **Context can plant wrong names.** In the talk, EG2 heard "the team at from…". Refine wrote "at Meta" because the description
  mentions Meta. The speaker said OpenAI. Treat names inserted by refine as unverified.
- **Speed:** refine takes about 2–6 min per audio hour (4 parallel calls). EG2 + refine is no longer faster than mlx-whisper
  (about 3.6 min/h). Its advantage is fixing names and terms using the title and description.

## Ideas to close the accuracy gap (not implemented)

1. Beam search over the 64 re-ranked candidates per frame, with a word n-gram LM (the vocabulary is closed, so an LM is cheap).
2. Use the v6 `letters_LS` character-CTC head to rescore low-margin words and spell out-of-vocabulary terms. The vendor ships this in v6, but app 1.6 does not use it.
3. Punctuate in overlapping 96-word windows instead of disjoint blocks.
4. Use EG2 as a fast first pass, then re-transcribe only low-confidence spans with Whisper (per-word `score` is already in the JSON output).

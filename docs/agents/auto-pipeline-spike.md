# `--backend auto` environment spike (2026-10-25)

Every model the `auto` pipeline needs was installed and run on this machine: **Apple M1 Pro, 32 GB, macOS 26.6.2,
Python 3.11.1**. Re-run with `benchmarks/suite/spike.py` (usage in its docstring). Raw numbers are in
`benchmarks/suite/spike-results.json`.

**`.venv` is shared with the main checkout** (the main checkout of this repo): it is a symlink to
that repo's venv, so every install here changed the main checkout's environment too. The state before this spike is
in `docs/agents/auto-pipeline-spike-prefreeze.txt` (`pip freeze`, usable for rollback). `requirements.txt` was only
verified inside this venv; it was not tested on a fresh install.

## Decisions (later tasks read these)

```
ENGINE_A[en] = qwen3-asr-1.7b      ENGINE_B[en] = whisper-large-v3
ENGINE_A[es] = qwen3-asr-1.7b      ENGINE_B[es] = whisper-large-v3
SUBSTITUTE   = parakeet-v3         (en+es; the fastest; the swap candidate if the suite shows complementary errors)
MEASURED_WER_PAIR = qwen3-asr-1.7b + parakeet-v3   (see below: this is what the WER numbers alone select)
EXCLUDED     = eg2                 (English only, 4.8 % mean WER vs about 1.6 % for the others)
PICKER_MODEL = mlx-community/gemma-4-26b-a4b-it-4bit      (fallback: mlx-community/Qwen3-30B-A3B-4bit)
VERIFIER     = mlx-whisper large-v3 (mlx-community/whisper-large-v3-mlx), teacher-forced scoring works
VAD          = silero-vad 6.2.3 (works)
DIARIZATION  = pyannote.audio 4.0.7 imports; HF_TOKEN is NOT set here, so diarization is skipped on this machine
WORD_ALIGN   = Qwen3-ForcedAligner-0.6B (mlx-community/Qwen3-ForcedAligner-0.6B-bf16) for engines without word times
               status: plumbing proven; it agrees with Whisper word times (median 0.15-0.16 s, p90 0.37-0.38 s);
               absolute timing accuracy is UNVERIFIED (no human word times exist for these clips)
```

**Plain statement on the pair.** On measured WER alone the pair would be **Qwen3-ASR + Parakeet v3** (en 1.55 % and
1.55 %, es 0.76 % and 1.52 %; Whisper is 1.67 % en and 2.41 % es). The recorded pair is **Qwen3-ASR + Whisper**, and
Whisper was chosen under the rule's architecture-preference clause (Whisper plus a non-Whisper engine), plus
three non-WER reasons: it is the only engine with word-level timestamps and word probabilities, the verifier
(S6) already needs Whisper loaded, and its measured deficit is a single trailing hallucinated word per clip that the
guards must remove anyway. The difference between the three engines is 1-3 words over about 300 words, within noise,
so this is a judgment call that the eval suite (spec section 9) must confirm or overturn.

### Why this engine pair

The rule: the two most accurate engines that both run, preferring a pair whose errors differ by architecture
(Whisper plus a non-Whisper engine).

Mean WER over the two clips per language (table below):

| Language | whisper-large-v3 | qwen3-asr-1.7b | parakeet-v3 | eg2 |
|---|---:|---:|---:|---:|
| English (en, en_hard) | 1.67 % | **1.55 %** | **1.55 %** | 4.76 % |
| Spanish (es, es_hard) | 2.41 % | **0.76 %** | 1.52 % | n/a |

- Qwen3-ASR is best or tied-best in both languages, so it is Engine A.
- Whisper is a few words behind **only because of one trailing hallucinated word on 4 of 4 clips** (`And.`,
  `Gracias.`, `that.`, `Gracias.`). Trailing silence after the last speech (silero-vad): en 0.0 s, es 1.9 s,
  en_hard 0.3 s, es_hard 0.7 s, so it happens both flush with speech (en) and over silence (es). Dropping that last word, Whisper's WER is
  en 0.0 %, en_hard 1.0 %, es 1.5 %, es_hard 0.0 % (mean en 0.5 %, es 0.8 %), which matches or beats the others.
  The S3 hallucination guards (VAD gating, no text over silence) must cover this.
- Whisper is the only engine with real word timestamps plus word probabilities, so it is the natural Engine B
  and the verifier, and it differs in architecture from Qwen3-ASR. Parakeet (TDT transducer) is the better
  non-Whisper alternative to Qwen3 on speed (about 45-55 x RT against about 9-13 x), with similar accuracy on English.
- EN and ES use the same pair. The spec's table already assumes this.
- **Caveat on the numbers:** 4 clips, about 300 reference words in total, so one word is 1-1.5 % WER. The ranking
  between Qwen3, Parakeet and Whisper is within noise; the pair is chosen by the rule, and the eval suite
  (spec section 9) confirms it. EG2's gap (8.3 % on `en_hard`) is large enough to trust.

## Results

Clips (`~/.cache/whisper_translate/suite/spike/`, 16 kHz mono, each with a sibling `.txt` reference):

| Clip | Source | Length | Ref words |
|---|---|---:|---:|
| `en_30s` | LibriSpeech test-clean (speaker 6930, chapter 75918) | 29.5 s | 79 |
| `es_30s` | FLEURS es_419 test | 31.7 s | 66 |
| `en_hard_30s` | LibriSpeech test-other | 30.3 s | 96 |
| `es_hard_30s` | Multilingual LibriSpeech Spanish test | 31.5 s | 56 |

The two `*_hard` clips are an addition to the brief: the first two clips are so clean that most engines score 0 %.
Utterances are joined with 0.3 s gaps; utterances that would overshoot about 33 s are skipped.

Measurements: WER after Whisper's normalizer (`EnglishTextNormalizer` for en, `BasicTextNormalizer` for es). Model
load is timed separately. x RT is steady state: the second transcription of the clip after the model is loaded
(the first one pays Metal kernel compilation; those numbers are in the JSON as `x_rt_first`). `RSS` is the process
peak resident set from `getrusage`; `MLX peak` is `mx.get_peak_memory()` (GPU/unified memory the RSS can miss).
Timestamps and probs: what the engine returned in this run.

| Engine | Clip | WER | Load (s) | x RT | RSS (GB) | MLX peak (GB) | Word timestamps | Word probs |
|---|---|---:|---:|---:|---:|---:|---|---|
| whisper-large-v3 | en | 1.27 % | 2.5 | 6.5 | 3.39 | 3.70 | yes | yes (`probability`) |
| whisper-large-v3 | es | 3.03 % | 2.5 | 6.6 | 3.39 | 3.70 | yes | yes |
| whisper-large-v3 | en_hard | 2.06 % | 2.2 | 4.8 | 3.39 | 3.69 | yes | yes |
| whisper-large-v3 | es_hard | 1.79 % | 2.4 | 5.4 | 3.39 | 3.69 | yes | yes |
| qwen3-asr-1.7b | en | 0.00 % | 6.5 | 11.2 | 3.30 | 5.15 | no | no |
| qwen3-asr-1.7b | es | 1.52 % | 5.9 | 10.3 | 3.70 | 5.20 | no | no |
| qwen3-asr-1.7b | en_hard | 3.09 % | 6.3 | 8.7 | 3.46 | 5.17 | no | no |
| qwen3-asr-1.7b | es_hard | 0.00 % | 5.3 | 13.6 | 4.21 | 5.20 | no | no |
| parakeet-v3 | en | 0.00 % | 2.0 | 53.0 | 0.92 | 2.52 | yes (token) | yes (token `confidence`) |
| parakeet-v3 | es | 3.03 % | 1.7 | 44.9 | 1.73 | 4.17 | yes (token) | yes (token) |
| parakeet-v3 | en_hard | 3.09 % | 1.8 | 44.1 | 1.35 | 2.96 | yes (token) | yes (token) |
| parakeet-v3 | es_hard | 0.00 % | 1.7 | 56.0 | 1.93 | 4.57 | yes (token) | yes (token) |
| eg2 | en | 1.27 % | 4.3 | 96.4 | 1.09 | n/a (Core ML) | yes | yes (CTC margin `score`) |
| eg2 | en_hard | 8.25 % | 3.3 | 117.7 | 1.14 | n/a (Core ML) | yes | yes (CTC margin) |
| eg2 | es, es_hard | not run | | | | | | English only |

Notes on the cells:

- Qwen3-ASR (`mlx-community/Qwen3-ASR-1.7B-bf16`, via `mlx-audio`) returns one text blob (1 segment): no word
  times and no probabilities. The separate **Qwen3-ForcedAligner-0.6B** supplies them. Evidence (`spike.py align`,
  fed the **Qwen3 hypothesis**, not the reference, and compared with Whisper's own word starts, matched words only):

  | Clip | Words aligned / matched | Median abs start diff | p90 abs start diff | Time | Aligner last end | Whisper last real word end |
  |---|---:|---:|---:|---:|---:|---:|
  | en_30s (29.5 s) | 79 / 79 | 0.16 s | 0.38 s | 0.4 s | 29.36 s | 29.28 s |
  | es_30s (31.7 s) | 66 / 66 | 0.15 s | 0.37 s | 0.4 s | 29.68 s | 29.48 s |

  The two engines agree to about 0.15 s (median), 0.4 s at p90. That fits the +-0.3 s alignment window in S4 at the median
  only; p90 is larger than the window. Which engine is closer to
  the true word boundary is **not known**: there are no human word times for these clips, so absolute accuracy is
  unverified; only plumbing and cross-engine agreement are proven. Peak across the align run 8.1 GB MLX (Qwen3-ASR
  and the aligner, loaded one after the other).
- The Spanish clip's 2 s tail gap is explained, not an alignment error: `es_30s` ends with 1.9 s of silence (silero-vad
  last speech end 29.8 s; RMS in the last 2 s about 0.002 against 0.02-0.07 in speech). The aligner's last word
  (`cierto`) ends at 29.68 s, Whisper's at 29.48 s; Whisper then wrote `Gracias.` at 30.0-31.0 s, which is text
  invented over silence.
- Parakeet v3 (`mlx-community/parakeet-tdt-0.6b-v3`, `parakeet-mlx` 0.5.3) **does** expose per-token `confidence`
  (the prior research said no word probabilities; that is out of date for 0.5.3). These are token-level, not word-level.
- Speed budget: the 20 min per audio hour target is **at risk**, not comfortable. Whisper at 4.8-6.6 x RT is
  9.1-12.5 min/hour and Qwen3-ASR at 8.7-13.6 x RT is 4.4-6.9 min/hour. Run one after the other (one stage's models
  at a time), the two engines alone are **13.5-19.4 min per audio hour**, before the aligner (about 0.8 min/hour at
  its measured 0.4 s per 30 s), the Whisper verifier, the picker, diarization and Claude refine. If the suite needs
  speed, swap Qwen3-ASR for Parakeet (44-56 x RT, about 1.1-1.4 min/hour), which would bring the two engines to about 10.2-13.9 min/hour.
  The target is soft (reported, never blocking).
- EG2 load is 3-4 s only because the Core ML compile cache already existed on this machine; a fresh machine pays
  about 110 s once (see CLAUDE.md).
- Peak memory is far under the 24 GB budget for every ASR stage (largest is Qwen3-ASR at 5.2 GB MLX peak).

## Picker LLMs (mlx-lm 0.32.2)

16 multiple-choice items (two hypotheses A/B, 9 English and 7 Spanish; see `PICKER_ITEMS` in `spike.py`; the first
item is the brief's pipeline / pipe line prompt), each asking for JSON `{"choice": "A"|"B"|"unclear"}`. Thinking is
disabled through the chat template. The items are easy homophone/segmentation cases, so this proves the plumbing and
a quality floor, not the final ranking. tok/s is steady state (generation speed, first item excluded); prompt tok/s is
prefill.

| Model | Load (s) | tok/s | Prompt tok/s | RSS (GB) | MLX peak (GB) | Valid JSON | Correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlx-community/gemma-4-26b-a4b-it-4bit` | 5.4-6.1 | 52.2 | 204 | 9.0-9.8 | **13.42** | 16/16 | **16/16** |
| `mlx-community/Qwen3-30B-A3B-4bit` | 5.4 | 58.6 | 180 | 9.5-9.6 | 16.18 | 16/16 | 15/16 |

- Both fit in about 20 GB, so `Qwen3-14B-4bit` was not needed.
- **PICKER_MODEL = `mlx-community/gemma-4-26b-a4b-it-4bit`**: 16/16 correct, the lowest peak (13.4 GB), speed is
  adequate. The one Qwen3 miss was the Spanish `no sé si vamos a llegar` item (`si` vs `sí`).
- Gemma 4 wraps its JSON in a markdown fence (` ```json ... ``` `) even when told "JSON only"; Qwen3 returns bare
  JSON. The picker parser must extract the first `{...}` object instead of calling `json.loads` on the raw text.
- Gemma 4 26B-A4B is text-only here and loads in `mlx-lm` 0.32.2 (the prior research said the MLX builds were unverified: now verified).
- Load times are with the weights already on disk (downloaded in this task, about 15-17 GB each).

## Whisper teacher-forced scoring (verifier)

`mlx_whisper.load_models.load_model` + `log_mel_spectrogram` + `model.embed_audio` + `model.logits`, decoder prompt
`<|sot|><|en|><|transcribe|><|notimestamps|>` then the text tokens, mean token log-prob of the text tokens.

| Text (clip `en_30s`) | Mean token log-prob |
|---|---:|
| Reference | **-0.011** |
| Corrupted (3 word pairs swapped) | -1.475 |

Reference scores higher by 1.46 nats per token. `score_text` in `spike.py` is the working reference implementation:
note the leading space before the text (`encode(" " + text)`), the prefix is `sot_sequence_including_notimestamps`,
and only the text positions are averaged. Audio is padded/trimmed to 30 s (3000 mel frames), so Task 5 must score
span audio in 30 s windows or a cropped mel with the matching frame count.

## VAD and diarization

- silero-vad 6.2.3 on `en_30s` + 10 s of appended silence: 4 speech spans, first start 0.6 s, last end 29.5 s. The
  clip is 29.5 s long, so **the speech spans exclude the silence**.
- pyannote.audio 4.0.7 imports. `HF_TOKEN` is **not set** on this machine, so community-1 was not loaded and
  diarization will be skipped here (a flag, not an abort). `huggingface-cli login` or `HF_TOKEN` plus accepting the
  model terms on the Hub are needed.
- pyannote warns that torchcodec cannot decode (see fallbacks). Pass audio as in-memory
  `{"waveform": tensor(channel, time), "sample_rate": 16000}`, which is what the pipeline does anyway.

## Environment changes and fallbacks taken

1. **transformers 4.57.6 -> 5.20.2, huggingface_hub 0.36.2 -> 2.3.0, mlx 0.30.6 -> 0.32.4** (also tokenizers 0.23.3,
   safetensors 0.8.0). `mlx-lm` 0.32 needs `transformers>=5.7`; `mlx-audio` 0.6 needs `transformers>=5.14` and
   `huggingface_hub>=1`. The older mlx-lm/mlx-audio releases that accept transformers 4.x (mlx-lm 0.29,
   mlx-audio 0.2.10) predate Qwen3-ASR and Gemma 4, so downgrading was not an option. The 47 existing tests still
   pass (47 passed) and mlx-whisper 0.4.3 and EG2 both run on the new stack (they produced the numbers above).
   **Side effect:** `pip check` now reports that other packages in this shared venv that are not part of this
   repo's requirements pin the old versions: `vibevoice 1.0.0` (transformers<5), `diffusers 0.36.0`,
   `gradio 6.5.1` and `gradio-client 2.0.3` (huggingface-hub<2). They were not touched or tested.
2. **torch stayed at 2.10.0; numpy stayed at 2.3.5.** pip pulled `torchaudio 2.11.0` for pyannote, which does not
   match torch 2.10, so it was pinned back to `torchaudio==2.10.0` and `torchcodec==0.10.0`.
3. **torchcodec cannot load** (needs FFmpeg 4-7; this machine has FFmpeg 9). Fallback: decode audio with `soundfile`
   (`datasets.Audio(decode=False)` for the clip prep; in-memory waveforms for pyannote). Nothing in the pipeline
   needs torchcodec.
4. **FLEURS** `google/fleurs` es_419 loads through `datasets` 5.x streaming, as does `openslr/librispeech_asr` and
   `facebook/multilingual_librispeech`, so no alternate dataset was needed; `datasets` is used with
   `Audio(decode=False)` because of item 3. The `datasets` streaming process may hang at interpreter exit after the
   last clip is written (observed once; kill it).
5. **Gemma 4**: an MLX 4-bit build exists and works (`gemma-4-26b-a4b-it-4bit`), so the `Qwen3-14B-4bit` fallback was
   not taken.
6. Extra clips (`en_hard_30s`, `es_hard_30s`) added because the brief's two clips do not separate the engines.
7. `mlx_lm` prompts use `enable_thinking=False` in the chat template (Qwen3 and Gemma 4 both accept it).

## What the next tasks can rely on

- `python -m benchmarks.suite.spike asr --engine <whisper-large-v3|qwen3-asr-1.7b|parakeet-v3|eg2> --clip <en|es|en_hard|es_hard>`
  reproduces a table row. `all-asr`, `picker --model <repo>`, `score`, `vad`, `align` reproduce the rest.
- Slow tests can use `~/.cache/whisper_translate/suite/spike/en_30s.wav` and `es_30s.wav` with sibling `.txt` references.
- Model loading APIs that worked: `mlx_whisper.transcribe(..., word_timestamps=True, condition_on_previous_text=False)`,
  `mlx_audio.stt.utils.load_model(repo).generate(samples, language="English"|"Spanish")`,
  `parakeet_mlx.from_pretrained(repo).transcribe(path)` (file path only), `EG2Engine(workers=2)`,
  `mlx_lm.load` + `stream_generate`, `silero_vad.get_speech_timestamps`.

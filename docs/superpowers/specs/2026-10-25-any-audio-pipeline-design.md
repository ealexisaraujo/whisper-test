# Any-audio transcription pipeline: design (2026-10-25)

Status: draft for the user's approval. Research behind it: §11 (papers and model cards, verified by
research agents on 2026-10-25). Related: `docs/agents/refine-evals.md` (refine eval work so far).

## 1. Intent and why

The user transcribes about **1 hour of audio a month** (YouTube/X talks, interviews, podcasts) and uploads
the transcripts to Drive. Today they pick one engine by hand per run. An opt-in Claude "refine" stage fixes
ASR errors, but on 4 real talks about **13 % of its edits made the text worse** (19 % on a dense technical
lecture), mostly by inventing plausible words or changing words both engines heard. Every new kind of audio
brought a new failure, and everything tested so far is one English speaker.

The goal is a pipeline that is **trustworthy on any audio the user throws at it**, without the user
choosing settings or reviewing anything: any speaker or accent, English or Spanish, any domain, clean or
noisy, one speaker or several, minutes to hours. Here "scale" means *coverage of audio types*, not throughput.

## 2. Goals, non-goals, anti-goals

**Goals**
1. One command transcribes any English or Spanish audio with no per-file settings.
2. **Accuracy over speed.** A 1-hour file may take about 15–20 minutes end to end on the M1 Pro (user's call).
3. Two local ASR engines per job, combined (user's call). Words both engines agree on are locked.
4. Speaker labels ("Speaker 1", "Speaker 2", …) on multi-speaker audio (user's call).
5. Every transcript carries **visible uncertainty**: per-word confidence in JSON, `[unclear]`-style markers
   in TXT/SRT, and a file-level trust score.
6. The correction stages **never make the transcript worse than its own inputs** on the eval suite, measured
   per audio slice.
7. An eval suite that grades itself against **human reference transcripts from public datasets**, reported
   per slice (accent, language, domain, noise, speakers). It runs before any pipeline change ships. The user
   does no labeling (user's call).

**Non-goals**
- Real-time or streaming transcription.
- Training or fine-tuning speech models.
- A hosted service for other people.
- Languages other than English and Spanish in this round. The design must not *block* adding more.
- Real speaker names. Speakers stay generic; naming from metadata plants wrong names.
- Human labeling workflows. The existing review UI stays as an optional tool, not a required step.

**Anti-goals: failure even if it runs**
- Fluent text containing words nobody said. A visibly garbled or `[unclear]` span is better than a smooth wrong one.
- Uncertainty that is hidden or uncalibrated: a trust score that says "fine" on bad audio.
- Silent degradation. If a stage can't run (Claude unavailable, a model missing), the output must say so.
- Paid APIs. The user has no budget (user's call).

## 3. Constraints

- **Hardware:** Apple M1 Pro, 32 GB RAM, about 294 GB free disk. Models load one stage at a time; peak
  resident memory should stay under about 24 GB.
- **Local-only speech stack.** Free, permissively licensed models running on MLX / Core ML / PyTorch-MPS.
  The **OpenAI backend stays** as an explicit opt-in single-engine path, but it is not part of the new pipeline.
- **Claude only through the user's Claude Code subscription**, via the existing headless CLI call with
  `--safe-mode --strict-mcp-config --disable-slash-commands --tools ""` (see CLAUDE.md; dropping those flags
  costs ~300× more). Claude does refine only (user's call). It must see much less text than today.
- **Local picker model** for choosing between engine hypotheses (user's call). It must fit in memory beside nothing else.
- **Python 3.11, repo venv `.venv`.** Production code lives in the `transcribe/` package; keep the dependency
  graph acyclic (CLAUDE.md). Tests mock models and subprocesses and never call real Claude.
- **Repo rule change (user's call, 2026-10-25):** "exactly one backend per run, no automatic fallback" is
  replaced for the new pipeline by **deterministic routing plus fusion**. The probe decides the engines up
  front; if a chosen engine fails, the job raises `TranscriptionError`; nothing silently switches. The old
  single-backend modes (`--backend mlx-whisper|openai|eg2`) keep their current behavior. Update CLAUDE.md and
  `docs/agents/*` when this lands.
- **One-time user step:** a free Hugging Face account that accepts pyannote's terms, plus a token
  (`HF_TOKEN`). Without it, diarization is skipped and flagged with a message explaining the step (§8a).
- **Licenses:** model weights Apache-2.0 / MIT / CC-BY-4.0 only. Some eval datasets are non-commercial
  (TED-LIUM, CORAAL, Speech Accent Archive): fine for internal eval, never redistributed or committed.

## 4. Decided by the user

| Decision | Choice |
|---|---|
| Engines per job | Two, combined; agreed words locked |
| Paid models | None. Local engines only |
| Refine | Keep Claude (subscription) |
| Picking between engines | Local model |
| Languages | English and Spanish |
| Speed vs accuracy | Accuracy; ~15–20 min per audio hour is fine |
| Speaker labels | Yes |
| Human labeling | None; evals grade against reference transcripts |
| Eval policies (earlier) | Verbatim (keep fillers/false starts); shared-mishearing fixes pass when context makes them certain; no synonyms; numbers normalized |
| When the suite contradicts a choice above | **Data wins per audio type.** A stage or engine that makes a slice worse is turned off automatically for that audio profile (see §6, "Per-profile configuration") |

## 5. Pipeline architecture

```
ingest ─► probe ─► diarize ─► ASR-A ┐
                         └─► ASR-B ┴─► align+confidence ─► pick (local LLM) ─► verify (audio score)
                                                        ─► refine (Claude, gated) ─► verify ─► assemble ─► outputs
```

Each stage is a separate module with a typed input/output dataclass (extending `models.py`), so stages can
be tested, cached and evaluated alone. Intermediate artifacts are cached per job (keyed by audio hash and
stage config), so re-running a later stage doesn't redo ASR.

**S0 Ingest (exists).** `InputResolver` → `MediaDownloader` → `AudioNormalizer` (16 kHz mono). Add EBU R128
loudness normalization. **No denoising**: two 2025–26 studies found it raised WER in every configuration tested.

**S1 Probe.** Builds an `AudioProfile`:
- Silero VAD speech map; speech / music / silence shares.
- Language ID on 3–5 speech windows spread across the file (not just the first 30 s), giving a per-window
  language and a code-switching flag when windows disagree.
- Duration, original sample rate (phone-band detection), and later the speaker count from S2.

The profile picks the engine pair and settings deterministically (table below). The decision and its
reasons are written into the JSON metadata.

**S2 Diarize.** pyannote community-1 gives speaker turns. Words are assigned to speakers by timestamp
overlap in S8. Skipped when the probe or diarization finds one speaker.

**S3 Dual ASR.** Both engines run on the same VAD-gated chunks (≤30 s, with overlap, cut at pauses).
Starting pairs (builder confirms with the eval suite; §9):

| Profile | Engine A | Engine B |
|---|---|---|
| English | Qwen3-ASR 1.7B | Whisper large-v3 (full, not turbo) |
| Spanish | Qwen3-ASR 1.7B | Whisper large-v3 |
| Code-switching or unsupported language | Whisper large-v3 (per-chunk language, nothing forced) | Qwen3-ASR (built-in language ID) |

- EG2 and Parakeet v3 (parakeet-mlx) are candidate substitutes for a slot if the suite shows they add
  complementary errors. EG2 is English-only.
- Hallucination guards per engine:
  - Whisper: compression ratio > 2.4, avg logprob < −1.0, no-speech probability > 0.6, and
    `condition_on_previous_text=False`.
  - All engines: a known-hallucination phrase list ("thanks for watching", …), a repetition-loop detector,
    and any text produced over VAD silence.

**S4 Align and confidence.**
- Put both outputs on one timeline with word timestamps (engine timestamps, or Qwen3-ForcedAligner-0.6B
  where an engine has none).
- Align with Levenshtein on normalized words, constrained to about ±0.3 s.
- Each slot is `agree`, `substitute`, `A-only` or `B-only`.
- Per-word confidence comes from: agreement, each engine's word probability where available, hallucination
  flags, and out-of-vocabulary/proper-noun status.
- Tiers: **solid** (agree, confident), **disputed**, **missing** (one-sided).
- Start with rule-based tiers. Fit a logistic model on suite data later if the rules calibrate poorly.

**S5 Pick (local LLM).**
- Input: disputed and missing spans, each with surrounding locked context, both engine versions and the
  term list. Output per span: `A`, `B`, or `unclear`. Structured JSON; no free text allowed.
- Model: a ~30B mixture-of-experts at 4-bit in MLX (Qwen3-30B-A3B or Gemma 4 26B-A4B, about 16 GB); the
  builder picks by suite results.
- Research basis: constraining an LLM to choose between hypotheses (ClozeGER, Ma et al. 2024) avoids the
  over-correction of free rewriting; small local models are unreliable at free rewriting.

**S6 Verify (audio score).**
- For each changed span, score the candidates against the span's audio with Whisper teacher-forced,
  length-normalized log-likelihood (`mlx_whisper` exposes `Whisper.logits(tokens, audio_features)`).
- A pick or edit is kept only if it beats the alternative by a margin; otherwise the span becomes `unclear`.
- Whisper may favor its own wording. The builder measures that bias on the suite and adjusts (margin, or a
  second scorer such as Qwen3-ASR).

**S7 Refine (Claude, gated).**
- Claude sees only:
  - `unclear` spans,
  - suspected shared errors: solid words that are low-probability for both engines, or that are
    phonetically close to a term-list entry,
  - a **term list** built from title, description and chapters (proper nouns, products, jargon).
- It returns a structured edit list, not rewritten segments.
- Code gates, applied after Claude answers (prompt-only gating is known not to work):
  1. Never change a locked solid word unless it is a term-list match within phonetic distance.
  2. Each substitution must be phonetically close to what an engine heard (g2p: CMUdict/ARPAbet for
     English, rule-based for Spanish).
  3. Insertions only inside missing or low-confidence spans.
  4. No noun→pronoun swaps; no removing fillers or false starts (verbatim policy).
  5. Segment length within ±20 %.
- Every surviving edit is re-scored in S6.
- Rejected edits are reverted and the span stays `unclear`.
- If Claude is unavailable, S7 is skipped and the output metadata says `refine: skipped (<reason>)`.

**S8 Assemble.**
- Words go into segments using speaker turns and pauses.
- Outputs:
  - **JSON:** per word: text, start, end, speaker, tier, confidence, source (A/B/pick/refine). Plus the probe
    profile, engines used, stage statuses, and a file **trust score** (calibrated estimate of WER).
  - **TXT:** speaker labels; `[unclear: A | B]` markers where unresolved. The builder picks the exact marker
    format, but it must be greppable.
  - **SRT:** speaker prefix per cue; unresolved spans marked.
- Existing TXT/SRT consumers and the Drive upload flow must keep working.

**CLI.** A new `--backend auto` runs the full pipeline. Existing backends are unchanged. Options default to
the accurate settings; there are flags for the term list, skipping refine, and the number of speakers when known.

## 6. Evaluation system

The user does no labeling, so the suite grades itself against human reference transcripts.

**Slices.** Report each slice separately, worst slice first. Target about 6–8 hours, scaled down if downloads
are impractical (the builder records what was used):
- **English:** clean read (LibriSpeech test-clean/other); talks (TED-LIUM 3); accented business speech
  (Earnings-22, by region); meetings with overlap (AMI close-talk + distant mic); accents (Common Voice en,
  by accent label).
- **Spanish:** Spain and Latin America (Common Voice es, FLEURS es); parliament (VoxPopuli es).
- **Conditions:** noise/music mixes (LibriSpeech + MUSAN at 0/5/10 dB); simulated phone (8 kHz μ-law);
  speech-free clips (music/noise/silence; metric: words per minute, target 0); Spanish–English
  code-switching (Bangor Miami, if obtainable).
- **Real-world:** the 4 existing talks (no references). Engine disagreement and refine stats are tracked,
  not graded.

**Metrics per slice.**
- Normalized WER and CER (English: Whisper `EnglishTextNormalizer`; Spanish: `BasicTextNormalizer` plus
  accent-insensitive variant reported).
- Entity and number error rate.
- Hallucination insertions per minute on speech-free clips.
- DER and cpWER where speaker references exist (AMI).
- Per-stage harm/help, computed automatically from the reference: for every pick or refine edit, did it move
  the text toward or away from the reference? This replaces human pass/fail labels.
- Unclear marking quality: share of `[unclear]` spans that really contain an error (precision), and share of
  errors that are marked (recall).
- Trust score calibration: correlation between file trust score and true WER.
- Speed (×RT on the M1 Pro) and peak memory.

**Ablations.** Each slice is also run as: single engine A, single engine B, fused (S4 with tie-break only),
+pick, +verify, +refine. Every stage must earn its place.

**Per-profile configuration (user's call: data wins per audio type).**
- The full gate's ablation results produce a **profile config file** (checked in, e.g.
  `transcribe/profiles.json`). For each audio profile (English, Spanish, code-switching/other, plus
  conditions the probe can detect: phone-band, heavy music, multi-speaker) it records which engines and
  stages are on.
- A stage or second engine is turned **off** for a profile when, on the slices that map to it, enabling it
  makes WER worse beyond the bootstrap interval or raises the harmful-edit rate. Routing must therefore
  support a **single engine** per profile. Alignment and tiers still run on one engine's own confidences,
  with no agreement signal.
- This happens when the suite runs, not per file at runtime. Runtime only reads the config. Each entry links
  to the result file that justified it, and the CLI JSON metadata records which entry was used.
- A profile with no matching slice uses the full default pipeline and is flagged as "unvalidated profile".

**Regression gate.** Run on a frozen dev/test split. Any prompt, model or threshold change ships only if,
on the test split:
- no slice's WER gets worse beyond its bootstrap 95 % interval,
- no stage's harmful-edit rate rises,
- speech-free insertions stay ≤ the baseline.

Results go to versioned JSON under `benchmarks/` with the config that produced them.

**Real-audio monitoring (no references).** On every run, record agreement rate, tier shares, number of
`unclear` spans, refine accept/reject counts and stage statuses in the JSON metadata. The trust score is
validated on the suite. A run whose trust score falls below a threshold says so in the CLI output.

The existing refine-review UI and the 1,424 pre-labeled items stay available, but they are no longer on the
critical path.

## 7. Success criteria (initial targets; the builder sets final numbers from the baseline)

- Configured pipeline (after per-profile switching) WER ≤ the best single engine on **every** slice, and
  clearly lower on the average.
- Correction stages: harmful-edit rate under 2 % of edits on the suite (today's refine: ~13 % on real talks).
- Speech-free clips: about 0 inserted words per minute.
- Trust score: Spearman correlation ≥ 0.7 with true per-file WER.
- 1 hour of English audio in ≤ 20 minutes end to end on the M1 Pro; Spanish similar.
- `[unclear]` precision ≥ 0.6 (most marked spans really are wrong).

## 8. Build order (milestones; the plan details each)

1. **Eval harness first:** dataset fetch/prepare scripts, slice manifests, metrics, a baseline for the current
   single backends. Nothing else is judged without it.
2. Dual ASR (Qwen3-ASR + Whisper large-v3) with hallucination guards, plus alignment, confidence and tiers.
   Graded by the suite.
3. Pick (local LLM) + audio verify.
4. Gated refine (Claude) with term list and code gates.
5. Probe + routing + Spanish slices.
6. Diarization + speaker-aware assembly and outputs.
7. CLI `--backend auto`, docs/CLAUDE.md update, real-audio run on the 4 talks plus one Spanish talk.

## 8a. Calls made by the designer (not the user; review these)

1. **How `[unclear]` is scored.**
   - An unresolved span still carries a **best guess**: the candidate with the higher audio score.
     WER is computed on the best-guess text, so marking a span never hides an error from the metric, and
     marking can't be used to game WER.
   - The marker shows the alternatives: `[unclear: best | other]`.
   - Unclear quality is reported as precision (marked spans that are really wrong, target ≥ 0.6) and recall
     (errors that are marked, reported, no floor in round one), plus a **density cap**: at most 3 % of words
     marked per file on the suite, so the TXT stays readable.
2. **Eval cost.**
   - The full gate (all slices, all ablations) runs once per milestone and before shipping a change.
   - A **smoke subset** (about 30 minutes of audio across all slices) runs on every prompt or threshold change.
   - ASR outputs are cached per audio and engine config, so ablations and re-runs don't redo ASR. This caching
     is a must-have.
   - Refine responses are cached per (prompt version, input), so re-scoring doesn't spend subscription quota.
     A new prompt version re-runs refine on the smoke subset only, until it passes.
   - The 20 min/hour speed is a **soft target**: reported, and a regression above it gets flagged, but it
     doesn't block a change that improves accuracy (the user ranked accuracy first).
3. **Failure policy.**
   - **Core stages abort** the job with `TranscriptionError`/`CapabilityError`: ingest, probe, and the ASR
     engines the probe chose. An engine failure never switches engines silently.
   - **Enhancement stages skip and flag**: diarization, picker, audio verify, refine. A skipped stage shows
     `stage: skipped (<reason>)` in the JSON, a CLI warning, and a lower trust score. When the picker or verify
     is skipped, disputed spans keep the higher-confidence engine's word and are marked `[unclear]`.
   - A missing `HF_TOKEN` skips diarization (single "Speaker 1") with a message explaining the one-time step.
     It does not abort.
   - **Audio that is neither English nor Spanish:** best effort with Whisper large-v3 + Qwen3-ASR and
     alignment/tiers. Picker and verify still run, since they are language-neutral. Refine and the phonetic
     gates are skipped (no g2p) and flagged.

## 9. Left to the builder

- Exact engine pair per profile, picker model, thresholds and margins, all chosen by suite results. Record
  the evidence in `docs/agents/`.
- Chunking parameters, cache layout, marker format, module boundaries inside `transcribe/`.
- Which dataset subsets and sizes, within the slice list above.
- Whether EG2 or Parakeet replaces an engine slot.
- Packaging of new dependencies (mlx-audio / parakeet-mlx / mlx-lm / pyannote). Verify each actually runs on
  the M1 Pro before building on it. Several speeds and features in §11 are unverified.

## 10. Risks and unknowns

- Unverified: M1 Pro speeds, Whisper/Gemma Spanish WER, Qwen3-ASR's transformers requirement (the venv has
  4.57.6; MLX via mlx-audio may avoid an upgrade).
- pyannote on MPS speed for 1-hour files; clustering cost.
- Memory with a 16 GB picker plus ASR models: keep stages sequential and unload between them.
- Whisper-as-verifier self-preference.
- Eval datasets may need accounts or licenses (Common Voice, Earnings-22, AMI are free; Bangor Miami needs
  TalkBank terms). Fall back to the available ones and record the gap.

## 11. Research basis (abbreviated)

- **LLM correction:**
  - Unconstrained 1-best correction hurts strong ASR (Ma et al. 2023, arXiv 2307.04172).
  - Restricting choices to hypotheses helps (Ma 2024, 2409.09554; HyPoradise 2309.15701; ClozeGER 2405.10025).
  - Prompt-only confidence gating fails (Naderi 2024, 2407.21414).
  - Phonetic acceptance (Udagawa 2024, 2407.13300).
  - Entity retrieval (Pusateri 2024, 2409.06062).
  - Multi-ASR + LLM fusion (Prakash 2025, 2506.11089).
- **Fusion/confidence:**
  - ROVER (Fiscus 1997); confusion networks (Mangu 2000).
  - Whisper decode thresholds (openai/whisper).
  - Reference-free WER: e-WER (ACL 2018), NoRefER (2306.12577); cross-model disagreement for review
    prioritisation (2604.14152).
- **Hallucination:** Careless Whisper (2402.08021); Bag of Hallucinations (2501.11378).
- **Robust ASR:** Open ASR Leaderboard (2510.06961); Qwen3-ASR (2601.21337); Parakeet v3; Silero VAD;
  pyannote community-1; denoising harms ASR (2512.17562, 2603.04710).
- **Evals:**
  - Hamel Husain / Shreya Shankar error analysis; EvalGen (2404.12272); self-preference bias (2404.13076).
  - Judge-error correction / PPI (2301.09633).
  - Error bars on evals (2411.00640).
  - TTS-only data is unsafe for prevalence estimates (2305.17445).

# Refine-stage evals (started 2026-10-24)

> **Superseded (2026-10-25).** The graded eval suite (`benchmarks/suite/`, human reference transcripts per slice)
> now measures every correction stage's help/harm automatically and replaces the human audit below as the
> critical path. Results: `auto-pipeline-results.md`. The review tool and the 1,424 pre-labeled items stay
> available for spot checks.

Goal: a trustworthy, repeatable score for the `--refine` stage (`transcribe/refine.py`), so prompt or
model changes are judged by data instead of by eye. Method: error analysis as taught by Shreya Shankar
and Hamel Husain ("How to Build Better AI Evals with Claude Code in 5 Steps", YouTube `bdMHQLvtVaQ`):

1. Look at real outputs before writing any metric.
2. Top-down criteria (a domain expert's checklist; Claude is good at these) plus bottom-up criteria
   (patterns a human notices while reading; Claude is bad at these).
3. Binary pass/fail with a free-text note, never 1–5 scores.
4. One judge per failure mode, aggregated into a counts table.
5. Validate every judge against human labels (precision and recall). Off-the-shelf auto-eval tools
   reach about 80–90 % precision and miss the errors that need product taste.

## Status

| | |
|---|---|
| **Done** | Review tool `benchmarks/refine_review.py` (build/serve all datasets, audio per segment, saves labels on change). 4 datasets, 1,424 items (1,264 refine edits, 160 untouched). Claude pre-labels in each `suggestions.json` (text only, no audio): `karpathy` by the lead labeler, the other 3 by 5 parallel subagents calibrated on `karpathy`. Guidelines v2 (the user's policy decisions, 2026-10-25) applied. |
| **Pending** | Human audit of the pre-labels: `audit` filter, 380 items (fails + non-high confidence). Then rubric → deterministic checks → judges → eval harness (roadmap below). |
| **Interpretation** | Only labels with `source: human` or `accepted` (seen by a person) count as gold. `bulk-accepted` is a provisional label. Whisper is a second opinion, not ground truth. All four talks feature the same speaker (Andrej Karpathy), so accent diversity is still missing. |

### All datasets (pre-labels, guidelines v2, not yet human-audited)

| Dataset | Source | Length | EG2 raw → refined (disagreement vs Whisper) | Refine cost |
|---|---|---|---|---|
| `karpathy` | X `kyr0stack/2114091355802657181` (interview) | 30:40 | 7.78 % → 6.38 % | $0.51 |
| `yc-software` | YouTube `LCEmiRjPEtQ` (YC keynote "Software Is Changing (Again)") | 39:31 | 7.38 % → 4.90 % | $0.71 |
| `lecture` | YouTube `EsQ_bmXKA4A` (technical lecture: transformers, GPT) | 1:08:30 | 14.30 % → 11.97 % | $1.33 |
| `no-priors` | YouTube `kwSVtQ7dziU` (No Priors podcast, 3 speakers) | 1:06:31 | 8.87 % → 7.51 % | $1.25 |

| Dataset | Edits | Harmful edits | Missed fixes (untouched) | Fails by mode | To audit |
|---|---|---|---|---|---|
| `karpathy` | 140 | 14 (10 %) | 3 / 40 | F1 5 · F2 5 · F3 3 · F4 3 · F5 1 | 51 |
| `yc-software` | 263 | 26 (10 %) | 2 / 40 | F1 10 · F2 14 · F3 2 · F4 2 | 77 |
| `lecture` | 488 | 91 (19 %) | 4 / 40 | F1 42 · F2 37 · F3 4 · F4 4 · F5 8 | 164 |
| `no-priors` | 373 | 39 (10 %) | 2 / 40 | F1 17 · F2 17 · F3 2 · F4 3 · F5 2 | 88 |
| **Total** | **1,264** | **170 (13 %)** | **11 / 160** | F1 74 · F2 73 · F3 11 · F4 12 · F5 11 | **380** |

Reading: refine helps on every talk, but about 1 in 8 of its edits is harmful, and it is worst on the dense
technical lecture (19 %). There, EG2 drops about 10 % of words after minute 25 (word counts per 5 min vs
Whisper), and refine fills the gaps with plausible but unspoken words. F1 (plausible-but-wrong fill) and
F2 (changing what both ASRs heard) make up 87 % of all fails.

### Dataset `karpathy`

Source: X post `kyr0stack/status/2114091355802657181` (Karpathy interview, 30:40).
Inputs: `Kyrox_…_225045.json` (EG2 + refine, sonnet, $0.51, 59 s) and `Kyrox_…_225729.json` (mlx-whisper).

| | Disagreement vs mlx-whisper |
|---|---|
| EG2 raw | 7.78 % (S 183 / D 165 / I 134) |
| EG2 + refine | 6.38 % (S 124 / D 157 / I 114) |

Provisional numbers from the pre-labels (**not yet human-audited**):

| Metric | Value |
|---|---|
| Edit precision (good edits / all edits) | 126 / 140 = 90 % |
| Harmful edits | 14 / 140 = 10 % |
| Missed inferable fixes (among the 40 untouched candidates) | 3 |

## Labeling guidelines (v2, 2026-10-25)

v2 adds the user's policy decisions to the v1 rules below:

1. **Verbatim.** Fillers, false starts and repeats that were spoken must stay. Removing them, or smoothing
   grammar both ASRs heard (`it just`→`it's just`, `gets`→`get`), fails (F2). Removing a word only EG2
   produced (an ASR artifact, e.g. a stray `to`) passes.
2. **Shared mishearing.** When both ASRs mishear the same word and context makes the fix certain
   (`self tension`→`self attention`, `opening`→`OpenAI`, `bash`→`batch`, `Andre`→`Andrej`), the fix
   passes. This overrides F2. Uncertain cases still fail.
3. **No synonyms.** Writing a word that was not spoken fails, even when the meaning survives:
   `I view it` for "I see it", `matmul` for "matrix multiply", a pronoun for a dropped noun (F1/F3).
4. **Numbers and names: normalize.** `Software 3.0`, `Opus 4.7`, `CS231n` and the spoken form both pass.
   Dropping or changing a spoken number (`two to-dos`→`todo`, `78.7`→`78 percent`) fails.

v1 rules:

- **Edit → pass**: every change is supported (Whisper agrees, or it is an obvious ASR or grammar fix,
  e.g. `detail→detailed`, `Andre→Andrej`) and nothing is invented. Casing- and punctuation-only edits pass.
- **Edit → fail**: any change brings in a word that probably was not said, changes the meaning,
  changes a word **both** EG2 and Whisper agree on, or fills a dropped noun with a pronoun.
  Binary: a partly good, partly wrong edit fails.
- **Untouched → fail** only when the fix could be worked out from the text alone (`and someone and` →
  "and so on"). Dropped words that cannot be recovered without audio → pass: refine must not invent.
- Fill "what was actually said" when you are sure. It becomes ground truth for true WER.

The v1 open questions (verbatim, number style, pronoun fills) are settled by v2. Still open, from the subagent
labelers:
- How much phonetic inference counts as "from text alone" for an untouched fail (`fundamental link streams`
  → "fundamentally constrained")?
- A word moved across segment boundaries (lecture `s1152`/`s1153`): labels are per segment, so the joined text
  isn't judged.
- Gap fills that turn out right only through world knowledge (`Vim`, `capex`, `Steinberger`, `Qwen`) pass,
  but they are a risky pattern. Worth a dedicated judge to see whether they are lucky or reliable.

## Failure modes seen so far (bottom-up candidates)

| Code | Failure mode | Example |
|---|---|---|
| F1 | Plausible but wrong fill | `idea of AI as a new computer` (said "LLMs"); `point paradigm` → `new paradigm` (said "3.0") |
| F2 | Changes a word both ASRs agree on | `rename→renamed`, `assigned→assigns` |
| F3 | Pronoun fill for a dropped noun | `use it differently` (said "ChatGPT"); `prompt it` (said "an LLM") |
| F4 | Missed inferable fix | `so start, I'm writing code` (software 1.0); `and someone and` (and so on) |
| F5 | Wrong fix of a technical term | `our environments` → `or environments` (said "RL environments") |

Sub-patterns the subagents found across the new talks (filed under F1–F5 for now; candidates for
their own judges):
- **F2 verbatim cleanup:** dropped fillers, false starts and repeats (about a third of lecture F2s).
- **F2 number change:** `last two to-dos` → `last todo`; `seventy eight put seven` → `seventy eight percent`.
- **F1 term blind spot:** `message passing` → `math passing` (twice); refine overusing a just-recovered
  project name (`change` → `nanochat`, said "training").
- **F2 gap hiding:** refine deletes the residue word that marked a dropped noun (`talk about and how to` →
  `talk about how to`, said "LLMs and"), so the gap becomes invisible.
- **F5 direction swaps:** `encoder` ↔ `decoder`.
- **Sentence-type change:** `if that makes sense` → `Does that make sense?`

Good-edit classes worth keeping in the prompt: proper nouns and products (Andrej, MenuGen, OpenClaw,
Nano Banana), homophones (`docks→docs`, `jacket→jagged`), tense and number, and recovering a dropped
term the context makes certain (`vibe coding`).

## Roadmap to a production-grade eval

1. **Gold set.** Audit `karpathy`, then add 2–3 more talks that differ in accent, domain, audio
   quality and name density. Target 300–500 human-seen items with at least 50 fails. Freeze a
   **held-out test split** (about 30 %). Tune prompts and judges on dev only, and report on test.
2. **Rubric.** Group the notes into failure modes (start from F1–F5). Each mode gets a one-line
   definition plus pass and fail examples taken from the labels.
3. **Deterministic checks first.** These are cheap, exact and need no LLM:
   - segment count and timestamps unchanged
   - length ratio
   - **F2**: a refined token that changes a word where EG2 and Whisper agree, using the alignment
     already in `refine_review.py`
   - number-style consistency
4. **One binary LLM judge per remaining mode** (F1, F3, F4, F5). Run them through the same safe
   headless `claude -p` flags as refine. Inputs: EG2, refined, Whisper window, context. Validate each
   judge on dev: precision and recall against human labels, target ≥ 0.9 each. Report on test.
   Re-validate whenever the judge prompt changes.
5. **Eval harness** (`benchmarks/eval_refine.py`). Run refine on **frozen** EG2 segments 3 times to
   measure run-to-run variance. Then apply the checks and judges, and report:
   - fix rate, harm rate (fail edits / edits), miss rate
   - true WER on segments that have "what was said"
   - disagreement vs Whisper
   - cost and time

   Save each run to `benchmarks/refine_evals/results/<date>-<prompt-version>.json`.
6. **Regression gate.** A refine prompt or model change ships only if harm rate does not rise and
   fix rate does not fall beyond bootstrap confidence intervals on the test split.
7. **Production monitoring.** Run the deterministic checks and judges on every refined job. Sample a
   few segments per week into the review tool, and re-label periodically to catch drift. If more than
   one person labels, measure agreement (Cohen's kappa) on an overlapping slice before trusting either.

## Commands

```bash
# Build a review set from an EG2+refine run and an mlx-whisper run of the same source
.venv/bin/python benchmarks/refine_review.py build --name <ds> --refined <eg2_refine>.json --whisper <mlx>.json
# Review all datasets (dropdown switcher; keys: space play, p/f pass/fail, a accept suggestion, j/k move, n note)
.venv/bin/python benchmarks/refine_review.py serve   # http://127.0.0.1:8765
```

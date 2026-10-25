"""Assembly stage: speaker assignment, segment building and the trust score.

Unclear marker rendering (deterministic):

* Consecutive unclear words form one *group*. Alternatives live on the first word of each
  original span, so a group is split into one marker per word that carries a non-empty
  `alternatives` list (the first word of a group always starts a marker).
* A marker renders as `[unclear: <best words> | <alternative>]`, where the alternative is the
  marker's first word's `alternatives[0]`. An empty alternative (`""`: the other engine heard
  nothing) renders as `UNCLEAR_NONE` (an em dash, "possibly nothing"). A marker whose first word
  has no alternatives at all renders `[unclear: <best words>]`.
* A segment is never split inside a marker (pause, speaker change, sentence end and the
  `max_seconds` cap are all suppressed there).

`trust_score` honesty rules: the score is `"uncalibrated"` (no estimate) when the profile entry has no validated trust fit
(`trust_spearman` missing or < 0.7) or when the features cannot vary (a single engine without word probabilities).
Every skipped stage adds 0.05 to a calibrated estimate and lowers the level one step; skipped stages are always listed
in `warnings`.

`trust_score` coefficient format (written by the calibration task into profiles.json):

    {"intercept": float, "weights": {feature_name: float, ...}}

with feature names from `features` (`agreement_rate`, `unclear_rate`, `mean_confidence`,
`hallucination_flag_rate`, `skipped_stages`). `estimated_wer = intercept + sum(weight * feature)`,
clipped to [0, 1]; weights naming an unknown feature are ignored.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional, Sequence

from .engines import ENGINES_WITHOUT_WORD_PROBS
from .resolve import FusedWord
from .types import StageStatus

UNCLEAR_NONE = "—"  # em dash: the other engine heard nothing
SENTENCE_END_CHARS = ".?!…"
CLOSERS = "\"')]}”’»"
MIN_SENTENCE_SECONDS = 3.0
LEVEL_HIGH_BELOW = 0.08
LEVEL_MEDIUM_BELOW = 0.2
MIN_TRUST_SPEARMAN = 0.7  # a profile's trust fit must reach this rank correlation with per-file WER to be believed
SKIPPED_STAGE_PENALTY = 0.05
LEVEL_UNCALIBRATED = "uncalibrated"
_LEVELS = ("low", "medium", "high")


def assign_speakers(words: List[FusedWord], turns: List[dict]) -> None:
    """Set `word.speaker` from the diarization turn with the largest overlap (nearest turn if none)."""
    if not turns:
        for word in words:
            word.speaker = "1"
        return
    for word in words:
        best_speaker: Optional[str] = None
        best_overlap = 0.0
        for turn in turns:
            overlap = min(word.end, turn["end"]) - max(word.start, turn["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]
        if best_speaker is None:
            best_gap = float("inf")
            for turn in turns:
                gap = max(turn["start"] - word.end, word.start - turn["end"], 0.0)
                if gap < best_gap:
                    best_gap = gap
                    best_speaker = turn["speaker"]
        word.speaker = best_speaker


def _starts_marker(words: Sequence[FusedWord], index: int) -> bool:
    """True when `words[index]` is unclear and begins a new marker."""
    word = words[index]
    if not word.unclear:
        return False
    if index == 0 or not words[index - 1].unclear:
        return True
    return bool(word.alternatives)


def _inside_marker(words: Sequence[FusedWord], index: int) -> bool:
    """True when a split before `words[index]` would cut a marker."""
    return index > 0 and words[index].unclear and words[index - 1].unclear and not words[index].alternatives


def _render_text(words: Sequence[FusedWord]) -> str:
    tokens: List[str] = []
    i = 0
    while i < len(words):
        if not words[i].unclear:
            tokens.append(words[i].text)
            i += 1
            continue
        j = i + 1
        while j < len(words) and words[j].unclear and not words[j].alternatives:
            j += 1
        best = " ".join(w.text for w in words[i:j])
        alts = words[i].alternatives
        if alts:
            tokens.append(f"[unclear: {best} | {alts[0] if alts[0] != '' else UNCLEAR_NONE}]")
        else:
            tokens.append(f"[unclear: {best}]")
        i = j
    return " ".join(tokens)


def _ends_sentence(text: str) -> bool:
    stripped = text.rstrip(CLOSERS)
    return bool(stripped) and stripped[-1] in SENTENCE_END_CHARS


def build_segments(words: List[FusedWord], max_seconds: float = 12.0, pause: float = 0.8) -> List[dict]:
    """Group fused words into `{"start","end","speaker","text","words"}` segments."""
    segments: List[dict] = []
    current: List[FusedWord] = []

    def flush() -> None:
        if not current:
            return
        segments.append(
            {
                "start": current[0].start,
                "end": current[-1].end,
                "speaker": current[0].speaker,
                "text": _render_text(current),
                "words": [asdict(w) for w in current],
            }
        )
        current.clear()

    for index, word in enumerate(words):
        if current and not _inside_marker(words, index):
            prev = current[-1]
            if (
                word.speaker != prev.speaker
                or word.start - prev.end >= pause
                or (prev.end - current[0].start >= MIN_SENTENCE_SECONDS and _ends_sentence(prev.text))
                or word.end - current[0].start > max_seconds
            ):
                flush()
        current.append(word)
    flush()
    return segments


def _uncalibrated_reason(entry: Optional[Dict]) -> Optional[str]:
    """Why the trust score cannot be believed for this profile entry (None when it can; `entry=None` skips the check)."""
    if entry is None:
        return None
    engines = list(entry.get("engines") or [])
    if len(engines) == 1 and engines[0] in ENGINES_WITHOUT_WORD_PROBS:
        return f"trust features are constant for the single engine {engines[0]} (no word probabilities)"
    spearman = entry.get("trust_spearman")
    if isinstance(spearman, bool) or not isinstance(spearman, (int, float)):
        return "no validated trust fit for this profile"
    if spearman < MIN_TRUST_SPEARMAN:
        return f"trust fit for this profile is weak (Spearman {spearman:.2f} < {MIN_TRUST_SPEARMAN})"
    return None


def trust_score(
    words: List[FusedWord],
    stage_statuses: List[StageStatus],
    coef: Optional[Dict],
    flag_rate: float = 0.0,
    entry: Optional[Dict] = None,
) -> dict:
    """Estimate transcript quality; returns `{"estimated_wer", "features", "level", "warnings"}`.

    `flag_rate` is the fraction of (engine, chunk) pairs where a hallucination guard removed or collapsed text.
    `entry` is the profile entry the run used: without a validated trust fit (`trust_spearman` >= 0.7) or with
    constant features the result is `level="uncalibrated"`, `estimated_wer=None` and a `reason`. Each skipped stage
    adds 0.05 to a calibrated estimate, lowers its level one step and is listed in `warnings` (also when uncalibrated).
    """
    total = len(words)
    if total:
        agreement = sum(1 for w in words if w.source == "agree") / total
        unclear = sum(1 for w in words if w.unclear) / total
        mean_conf = sum(w.confidence for w in words) / total
    else:
        agreement = unclear = mean_conf = 0.0
    skipped = [s for s in stage_statuses if s.status.startswith("skipped")]
    features = {
        "agreement_rate": agreement,
        "unclear_rate": unclear,
        "mean_confidence": mean_conf,
        "hallucination_flag_rate": float(flag_rate),
        "skipped_stages": len(skipped),
    }
    warnings = [f"stage '{s.name}' was skipped: {s.status}" for s in skipped]
    reason = _uncalibrated_reason(entry)
    if reason is not None:
        return {
            "estimated_wer": None,
            "features": features,
            "level": LEVEL_UNCALIBRATED,
            "reason": reason,
            "warnings": warnings,
        }
    if coef:
        estimate = float(coef.get("intercept", 0.0)) + sum(
            float(weight) * features[name]
            for name, weight in (coef.get("weights") or {}).items()
            if name in features and name != "skipped_stages"  # the fixed per-stage penalty below replaces it
        )
    else:
        estimate = 0.5 * (1.0 - agreement) + unclear
    estimate = min(1.0, max(0.0, estimate))
    if estimate < LEVEL_HIGH_BELOW:
        level_index = 2
    elif estimate < LEVEL_MEDIUM_BELOW:
        level_index = 1
    else:
        level_index = 0
    if skipped:
        estimate = min(1.0, estimate + SKIPPED_STAGE_PENALTY * len(skipped))
        level_index = max(0, level_index - len(skipped))
    return {"estimated_wer": estimate, "features": features, "level": _LEVELS[level_index], "warnings": warnings}

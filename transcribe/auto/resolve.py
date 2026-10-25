"""Resolution stage: turn aligned spans (+ picker choices + audio verifier) into fused words.

Per non-solid span the two candidate sides ("a", "b") are compared by scoring the FULL window
text (left context + candidate + right context, whitespace-collapsed identically for both
sides) against the audio with `verify.compare` (summed log-likelihoods). A bare "" is never
scored: an empty side is always scored with its surrounding context.

Decision table (`sa`, `sb` = window scores, `margin` applies to these summed values):

    picker X, s_X >= s_other - margin   -> X, resolved, source "pick"      (confidence 0.7)
    picker X, s_other > s_X + margin    -> other, resolved, source "verify" (0.6)
    unclear/None, |sa - sb| > margin    -> higher side, resolved, "verify"  (0.6)
    unclear/None, |sa - sb| <= margin   -> higher side as best guess, unclear, alternatives=[other] (0.3)
    scorer None / span unverifiable     -> picker choice (or default_side), unclear (0.3)

Failure handling for the scorer: a `ValueError` for one span (degenerate window, no usable
time, non-finite score) only sends that span to the "unverified" row. Any other exception
propagates if no span has been scored successfully yet (a wholesale failure such as a model
load error, so the orchestrator can mark the verify stage skipped); once at least one span
has been scored, other exceptions are also treated as a per-span failure. If every scorable
span fails with ValueError (checked after `MAX_LEADING_VALUE_ERRORS` failures with no success, or at the end if there are fewer scorable spans), the last error is re-raised so the
orchestrator can mark verify skipped instead of silently running unverified.

Dropping words is risky, so for single-engine disputed spans (a low-confidence word whose
other side is empty) the empty side only wins when it beats the word by >= 2 * margin; an
unverified (scorer missing/failed) empty choice is not honoured and the word is kept as
unclear. In genuine two-engine spans (e.g. "missing" spans) the normal table applies and an
empty winner produces no words. For an unclear span the best guess follows the table: it is
the higher-scoring side even if that side is empty, in which case no words are emitted (never invent words nobody said); when the best
guess has words, `alternatives` is `[other_text]`, i.e. `[""]` if the other side is empty.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .align import Span, word_confidence
from .picker import span_text
from .types import Word
from .verify import AudioScorer, compare

logger = logging.getLogger(__name__)

CONF_AGREED = 0.7
CONF_VERIFIED = 0.6
CONF_UNCLEAR = 0.3
EMPTY_WIN_MARGIN_FACTOR = 2.0
MAX_LEADING_VALUE_ERRORS = 5


@dataclass
class FusedWord:
    text: str
    start: float
    end: float
    tier: str  # solid | resolved | unclear
    confidence: float
    source: str  # agree | a | b | pick | verify | refine
    alternatives: List[str] = field(default_factory=list)
    unclear: bool = False
    speaker: Optional[str] = None


def _other(side: str) -> str:
    return "b" if side == "a" else "a"


def _side_words(span: Span, side: str) -> List[Word]:
    words = []
    for slot in span.slots:
        word = slot.a if side == "a" else slot.b
        if word is not None:
            words.append(word)
    return words


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _window_text(span: Span, side: str) -> str:
    return _collapse(f"{span.left_context} {span_text(span, side)} {span.right_context}")


def _timed(word: Word) -> bool:
    return math.isfinite(word.start) and math.isfinite(word.end)


def _span_times(spans: List[Span], index: int) -> Tuple[float, float]:
    """Union of a and b word times; else the neighbouring spans' words; else ValueError."""
    span = spans[index]
    words = [w for w in _side_words(span, "a") + _side_words(span, "b") if _timed(w)]
    if words:
        return min(w.start for w in words), max(w.end for w in words)
    before: Optional[float] = None
    for prev in reversed(spans[:index]):
        ws = [w for w in _side_words(prev, "a") + _side_words(prev, "b") if _timed(w)]
        if ws:
            before = max(w.end for w in ws)
            break
    after: Optional[float] = None
    for nxt in spans[index + 1:]:
        ws = [w for w in _side_words(nxt, "a") + _side_words(nxt, "b") if _timed(w)]
        if ws:
            after = min(w.start for w in ws)
            break
    if before is None and after is None:
        raise ValueError("span has no timed words and no timed neighbours")
    bounds = [t for t in (before, after) if t is not None]
    return min(bounds), max(bounds)


def _window_times(span: Span, start: float, end: float) -> Tuple[float, float]:
    """The scored text is left context + candidate + right context, so the audio window must span all of it:
    from the first left-context word's start to the last right-context word's end (same chunk). An untimed side
    falls back to the span's own time."""
    lo, hi = start, end
    if span.left_context_start is not None and math.isfinite(span.left_context_start):
        lo = min(lo, span.left_context_start)
    if span.right_context_end is not None and math.isfinite(span.right_context_end):
        hi = max(hi, span.right_context_end)
    return lo, hi


def _score_span(scorer: AudioScorer, samples, spans: List[Span], index: int, lang) -> Tuple[float, float]:
    span = spans[index]
    texts = [_window_text(span, "a"), _window_text(span, "b")]
    if not all(texts):
        raise ValueError("empty window text (nothing to score)")
    start, end = _window_times(span, *_span_times(spans, index))
    sa, sb = compare(scorer, samples, start, end, texts, lang)
    if not (math.isfinite(sa) and math.isfinite(sb)):
        raise ValueError("non-finite score")
    return sa, sb


def _decide(
    picker: Optional[str],
    sa: float,
    sb: float,
    margin: float,
    default_side: str,
    guarded_empty: Optional[str],
) -> Tuple[str, str, str, float]:
    """Return (winning side, tier, source, confidence)."""
    scores = {"a": sa, "b": sb}
    if picker in ("a", "b"):
        x, o = picker, _other(picker)
        if scores[o] > scores[x] + margin:
            win, source = o, "verify"
        else:
            win, source = x, "pick"
        unclear = False
    else:
        diff = sa - sb
        win = "a" if diff > 0 else "b" if diff < 0 else default_side
        source = "verify"
        unclear = abs(diff) <= margin
    if guarded_empty is not None and win == guarded_empty:
        nonempty = _other(guarded_empty)
        if scores[guarded_empty] - scores[nonempty] < EMPTY_WIN_MARGIN_FACTOR * margin:
            win = nonempty
            # Keeping the word is the pick if the picker chose it; otherwise the verifier
            # (or nobody, for an undecided span) kept it.
            source = "pick" if picker == nonempty else "verify"
            unclear = unclear or picker not in ("a", "b")
    if unclear:
        return win, "unclear", "verify", CONF_UNCLEAR
    return win, "resolved", source, CONF_AGREED if source == "pick" else CONF_VERIFIED


def _emit(span: Span, side: str, tier: str, source: str, conf: float, unclear: bool, alternatives: List[str]):
    out = []
    for i, w in enumerate(_side_words(span, side)):
        out.append(
            FusedWord(
                text=w.text,
                start=w.start,
                end=w.end,
                tier=tier,
                confidence=conf,
                source=source,
                alternatives=list(alternatives) if i == 0 else [],
                unclear=unclear,
            )
        )
    return out


def _solid_words(span: Span) -> List[FusedWord]:
    out = []
    for slot in span.slots:
        w = slot.a if slot.a is not None else slot.b
        if w is None:
            continue
        out.append(
            FusedWord(
                text=w.text,
                start=w.start,
                end=w.end,
                tier="solid",
                confidence=word_confidence(slot),
                source="agree",
            )
        )
    return out


def resolve(
    spans: List[Span],
    picks: Optional[Dict[int, str]],
    scorer: Optional[AudioScorer],
    samples,
    lang,
    margin: float = 0.05,
    default_side: str = "a",
) -> List[FusedWord]:
    """Fuse spans into one word list (see module docstring for the rules)."""
    picks = picks or {}
    fused: List[FusedWord] = []
    scored_ok = False
    attempts = 0
    last_error: Optional[ValueError] = None
    for index, span in enumerate(spans):
        if span.tier == "solid":
            fused.extend(_solid_words(span))
            continue

        picker = picks.get(span.id)
        if picker not in ("a", "b"):
            picker = None
        text = {"a": span_text(span, "a"), "b": span_text(span, "b")}
        empty_side = next((s for s in ("a", "b") if not text[s]), None)
        # Single-engine disputed span: dropping the word needs extra evidence.
        # Depends on align's tier semantics: `build_spans` only yields "disputed" for runs with
        # a substitute/low-confidence agree slot (both sides non-empty), and "missing" for
        # purely one-sided runs; `single_engine_spans` forces "disputed" with `b` always None.
        # So an empty side in a "disputed" span <=> single-engine. If align changes, revisit.
        guarded_empty = empty_side if span.tier == "disputed" else None
        if empty_side is not None and not text[_other(empty_side)]:
            # Nothing on either side: nothing to emit.
            continue

        scores: Optional[Tuple[float, float]] = None
        if scorer is not None:
            attempts += 1
            try:
                scores = _score_span(scorer, samples, spans, index, lang)
                scored_ok = True
            except ValueError as exc:
                logger.warning("span %s unverifiable: %s", span.id, exc)
                last_error = exc
                if not scored_ok and attempts >= MAX_LEADING_VALUE_ERRORS:
                    raise
            except Exception as exc:
                if not scored_ok:
                    raise
                logger.warning("span %s scorer failed: %s", span.id, exc)

        if scores is None:
            side = picker or default_side
            if guarded_empty is not None and side == guarded_empty:
                side = _other(side)
            fused.extend(
                _emit(span, side, "unclear", "pick" if picker else side, CONF_UNCLEAR, True, [])
            )
            continue

        win, tier, source, conf = _decide(picker, scores[0], scores[1], margin, default_side, guarded_empty)
        alternatives: List[str] = []
        if tier == "unclear":
            # Table row 4: best guess = higher side; an empty best guess emits no words.
            alternatives = [text[_other(win)]]
        fused.extend(_emit(span, win, tier, source, conf, tier == "unclear", alternatives))
    if scorer is not None and attempts and not scored_ok and last_error is not None:
        raise last_error
    return fused

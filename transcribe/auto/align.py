"""Word alignment between two engines over the same chunk, plus confidence and tiers.

Both engines transcribe the identical chunk audio, so alignment is per chunk and on text
only (no time window). Words both engines agree on (with enough confidence) become
"solid" spans that later stages leave untouched; everything else is grouped into
"disputed" / "missing" spans for the picker, verifier and refine stages.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .types import EngineOutput, Word

_SOLID, _DISPUTED, _MISSING = "solid", "disputed", "missing"
_DEFAULT_AGREE_CONF = 0.9

# Intentionally small: English/Spanish number words 0-20 plus tens. Richer number handling
# (hyphenated compounds, "twenty one" vs "21") lives in benchmarks/suite/metrics.py, which
# production code must not import.
_EN_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_ES_NUMBERS = {
    "cero": 0, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7,
    "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14,
    "quince": 15, "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19,
    "veinte": 20, "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60,
    "setenta": 70, "ochenta": 80, "noventa": 90,
}


@dataclass
class Slot:
    """One aligned position. `kind`: agree | substitute | a_only | b_only."""

    chunk: int
    kind: str
    a: Optional[Word]
    b: Optional[Word]


@dataclass
class Span:
    """A run of slots handled together. `tier`: solid | disputed | missing.

    Contexts are the best-guess text (engine A's word where both exist) of up to
    `context_words` words on each side, within the same chunk. `left_context_start` / `right_context_end` are the
    start of the first left-context word and the end of the last right-context word (None when the side has no
    timed context word): the audio window a verifier scores must cover them because the scored text includes the
    context.
    """

    id: int
    chunk: int
    slots: List[Slot] = field(default_factory=list)
    tier: str = _SOLID
    left_context: str = ""
    right_context: str = ""
    left_context_start: Optional[float] = None
    right_context_end: Optional[float] = None


def norm_token(text: str, lang: str) -> str:
    """Lowercase, strip accents and punctuation (incl. ¿¡ and quotes), map number words to digits.

    Keeps every alphanumeric character of any script (Cyrillic, CJK, Arabic, ...) so different
    non-Latin words never collapse to the same token. Punctuation-only text normalises to "".
    """
    folded = unicodedata.normalize("NFKD", text.lower())
    token = "".join(ch for ch in folded if ch.isalnum() and not unicodedata.combining(ch))
    table = _ES_NUMBERS if (lang or "").lower().startswith("es") else _EN_NUMBERS
    if token in table:
        return str(table[token])
    return token


def _is_blank(word: Word) -> bool:
    """True for punctuation-only words ("—", "...", "-") that normalise to an empty token."""
    return norm_token(word.text, "en") == ""


def _align_tokens(a: List[Word], b: List[Word], na: List[str], nb: List[str]) -> List[Slot]:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = dp[i - 1][j - 1] + (0 if na[i - 1] == nb[j - 1] else 1)
            dp[i][j] = min(sub, dp[i - 1][j] + 1, dp[i][j - 1] + 1)

    # Backtrace from the end, trying the diagonal first so substitution wins ties.
    rev: List[Slot] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            same = na[i - 1] == nb[j - 1]
            if dp[i][j] == dp[i - 1][j - 1] + (0 if same else 1):
                rev.append(Slot(-1, "agree" if same else "substitute", a[i - 1], b[j - 1]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            rev.append(Slot(-1, "a_only", a[i - 1], None))
            i -= 1
        else:
            rev.append(Slot(-1, "b_only", None, b[j - 1]))
            j -= 1
    rev.reverse()
    return rev


def _split_blank(words: List[Word]):
    """Return (non-blank words, [(gap, blank word)]) where gap = non-blank words before it."""
    kept: List[Word] = []
    blanks = []
    for w in words:
        if _is_blank(w):
            blanks.append((len(kept), w))
        else:
            kept.append(w)
    return kept, blanks


def align_chunk(a: List[Word], b: List[Word], lang: str) -> List[Slot]:
    """Levenshtein alignment over normalised tokens; ties prefer substitution over insert+delete.

    Punctuation-only words (normalising to "") stay out of the DP and are re-attached in
    place as neutral `agree` slots that `build_spans` always treats as solid, so the text is
    preserved without creating missing/disputed spans. Blank words of A are kept (paired
    positionally with B's blanks in the same gap); B-only blanks are dropped, since the
    best-guess text comes from A.
    """
    ka, blanks_a = _split_blank(a)
    kb, blanks_b = _split_blank(b)
    core = _align_tokens(
        ka, kb, [norm_token(w.text, lang) for w in ka], [norm_token(w.text, lang) for w in kb]
    )
    if not blanks_a and not blanks_b:
        return core

    out: List[Slot] = []
    pa = pb = 0  # next unflushed blank in each list

    def flush(limit_a: Optional[int], limit_b: Optional[int]) -> None:
        nonlocal pa, pb
        got_a: List[Word] = []
        got_b: List[Word] = []
        while pa < len(blanks_a) and limit_a is not None and blanks_a[pa][0] <= limit_a:
            got_a.append(blanks_a[pa][1])
            pa += 1
        while pb < len(blanks_b) and limit_b is not None and blanks_b[pb][0] <= limit_b:
            got_b.append(blanks_b[pb][1])
            pb += 1
        for k, wa in enumerate(got_a):
            out.append(Slot(-1, "agree", wa, got_b[k] if k < len(got_b) else None))

    ca = cb = 0  # ordinals of the next non-blank token in a and b
    for slot in core:
        flush(ca if slot.a is not None else None, cb if slot.b is not None else None)
        out.append(slot)
        ca += slot.a is not None
        cb += slot.b is not None
    flush(len(ka), len(kb))
    return out


def _probs(slot: Slot) -> List[float]:
    return [w.prob for w in (slot.a, slot.b) if w is not None and w.prob is not None]


def word_confidence(slot: Slot) -> float:
    """agree: mean of available probs (0.9 if none); substitute: 0.3 * max prob; one-sided: 0.2."""
    probs = _probs(slot)
    if slot.kind == "agree":
        return sum(probs) / len(probs) if probs else _DEFAULT_AGREE_CONF
    if slot.kind == "substitute":
        return 0.3 * (max(probs) if probs else _DEFAULT_AGREE_CONF)
    return 0.2


def _best_text(slot: Slot) -> str:
    word = slot.a if slot.a is not None else slot.b
    return word.text if word is not None else ""


def _timed_word(word: Optional[Word]) -> bool:
    return word is not None and math.isfinite(word.start) and math.isfinite(word.end)


def _first_start(words: List[Optional[Word]]) -> Optional[float]:
    return next((w.start for w in words if _timed_word(w)), None)


def _last_end(words: List[Optional[Word]]) -> Optional[float]:
    return next((w.end for w in reversed(words) if _timed_word(w)), None)


def _solid_agree(slot: Slot, solid_min_prob: float) -> bool:
    if slot.kind != "agree":
        return False
    word = slot.a if slot.a is not None else slot.b
    if word is not None and _is_blank(word):
        return True  # neutral punctuation slot: never disputed
    return all(p >= solid_min_prob for p in _probs(slot))


def _group(
    slots_by_chunk: Dict[int, List[Slot]],
    is_solid,
    force_disputed: bool,
    context_words: int,
) -> List[Span]:
    spans: List[Span] = []
    for chunk in sorted(slots_by_chunk):
        slots = slots_by_chunk[chunk]
        texts = [_best_text(s) for s in slots]
        best_words = [s.a if s.a is not None else s.b for s in slots]
        runs: List[List[Slot]] = []
        bounds: List[tuple] = []
        solid_flags: List[bool] = []
        pos = 0
        while pos < len(slots):
            if is_solid(slots[pos]):
                runs.append([slots[pos]])
                bounds.append((pos, pos + 1))
                solid_flags.append(True)
                pos += 1
                continue
            end = pos
            while end < len(slots) and not is_solid(slots[end]):
                end += 1
            runs.append(slots[pos:end])
            bounds.append((pos, end))
            solid_flags.append(False)
            pos = end
        for run, (lo, hi), solid in zip(runs, bounds, solid_flags):
            if solid:
                tier = _SOLID
            elif force_disputed or any(s.kind in ("substitute", "agree") for s in run):
                tier = _DISPUTED
            else:
                tier = _MISSING
            spans.append(
                Span(
                    id=len(spans),
                    chunk=chunk,
                    slots=run,
                    tier=tier,
                    left_context=" ".join(texts[max(0, lo - context_words):lo]),
                    right_context=" ".join(texts[hi:hi + context_words]),
                    left_context_start=_first_start(best_words[max(0, lo - context_words):lo]),
                    right_context_end=_last_end(best_words[hi:hi + context_words]),
                )
            )
    return spans


def build_spans(
    slots_by_chunk: Dict[int, List[Slot]],
    solid_min_prob: float = 0.5,
    context_words: int = 8,
) -> List[Span]:
    """Group slots into spans, ordered by chunk then position; `Span.id` is sequential file-wide.

    An agree slot is solid only if every available prob >= `solid_min_prob`; otherwise it is
    disputed (a refine candidate). Consecutive non-solid slots merge into one span: disputed
    if it holds any substitution or low-confidence agreement, missing if purely one-sided.
    """
    for chunk, slots in slots_by_chunk.items():
        for s in slots:
            s.chunk = chunk
    return _group(
        slots_by_chunk,
        lambda s: _solid_agree(s, solid_min_prob),
        force_disputed=False,
        context_words=context_words,
    )


def single_engine_spans(
    out: EngineOutput, lang: str, solid_min_prob: float = 0.5, context_words: int = 8
) -> List[Span]:
    """Spans for profiles running one engine.

    Every word becomes a slot with only side `a` set. Words with prob >= threshold (or no prob)
    are solid (kind "agree", `b` None); words below it are disputed (kind "a_only") with the
    only candidate. Punctuation-only words are always solid.

    An `a_only` slot inside a forced-disputed span therefore means "low-confidence
    single-engine word", not "engine B missed it": the picker (Task 8) must offer an empty B
    option. `lang` is accepted for symmetry with `align_chunk` and is unused.
    """
    slots_by_chunk: Dict[int, List[Slot]] = {}
    low = set()
    for chunk, words in out.chunks.items():
        slots = []
        for w in words:
            is_low = w.prob is not None and w.prob < solid_min_prob and not _is_blank(w)
            slot = Slot(chunk, "a_only" if is_low else "agree", w, None)
            if is_low:
                low.add(id(slot))
            slots.append(slot)
        slots_by_chunk[chunk] = slots
    return _group(
        slots_by_chunk,
        lambda s: id(s) not in low,
        force_disputed=True,
        context_words=context_words,
    )

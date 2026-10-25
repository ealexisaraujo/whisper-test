"""Hallucination guards for ASR output: known boilerplate, repetition loops, silence, trailing ghosts.

Whisper-family models invent text over silence ("Thanks for watching", a lone "And." after the last
real word, endless "we go we go ..."). The guards drop or collapse such text and record why in
`EngineOutput.flags[chunk]` so later stages can see what was removed. Input is never mutated.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .types import Chunk, EngineOutput, Word

HALLUCINATION_PHRASES: Tuple[str, ...] = (
    "thanks for watching",
    "thank you for watching",
    "thank you for listening",
    "thanks for listening",
    "please subscribe",
    "subscribe",
    "like and subscribe",
    "subtitles by",
    "subtitles made by",
    "captions by",
    "gracias por ver",
    "gracias por ver el video",
    "gracias por escuchar",
    "suscríbete",
    "suscribete",
    "subtítulos por",
    "subtitulos por",
    "amara.org",
    "subtítulos realizados por la comunidad de amara.org",
    "subtitulos realizados por la comunidad de amara.org",
    "subtitulado por la comunidad de amara.org",
    "subtítulos en español de amara.org",
    "subtitles by the amara.org community",
    "captions by the amara.org community",
    "transcribed by",
    "transcripción por",
    "translated by",
    "traducido por",
    "gracias por ver este video",
    "gracias por vernos",
    "thank you for watching this video",
    "thanks for watching this video",
    "thanks for watching the video",
)

SILENCE_DBFS = -50.0
REPEAT_NGRAM = 4
REPEAT_MIN_OCCURRENCES = 3
MAX_REPEAT_PERIOD = 32
MAX_SINGLE_WORD_REPEATS = 10  # same word this many times in a row is a loop, fewer is a stutter
PHRASE_TOLERANCE_MIN_TOKENS = 3  # only phrases this long tolerate one extra courtesy word
TRAILING_PHRASE_MIN_TOKENS = 3  # only phrases this long are dropped as a trailing run (short ones: whole chunk only)
COURTESY_WORDS = frozenset({"everyone", "guys", "all", "folks", "friends", "todos", "amigos"})
LOW_PROB_TAIL = 0.15
VAD_TAIL_TOLERANCE = 0.25  # seconds a trailing word may stray from VAD speech before it is dropped

_EDGE_PUNCT = ".,!?¡¿;:\"'…()[]"


def _tokens(words: Sequence[Word]) -> List[str]:
    return [w.text.strip().lower().strip(_EDGE_PUNCT) for w in words]


def _phrase_tokens(phrase: str) -> List[str]:
    return [t.strip(_EDGE_PUNCT) for t in phrase.lower().split()]


_PHRASES = [(p, _phrase_tokens(p)) for p in HALLUCINATION_PHRASES]


def _is_known_phrase_chunk(words: Sequence[Word]) -> bool:
    """True when the chunk is exactly a known phrase.

    Real speech such as "I subscribe" or "did you subscribe" must survive, so the normalized
    tokens must equal the phrase tokens. Only a phrase of 3+ tokens tolerates one extra
    courtesy word ("thanks for watching everyone").
    """
    tokens = [t for t in _tokens(words) if t]
    if not tokens:
        return False
    for _, phrase in _PHRASES:
        if tokens == phrase:
            return True
        if len(phrase) >= PHRASE_TOLERANCE_MIN_TOKENS and len(tokens) == len(phrase) + 1 and (
            (tokens[:-1] == phrase and tokens[-1] in COURTESY_WORDS)
            or (tokens[1:] == phrase and tokens[0] in COURTESY_WORDS)
        ):
            return True
    return False


def _drop_trailing_phrases(words: List[Word]) -> Tuple[List[Word], bool]:
    """Drop known phrases of 3+ tokens that end the chunk ("... Subtítulos realizados por la comunidad de Amara.org").

    Short phrases ("subscribe", "please subscribe") are only dropped when they are the whole chunk (see
    `_is_known_phrase_chunk`): "I subscribe" inside or at the end of real speech stays verbatim. Repeated trailing
    phrases are dropped one after another; a phrase that is the whole remaining text empties the chunk.
    """
    kept = list(words)
    dropped = False
    while kept:
        tokens = [(i, t) for i, t in enumerate(_tokens(kept)) if t]
        cut: Optional[int] = None
        for _, phrase in sorted(_PHRASES, key=lambda p: -len(p[1])):
            n = len(phrase)
            if n >= TRAILING_PHRASE_MIN_TOKENS and len(tokens) >= n and [t for _, t in tokens[-n:]] == phrase:
                cut = tokens[-n][0]
                break
        if cut is None:
            break
        kept = kept[:cut]
        dropped = True
    return kept, dropped


def _collapse_repetition(words: List[Word]) -> Tuple[List[Word], bool]:
    """Collapse repetition loops, keeping the first occurrence.

    - Period 1 (the same single word): only runs of >= 10 consecutive repeats collapse, so real
      stutters ("no no no") stay verbatim.
    - Period p >= 2: a run where a 4-gram occurs >= 3 times back to back (span >= 4 + 2p tokens)
      collapses to its first p tokens. Units made of one repeated word belong to period 1.
    """
    tokens = _tokens(words)
    n = len(tokens)
    kept: List[Word] = []
    collapsed = False
    i = 0
    while i < n:
        span = 0
        period = 0
        j = i
        while j + 1 < n and tokens[j] == tokens[j + 1]:
            j += 1
        if j - i + 1 >= MAX_SINGLE_WORD_REPEATS:
            period, span = 1, j - i + 1
        else:
            for p in range(2, MAX_REPEAT_PERIOD + 1):
                if i + REPEAT_NGRAM + (REPEAT_MIN_OCCURRENCES - 1) * p > n:
                    break
                if len(set(tokens[i:i + p])) == 1:
                    continue
                j = i
                while j + p < n and tokens[j] == tokens[j + p]:
                    j += 1
                if j - i + p >= REPEAT_NGRAM + (REPEAT_MIN_OCCURRENCES - 1) * p:
                    period, span = p, j - i + p
                    break
        if period:
            kept.extend(words[i:i + period])
            collapsed = True
            i += span
        else:
            kept.append(words[i])
            i += 1
    return kept, collapsed


def _rms_dbfs(rms: float) -> float:
    return 20.0 * math.log10(max(float(rms), 1e-12))


def _outside_speech(word: Word, spans: Sequence[Tuple[float, float]]) -> bool:
    return all(
        word.end < s - VAD_TAIL_TOLERANCE or word.start > e + VAD_TAIL_TOLERANCE for s, e in spans
    )


def _drop_trailing_ghosts(
    words: List[Word], chunk_spans: Optional[List[Tuple[float, float]]]
) -> Tuple[List[Word], bool]:
    """Drop trailing words outside the chunk's VAD speech, or with prob < 0.15 at the chunk's end."""
    kept = list(words)
    while kept:
        last = kept[-1]
        ghost = (chunk_spans is not None and _outside_speech(last, chunk_spans)) or (
            last.prob is not None and last.prob < LOW_PROB_TAIL
        )
        if not ghost:
            break
        kept.pop()
    return kept, len(kept) != len(words)


def apply_guards(
    out: EngineOutput,
    chunk_audio_rms: Dict[int, float],
    speech_spans: Optional[Sequence[Tuple[float, float]]] = None,
    chunks: Optional[Sequence[Chunk]] = None,
) -> EngineOutput:
    """Return a guarded copy of `out`; each applied guard appends its reason to `flags[chunk]`.

    `chunk_audio_rms` is the linear RMS of the chunk's float samples (full scale = 1.0); chunks
    below -50 dBFS lose all words. `speech_spans` (file-time VAD spans) enables the trailing-word
    rule against the speech that overlaps the chunk (`None` turns the rule off; spans given but none
    overlapping the chunk drop every word, flag `outside-speech`); `chunks` supplies the chunk bounds (without
    them a chunk is taken to span its own words).
    """
    bounds = {c.index: (c.start, c.end) for c in chunks or []}
    result = EngineOutput(
        engine=out.engine,
        language=out.language,
        chunks={},
        flags={k: list(v) for k, v in out.flags.items()},
    )

    def flag(index: int, reason: str) -> None:
        reasons = result.flags.setdefault(index, [])
        if reason not in reasons:
            reasons.append(reason)

    for index, words in out.chunks.items():
        current = list(words)
        if index in chunk_audio_rms and _rms_dbfs(chunk_audio_rms[index]) < SILENCE_DBFS:
            if current:
                flag(index, "low-energy")
            result.chunks[index] = []
            continue
        if current and _is_known_phrase_chunk(current):
            flag(index, "known-phrase")
            result.chunks[index] = []
            continue
        current, phrase_dropped = _drop_trailing_phrases(current)
        if phrase_dropped:
            flag(index, "known-phrase-trailing")
            if not current:
                result.chunks[index] = []
                continue
        current, collapsed = _collapse_repetition(current)
        if collapsed:
            flag(index, "repetition")
        chunk_spans = None
        if speech_spans is not None and current:
            lo, hi = bounds[index] if index in bounds else (
                min(w.start for w in words), max(w.end for w in words))
            chunk_spans = [(s, e) for s, e in speech_spans if e > lo and s < hi]
        if chunk_spans is not None and not chunk_spans:
            # VAD found no speech anywhere in this chunk: nothing here can be real words.
            flag(index, "outside-speech")
            result.chunks[index] = []
            continue
        current, trimmed = _drop_trailing_ghosts(current, chunk_spans)
        if trimmed:
            flag(index, "trailing")
        result.chunks[index] = current
    return result

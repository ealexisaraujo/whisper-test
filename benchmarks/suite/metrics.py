"""Scoring metrics for the transcription eval suite.

Pure functions, no model or network access. Text metrics normalize first
(`normalize`), then align with jiwer. Numeric helpers (spearman, bootstrap)
depend only on numpy.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections import Counter
from functools import lru_cache

import jiwer
import numpy as np

try:  # scipy is optional: without it, speaker mapping falls back to greedy
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment
except ImportError:  # pragma: no cover - depends on the environment
    _linear_sum_assignment = None

# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

_ES_UNITS = {
    "cero": 0, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8,
    "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
    "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiuno": 21,
    "veintidos": 22, "veintitres": 23, "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26,
    "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
}
_ES_TENS = {"treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70, "ochenta": 80,
            "noventa": 90}
_ES_HUNDREDS = {"cien": 100, "ciento": 100, "doscientos": 200, "trescientos": 300, "cuatrocientos": 400,
                "quinientos": 500, "seiscientos": 600, "setecientos": 700, "ochocientos": 800,
                "novecientos": 900}
_ES_MULT = {"mil": 1000, "millon": 1_000_000, "millones": 1_000_000}
_ES_NUMBER_WORDS = {**_ES_UNITS, **_ES_TENS, **_ES_HUNDREDS, **_ES_MULT}

_EN_NUMBER_WORDS = set(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand "
    "million billion".split()
)


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return unicodedata.normalize("NFC", "".join(c for c in decomposed if not unicodedata.combining(c)))


@lru_cache(maxsize=1)
def _english_normalizer():
    from whisper_normalizer.english import EnglishTextNormalizer

    return EnglishTextNormalizer()


@lru_cache(maxsize=1)
def _basic_normalizer():
    from whisper_normalizer.basic import BasicTextNormalizer

    return BasicTextNormalizer()


def _es_word_key(token: str) -> str:
    return _strip_accents(token)


def _es_class(word: str) -> str:
    if word in _ES_MULT:
        return "mult"
    if word in _ES_HUNDREDS:
        return "hundreds"
    if word in _ES_TENS:
        return "tens"
    return "unit"


def _es_split_run(run: list[str]) -> list[list[str]]:
    """Split a run of number words into valid numbers ("tres cuatro" -> [["tres"], ["cuatro"]])."""
    groups: list[list[str]] = []
    group: list[str] = []
    state = "start"  # class of the last word since the group start or last multiplier
    for word in run:
        cls = _es_class(word)
        if word == "un":
            ok = True
        elif cls == "mult":
            ok = state != "start" or not group or group == ["un"] or group[-1] in _ES_MULT
        elif state in ("start", "mult"):
            ok = not (group and group[-1] == "cero") and not (word == "cero" and group and state != "start")
        elif state == "hundreds":
            ok = cls in ("tens", "unit") and word != "cero"
        elif state == "tens":
            ok = cls == "unit" and 1 <= _ES_UNITS[word] <= 9
        else:  # unit: nothing but a multiplier may follow
            ok = False
        if not ok:
            if group:
                groups.append(group)
            group, state = [], "start"
        group.append(word)
        state = "mult" if cls == "mult" else cls if word != "un" else state
    if group:
        groups.append(group)
    return groups


def _es_group_value(group: list[str]) -> int:
    total = current = 0
    for k in group:
        if k == "un":
            continue
        if k == "mil":
            total += (current or 1) * 1000
            current = 0
        elif k in ("millon", "millones"):
            total = (total + current or 1) * 1_000_000
            current = 0
        else:
            current += _ES_NUMBER_WORDS[k]
    return total + current


def _es_numbers_to_digits(tokens: list[str]) -> list[str]:
    """Replace runs of Spanish number words with digits ("dos mil veintiuno" -> "2021").

    A lone "uno" is left alone (often a pronoun); "un"/"una" are never numbers except a leading "un"
    directly before "mil"/"millon(es)", which is absorbed into the number.
    """
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        key = _es_word_key(tokens[i])
        starts_with_un = key == "un" and i + 1 < n and _es_word_key(tokens[i + 1]) in _ES_MULT
        if key not in _ES_NUMBER_WORDS and not starts_with_un:
            out.append(tokens[i])
            i += 1
            continue
        j = i + 1 if starts_with_un else i
        run = [] if not starts_with_un else ["un"]
        while j < n:
            k = _es_word_key(tokens[j])
            if k in _ES_NUMBER_WORDS:
                run.append(k)
                j += 1
            elif (k == "y" and run and run[-1] in _ES_TENS and j + 1 < n
                  and _es_word_key(tokens[j + 1]) in _ES_UNITS):
                j += 1  # "treinta y cinco"
            else:
                break
        for group in _es_split_run(run):
            if group == ["uno"]:
                out.append(tokens[i])
            else:
                out.append(str(_es_group_value(group)))
        i = j
    return out


def normalize(text: str, lang: str, strip_accents: bool = False) -> str:
    """Normalize text for scoring.

    en: whisper EnglishTextNormalizer (lowercases, expands contractions, converts spelled numbers to digits,
    and drops fillers such as "um"/"uh", so fillers never count as WER errors).
    es (and any other language): BasicTextNormalizer + lowercase, thousands separators removed and
    spelled-out Spanish numbers converted to digits. `strip_accents=True` also drops diacritics
    (so "señor" -> "senor"), the accent-insensitive variant.
    """
    text = text or ""
    if lang == "en":
        out = _english_normalizer()(text)
    else:
        # thousands separators only ("1.000", "1,000,000"); decimals like "0,125" or "2,5" are kept
        text = re.sub(r"(?<![\d.,])([1-9]\d{0,2})(([.,])\d{3}(?:\3\d{3})*)(?!\d)",
                      lambda m: m.group(1) + re.sub(r"[.,]", "", m.group(2)), text)
        out = _basic_normalizer()(text).lower()
        if lang == "es":
            out = " ".join(_es_numbers_to_digits(out.split()))
    if strip_accents:
        out = _strip_accents(out)
    return " ".join(out.split())


# --------------------------------------------------------------------------
# Alignment helpers
# --------------------------------------------------------------------------


def _align(ref: list[str], hyp: list[str]) -> list[tuple[str, int, int, int, int]]:
    """Word alignment as (type, ref_start, ref_end, hyp_start, hyp_end); handles empty sides."""
    if not ref and not hyp:
        return []
    if not ref:
        return [("insert", 0, 0, 0, len(hyp))]
    if not hyp:
        return [("delete", 0, len(ref), 0, 0)]
    out = jiwer.process_words(" ".join(ref), " ".join(hyp))
    return [(c.type, c.ref_start_idx, c.ref_end_idx, c.hyp_start_idx, c.hyp_end_idx)
            for c in out.alignments[0]]


def wer_cer(refs: list[str], hyps: list[str], lang: str) -> dict:
    """Corpus-level WER/CER over paired lists.

    Returns {wer, cer, sub, del, ins, words} where sub/del/ins/words are word-level counts
    (`words` = reference word count) and `chars` the reference character count. References that
    normalize to nothing (no-speech items) contribute only insertions and no divide-by-zero
    occurs: with zero reference words, wer/cer are 0.0 and callers should use insertions_per_minute.
    """
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps must have the same length")
    sub = dele = ins = words = 0
    csub = cdel = cins = chars = 0
    for ref, hyp in zip(refs, hyps):
        r = normalize(ref, lang)
        h = normalize(hyp, lang)
        if not r:
            ins += len(h.split())
            cins += len(h)
            continue
        if not h:
            dele += len(r.split())
            words += len(r.split())
            cdel += len(r)
            chars += len(r)
            continue
        w = jiwer.process_words(r, h)
        c = jiwer.process_characters(r, h)
        sub += w.substitutions
        dele += w.deletions
        ins += w.insertions
        words += w.hits + w.substitutions + w.deletions
        csub += c.substitutions
        cdel += c.deletions
        cins += c.insertions
        chars += c.hits + c.substitutions + c.deletions
    return {
        "wer": (sub + dele + ins) / words if words else 0.0,
        "cer": (csub + cdel + cins) / chars if chars else 0.0,
        "sub": sub, "del": dele, "ins": ins, "words": words, "chars": chars,
    }


# --------------------------------------------------------------------------
# Entities and numbers
# --------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"[.!?¿¡:]$")
_PRONOUN_I = re.compile(r"^I(['’][a-z]+)?$")


def _is_number_token(raw: str, lang: str) -> str | None:
    """Return 'digit', 'word' or None for a raw token."""
    stripped = raw.strip(".,;:!?¿¡\"'()«»“”")
    if not stripped:
        return None
    if any(ch.isdigit() for ch in stripped):
        return "digit"
    words = _ES_NUMBER_WORDS if lang == "es" else _EN_NUMBER_WORDS
    key = _strip_accents(stripped.lower()) if lang == "es" else stripped.lower()
    if key in words and not (lang == "es" and key == "uno"):
        return "word"
    return None


def _core(token: str) -> str:
    """Drop currency/percent symbols the English normalizer attaches to numbers ("$300", "50%")."""
    return token.strip("$%£€¢") or token


def _contains_subsequence(haystack: list[str], needle: list[str]) -> bool:
    n = len(needle)
    if n == 0:
        return True
    return any(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1))


def entity_number_error(ref: str, hyp: str, lang: str) -> dict:
    """Missed entities/numbers: {"total", "missed"}.

    Entities are taken from the *raw* ref: tokens containing digits, runs of consecutive spelled number
    words (kept together so "twenty twenty one" is one entity), and capitalized tokens that are not
    sentence-initial (skipped when the ref is all upper-case, e.g. LibriSpeech, or has no lower-case
    letters at all; the pronoun "I" is ignored). Each entity is normalized and counts as found when its
    token sequence occurs contiguously in the normalized hyp (presence check, no multiplicity; currency and
    percent symbols are ignored so "300 dollars" matches "$300").
    """
    tokens = (ref or "").split()
    hyp_tokens = [_core(t) for t in normalize(hyp, lang).split()]
    use_caps = any(c.islower() for c in (ref or "")) and not (ref or "").isupper()
    entities: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if run:
            entities.append(" ".join(run))
            run.clear()

    sentence_initial = True
    for idx, raw in enumerate(tokens):
        kind = _is_number_token(raw, lang)
        if (kind is None and lang == "es" and raw.lower() == "y" and run
                and _es_word_key(run[-1].strip(".,;:!?¿¡\"'()«»“”").lower()) in _ES_TENS
                and idx + 1 < len(tokens)
                and _es_word_key(tokens[idx + 1].strip(".,;:!?¿¡\"'()«»“”").lower()) in _ES_UNITS):
            run.append(raw)  # "treinta y cinco": the connector keeps the number in one run
            sentence_initial = False
            continue
        if kind == "word":
            run.append(raw)
        else:
            flush()
            if kind == "digit":
                entities.append(raw)
            elif (use_caps and not sentence_initial and raw.strip(".,;:!?¿¡\"'()«»“”")[:1].isupper()
                  and not _PRONOUN_I.match(raw.strip(".,;:!?¿¡\"'()«»“”"))):
                entities.append(raw)
        sentence_initial = bool(_SENTENCE_END.search(raw))
    flush()

    total = missed = 0
    for entity in entities:
        norm = [_core(t) for t in normalize(entity, lang).split()]
        if not norm:
            continue
        total += 1
        if not _contains_subsequence(hyp_tokens, norm):
            missed += 1
    return {"total": total, "missed": missed}


def insertions_per_minute(hyp: str, duration: float) -> float:
    """Hypothesis words per minute of audio; the hallucination rate on no-speech or empty-ref audio."""
    if duration <= 0:
        return 0.0
    return len((hyp or "").split()) / (duration / 60.0)


# --------------------------------------------------------------------------
# Edit effects
# --------------------------------------------------------------------------


def _slots(ref: list[str], hyp: list[str]) -> list[tuple[object, int]]:
    """Per ref-anchored slot (gap0, word0, gap1, ..., word_{N-1}, gap_N): (hyp content, error count).

    Word slot: (aligned hyp word or None for a deletion, 1 if wrong else 0).
    Gap slot: (tuple of inserted hyp words, number inserted).
    """
    n = len(ref)
    slots: list[list] = []
    for i in range(n + 1):
        slots.append([(), 0])  # gap i
        if i < n:
            slots.append([None, 0])  # word i
    for typ, r0, r1, h0, h1 in _align(ref, hyp):
        if typ == "equal":
            for k in range(r1 - r0):
                slots[2 * (r0 + k) + 1] = [hyp[h0 + k], 0]
        elif typ == "substitute":
            for k in range(r1 - r0):
                slots[2 * (r0 + k) + 1] = [hyp[h0 + k] if h0 + k < h1 else None, 1]
        elif typ == "delete":
            for k in range(r1 - r0):
                slots[2 * (r0 + k) + 1] = [None, 1]
        elif typ == "insert":
            slots[2 * r0] = [tuple(hyp[h0:h1]), h1 - h0]
    return [(c, e) for c, e in slots]


def edit_effects(ref: str, before: str, after: str, lang: str) -> dict:
    """Classify each changed region between `before` and `after` against the reference.

    Both hypotheses are aligned to the ref. Alignment slots (ref words and the gaps between them) whose
    hyp content differs between before/after are grouped into contiguous regions; a region counts as
    "help" if its error count went down, "harm" if up, "neutral" if unchanged (e.g. one wrong word
    swapped for another wrong word).
    """
    r = normalize(ref, lang).split()
    b = _slots(r, normalize(before, lang).split())
    a = _slots(r, normalize(after, lang).split())
    result = {"help": 0, "harm": 0, "neutral": 0}
    in_region = False
    eb = ea = 0
    for idx, ((cb, e_b), (ca, e_a)) in enumerate(zip(b, a, strict=True)):
        if cb != ca:
            in_region = True
            eb += e_b
            ea += e_a
            continue
        if in_region and idx % 2 == 0 and cb == ():
            continue  # unchanged empty gap between changed words: same region
        if in_region:
            result["help" if ea < eb else "harm" if ea > eb else "neutral"] += 1
            in_region = False
            eb = ea = 0
    if in_region:
        result["help" if ea < eb else "harm" if ea > eb else "neutral"] += 1
    return result


# --------------------------------------------------------------------------
# [unclear] marker quality
# --------------------------------------------------------------------------


def unclear_quality(ref: str, words: list[dict], lang: str) -> dict:
    """Quality of `[unclear]` flags: {"precision", "recall", "density"}.

    `words` is the best-guess transcript as {"text", "unclear": bool}. A hyp word is an error if the
    ref alignment marks it substituted or inserted (deleted ref words have no hyp word to flag).
    precision = flagged errors / flagged; recall = flagged errors / errors; density = flagged / words.
    The integer counts `flagged`, `flagged_errors`, `errors` and `words` (scored hyp tokens) are returned too.
    Undefined ratios (nothing flagged, no errors) are reported as 1.0 (vacuously perfect); density of
    an empty transcript is 0.0.

    The joined transcript is normalized once (so "twenty five" == "25" and contractions behave as in the
    other metrics) and each normalized token is mapped back to its source word(s): the per-word
    normalization is aligned to the joined one; where several words merge into one token (or a token has
    no 1:1 partner), the token is flagged if any of those words is flagged. Words the normalizer removes
    entirely (fillers such as "um" in English) are not scored, so their flags are ignored, consistently
    in precision, recall and density.
    """
    per_word: list[tuple[str, int]] = []  # (token, source word index)
    for wi, w in enumerate(words):
        for tok in normalize(w.get("text", ""), lang).split():
            per_word.append((tok, wi))
    hyp_tokens = normalize(" ".join(w.get("text", "") for w in words), lang).split()
    word_flag = [bool(w.get("unclear")) for w in words]
    flags = [False] * len(hyp_tokens)
    matcher = difflib.SequenceMatcher(None, [t for t, _ in per_word], hyp_tokens, autojunk=False)
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        if op == "equal":
            for k in range(b1 - b0):
                flags[b0 + k] = word_flag[per_word[a0 + k][1]]
        elif op in ("replace", "insert") and b1 > b0:
            if a1 > a0:
                src = {per_word[k][1] for k in range(a0, a1)}
            else:  # token with no per-word partner: inherit the preceding word
                src = {per_word[a0 - 1][1]} if a0 > 0 else set()
            flag = any(word_flag[wi] for wi in src)
            for k in range(b0, b1):
                flags[k] = flag
    r = normalize(ref, lang).split()
    error = [False] * len(hyp_tokens)
    for typ, _r0, _r1, h0, h1 in _align(r, hyp_tokens):
        if typ in ("substitute", "insert"):
            for k in range(h0, h1):
                error[k] = True
    flagged = sum(flags)
    errors = sum(error)
    both = sum(1 for f, e in zip(flags, error) if f and e)
    return {
        "precision": both / flagged if flagged else 1.0,
        "recall": both / errors if errors else 1.0,
        "density": flagged / len(hyp_tokens) if hyp_tokens else 0.0,
        # raw counts so callers can pool across items: flagged, flagged errors, errors, scored hyp tokens
        "flagged": flagged, "flagged_errors": both, "errors": errors, "words": len(hyp_tokens),
    }


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation (average ranks for ties). 0.0 when undefined (n < 2 or constant input)."""
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) < 2:
        return 0.0
    rx = _average_ranks(np.asarray(xs, dtype=float))
    ry = _average_ranks(np.asarray(ys, dtype=float))
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom else 0.0


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap CI of the mean (deterministic for a given seed)."""
    if len(values) == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------------
# Diarization error rate
# --------------------------------------------------------------------------

_FRAME = 0.01


def _merge_by_speaker(turns: list[dict]) -> dict[str, list[tuple[float, float]]]:
    per: dict[str, list[tuple[float, float]]] = {}
    for t in sorted(turns, key=lambda t: (t["start"], t["end"])):
        spans = per.setdefault(str(t["speaker"]), [])
        if spans and t["start"] <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], t["end"]))
        else:
            spans.append((t["start"], t["end"]))
    return per


def _activity(per: dict[str, list[tuple[float, float]]], frames: int) -> np.ndarray:
    mat = np.zeros((len(per), frames), dtype=bool)
    for row, spans in enumerate(per.values()):
        for start, end in spans:
            mat[row, int(round(start / _FRAME)):int(round(end / _FRAME))] = True
    return mat


def _assign(overlap: np.ndarray) -> int:
    """Maximum total overlap over a one-to-one speaker mapping (Hungarian, else greedy)."""
    if overlap.size == 0:
        return 0
    if _linear_sum_assignment is not None:
        rows, cols = _linear_sum_assignment(overlap, maximize=True)
        return int(overlap[rows, cols].sum())
    work = overlap.astype(np.int64).copy()
    total = 0
    for _ in range(min(work.shape)):
        r, c = np.unravel_index(np.argmax(work), work.shape)
        if work[r, c] <= 0:
            break
        total += int(work[r, c])
        work[r, :] = -1
        work[:, c] = -1
    return total


def der(ref_turns: list[dict], hyp_turns: list[dict], collar: float = 0.25) -> float:
    """Frame-based (10 ms) diarization error rate with optimal one-to-one speaker mapping.

    Turns are {"start", "end", "speaker"}; consecutive or overlapping turns of one speaker are merged first.
    Overlap: a frame with k active reference speakers counts k times in the denominator (standard DER);
    per frame, missed = max(0, n_ref - n_hyp), false alarm = max(0, n_hyp - n_ref), confusion =
    min(n_ref, n_hyp) - correctly mapped speakers. Frames within `collar` seconds of any (merged)
    reference turn boundary are not scored. The speaker mapping maximizes overlap (scipy Hungarian when
    available, otherwise greedy). With no reference speech the result is 0.0 if the hyp is also silent,
    else 1.0.
    """
    ref = _merge_by_speaker(ref_turns)
    hyp = _merge_by_speaker(hyp_turns)
    ends = [e for spans in (*ref.values(), *hyp.values()) for _, e in spans]
    frames = int(round(max(ends) / _FRAME)) + 1 if ends else 0
    if frames == 0:
        return 0.0
    ref_act = _activity(ref, frames)
    hyp_act = _activity(hyp, frames)

    scored = np.ones(frames, dtype=bool)
    if collar > 0:
        half = int(round(collar / _FRAME))
        for spans in ref.values():
            for start, end in spans:
                for boundary in (start, end):
                    centre = int(round(boundary / _FRAME))
                    scored[max(0, centre - half):min(frames, centre + half)] = False
    ref_act = ref_act[:, scored]
    hyp_act = hyp_act[:, scored]

    n_ref = ref_act.sum(axis=0)
    n_hyp = hyp_act.sum(axis=0)
    total_ref = int(n_ref.sum())
    miss = int(np.maximum(0, n_ref - n_hyp).sum())
    false_alarm = int(np.maximum(0, n_hyp - n_ref).sum())
    if total_ref == 0:
        return 0.0 if false_alarm == 0 else 1.0
    overlap = ref_act.astype(np.int64) @ hyp_act.astype(np.int64).T if ref_act.size and hyp_act.size \
        else np.zeros((ref_act.shape[0], hyp_act.shape[0]), dtype=np.int64)
    correct = _assign(overlap)
    confusion = int(np.minimum(n_ref, n_hyp).sum()) - correct
    return (miss + false_alarm + confusion) / total_ref

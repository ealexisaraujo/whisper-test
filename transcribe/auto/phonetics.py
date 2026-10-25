"""Phonetic comparison used as a code-enforced gate for the refine stage.

`phonemes(text, lang)` returns the phonemes of every word joined by a single space.
`phonetic_distance(a, b, lang)` is a normalized Levenshtein distance (0..1) over the phoneme
sequence of each text with word boundaries ignored, so a word-boundary split such as
"a plazar" vs "aplazar" is as close as the same word written once.

English: CMUdict through `pronouncing` (first pronunciation, stress digits stripped).
  - Lookup is lowercase with punctuation stripped; for words with apostrophes the full form is
    tried first ("don't"), then the form without apostrophes, then a possessive "'s" suffix is
    split off and a "Z" appended.
  - Hyphenated words are split into separate words.
  - Out-of-vocabulary (OOV) words use `jellyfish.metaphone`. `phonemes()` maps the metaphone code
    into the same uppercase symbol space as best it can (e.g. "0" -> TH, "X" -> SH, vowels -> AH).
    CMUdict phonemes and metaphone codes are not comparable symbol by symbol (metaphone drops
    most vowels), so `phonetic_distance` switches both sides to metaphone codes (one character
    per symbol) whenever EITHER text contains an OOV word.
  - Digits are not expanded to number words: they stay as single-character symbols ("4" is "4").
Spanish: rule based on the lowercased, accent-stripped text (n-tilde stays distinct from n):
  c/z/s -> s (seseo, except c before a/o/u -> k and "ch" -> C), b/v -> b, ll/y -> y, silent h,
  qu/k/c(a,o,u) -> k, g(e,i)/j -> x, rr -> r. Additions beyond that list: "gu" before e/i drops
  the u (guerra, guitarra), and the letter x is spelled k+s. Digits stay as characters.
Any other language: lowercased, accent-stripped letters (best effort, no phonetic knowledge).
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import List, Sequence, Tuple

_METAPHONE_TO_SYMBOLS = {"0": "TH", "X": "SH", "J": "JH", "H": "HH", "A": "AH", "E": "AH", "I": "AH", "O": "AH", "U": "AH"}
_STRESS = re.compile(r"\d")
_KEEP_EN = re.compile(r"[^a-z0-9']")


def _lang(lang: str) -> str:
    return (lang or "").strip().lower()[:2]


# ------------------------------------------------------------------ English


def _cmu(word: str) -> List[str] | None:
    import pronouncing  # lazy: only English phonetic gates need it

    phones = pronouncing.phones_for_word(word)
    if not phones:
        return None
    return _STRESS.sub("", phones[0]).split()


def _metaphone(letters: str) -> str:
    import jellyfish  # lazy: a missing package must fail loudly, not look like "no phonemes"

    try:
        return jellyfish.metaphone(letters) or ""
    except Exception:  # pragma: no cover - defensive, metaphone is total on ascii letters
        return ""


def _en_words(text: str) -> List[str]:
    text = unicodedata.normalize("NFKD", text.lower().replace("’", "'"))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    words = []
    for raw in re.split(r"[\s\-_/]+", text):
        word = _KEEP_EN.sub("", raw).strip("'")
        if word:
            words.append(word)
    return words


def _alpha_groups(word: str) -> List[str]:
    """Split a word into maximal letter/apostrophe groups and single digit characters."""
    return re.findall(r"[a-z']+|\d", word)


def _en_alpha(group: str) -> Tuple[List[str], bool]:
    """Phonemes of one letter group, and whether it was out of vocabulary."""
    found = _cmu(group)
    if found is not None:
        return found, False
    if "'" in group:
        found = _cmu(group.replace("'", ""))
        if found is not None:
            return found, False
        if group.endswith("'s") and len(group) > 2:
            base = _cmu(group[:-2])
            if base is not None:
                return base + ["Z"], False
    letters = group.replace("'", "")
    code = _metaphone(letters.upper())
    symbols: List[str] = []
    for ch in code:
        symbols.append(_METAPHONE_TO_SYMBOLS.get(ch, ch))
    return symbols, True


@lru_cache(maxsize=4096)
def _en_word(word: str) -> Tuple[Tuple[str, ...], str, bool]:
    """(phoneme symbols, metaphone-style code, is_oov) for one cleaned English word."""
    symbols: List[str] = []
    code: List[str] = []
    oov = False
    for group in _alpha_groups(word):
        if group.isdigit():
            symbols.append(group)
            code.append(group)
            continue
        phones, is_oov = _en_alpha(group)
        symbols.extend(phones)
        oov = oov or is_oov
        code.append(_metaphone(group.replace("'", "").upper()))
    return tuple(symbols), "".join(code), oov


def _en_analyse(text: str) -> Tuple[List[List[str]], List[str], bool]:
    per_word, codes, oov = [], [], False
    for word in _en_words(text):
        symbols, code, is_oov = _en_word(word)
        if symbols:
            per_word.append(list(symbols))
        if code:
            codes.append(code)
        oov = oov or is_oov
    return per_word, codes, oov


# ------------------------------------------------------------------ Spanish


def _strip_accents_keep_enye(text: str) -> str:
    text = text.lower().replace("ñ", "\0")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.replace("\0", "ñ")


def _es_word(word: str) -> str:
    out: List[str] = []
    i, n = 0, len(word)
    while i < n:
        ch = word[i]
        nxt = word[i + 1] if i + 1 < n else ""
        if ch == "h":
            i += 1
            continue
        if ch == "c":
            if nxt == "h":
                out.append("C")
                i += 2
                continue
            out.append("s" if nxt in ("e", "i") else "k")
        elif ch == "z" or ch == "s":
            out.append("s")
        elif ch == "q":
            out.append("k")
            if nxt == "u":
                i += 1
        elif ch == "k":
            out.append("k")
        elif ch == "g":
            if nxt == "u" and word[i + 2 : i + 3] in ("e", "i"):
                out.append("g")
                i += 2
                continue
            out.append("x" if nxt in ("e", "i") else "g")
        elif ch == "j":
            out.append("x")
        elif ch in ("b", "v"):
            out.append("b")
        elif ch == "l" and nxt == "l":
            out.append("y")
            i += 2
            continue
        elif ch == "y":
            out.append("y")
        elif ch == "r" and nxt == "r":
            out.append("r")
            i += 2
            continue
        elif ch == "x":
            out.extend(["k", "s"])
        elif ch.isalnum() or ch == "ñ":
            out.append(ch)
        i += 1
    return "".join(out)


def _es_words(text: str) -> List[str]:
    words = []
    for raw in re.split(r"[\s\-_/]+", _strip_accents_keep_enye(text)):
        word = _es_word(re.sub(r"[^a-zñ0-9]", "", raw))
        if word:
            words.append(word)
    return words


# ------------------------------------------------------------------ generic


def _generic_words(text: str) -> List[str]:
    text = _strip_accents_keep_enye(text).replace("ñ", "n")
    return [w for w in (re.sub(r"[^a-z0-9]", "", raw) for raw in text.split()) if w]


def phonemes(text: str, lang: str) -> str:
    """Phonemes of every word of `text`, words joined by a single space.

    English symbols are CMUdict (or metaphone-mapped) tokens separated by spaces; Spanish and other
    languages use one character per phoneme with the words separated by spaces.
    """
    code = _lang(lang)
    text = text or ""
    if code == "en":
        per_word, _, _ = _en_analyse(text)
        return " ".join(" ".join(symbols) for symbols in per_word)
    if code == "es":
        return " ".join(_es_words(text))
    return " ".join(_generic_words(text))


def _levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def _normalized(a: Sequence[str], b: Sequence[str]) -> float:
    longest = max(len(a), len(b))
    if longest == 0:
        return 0.0
    return _levenshtein(a, b) / longest


def phonetic_distance(a: str, b: str, lang: str) -> float:
    """Normalized Levenshtein distance (0 = same sound, 1 = nothing shared) over phonemes.

    Word boundaries are ignored. See the module docstring for the English OOV (metaphone) mode.
    """
    code = _lang(lang)
    a, b = a or "", b or ""
    if code == "en":
        words_a, codes_a, oov_a = _en_analyse(a)
        words_b, codes_b, oov_b = _en_analyse(b)
        if oov_a or oov_b:
            return _normalized(list("".join(codes_a)), list("".join(codes_b)))
        return _normalized([s for w in words_a for s in w], [s for w in words_b for s in w])
    if code == "es":
        return _normalized(list("".join(_es_words(a))), list("".join(_es_words(b))))
    return _normalized(list("".join(_generic_words(a))), list("".join(_generic_words(b))))

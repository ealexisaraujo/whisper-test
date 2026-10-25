"""Term list built from video metadata: names, products and acronyms that ASR tends to get wrong.

`build_terms` feeds the picker and the refine stage. It is deterministic: terms come out in order
of first appearance (extras first, then title, description, chapter titles, tags), deduplicated
case-insensitively keeping the first-seen casing, capped at 200.

What is extracted (a precision-first heuristic, not an NER model):
  - CamelCase tokens (MenuGen, iPhone), ALL-CAPS acronyms of 2+ letters with an optional plural s
    (NASA, LLMs), and tokens that mix letters and digits (GPT-4, Claude3).
  - Multi-word names: 2-3 consecutive capitalized words. Function words (Is/The/And/Of/De/La/El/Y
    ...) break a run, so a Title Case title like "Software Is Changing" yields no phrase name.
    Runs of 4+ capitalized words are dropped (almost always a Title Case phrase, not a name).
  - In the description (prose) a run must not start the sentence; if it does, its first word is
    dropped and the rest is kept when at least 2 words remain. Titles, chapter titles and tags are
    "title-like": a run may start the text but single capitalized words are never terms.
  - Single capitalized common words ("Today", "Welcome") are never terms, and a chunk-initial
    "-ing" word ("Using Claude Code") does not join a name run.
Excluded: URLs, emails, #hashtags, @handles, pure numbers, times, and unit-like tokens (5pm, 3rd, 10x).
Channel/uploader names are not used. `extra` (the `--terms` option) is included verbatim, first.
Metadata keys may be plain (`title`) or platform-prefixed as the downloader writes them
(`youtube_title`, `x_description`); missing keys and None values are fine.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterator, List, Optional

MAX_TERMS = 200
MAX_NAME_WORDS = 3

STOPWORDS = frozenset(
    """
    a an the is are was were be been am and or but nor of in on at to for with from by as into onto
    about over under how why what when where who whom which that this these those it its you your
    yours my our we they he she his her their not no vs versus do does did can will would should
    could may might if then than so just also very more most new
    el la los las un una unos unas de del y e o u en con por para que como es son se su sus al lo
    le les mi tu mas pero sin sobre entre desde hasta ante bajo tras segun
    """.split()
)

_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_OR_HANDLE = re.compile(r"\S*@\S+")
_HASHTAG = re.compile(r"(?<!\w)#\w+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CHUNK_SPLIT = re.compile(r"[,;:()\[\]{}|\"“”«»—–…]|\s-\s|\s/\s")
_WORD = re.compile(r"[^\W_]+(?:(?:['’\-]|\.(?=\d))[^\W_]+)*")
_CAMEL = re.compile(r"[a-z][A-Z]")
_ACRONYM = re.compile(r"[A-Z]{2,}s?")
_UNIT_LIKE = re.compile(
    r"\d[\d.,]*(?:st|nd|rd|th|am|pm|s|k|m|x|h|hr|min|sec|ms|gb|mb|kb|tb|hz)", re.IGNORECASE
)


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        title = value.get("title")
        if isinstance(title, str):
            yield title
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _field(metadata: Dict[str, Any], name: str) -> Iterator[str]:
    keys = [name] + sorted(k for k in metadata if isinstance(k, str) and k.endswith("_" + name))
    for key in keys:
        yield from _strings(metadata.get(key))


def _clean(text: str) -> str:
    text = _URL.sub(" ", text)
    text = _EMAIL_OR_HANDLE.sub(" ", text)
    return _HASHTAG.sub(" ", text)


def _is_cap_word(word: str) -> bool:
    parts = word.split("-")
    return all(len(p) >= 2 and p[0].isupper() and p[1:].replace("'", "").replace("’", "").islower() for p in parts)


def _strip_possessive(word: str) -> str:
    return word[:-2] if word[-2:] in ("'s", "’s") and len(word) > 2 else word


def _is_stop(word: str) -> bool:
    return word.lower().replace("’", "'") in STOPWORDS


def _single_token_term(word: str, allow_acronym: bool) -> bool:
    word = _strip_possessive(word)
    has_digit = any(ch.isdigit() for ch in word)
    has_alpha = any(ch.isalpha() for ch in word)
    if has_digit:
        return has_alpha and not _UNIT_LIKE.fullmatch(word)
    if not has_alpha:
        return False
    if allow_acronym and _ACRONYM.fullmatch(word):
        return True
    return bool(_CAMEL.search(word))


def _is_gerund(word: str) -> bool:
    """Text-initial "Using"/"Building" in a Title Case title is a verb, not part of a name."""
    return len(word) >= 5 and word.lower().endswith("ing")


def _is_shouting(words: List[str]) -> bool:
    letters = [w for w in words if sum(c.isalpha() for c in w) >= 2]
    return len(letters) >= 3 and all(w.upper() == w for w in letters)


def _extract(text: str, title_like: bool) -> Iterator[str]:
    for line in _clean(text).splitlines():
        for sentence in _SENTENCE_SPLIT.split(line):
            for index, chunk in enumerate(_CHUNK_SPLIT.split(sentence)):
                words = _WORD.findall(chunk)
                if not words:
                    continue
                shouting = _is_shouting(words)
                run: List[str] = []
                run_starts_sentence = False

                def flush() -> Iterator[str]:
                    nonlocal run, run_starts_sentence
                    names = run
                    if run_starts_sentence and not title_like:
                        names = names[1:]
                    if 2 <= len(names) <= MAX_NAME_WORDS:
                        yield " ".join(names)
                    run, run_starts_sentence = [], False

                for position, word in enumerate(words):
                    if _single_token_term(word, allow_acronym=not shouting):
                        yield _strip_possessive(word)
                    if _is_cap_word(word) and not _is_stop(word) and not (position == 0 and _is_gerund(word)):
                        if not run:
                            run_starts_sentence = index == 0 and position == 0
                        run.append(word)
                    else:
                        yield from flush()
                yield from flush()


def build_terms(metadata: Dict[str, Any], extra: Optional[List[str]] = None) -> List[str]:
    """Candidate names, products and acronyms from video metadata (see module docstring)."""
    metadata = metadata or {}
    candidates: List[str] = []
    for item in extra or []:
        if isinstance(item, str) and item.strip():
            candidates.append(item.strip())
    sources: List[tuple] = []
    for text in _field(metadata, "title"):
        sources.append((text, True))
    for text in _field(metadata, "description"):
        sources.append((text, False))
    for text in _field(metadata, "chapters"):
        sources.append((text, True))
    for text in _field(metadata, "tags"):
        sources.append((text, True))
    for text, title_like in sources:
        candidates.extend(_extract(text, title_like))

    seen = set()
    terms: List[str] = []
    for term in candidates:
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
        if len(terms) >= MAX_TERMS:
            break
    return terms

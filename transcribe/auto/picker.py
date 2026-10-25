"""Local LLM picker: choose between two ASR candidates for a disputed span, never rewrite.

The model only answers option 1 / option 2 / unclear per span; the choice is mapped back to
the engine side ("a" / "b") after a deterministic per-span shuffle that removes position
bias. Anything missing, invalid or duplicated in the model's answer becomes "unclear".
"""

from __future__ import annotations

import gc
import json
import re
import random
from typing import Callable, Dict, List, Optional, Protocol, runtime_checkable

from .align import Span, norm_token

PICKER_MODEL = "mlx-community/gemma-4-26b-a4b-it-4bit"
MAX_TERMS = 50
MAX_DISPLAY_WORDS = 40
_NOTHING = "(nothing)"
_TOKENS_PER_SPAN, _TOKENS_BASE = 30, 50

_PROMPT = (
    "Two automatic transcripts of the same {lang} audio disagree on a span. For each numbered "
    "item pick the option that is the more plausible transcription of natural speech in its "
    "context.\n"
    "Rules:\n"
    '- Answer "unclear" when neither option fits or they are equally plausible.\n'
    "- Never invent text; only choose between the two options.\n"
    "- Fillers, false starts and repetitions are normal speech. Do not prefer an option just "
    "because it is cleaner or more grammatical.\n"
    "Transcript language: {lang}\n"
    "Known terms: {terms}\n\n"
    "Items (left context [option 1 | option 2] right context):\n"
    "{items}\n\n"
    'Answer with JSON only: {{"choices": [{{"id": <int>, "choice": "1"|"2"|"unclear"}}]}}'
)


@runtime_checkable
class Picker(Protocol):
    def pick(self, spans: List[Span], lang: str, terms: List[str]) -> Dict[int, str]: ...

    def release(self) -> None: ...


def span_text(span: Span, side: str) -> str:
    """Candidate text of one side ("a" or "b"): slot words in order, skipping missing ones."""
    words = []
    for slot in span.slots:
        word = slot.a if side == "a" else slot.b
        if word is not None:
            words.append(word.text)
    return " ".join(words)


def _norm_text(text: str, lang: str) -> str:
    return " ".join(t for t in (norm_token(w, lang) for w in text.split()) if t)


def _balanced_end(text: str, start: int) -> Optional[int]:
    """Index just past the `}` closing the `{` at `start` (string/escape-aware), or None."""
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _choices_list(text: str) -> Optional[list]:
    """Try every "{" as a start; return the `choices` list of the first balanced object that
    parses as JSON and has one. Prose braces, `{id}` placeholders and stray "{" are skipped."""
    pos = text.find("{")
    while pos >= 0:
        end = _balanced_end(text, pos)
        if end is not None:
            try:
                obj = json.loads(text[pos:end])
            except Exception:
                obj = None
            if isinstance(obj, dict) and isinstance(obj.get("choices"), list):
                return obj["choices"]
        pos = text.find("{", pos + 1)
    return None


def _entries_fallback(text: str) -> list:
    """Complete `{... "id" ... "choice" ...}` entries, for output with no parseable wrapper
    (e.g. truncated by max_tokens): only the cut-off tail is lost."""
    items = []
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(obj, dict) and "id" in obj and "choice" in obj:
            items.append(obj)
    return items


def parse_choices(text: str) -> Dict[int, str]:
    """Parse the model output into {id: "1"|"2"|"unclear"}.

    Invalid choice values and duplicated ids become "unclear"; ids the model skipped are
    absent (the caller treats them as unclear). Output without a usable wrapper falls back to
    individually complete entries; nothing parseable returns {}.
    """
    text = text or ""
    items = _choices_list(text)
    if items is None:
        items = _entries_fallback(text)
    seen: Dict[int, str] = {}
    dup = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_id, raw_choice = item.get("id"), item.get("choice")
        if isinstance(raw_id, str) and raw_id.strip().isdigit():
            raw_id = int(raw_id)
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            continue
        if isinstance(raw_choice, int) and not isinstance(raw_choice, bool):
            raw_choice = str(raw_choice)
        choice = raw_choice.strip().lower() if isinstance(raw_choice, str) else None
        if raw_id in seen:
            dup.add(raw_id)
        seen[raw_id] = choice if choice in ("1", "2", "unclear") else "unclear"
    for i in dup:
        seen[i] = "unclear"
    return seen


def _clip(text: str, limit: int = MAX_DISPLAY_WORDS, keep_end: bool = False) -> str:
    """Collapse all whitespace (newlines would break the one-item-per-line format) and cap at
    `limit` words, keeping the end for left contexts and the start otherwise."""
    words = text.split()
    if len(words) > limit:
        words = words[-limit:] if keep_end else words[:limit]
    return " ".join(words)


class MlxPicker:
    """Gemma 4 26B-A4B (4-bit) via mlx-lm, loaded lazily on the first LLM call.

    `generate(prompt, max_tokens) -> str` can be injected (tests); by default it runs the
    real model with temperature 0 and the chat template applied.
    """

    def __init__(
        self,
        repo: str = PICKER_MODEL,
        batch_size: int = 15,
        seed: int = 0,
        generate: Optional[Callable[[str, int], str]] = None,
    ) -> None:
        self.repo = repo
        self.batch_size = max(1, int(batch_size))
        self.seed = seed
        self._generate = generate
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------ model plumbing
    def _load(self) -> None:
        if self._model is None:
            from mlx_lm import load

            self._model, self._tokenizer = load(self.repo)

    def _run_model(self, prompt: str, max_tokens: int) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        self._load()
        tok = self._tokenizer
        messages = [{"role": "user", "content": prompt}]
        try:
            ids = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            ids = tok.apply_chat_template(messages, add_generation_prompt=True)
        return generate(
            self._model, tok, prompt=ids, max_tokens=max_tokens, sampler=make_sampler(temp=0.0)
        )

    def release(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------ picking
    def pick(self, spans: List[Span], lang: str, terms: List[str]) -> Dict[int, str]:
        """Return {span.id: "a" | "b" | "unclear"} for exactly the input spans.

        All spans must be in one language (`lang`); batch by language upstream. Exceptions from
        model load or generation propagate: the orchestrator marks the pick stage skipped.
        """
        result: Dict[int, str] = {}
        todo: List[Span] = []
        for span in spans:
            if _norm_text(span_text(span, "a"), lang) == _norm_text(span_text(span, "b"), lang):
                result[span.id] = "a"
            else:
                todo.append(span)
        run = self._generate or self._run_model
        for start in range(0, len(todo), self.batch_size):
            batch = todo[start : start + self.batch_size]
            result.update(self._pick_batch(batch, lang, terms, run))
        return {span.id: result.get(span.id, "unclear") for span in spans}

    def _pick_batch(self, batch, lang, terms, run) -> Dict[int, str]:
        lines, flipped = [], {}
        for span in batch:
            a = _clip(span_text(span, "a")) or _NOTHING
            b = _clip(span_text(span, "b")) or _NOTHING
            flipped[span.id] = random.Random(self.seed + span.id).random() < 0.5
            first, second = (b, a) if flipped[span.id] else (a, b)
            left_ctx = _clip(span.left_context, keep_end=True)
            right_ctx = _clip(span.right_context)
            left = f"{left_ctx} " if left_ctx else ""
            right = f" {right_ctx}" if right_ctx else ""
            lines.append(f"{span.id}. {left}[{first} | {second}]{right}")
        prompt = _PROMPT.format(
            lang=lang or "unknown",
            terms=", ".join(terms[:MAX_TERMS]) if terms else "(none)",
            items="\n".join(lines),
        )
        text = run(prompt, _TOKENS_PER_SPAN * len(batch) + _TOKENS_BASE)
        parsed = parse_choices(text)
        out: Dict[int, str] = {}
        for span in batch:
            choice = parsed.get(span.id)
            if choice == "1":
                out[span.id] = "b" if flipped[span.id] else "a"
            elif choice == "2":
                out[span.id] = "a" if flipped[span.id] else "b"
            else:
                out[span.id] = "unclear"
        return out

"""Gated Claude refine: fix only what the two local engines could not settle.

After fusion and dispute resolution, `refine_targets` picks the words that are still doubtful:

    unclear         a maximal run of consecutive unclear words (alternatives from the run's first word)
    term_near       a 1-3 word n-gram phonetically close (<= 0.35) to a metadata term but not equal to it
    low_conf_agree  a word both engines agreed on (source "agree") with confidence < 0.5

Targets never overlap (priority: unclear, then term_near, then low_conf_agree). `end_idx` is
exclusive: the target covers `words[start_idx:end_idx]`.

`GatedRefiner` sends the targets to the headless Claude Code CLI and gets back proposed edits.
Prompt-only gating is not trusted: every edit passes `gate_edit` (empty, length, phonetic, filler,
pronoun, locked) and then an audio check in code. The audio check scores the FULL window text
(left context + candidate + right context, identical context for both candidates) with
`verify.compare` (summed log-likelihoods): an edit needs `s_repl >= s_orig + 0.05`, or
`s_repl >= s_orig - 0.10` for a genuine term correction (see `is_term_match`: relational, the original
must also sound like the term). A `ValueError` (degenerate
window, non-finite score) for one target rejects that edit as "audio_unverifiable". With no scorer
(verify skipped) only term-match edits that pass every gate are accepted.

This is an enhancement stage: it never raises. Failures are reported through `stats["status"]`
("ok", "skipped: <reason>", "partial: <n> batches failed") and the words come back unchanged.
"""

from __future__ import annotations

import difflib
import json
import logging
import math
import os
import re
import subprocess
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..constants import DEFAULT_REFINE_MODEL
from ..models import CapabilityError, TranscriptionError
from ..refine import resolve_claude_bin
from .phonetics import phonetic_distance
from .resolve import FusedWord
from .verify import compare

logger = logging.getLogger(__name__)

SUPPORTED_LANGS = ("en", "es")
LOW_CONF_AGREE_MAX = 0.5
CONTEXT_WORDS = 8
NGRAM_MAX = 3
TERM_NEAR_MAX_DISTANCE = 0.35
GATE_PHONETIC_MAX_DISTANCE = 0.4
LENGTH_TOLERANCE = 0.2
MIN_TERM_LETTERS = 3
AUDIO_MARGIN = 0.05
AUDIO_TERM_MARGIN = -0.10
ACCEPTED_CONFIDENCE = 0.6
BATCH_SIZE = 40
CALL_TIMEOUT_SECONDS = 300
ATTEMPTS = 2
PARTIAL_PREFIX = "partial: "

FILLERS = {
    "en": ("um", "uh", "like", "you know", "so", "right", "i mean"),
    "es": ("eh", "este", "o sea", "pues", "bueno", "¿no?"),
}
PRONOUNS = {
    "en": {"it", "they", "them", "this", "that", "he", "she"},
    "es": {"lo", "la", "eso", "esto", "él", "ella", "ellos"},
}

REFINE_GATED_SYSTEM_PROMPT = """You review a few doubtful spots in a speech transcript produced by two automatic speech recognition (ASR) engines. Each target is a short span the engines could not settle, a low-confidence word, or a phrase that sounds close to a name or term.

Rules:
- Verbatim policy: the transcript keeps fillers (um, uh, like, you know), false starts and repeats exactly as spoken. Never remove them and never clean up the speaker's grammar.
- Never substitute synonyms and never paraphrase, summarize, translate or censor. A replacement must sound like what the audio contained.
- Only fix words that are likely ASR mishearings: homophones, wrong word boundaries, garbled names and technical terms. Use the "alternatives" (what the other engine heard) and the surrounding context to decide.
- The listed terms come from the video's metadata (title, description, channel). They may be wrong or irrelevant: use one only when the speaker plausibly said it at that spot, never to insert information.
- Return an edit only when you are sure. If you are unsure, leave the target out.
- Return an empty list of edits if nothing needs fixing.
- Each edit is {"id", "replacement"}, where replacement is the full corrected text for that target (the same span, not the context). Keep the transcript's language."""

REFINE_GATED_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "replacement": {"type": "string"},
                },
                "required": ["id", "replacement"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["edits"],
    "additionalProperties": False,
}

_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)


# ------------------------------------------------------------------ text helpers


def _lang_code(lang: Optional[str]) -> str:
    return (lang or "").strip().lower()[:2]


def _norm(text: str) -> str:
    """Lowercase words without punctuation, single-spaced (used for comparisons and phonetics)."""
    return " ".join(_TOKEN_RE.findall((text or "").lower()))


def _letters(text: str) -> int:
    return sum(len(tok) for tok in _TOKEN_RE.findall(text or ""))


def _collapse(text: str) -> str:
    return " ".join((text or "").split())


@lru_cache(maxsize=65536)
def _distance(a: str, b: str, lang: str) -> float:
    return phonetic_distance(a, b, lang)


def _ngrams(text: str) -> List[str]:
    """All 1-3 word n-grams of `text` (normalized, punctuation-free)."""
    tokens = _norm(text).split()
    return [" ".join(tokens[i : i + n]) for n in range(1, NGRAM_MAX + 1) for i in range(len(tokens) - n + 1)]


def _near(grams: Sequence[str], term: str, lang: str) -> bool:
    return any(g == term or _distance(g, term, lang) <= TERM_NEAR_MAX_DISTANCE for g in grams)


def is_term_match(target: Dict[str, Any], replacement: str, terms: Sequence[str], lang: str) -> bool:
    """True only for a genuine term correction: the edit turns something that sounds like a term into it.

    Relational: some term t must be within 0.35 (or equal, case-insensitively) of a 1-3-gram of the
    REPLACEMENT that is new (not already an n-gram of the original text) AND of a 1-3-gram of the
    original target text or of one of its alternatives. A replacement that merely equals some term,
    with nothing in the original sounding like it, is a planted name and is not a match.
    """
    original_grams = _ngrams(str(target.get("text") or ""))
    source_grams = list(original_grams)
    for alt in target.get("alternatives") or []:
        source_grams.extend(_ngrams(str(alt)))
    unchanged = set(original_grams)
    repl_grams = [g for g in _ngrams(replacement) if g not in unchanged]
    if not repl_grams or not source_grams:
        return False
    for term in terms:
        t = _norm(term)
        if t and _near(repl_grams, t, lang) and _near(source_grams, t, lang):
            return True
    return False


# ------------------------------------------------------------------ targets


def _unclear_runs(words: Sequence[FusedWord]) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    i = 0
    while i < len(words):
        if words[i].unclear:
            j = i
            while j < len(words) and words[j].unclear:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _term_near_candidates(
    words: Sequence[FusedWord], terms: Sequence[str], lang: str, blocked: List[bool]
) -> List[Tuple[int, int]]:
    """Non-overlapping (start, end) n-grams near a term, best (lowest distance) first-come."""
    norm_terms = []
    for term in terms:
        t = _norm(term)
        if t and _letters(t) >= MIN_TERM_LETTERS and t not in norm_terms:
            norm_terms.append(t)
    if not norm_terms:
        return []
    exact = set(norm_terms)
    texts = [_norm(w.text) for w in words]
    n_words = len(words)

    protected = [False] * n_words  # exact occurrences of a term are correct as written
    for n in range(1, NGRAM_MAX + 1):
        for i in range(n_words - n + 1):
            if " ".join(texts[i : i + n]) in exact:
                for k in range(i, i + n):
                    protected[k] = True

    found: List[Tuple[float, int, int, int]] = []
    for n in range(1, NGRAM_MAX + 1):
        for i in range(n_words - n + 1):
            span = range(i, i + n)
            if any(blocked[k] or protected[k] for k in span):
                continue
            gram = " ".join(texts[i : i + n])
            letters = _letters(gram)
            if not gram or letters < MIN_TERM_LETTERS:
                continue
            best: Optional[float] = None
            for t in norm_terms:
                tl = _letters(t)
                if min(letters, tl) / max(letters, tl) < 0.4:  # loose length prefilter (cheap)
                    continue
                d = _distance(gram, t, lang)
                if d <= TERM_NEAR_MAX_DISTANCE and (best is None or d < best):
                    best = d
            if best is not None:
                found.append((best, -n, i, i + n))

    found.sort()
    taken = [False] * n_words
    chosen: List[Tuple[int, int]] = []
    for _, _, start, end in found:
        if any(taken[k] for k in range(start, end)):
            continue
        for k in range(start, end):
            taken[k] = True
        chosen.append((start, end))
    return chosen


def refine_targets(words: List[FusedWord], terms: List[str], lang: str) -> List[Dict[str, Any]]:
    """Doubtful spans for Claude, ordered by position; ids are 0..n-1 in that order."""
    if not words:
        return []
    code = _lang_code(lang)
    claimed = [False] * len(words)
    picked: List[Tuple[int, int, str]] = []

    for start, end in _unclear_runs(words):
        picked.append((start, end, "unclear"))
        for k in range(start, end):
            claimed[k] = True

    for start, end in _term_near_candidates(words, terms or [], code, claimed):
        picked.append((start, end, "term_near"))
        for k in range(start, end):
            claimed[k] = True

    for i, w in enumerate(words):
        if not claimed[i] and w.source == "agree" and w.confidence < LOW_CONF_AGREE_MAX:
            picked.append((i, i + 1, "low_conf_agree"))
            claimed[i] = True

    picked.sort()
    targets: List[Dict[str, Any]] = []
    for idx, (start, end, reason) in enumerate(picked):
        alternatives = list(words[start].alternatives) if reason == "unclear" else []
        targets.append(
            {
                "id": idx,
                "start_idx": start,
                "end_idx": end,
                "text": " ".join(w.text for w in words[start:end]),
                "alternatives": alternatives,
                "left": " ".join(w.text for w in words[max(0, start - CONTEXT_WORDS) : start]),
                "right": " ".join(w.text for w in words[end : end + CONTEXT_WORDS]),
                # audio extent of the context words: the scored window text includes them, so must the audio window
                "left_start": words[max(0, start - CONTEXT_WORDS)].start if start > 0 else None,
                "right_end": words[min(len(words), end + CONTEXT_WORDS) - 1].end if end < len(words) else None,
                "reason": reason,
            }
        )
    return targets


# ------------------------------------------------------------------ gates


def _filler_counts(text: str, lang: str) -> Counter:
    low = " " + (text or "").lower() + " "
    counts: Counter = Counter()
    for filler in FILLERS.get(lang, ()):
        if filler == "¿no?":
            pattern = r"(?<![\w])¿?no\?"
        else:
            pattern = r"(?<![\w'])" + re.escape(filler).replace(r"\ ", r"[\s,.]+") + r"(?![\w'])"
        found = len(re.findall(pattern, low))
        if found:
            counts[filler] = found
    return counts


def _swaps_in_pronoun(original: str, replacement: str, lang: str) -> bool:
    pronouns = PRONOUNS.get(lang, set())
    if not pronouns:
        return False
    a = _TOKEN_RE.findall(original.lower())
    b = _TOKEN_RE.findall(replacement.lower())
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "replace":
            continue
        if any(tok in pronouns for tok in b[j1:j2]) and any(tok not in pronouns for tok in a[i1:i2]):
            return True
    return False


def gate_edit(target: Dict[str, Any], replacement: str, terms: List[str], lang: str) -> Optional[str]:
    """Name of the first failing gate, or None when the edit passes every gate."""
    code = _lang_code(lang)
    original = str(target.get("text") or "")
    repl = _collapse(replacement)

    if not repl:
        return "empty"

    n_orig = len(original.split())
    n_repl = len(repl.split())
    tolerance = max(1, math.floor(LENGTH_TOLERANCE * n_orig + 1e-9))
    if abs(n_repl - n_orig) > tolerance:
        return "length"

    term_match = is_term_match(target, repl, terms, code)
    references = [original] + [a for a in (target.get("alternatives") or []) if a]
    repl_norm = _norm(repl)
    if not term_match and all(
        _distance(repl_norm, _norm(ref), code) > GATE_PHONETIC_MAX_DISTANCE for ref in references
    ):
        return "phonetic"

    removed = _filler_counts(original, code) - _filler_counts(repl, code)
    if removed:
        return "filler_removed"

    if _swaps_in_pronoun(original, repl, code):
        return "pronoun_swap"

    if target.get("reason") == "low_conf_agree" and not term_match:
        return "locked"
    return None


# ------------------------------------------------------------------ refiner


def _replacement_words(original: Sequence[FusedWord], replacement: str) -> List[FusedWord]:
    """Split `replacement` into words with times spread (by characters) over the original span."""
    pieces = replacement.split()
    start, end = original[0].start, original[-1].end
    total = sum(len(p) for p in pieces) or 1
    speaker = original[0].speaker
    out: List[FusedWord] = []
    cursor = start
    consumed = 0
    for k, piece in enumerate(pieces):
        consumed += len(piece)
        stop = end if k == len(pieces) - 1 else start + (end - start) * consumed / total
        out.append(
            FusedWord(
                text=piece,
                start=cursor,
                end=stop,
                tier="resolved",
                confidence=ACCEPTED_CONFIDENCE,
                source="refine",
                alternatives=[],
                unclear=False,
                speaker=speaker,
            )
        )
        cursor = stop
    return out


def _parse_edits(envelope: Dict[str, Any]) -> List[Tuple[int, str]]:
    if envelope.get("is_error"):
        raise TranscriptionError(
            f"claude CLI reported an error (subtype={envelope.get('subtype')}): {str(envelope.get('result'))[:300]}"
        )
    structured = envelope.get("structured_output")
    items = structured.get("edits") if isinstance(structured, dict) else None
    if not isinstance(items, list):
        raise TranscriptionError("claude CLI response is missing structured_output.edits")
    edits: List[Tuple[int, str]] = []
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("id"), int) and isinstance(item.get("replacement"), str):
            edits.append((item["id"], item["replacement"]))
    return edits


class GatedRefiner:
    """Claude proposes edits for doubtful targets; code decides which ones are applied."""

    def __init__(
        self,
        model: Optional[str] = None,
        claude_bin: Optional[str] = None,
        batch_size: int = BATCH_SIZE,
        timeout_seconds: int = CALL_TIMEOUT_SECONDS,
    ):
        self.model = model or os.getenv("TRANSCRIBE_REFINE_MODEL") or DEFAULT_REFINE_MODEL
        self.claude_bin = claude_bin
        self.batch_size = max(1, batch_size)
        self.timeout_seconds = timeout_seconds

    def build_command(self) -> List[str]:
        # Same mandatory flags as ClaudeRefiner.build_command: without --safe-mode/--strict-mcp-config/
        # --disable-slash-commands user plugins and MCP load (~300x cost); --bare refuses OAuth.
        return [
            resolve_claude_bin(self.claude_bin),
            "-p",
            "--safe-mode",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--tools",
            "",
            "--no-session-persistence",
            "--model",
            self.model,
            "--output-format",
            "json",
            "--system-prompt",
            REFINE_GATED_SYSTEM_PROMPT,
            "--json-schema",
            json.dumps(REFINE_GATED_JSON_SCHEMA, separators=(",", ":")),
        ]

    # -- Claude calls

    def build_prompt(self, batch: Sequence[Dict[str, Any]], terms: Sequence[str], lang: str) -> str:
        items = [
            {
                "id": t["id"],
                "reason": t["reason"],
                "text": t["text"],
                "alternatives": list(t.get("alternatives") or []),
                "left": t.get("left", ""),
                "right": t.get("right", ""),
            }
            for t in batch
        ]
        parts = [
            f"Transcript language: {lang}",
            "Terms from the video metadata (may be wrong or irrelevant): "
            + json.dumps(list(terms), ensure_ascii=False),
            "",
            f"Review the {len(items)} targets below. Return edits only for the ones you are sure about; "
            "an empty edits list is fine.",
            "Targets:",
        ]
        return "\n".join(parts) + "\n" + json.dumps(items, ensure_ascii=False, indent=1)

    def _call(self, prompt: str) -> Dict[str, Any]:
        try:
            command = self.build_command()
            proc = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds or None,
            )
        except CapabilityError:
            raise
        except subprocess.TimeoutExpired as exc:
            raise TranscriptionError(f"claude CLI timed out after {self.timeout_seconds}s") from exc
        except OSError as exc:
            raise TranscriptionError(f"claude CLI could not be started: {exc}") from exc
        except ValueError as exc:  # includes UnicodeDecodeError from decoding the CLI output
            raise TranscriptionError(f"claude CLI output could not be read: {exc}") from exc

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()[-500:]
            raise TranscriptionError(f"claude CLI exited with code {proc.returncode}: {detail}")
        try:
            envelope = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise TranscriptionError(f"claude CLI returned non-JSON output: {proc.stdout[:200]!r}") from exc
        if not isinstance(envelope, dict):
            raise TranscriptionError("claude CLI returned an unexpected JSON payload")
        return envelope

    def _run_batch(
        self, batch: Sequence[Dict[str, Any]], terms: Sequence[str], lang: str
    ) -> Tuple[List[Tuple[int, str]], float]:
        prompt = self.build_prompt(batch, terms, lang)
        last_error = ""
        for attempt in range(1, ATTEMPTS + 1):
            try:
                envelope = self._call(prompt)
                cost = envelope.get("total_cost_usd")
                return _parse_edits(envelope), float(cost) if isinstance(cost, (int, float)) else 0.0
            except TranscriptionError as exc:
                last_error = str(exc)
                logger.warning("Gated refine batch attempt %s/%s failed: %s", attempt, ATTEMPTS, last_error)
        raise TranscriptionError(last_error)

    # -- audio check

    def _audio_accepts(
        self,
        words: Sequence[FusedWord],
        target: Dict[str, Any],
        replacement: str,
        term_match: bool,
        scorer: Any,
        samples: Any,
        lang: str,
    ) -> Optional[bool]:
        """True/False from the audio score; None when the edit cannot be verified."""
        start = words[target["start_idx"]].start
        end = words[target["end_idx"] - 1].end
        left_start, right_end = target.get("left_start"), target.get("right_end")
        if isinstance(left_start, (int, float)) and math.isfinite(left_start):
            start = min(start, float(left_start))  # window = first left-context word .. last right-context word
        if isinstance(right_end, (int, float)) and math.isfinite(right_end):
            end = max(end, float(right_end))
        left, right = target.get("left", ""), target.get("right", "")
        original_window = _collapse(f"{left} {target['text']} {right}")
        replaced_window = _collapse(f"{left} {replacement} {right}")
        try:
            s_orig, s_repl = compare(scorer, samples, start, end, [original_window, replaced_window], lang)
            if not (math.isfinite(s_orig) and math.isfinite(s_repl)):
                raise ValueError("non-finite score")
        except Exception as exc:  # per-target failure: never let one window abort the stage
            logger.warning("Gated refine audio check failed for target %s: %s", target.get("id"), exc)
            return None
        margin = AUDIO_TERM_MARGIN if term_match else AUDIO_MARGIN
        return s_repl >= s_orig + margin

    # -- main entry

    def refine(
        self,
        words: List[FusedWord],
        targets: List[Dict[str, Any]],
        terms: List[str],
        lang: str,
        scorer: Any,
        samples: Any,
    ) -> Tuple[List[FusedWord], Dict[str, Any]]:
        stats: Dict[str, Any] = {
            "targets": len(targets),
            "proposed": 0,
            "accepted": 0,
            "rejected_by_gate": {},
            "rejected_by_audio": 0,
            "audio_unverifiable": 0,
            "audio_check": "performed" if scorer is not None else "skipped",
            "batches": 0,
            "batches_failed": 0,
            "model": self.model,
            "cost_usd": 0.0,
            "status": "ok",
        }
        out_words = list(words)
        code = _lang_code(lang)
        if code not in SUPPORTED_LANGS:
            stats["status"] = "skipped: unsupported language"
            return out_words, stats
        valid = [
            t for t in targets if 0 <= t["start_idx"] < t["end_idx"] <= len(words)
        ]
        if not valid:
            return out_words, stats
        try:
            resolve_claude_bin(self.claude_bin)
        except CapabilityError as exc:
            stats["status"] = f"skipped: {exc}"
            return out_words, stats

        by_id = {t["id"]: t for t in valid}
        batches = [valid[i : i + self.batch_size] for i in range(0, len(valid), self.batch_size)]
        stats["batches"] = len(batches)
        proposals: Dict[int, str] = {}
        last_error = ""
        for batch in batches:
            batch_ids = {t["id"] for t in batch}
            try:
                edits, cost = self._run_batch(batch, terms, code)
            except Exception as exc:  # enhancement stage: any batch failure is recorded, never raised
                stats["batches_failed"] += 1
                last_error = str(exc)
                continue
            stats["cost_usd"] += cost
            for edit_id, replacement in edits:
                if edit_id in batch_ids and edit_id not in proposals:
                    proposals[edit_id] = replacement

        if stats["batches_failed"] == len(batches):
            stats["status"] = f"skipped: {last_error or 'claude CLI failed'}"
            return out_words, stats
        if stats["batches_failed"]:
            stats["status"] = f"{PARTIAL_PREFIX}{stats['batches_failed']} batches failed"

        accepted: Dict[int, str] = {}
        for edit_id in sorted(proposals):
            target = by_id[edit_id]
            replacement = _collapse(proposals[edit_id])
            if replacement == _collapse(target["text"]):
                continue  # no-op: nothing to apply or verify
            stats["proposed"] += 1
            failed = gate_edit(target, replacement, terms, code)
            if failed:
                stats["rejected_by_gate"][failed] = stats["rejected_by_gate"].get(failed, 0) + 1
                continue
            term_match = is_term_match(target, replacement, terms, code)
            if scorer is None:
                if not term_match:
                    stats["rejected_by_gate"]["no_audio_check"] = stats["rejected_by_gate"].get("no_audio_check", 0) + 1
                    continue
                accepted[edit_id] = replacement
                continue
            verdict = self._audio_accepts(words, target, replacement, term_match, scorer, samples, code)
            if verdict is None:
                stats["audio_unverifiable"] += 1
            elif verdict:
                accepted[edit_id] = replacement
            else:
                stats["rejected_by_audio"] += 1

        if accepted:
            rebuilt: List[FusedWord] = []
            pos = 0
            for edit_id in sorted(accepted, key=lambda i: by_id[i]["start_idx"]):
                target = by_id[edit_id]
                if target["start_idx"] < pos:
                    continue  # overlapping target (refine_targets never produces this)
                rebuilt.extend(words[pos : target["start_idx"]])
                rebuilt.extend(_replacement_words(words[target["start_idx"] : target["end_idx"]], accepted[edit_id]))
                pos = target["end_idx"]
                stats["accepted"] += 1
            rebuilt.extend(words[pos:])
            out_words = rebuilt
        stats["cost_usd"] = round(stats["cost_usd"], 6)
        logger.info(
            "Gated refine: targets=%s proposed=%s accepted=%s gate_rejected=%s audio_rejected=%s status=%s",
            stats["targets"],
            stats["proposed"],
            stats["accepted"],
            stats["rejected_by_gate"],
            stats["rejected_by_audio"],
            stats["status"],
        )
        return out_words, stats

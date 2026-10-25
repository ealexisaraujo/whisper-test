"""Opt-in post-transcription refine stage: fixes ASR errors with Claude via the Claude Code CLI.

Segment texts are sent in batches to `claude -p` (headless, no tools/plugins/MCP) and come back
as structured JSON. Timestamps, speakers and segment count are never touched; only `text` changes.
"""

from __future__ import annotations

import bisect
import difflib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .backends import _to_float, combine_text_from_segments
from .constants import DEFAULT_REFINE_MODEL
from .models import BackendResult, CapabilityError, TranscriptionError

LOGGER = logging.getLogger("transcribe_youtube")

REFINE_BATCH_SIZE = 60
REFINE_CONTEXT_SEGMENTS = 5  # read-only segments from the previous batch
REFINE_MAX_WORKERS = 4
REFINE_ATTEMPTS = 2  # one retry per batch, then fail loudly
REFINE_CALL_TIMEOUT_SECONDS = 600
REFINE_DESCRIPTION_CHARS = 1500

# Divergence guard: word-level similarity for normal segments, char-level for very short ones.
REFINE_GUARD_MIN_WORDS = 4
REFINE_GUARD_MIN_WORD_RATIO = 0.5
REFINE_GUARD_MIN_CHAR_RATIO = 0.4

# Uncertain-word hints: words in the lowest score quantile, at most N per segment.
REFINE_UNCERTAIN_QUANTILE = 0.10
REFINE_UNCERTAIN_PER_SEGMENT = 3

REFINE_SYSTEM_PROMPT = """You correct automatic speech recognition (ASR) errors in transcript segments.

Rules:
- Fix only recognition errors: misheard words, homophones, wrong word boundaries (e.g. "in graphic" -> "infographic", "boring park" -> "boring part"), dropped or wrong word endings, garbled names of people, brands, products and technical terms, and clearly wrong casing or punctuation.
- Never paraphrase, summarize, translate, censor, add or drop content. Keep filler words, repetitions and the speaker's own grammar unless they are clearly recognition errors.
- Never merge, split or reorder segments. A sentence may continue across segments; keep each segment's words in that segment.
- Return every segment id you were asked to correct exactly once, with its full text (unchanged if nothing needs fixing). Never return the read-only context segments.
- Keep the original language of the transcript.
- If unsure, leave the text unchanged.
- Use the transcript context (title, channel, description, notes) only to fix names and terms that are plausibly what was said; never insert information from it.
- Words listed in "uncertain" had low recognizer confidence: check them closely, but change them only when a better reading is clear."""

REFINE_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["id", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["segments"],
    "additionalProperties": False,
}

_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def resolve_claude_bin(explicit: Optional[str] = None) -> str:
    """Locate the Claude Code CLI: explicit path, then CLAUDE_BIN, then PATH."""
    candidate = explicit or os.getenv("CLAUDE_BIN") or "claude"
    resolved = shutil.which(candidate)
    if not resolved:
        raise CapabilityError(
            f"--refine requires the Claude Code CLI, but '{candidate}' was not found. "
            "Install Claude Code and log in (`claude` once interactively), or set CLAUDE_BIN to its path."
        )
    return resolved


def build_refine_context(
    metadata: Mapping[str, Any],
    *,
    title: Optional[str] = None,
    language: Optional[str] = None,
    notes: Optional[str] = None,
) -> Dict[str, str]:
    """Collect title/channel/description/language/notes for the refine prompt (empty keys dropped)."""
    platform = metadata.get("platform")
    context = {
        "title": metadata.get(f"{platform}_title") if platform else None,
        "channel": metadata.get(f"{platform}_channel") if platform else None,
        "description": metadata.get(f"{platform}_description") if platform else None,
        "language": language,
        "notes": notes,
    }
    context["title"] = context["title"] or title
    description = str(context["description"] or "").strip()
    if len(description) > REFINE_DESCRIPTION_CHARS:
        context["description"] = description[:REFINE_DESCRIPTION_CHARS].rstrip() + " ..."
    return {key: str(value).strip() for key, value in context.items() if value and str(value).strip()}


def map_uncertain_words(
    segments: Sequence[Mapping[str, Any]],
    words: Sequence[Any],
    *,
    quantile: float = REFINE_UNCERTAIN_QUANTILE,
    per_segment: int = REFINE_UNCERTAIN_PER_SEGMENT,
) -> List[List[str]]:
    """Map low-confidence words (e.g. EG2 `raw.words`) onto segments by start time.

    A word belongs to the last segment starting at or before it. Only words in the lowest
    `quantile` of all scores are kept, at most `per_segment` per segment, in time order.
    """
    buckets: List[List[Tuple[float, float, str]]] = [[] for _ in segments]
    starts = [_to_float(seg.get("start")) for seg in segments]
    if not segments or any(s is None for s in starts) or starts != sorted(starts):
        return [[] for _ in segments]

    scored: List[Tuple[float, float, str]] = []
    for item in words or []:
        if not isinstance(item, Mapping):
            continue
        score = _to_float(item.get("score"))
        start = _to_float(item.get("start"))
        token = str(item.get("word") or "").strip(" .,?!;:\"()[]")
        if score is None or start is None or not token:
            continue
        scored.append((score, start, token))
    if not scored:
        return [[] for _ in segments]

    ordered = sorted(score for score, _, _ in scored)
    threshold = ordered[int(quantile * (len(ordered) - 1))]
    for score, start, token in scored:
        if score > threshold:
            continue
        idx = bisect.bisect_right(starts, start + 1e-6) - 1
        if idx >= 0:
            buckets[idx].append((score, start, token))

    hints: List[List[str]] = []
    for bucket in buckets:
        lowest = sorted(bucket)[:per_segment]
        hints.append([token for _, _, token in sorted(lowest, key=lambda item: item[1])])
    return hints


def refine_diverges(original: str, refined: str) -> bool:
    """True when the refined text drifts too far from the original to be an ASR fix."""
    orig_words = _WORD_RE.findall(original.lower())
    new_words = _WORD_RE.findall(refined.lower())
    if len(orig_words) >= REFINE_GUARD_MIN_WORDS:
        ratio = difflib.SequenceMatcher(None, orig_words, new_words, autojunk=False).ratio()
        return ratio < REFINE_GUARD_MIN_WORD_RATIO
    ratio = difflib.SequenceMatcher(None, " ".join(orig_words), " ".join(new_words), autojunk=False).ratio()
    return ratio < REFINE_GUARD_MIN_CHAR_RATIO


class ClaudeRefiner:
    """Corrects segment texts with Claude through the headless Claude Code CLI."""

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        batch_size: int = REFINE_BATCH_SIZE,
        max_workers: int = REFINE_MAX_WORKERS,
        context_segments: int = REFINE_CONTEXT_SEGMENTS,
        timeout_seconds: int = REFINE_CALL_TIMEOUT_SECONDS,
        claude_bin: Optional[str] = None,
    ):
        self.model = model or os.getenv("TRANSCRIBE_REFINE_MODEL") or DEFAULT_REFINE_MODEL
        self.batch_size = max(1, batch_size)
        self.max_workers = max(1, max_workers)
        self.context_segments = max(0, context_segments)
        self.timeout_seconds = timeout_seconds
        self.claude_bin = resolve_claude_bin(claude_bin)

    def build_command(self) -> List[str]:
        # --safe-mode/--strict-mcp-config/--disable-slash-commands keep user plugins, MCP servers and
        # skills out of the call (~300x cheaper); --bare is avoided because it refuses OAuth auth.
        return [
            self.claude_bin,
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
            REFINE_SYSTEM_PROMPT,
            "--json-schema",
            json.dumps(REFINE_JSON_SCHEMA, separators=(",", ":")),
        ]

    def refine(self, result: BackendResult, context: Optional[Mapping[str, Any]] = None) -> BackendResult:
        segments = result.segments
        if not segments:
            raise CapabilityError(
                "--refine requires segmented output, but this backend/model returned text only. "
                "Use a timestamp-capable model or run without --refine."
            )

        texts = [str(seg.get("text") or "").strip() for seg in segments]
        raw_words = result.raw.get("words") if isinstance(result.raw, dict) else None
        uncertain = map_uncertain_words(segments, raw_words or [])
        bounds = [(lo, min(lo + self.batch_size, len(texts))) for lo in range(0, len(texts), self.batch_size)]
        LOGGER.info(
            "Refine started: model=%s segments=%s batches=%s workers=%s uncertain_hints=%s",
            self.model,
            len(texts),
            len(bounds),
            min(self.max_workers, len(bounds)),
            sum(len(h) for h in uncertain),
        )

        started_at = time.perf_counter()
        refined: Dict[int, str] = {}
        cost_usd = 0.0
        pool = ThreadPoolExecutor(max_workers=min(self.max_workers, len(bounds)))
        try:
            futures = {
                pool.submit(self._run_batch, idx, len(bounds), lo, hi, texts, uncertain, context or {}): idx
                for idx, (lo, hi) in enumerate(bounds, start=1)
            }
            for future in as_completed(futures):
                batch_texts, batch_cost = future.result()
                refined.update(batch_texts)
                cost_usd += batch_cost
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        duration_s = time.perf_counter() - started_at

        new_segments: List[Dict[str, Any]] = []
        changed = 0
        guarded = 0
        for idx, (seg, original) in enumerate(zip(segments, texts)):
            candidate = refined[idx].strip()
            final = original
            if original and not candidate:
                guarded += 1
                LOGGER.debug("Refine guard (empty): segment %s kept original %r", idx, original)
            elif original and refine_diverges(original, candidate):
                guarded += 1
                LOGGER.debug("Refine guard (diverged): segment %s kept %r over %r", idx, original, candidate)
            elif original and candidate != original:
                final = candidate
                changed += 1
            item = dict(seg)
            item["text"] = final
            new_segments.append(item)

        raw = dict(result.raw) if isinstance(result.raw, dict) else {}
        raw["refine"] = {
            "model": self.model,
            "batches": len(bounds),
            "changed_segments": changed,
            "guarded_segments": guarded,
            "cost_usd": round(cost_usd, 6),
            "duration_s": round(duration_s, 3),
            "original_texts": texts,
        }
        LOGGER.info(
            "Refine finished: changed=%s guarded=%s of %s segments cost=$%.4f duration=%.1fs",
            changed,
            guarded,
            len(texts),
            cost_usd,
            duration_s,
        )
        return BackendResult(
            text=combine_text_from_segments(new_segments),
            segments=new_segments,
            language=result.language,
            duration_sec=result.duration_sec,
            raw=raw,
        )

    def build_prompt(
        self,
        lo: int,
        hi: int,
        texts: Sequence[str],
        uncertain: Sequence[Sequence[str]],
        context: Mapping[str, Any],
    ) -> str:
        items: List[Dict[str, Any]] = []
        for idx in range(lo, hi):
            item: Dict[str, Any] = {"id": idx, "text": texts[idx]}
            if uncertain[idx]:
                item["uncertain"] = list(uncertain[idx])
            items.append(item)
        before = list(texts[max(0, lo - self.context_segments):lo])

        parts = ["Transcript context:", json.dumps(dict(context), ensure_ascii=False, indent=1)]
        if before:
            parts += [
                "",
                "Read-only context: the segments right before this batch, for continuity. Do NOT return them.",
                json.dumps(before, ensure_ascii=False, indent=1),
            ]
        parts += [
            "",
            f"Segments to correct ({len(items)}): return each id exactly once.",
            json.dumps(items, ensure_ascii=False, indent=1),
        ]
        return "\n".join(parts) + "\n"

    def _run_batch(
        self,
        batch_no: int,
        batch_count: int,
        lo: int,
        hi: int,
        texts: Sequence[str],
        uncertain: Sequence[Sequence[str]],
        context: Mapping[str, Any],
    ) -> Tuple[Dict[int, str], float]:
        prompt = self.build_prompt(lo, hi, texts, uncertain, context)
        expected = list(range(lo, hi))
        cost = 0.0
        last_error = ""
        for attempt in range(1, REFINE_ATTEMPTS + 1):
            try:
                envelope = self._call(prompt)
                cost += _to_float(envelope.get("total_cost_usd")) or 0.0
                LOGGER.debug(
                    "Refine batch %s/%s attempt %s: duration_ms=%s cost=%s",
                    batch_no,
                    batch_count,
                    attempt,
                    envelope.get("duration_ms"),
                    envelope.get("total_cost_usd"),
                )
                return _parse_batch_output(envelope, expected), cost
            except TranscriptionError as exc:
                last_error = str(exc)
                LOGGER.warning(
                    "Refine batch %s/%s (segments %s-%s) attempt %s/%s failed: %s",
                    batch_no,
                    batch_count,
                    lo,
                    hi - 1,
                    attempt,
                    REFINE_ATTEMPTS,
                    last_error,
                )
        raise TranscriptionError(
            f"Refine batch {batch_no}/{batch_count} (segments {lo}-{hi - 1}) failed after "
            f"{REFINE_ATTEMPTS} attempts: {last_error}"
        )

    def _call(self, prompt: str) -> Dict[str, Any]:
        try:
            proc = subprocess.run(
                self.build_command(),
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds or None,
            )
        except subprocess.TimeoutExpired as exc:
            raise TranscriptionError(f"claude CLI timed out after {self.timeout_seconds}s") from exc
        except OSError as exc:
            raise TranscriptionError(f"claude CLI could not be started: {exc}") from exc

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


def _parse_batch_output(envelope: Mapping[str, Any], expected: Sequence[int]) -> Dict[int, str]:
    if envelope.get("is_error"):
        raise TranscriptionError(
            f"claude CLI reported an error (subtype={envelope.get('subtype')}): {str(envelope.get('result'))[:300]}"
        )
    structured = envelope.get("structured_output")
    items = structured.get("segments") if isinstance(structured, dict) else None
    if not isinstance(items, list):
        raise TranscriptionError("claude CLI response is missing structured_output.segments")

    out: Dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("id"), int) or not isinstance(item.get("text"), str):
            raise TranscriptionError(f"invalid segment in structured_output: {item!r}"[:300])
        if item["id"] in out:
            raise TranscriptionError(f"segment id {item['id']} returned more than once")
        out[item["id"]] = item["text"]

    if sorted(out) != list(expected):
        missing = sorted(set(expected) - set(out))
        extra = sorted(set(out) - set(expected))
        raise TranscriptionError(f"segment ids mismatch: missing={missing[:10]} unexpected={extra[:10]}")
    return out

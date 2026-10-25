"""Tests for the gated Claude refine stage (mocked subprocess.run; fake scorer; no models)."""

from __future__ import annotations

import json
import subprocess
from typing import Any, Dict, List

import numpy as np
import pytest

import transcribe.refine as refine_mod
from transcribe.auto.refine_gated import (
    GatedRefiner,
    PARTIAL_PREFIX,
    gate_edit,
    refine_targets,
)
from transcribe.auto.resolve import FusedWord

SAMPLES = np.zeros(16000 * 10, dtype=np.float32)


# ------------------------------------------------------------------ helpers


def FW(text, i=0, *, tier="solid", source="agree", confidence=0.9, unclear=False, alternatives=None, speaker=None):
    return FusedWord(
        text=text,
        start=float(i),
        end=float(i) + 0.5,
        tier=tier,
        confidence=confidence,
        source=source,
        alternatives=list(alternatives or []),
        unclear=unclear,
        speaker=speaker,
    )


def words_of(text: str, **kw) -> List[FusedWord]:
    return [FW(w, i, **kw) for i, w in enumerate(text.split())]


def T(text, alternatives=None, left="", right="", reason="unclear", id=0, start_idx=0, end_idx=None):
    n = len(text.split())
    return {
        "id": id,
        "start_idx": start_idx,
        "end_idx": start_idx + n if end_idx is None else end_idx,
        "text": text,
        "alternatives": list(alternatives or []),
        "left": left,
        "right": right,
        "reason": reason,
    }


class FakeScorer:
    """Scores by window text via `fn(text) -> float`; `errors` maps text -> exception."""

    def __init__(self, fn=lambda t: -10.0, errors=None):
        self.fn = fn
        self.errors = errors or {}
        self.calls: List[tuple] = []

    def score(self, samples, start, end, text, lang):
        return self.score_detail(samples, start, end, text, lang)[0]

    def score_detail(self, samples, start, end, text, lang):
        self.calls.append((start, end, text))
        if text in self.errors:
            raise self.errors[text]
        return self.fn(text), len(text.split())

    def release(self):
        pass


def _envelope(edits: List[Dict[str, Any]], cost: float = 0.01) -> str:
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "",
            "duration_ms": 900,
            "total_cost_usd": cost,
            "structured_output": {"edits": edits},
        }
    )


class FakeRun:
    """Mocks subprocess.run; `responder(prompt_targets, call_no) -> (returncode, stdout)`."""

    def __init__(self, responder):
        self.responder = responder
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, cmd, input=None, capture_output=None, text=None, timeout=None):
        self.calls.append({"cmd": cmd, "input": input})
        code, stdout = self.responder(prompt_targets(input), len(self.calls))
        return subprocess.CompletedProcess(cmd, code, stdout=stdout, stderr="boom" if code else "")


def prompt_targets(prompt: str) -> List[Dict[str, Any]]:
    tail = prompt.split("Targets:\n", 1)[1]
    return json.loads(tail)


def answer(mapping: Dict[int, str]):
    return lambda targets, n: (0, _envelope([{"id": i, "replacement": r} for i, r in mapping.items()]))


@pytest.fixture
def refiner(monkeypatch: pytest.MonkeyPatch) -> GatedRefiner:
    monkeypatch.setattr(refine_mod.shutil, "which", lambda name: f"/usr/local/bin/{name}")
    return GatedRefiner(model="sonnet", batch_size=40)


def install(monkeypatch, responder) -> FakeRun:
    fake = FakeRun(responder)
    monkeypatch.setattr(refine_mod.subprocess, "run", fake)
    return fake


# ------------------------------------------------------------------ gates (brief)


def test_gate_rejects_filler_removal():
    # Dropping only "um" stays inside the length tolerance, so the filler gate is what fires.
    target = T("so um we built it")
    assert gate_edit(target, "so we built it", [], "en") == "filler_removed"


def test_gate_length_fires_before_filler_for_big_removal():
    assert gate_edit(T("so um we built it"), "we built it", [], "en") == "length"


def test_gate_rejects_pronoun_swap():
    target = T("prompt an LLM", alternatives=["prompt in"])
    assert gate_edit(target, "prompt it", [], "en") == "pronoun_swap"


def test_gate_allows_term_fix():
    target = T("self tension", alternatives=["self attention"])
    assert gate_edit(target, "self attention", ["self attention"], "en") is None


def test_gate_phonetic_rejects_unrelated():
    assert gate_edit(T("LLMs"), "AI", [], "en") == "phonetic"


def test_gate_empty_and_length():
    assert gate_edit(T("we built it"), "   ", [], "en") == "empty"
    assert gate_edit(T("we built it"), "we built it and then some", [], "en") == "length"
    # ±1 word minimum for short targets
    assert gate_edit(T("tension"), "attention", [], "en") is None


def test_gate_planted_term_does_not_rescue_phonetic_gate():
    # The replacement merely equals a term; nothing in the original sounds like it.
    assert gate_edit(T("LLMs"), "AI", ["AI"], "en") == "phonetic"
    assert gate_edit(T("LLMs"), "AI", ["LLMs"], "en") == "phonetic"
    assert gate_edit(T("AI"), "LLMs", ["LLMs"], "en") == "phonetic"
    assert gate_edit(T("AI"), "Karpathy", ["Karpathy"], "en") == "phonetic"


def test_gate_low_conf_agree_cannot_be_planted_with_a_term():
    target = T("the", reason="low_conf_agree")
    assert gate_edit(target, "Karpathy", ["Karpathy"], "en") in ("phonetic", "locked")


def test_gate_term_match_is_relational_via_alternatives():
    # The original sounds nothing like the term, but the other engine's alternative does.
    target = T("sell fatten shun", alternatives=["self attention"])
    assert gate_edit(target, "self attention", ["self attention"], "en") in (None, "length")
    target = T("self tension", alternatives=[""])
    assert gate_edit(target, "self attention", ["self attention"], "en") is None


def test_is_term_match_requires_both_sides():
    from transcribe.auto.refine_gated import is_term_match

    assert is_term_match(T("self tension"), "self attention", ["self attention"], "en")
    assert not is_term_match(T("LLMs"), "AI", ["AI"], "en")
    assert not is_term_match(T("AI"), "LLMs", ["LLMs"], "en")
    # unchanged n-grams of the original do not count as a correction
    assert not is_term_match(T("LLMs are great"), "LLMs is great", ["LLMs"], "en")
    # an equal (case-insensitive) alternative counts as the source side
    assert is_term_match(T("ship", alternatives=["Sheep"]), "sheep", ["SHEEP"], "en")


def test_gate_locked_requires_term_match_for_low_conf_agree():
    target = T("tension", reason="low_conf_agree")
    assert gate_edit(target, "attention", [], "en") == "locked"
    assert gate_edit(target, "attention", ["attention"], "en") is None


def test_gate_spanish_filler_and_pronoun():
    assert gate_edit(T("o sea que si"), "que si", [], "es") in ("filler_removed", "length")
    assert gate_edit(T("eh vamos a ver"), "vamos a ver", [], "es") in ("filler_removed", "length")
    assert gate_edit(T("pues vamos a ver"), "vamos a ver", [], "es") in ("filler_removed", "length")
    assert gate_edit(T("dame la casa"), "dame la casa", [], "es") is None
    assert gate_edit(T("dame casa", alternatives=["dame cosa"]), "dame eso", [], "es") == "pronoun_swap"


# ------------------------------------------------------------------ refine_targets


def test_targets_unclear_run_is_one_target_with_first_word_alternatives():
    words = words_of("we saw the")
    words += [
        FW("self", 3, unclear=True, tier="unclear", source="verify", alternatives=["elf"], confidence=0.3),
        FW("tension", 4, unclear=True, tier="unclear", source="verify", confidence=0.3),
    ]
    words += words_of("matter a lot")[0:0]
    words += [FW("is", 5), FW("key", 6)]
    targets = refine_targets(words, [], "en")
    assert len(targets) == 1
    t = targets[0]
    assert t["reason"] == "unclear"
    assert (t["start_idx"], t["end_idx"]) == (3, 5)
    assert t["text"] == "self tension"
    assert t["alternatives"] == ["elf"]
    assert t["left"] == "we saw the"
    assert t["right"] == "is key"


def test_targets_context_is_capped_at_eight_words():
    words = words_of("a1 a2 a3 a4 a5 a6 a7 a8 a9 a10") + [FW("xx", 10, unclear=True, tier="unclear", confidence=0.3)]
    words += words_of("b1 b2 b3 b4 b5 b6 b7 b8 b9")
    t = refine_targets(words, [], "en")[0]
    assert t["left"].split() == ["a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10"]
    assert t["right"].split() == ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"]


def test_targets_low_conf_agree_only_agree_below_half():
    words = [
        FW("one", 0, confidence=0.4),  # agree, low -> target
        FW("two", 1, confidence=0.5),  # exactly 0.5 -> not low
        FW("three", 2, confidence=0.3, source="pick"),  # not an agree word
        FW("four", 3, confidence=0.2),  # agree, low -> target
    ]
    targets = refine_targets(words, [], "en")
    assert [(t["reason"], t["start_idx"], t["end_idx"], t["text"]) for t in targets] == [
        ("low_conf_agree", 0, 1, "one"),
        ("low_conf_agree", 3, 4, "four"),
    ]
    assert [t["id"] for t in targets] == [0, 1]


def test_targets_term_near_finds_ngram_but_not_exact_match():
    words = words_of("we use self tension here and self attention there")
    targets = refine_targets(words, ["self attention"], "en")
    assert [(t["reason"], t["text"]) for t in targets] == [("term_near", "self tension")]
    assert (targets[0]["start_idx"], targets[0]["end_idx"]) == (2, 4)


def test_targets_priority_unclear_then_term_near_then_low_conf():
    words = [
        FW("we", 0),
        FW("self", 1, confidence=0.3),  # low-conf agree but part of a term_near n-gram
        FW("tension", 2, confidence=0.3),
        FW("and", 3),
        FW("self", 4, unclear=True, tier="unclear", source="verify", confidence=0.3, alternatives=["shelf"]),
        FW("tension", 5, unclear=True, tier="unclear", source="verify", confidence=0.3),
        FW("end", 6),
    ]
    targets = refine_targets(words, ["self attention"], "en")
    spans = [(t["reason"], t["start_idx"], t["end_idx"]) for t in targets]
    assert spans == [("term_near", 1, 3), ("unclear", 4, 6)]
    covered = [i for t in targets for i in range(t["start_idx"], t["end_idx"])]
    assert len(covered) == len(set(covered))


def test_targets_low_conf_word_alone_when_no_term_claims_it():
    words = words_of("hello", confidence=0.2)
    assert [t["reason"] for t in refine_targets(words, ["self attention"], "en")] == ["low_conf_agree"]


def test_targets_empty_inputs():
    assert refine_targets([], ["x"], "en") == []
    assert refine_targets(words_of("all fine here"), [], "en") == []


# ------------------------------------------------------------------ command / CLI


def test_command_keeps_safe_flags(refiner):
    cmd = refiner.build_command()
    for flag in ("--safe-mode", "--strict-mcp-config", "--disable-slash-commands", "--tools", "--no-session-persistence"):
        assert flag in cmd
    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--model") + 1] == "sonnet"
    assert cmd[cmd.index("--output-format") + 1] == "json"
    schema = json.loads(cmd[cmd.index("--json-schema") + 1])
    assert schema["properties"]["edits"]["items"]["properties"].keys() == {"id", "replacement"}


def test_system_prompt_states_the_policies(refiner):
    cmd = refiner.build_command()
    prompt = cmd[cmd.index("--system-prompt") + 1].lower()
    for needle in ("verbatim", "filler", "synonym", "paraphrase", "mishearing", "metadata", "empty list", "sure"):
        assert needle in prompt


def test_model_env_override(monkeypatch):
    monkeypatch.setattr(refine_mod.shutil, "which", lambda name: f"/usr/local/bin/{name}")
    monkeypatch.setenv("TRANSCRIBE_REFINE_MODEL", "haiku")
    assert GatedRefiner().model == "haiku"
    monkeypatch.delenv("TRANSCRIBE_REFINE_MODEL")
    assert GatedRefiner().model == "sonnet"


def test_refine_skips_when_cli_missing(monkeypatch):
    monkeypatch.setattr(refine_mod.shutil, "which", lambda name: None)
    monkeypatch.delenv("CLAUDE_BIN", raising=False)
    words = words_of("self tension")
    targets = [T("self tension")]
    out, stats = GatedRefiner().refine(words, targets, ["self attention"], "en", None, SAMPLES)
    assert out == words
    assert stats["status"].startswith("skipped")


def test_refine_skips_unsupported_language(refiner, monkeypatch):
    fake = install(monkeypatch, answer({0: "x"}))
    words = words_of("hallo welt")
    out, stats = refiner.refine(words, [T("hallo welt")], [], "de", None, SAMPLES)
    assert out == words
    assert stats["status"] == "skipped: unsupported language"
    assert fake.calls == []


# ------------------------------------------------------------------ refine


def _term_words():
    return words_of("we use self tension here")


def _term_target(words):
    return refine_targets(words, ["self attention"], "en")


def test_refine_accepts_term_fix_with_audio_support(refiner, monkeypatch):
    words = _term_words()
    targets = _term_target(words)
    install(monkeypatch, answer({0: "self attention"}))
    scorer = FakeScorer(lambda t: -2.0 if "self attention" in t else -5.0)
    out, stats = refiner.refine(words, targets, ["self attention"], "en", scorer, SAMPLES)
    assert [w.text for w in out] == ["we", "use", "self", "attention", "here"]
    assert stats["targets"] == 1 and stats["proposed"] == 1 and stats["accepted"] == 1
    assert stats["status"] == "ok"
    assert stats["rejected_by_audio"] == 0


def test_refine_audio_check_rejects(refiner, monkeypatch):
    words = _term_words()
    targets = _term_target(words)
    install(monkeypatch, answer({0: "self attention"}))
    # fake scorer prefers the original even with the term margin of -0.10
    scorer = FakeScorer(lambda t: -2.0 if "self tension" in t else -5.0)
    out, stats = refiner.refine(words, targets, ["self attention"], "en", scorer, SAMPLES)
    assert out == words
    assert stats["rejected_by_audio"] == 1
    assert stats["accepted"] == 0 and stats["proposed"] == 1


def test_refine_audio_margins(refiner, monkeypatch):
    # term match: -0.10 slack passes; non-term needs +0.05
    words = words_of("we saw the ship today")
    unclear = [
        FW("we", 0), FW("saw", 1), FW("the", 2),
        FW("ship", 3, unclear=True, tier="unclear", source="verify", confidence=0.3, alternatives=["sheep"]),
        FW("today", 4),
    ]
    targets = refine_targets(unclear, [], "en")
    assert len(targets) == 1
    install(monkeypatch, answer({0: "sheep"}))
    base = "we saw the ship today"

    # +0.04 is under the margin -> rejected; +0.06 -> accepted
    for delta, expected in ((0.04, 0), (0.06, 1)):
        scorer = FakeScorer(lambda t, d=delta: -3.0 + (d if "sheep" in t else 0.0))
        out, stats = refiner.refine(unclear, targets, [], "en", scorer, SAMPLES)
        assert stats["accepted"] == expected
        assert [c[2] for c in scorer.calls] == [base, "we saw the sheep today"]

    # term match with s_repl = s_orig - 0.09 passes, -0.11 fails
    for delta, expected in ((-0.09, 1), (-0.11, 0)):
        scorer = FakeScorer(lambda t, d=delta: -3.0 + (d if "sheep" in t else 0.0))
        _, stats = refiner.refine(unclear, targets, ["sheep"], "en", scorer, SAMPLES)
        assert stats["accepted"] == expected


def test_refine_audio_value_error_rejects_as_unverifiable(refiner, monkeypatch):
    words = _term_words()
    targets = _term_target(words)
    install(monkeypatch, answer({0: "self attention"}))
    scorer = FakeScorer(errors={"we use self tension here": ValueError("degenerate window")})
    out, stats = refiner.refine(words, targets, ["self attention"], "en", scorer, SAMPLES)
    assert out == words
    assert stats["audio_unverifiable"] == 1
    assert stats["accepted"] == 0


def test_refine_scorer_none_only_accepts_term_matches(refiner, monkeypatch):
    words = [
        FW("we", 0),
        FW("use", 1),
        FW("self", 2),
        FW("tension", 3),
        FW("and", 4),
        FW("ship", 5, unclear=True, tier="unclear", source="verify", confidence=0.3, alternatives=["sheep"]),
    ]
    targets = refine_targets(words, ["self attention"], "en")
    assert [t["reason"] for t in targets] == ["term_near", "unclear"]
    install(monkeypatch, answer({0: "self attention", 1: "sheep"}))
    out, stats = refiner.refine(words, targets, ["self attention"], "en", None, SAMPLES)
    assert [w.text for w in out] == ["we", "use", "self", "attention", "and", "ship"]
    assert out[-1].unclear is True  # rejected unclear target stays unclear
    assert stats["accepted"] == 1 and stats["proposed"] == 2
    assert stats["audio_check"] == "skipped"
    assert sum(stats["rejected_by_gate"].values()) == 1


def test_refine_gate_rejections_counted_and_words_unchanged(refiner, monkeypatch):
    words = words_of("so um we built it")
    words = [FW("so", 0), FW("um", 1, unclear=True, tier="unclear", source="verify", confidence=0.3), FW("we", 2)]
    targets = refine_targets(words, [], "en")
    install(monkeypatch, answer({0: ""}))
    out, stats = refiner.refine(words, targets, [], "en", FakeScorer(), SAMPLES)
    assert out == words
    assert stats["rejected_by_gate"] == {"empty": 1}
    assert stats["accepted"] == 0


def test_refine_accepted_word_fields_and_times(refiner, monkeypatch):
    words = [
        FW("go", 0, speaker="Speaker 2"),
        FusedWord("aplazar", 1.0, 2.2, "unclear", 0.3, "verify", ["a plazar"], True, "Speaker 1"),
        FW("now", 3, speaker="Speaker 2"),
    ]
    words[1].alternatives = ["a plazar"]
    targets = refine_targets(words, ["a plazar"], "es")
    assert targets[0]["reason"] == "unclear"
    install(monkeypatch, answer({0: "ab plazar"}))  # 2 words: "ab" (2 chars) + "plazar" (6 chars)
    scorer = FakeScorer(lambda t: -1.0 if "ab plazar" in t else -4.0)
    out, stats = refiner.refine(words, targets, ["a plazar"], "es", scorer, SAMPLES)
    assert stats["accepted"] == 1
    new = out[1:3]
    assert [w.text for w in new] == ["ab", "plazar"]
    assert new[0].start == pytest.approx(1.0)
    assert new[0].end == pytest.approx(1.0 + 1.2 * 2 / 8)
    assert new[1].start == pytest.approx(new[0].end)
    assert new[1].end == pytest.approx(2.2)
    for w in new:
        assert (w.tier, w.source, w.unclear, w.alternatives, w.confidence) == ("resolved", "refine", False, [], 0.6)
        assert w.speaker == "Speaker 1"
    assert [w.text for w in out] == ["go", "ab", "plazar", "now"]
    assert out[0].speaker == "Speaker 2" and out[3].speaker == "Speaker 2"


def test_refine_unknown_ids_and_noop_edits_are_ignored(refiner, monkeypatch):
    words = _term_words()
    targets = _term_target(words)
    install(monkeypatch, answer({99: "whatever", 0: "self tension"}))
    out, stats = refiner.refine(words, targets, ["self attention"], "en", FakeScorer(), SAMPLES)
    assert out == words
    assert stats["proposed"] == 0 and stats["status"] == "ok"


def test_refine_empty_edits_list_is_fine(refiner, monkeypatch):
    words = _term_words()
    targets = _term_target(words)
    install(monkeypatch, answer({}))
    out, stats = refiner.refine(words, targets, ["self attention"], "en", FakeScorer(), SAMPLES)
    assert out == words
    assert stats["proposed"] == 0 and stats["status"] == "ok"


def test_refine_cli_failure_retries_twice_then_skips(refiner, monkeypatch):
    words = _term_words()
    targets = _term_target(words)
    fake = install(monkeypatch, lambda t, n: (1, ""))
    out, stats = refiner.refine(words, targets, ["self attention"], "en", FakeScorer(), SAMPLES)
    assert out == words
    assert len(fake.calls) == 2
    assert stats["status"].startswith("skipped: ")


def test_refine_timeout_and_oserror_never_raise(refiner, monkeypatch):
    words = _term_words()
    targets = _term_target(words)

    def boom(*a, **k):
        raise subprocess.TimeoutExpired("claude", 1)

    monkeypatch.setattr(refine_mod.subprocess, "run", boom)
    out, stats = refiner.refine(words, targets, ["self attention"], "en", FakeScorer(), SAMPLES)
    assert out == words and stats["status"].startswith("skipped: ")

    def nope(*a, **k):
        raise OSError("no exec")

    monkeypatch.setattr(refine_mod.subprocess, "run", nope)
    out, stats = refiner.refine(words, targets, ["self attention"], "en", FakeScorer(), SAMPLES)
    assert out == words and stats["status"].startswith("skipped: ")


def test_refine_retry_recovers_after_one_failure(refiner, monkeypatch):
    words = _term_words()
    targets = _term_target(words)

    def responder(t, n):
        if n == 1:
            return 1, ""
        return 0, _envelope([{"id": 0, "replacement": "self attention"}])

    fake = install(monkeypatch, responder)
    scorer = FakeScorer(lambda t: -1.0 if "attention" in t else -3.0)
    out, stats = refiner.refine(words, targets, ["self attention"], "en", scorer, SAMPLES)
    assert len(fake.calls) == 2
    assert stats["accepted"] == 1 and stats["status"] == "ok"


def test_refine_partial_batch_failure_keeps_successful_edits(monkeypatch):
    monkeypatch.setattr(refine_mod.shutil, "which", lambda name: f"/usr/local/bin/{name}")
    refiner = GatedRefiner(model="sonnet", batch_size=1)
    # two separate term_near targets, one per batch
    words = words_of("we use self tension and self tension there")
    targets = refine_targets(words, ["self attention"], "en")
    assert len(targets) == 2

    def responder(items, n):
        ids = [it["id"] for it in items]
        if ids == [0]:
            return 0, _envelope([{"id": 0, "replacement": "self attention"}])
        return 1, ""  # second batch fails both attempts

    fake = install(monkeypatch, responder)
    scorer = FakeScorer(lambda t: -1.0 if "attention" in t else -3.0)
    out, stats = refiner.refine(words, targets, ["self attention"], "en", scorer, SAMPLES)
    assert [w.text for w in out] == "we use self attention and self tension there".split()
    assert stats["accepted"] == 1
    assert stats["status"] == f"{PARTIAL_PREFIX}1 batches failed"
    assert len(fake.calls) == 3  # batch 1 once, batch 2 twice


def test_refine_batches_split_by_batch_size(monkeypatch):
    monkeypatch.setattr(refine_mod.shutil, "which", lambda name: f"/usr/local/bin/{name}")
    refiner = GatedRefiner(model="sonnet", batch_size=2)
    words = words_of("alpha beta gamma delta epsilon", confidence=0.2)
    targets = refine_targets(words, [], "en")
    assert len(targets) == 5
    fake = install(monkeypatch, answer({}))
    refiner.refine(words, targets, [], "en", FakeScorer(), SAMPLES)
    sizes = [len(prompt_targets(c["input"])) for c in fake.calls]
    assert sizes == [2, 2, 1]


def test_refine_prompt_carries_targets_terms_and_context(refiner, monkeypatch):
    words = _term_words()
    targets = _term_target(words)
    fake = install(monkeypatch, answer({}))
    refiner.refine(words, targets, ["self attention"], "en", FakeScorer(), SAMPLES)
    prompt = fake.calls[0]["input"]
    assert "self attention" in prompt.split("Targets:\n")[0]
    sent = prompt_targets(prompt)
    assert sent[0]["id"] == 0 and sent[0]["text"] == "self tension"
    assert sent[0]["left"] == "we use" and sent[0]["right"] == "here"


def test_refine_no_targets_makes_no_call(refiner, monkeypatch):
    fake = install(monkeypatch, answer({}))
    words = words_of("all fine")
    out, stats = refiner.refine(words, [], [], "en", None, SAMPLES)
    assert out == words and fake.calls == [] and stats["status"] == "ok" and stats["targets"] == 0


def test_refine_stats_shape(refiner, monkeypatch):
    words = _term_words()
    targets = _term_target(words)
    install(monkeypatch, answer({}))
    _, stats = refiner.refine(words, targets, ["self attention"], "en", FakeScorer(), SAMPLES)
    for key in ("targets", "proposed", "accepted", "rejected_by_gate", "rejected_by_audio", "status"):
        assert key in stats


# ------------------------------------------------------------------ planted names (fix round 1)


def test_refine_flat_scorer_does_not_accept_planted_term(refiner, monkeypatch):
    words = [
        FW("we", 0),
        FW("like", 1),
        FW("AI", 2, unclear=True, tier="unclear", source="verify", confidence=0.3, alternatives=["a i"]),
        FW("models", 3),
    ]
    targets = refine_targets(words, ["LLMs"], "en")
    assert [t["reason"] for t in targets] == ["unclear"]
    install(monkeypatch, answer({0: "LLMs"}))
    scorer = FakeScorer()  # flat: every window scores the same, so the -0.10 leniency would accept
    out, stats = refiner.refine(words, targets, ["LLMs"], "en", scorer, SAMPLES)
    assert out == words
    assert stats["accepted"] == 0
    assert stats["rejected_by_gate"] == {"phonetic": 1}
    assert scorer.calls == []

    # and with no scorer at all
    out, stats = refiner.refine(words, targets, ["LLMs"], "en", None, SAMPLES)
    assert out == words and stats["accepted"] == 0


def test_refine_flat_scorer_accepts_genuine_term_near_correction(refiner, monkeypatch):
    words = _term_words()
    targets = _term_target(words)
    assert targets[0]["reason"] == "term_near"
    install(monkeypatch, answer({0: "self attention"}))
    # s_repl is 0.09 below s_orig: only the lenient term margin (-0.10) accepts it
    scorer = FakeScorer(lambda t: -3.0 - (0.09 if "attention" in t else 0.0))
    out, stats = refiner.refine(words, targets, ["self attention"], "en", scorer, SAMPLES)
    assert [w.text for w in out] == ["we", "use", "self", "attention", "here"]
    assert stats["accepted"] == 1


def test_refine_planted_term_on_low_conf_agree_is_rejected(refiner, monkeypatch):
    words = [FW("so", 0), FW("the", 1, confidence=0.2), FW("end", 2)]
    targets = refine_targets(words, ["Karpathy"], "en")
    assert [t["reason"] for t in targets] == ["low_conf_agree"]
    install(monkeypatch, answer({0: "Karpathy"}))
    out, stats = refiner.refine(words, targets, ["Karpathy"], "en", FakeScorer(), SAMPLES)
    assert out == words and stats["accepted"] == 0


# ------------------------------------------------------------------ never raises (output decoding)


def test_refine_undecodable_cli_output_never_raises(refiner, monkeypatch):
    words = _term_words()
    targets = _term_target(words)

    def bad_decode(*a, **k):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setattr(refine_mod.subprocess, "run", bad_decode)
    out, stats = refiner.refine(words, targets, ["self attention"], "en", FakeScorer(), SAMPLES)
    assert out == words and stats["status"].startswith("skipped: ")


def test_refine_unexpected_batch_error_is_recorded_not_raised(refiner, monkeypatch):
    words = _term_words()
    targets = _term_target(words)
    monkeypatch.setattr(refiner, "_run_batch", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    out, stats = refiner.refine(words, targets, ["self attention"], "en", FakeScorer(), SAMPLES)
    assert out == words and stats["status"].startswith("skipped: ")


def test_refine_audio_window_spans_the_context_words(refiner, monkeypatch):
    words = _term_words()  # we(0) use(1) self(2) tension(3) here(4): times [i, i + 0.5]
    targets = _term_target(words)
    t = targets[0]
    assert (t["left_start"], t["right_end"]) == (0.0, 4.5)  # first left-context word .. last right-context word
    install(monkeypatch, answer({0: "self attention"}))
    scorer = FakeScorer(lambda t_: -2.0 if "self attention" in t_ else -5.0)
    refiner.refine(words, targets, ["self attention"], "en", scorer, SAMPLES)
    assert scorer.calls and all((start, end) == (0.0, 4.5) for start, end, _ in scorer.calls)
    assert all(text.startswith("we use") and text.endswith("here") for _, _, text in scorer.calls)


def test_refine_audio_window_falls_back_to_target_times_without_context_times(refiner, monkeypatch):
    words = _term_words()
    targets = _term_target(words)
    targets[0]["left_start"] = targets[0]["right_end"] = None  # e.g. a target built without context timing
    install(monkeypatch, answer({0: "self attention"}))
    scorer = FakeScorer(lambda t_: -2.0 if "self attention" in t_ else -5.0)
    refiner.refine(words, targets, ["self attention"], "en", scorer, SAMPLES)
    assert all((start, end) == (2.0, 3.5) for start, end, _ in scorer.calls)

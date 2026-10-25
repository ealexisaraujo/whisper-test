"""Tests for the resolution stage (fake scorers; no models)."""

from __future__ import annotations

import numpy as np
import pytest

from transcribe.auto.align import Slot, Span
from transcribe.auto.resolve import resolve
from transcribe.auto.types import Word

SAMPLES = np.zeros(16000 * 10, dtype=np.float32)
MARGIN = 0.05


class FakeScorer:
    """Scores by window text; unknown text -> `default`. `errors` maps text -> exception."""

    def __init__(self, scores, default=-10.0, errors=None):
        self.scores = scores
        self.default = default
        self.errors = errors or {}
        self.calls = []

    def score(self, samples, start, end, text, lang):
        return self.score_detail(samples, start, end, text, lang)[0]

    def score_detail(self, samples, start, end, text, lang):
        self.calls.append((start, end, text))
        if text in self.errors:
            raise self.errors[text]
        return self.scores.get(text, self.default), len(text.split())

    def release(self):
        pass


def W(text, start, end, prob=0.9):
    return Word(text, start, end, prob)


def disputed(a="eat", b="heat", sid=0, left="I", right="it", start=1.0, end=1.5, tier="disputed"):
    slot = Slot(0, "substitute", W(a, start, end) if a else None, W(b, start, end) if b else None)
    if a is None:
        slot.kind = "b_only"
    elif b is None:
        slot.kind = "a_only"
    return Span(sid, 0, [slot], tier, left, right)


def win(left, cand, right):
    return " ".join(f"{left} {cand} {right}".split())


def one(spans, picks, scorer, **kw):
    return resolve(spans, picks, scorer, SAMPLES, "en", **kw)


# ---- table rows ---------------------------------------------------------------------------


def test_pick_confirmed_within_margin():
    sc = FakeScorer({win("I", "eat", "it"): -5.00, win("I", "heat", "it"): -4.97})  # other better by 0.03
    (w,) = one([disputed()], {0: "a"}, sc)
    assert (w.text, w.tier, w.source, w.unclear) == ("eat", "resolved", "pick", False)
    assert w.confidence == 0.7


def test_verify_overrides_pick():
    sc = FakeScorer({win("I", "eat", "it"): -9.0, win("I", "heat", "it"): -5.0})
    (w,) = one([disputed()], {0: "a"}, sc)
    assert (w.text, w.tier, w.source, w.unclear) == ("heat", "resolved", "verify", False)
    assert w.confidence == 0.6


@pytest.mark.parametrize("picks", [None, {0: "unclear"}, {}])
def test_unclear_picker_verifier_decides(picks):
    sc = FakeScorer({win("I", "eat", "it"): -9.0, win("I", "heat", "it"): -5.0})
    (w,) = one([disputed()], picks, sc)
    assert (w.text, w.tier, w.source, w.unclear) == ("heat", "resolved", "verify", False)
    assert w.alternatives == []


def test_unclear_picker_close_scores_marks_unclear_with_alternative():
    sc = FakeScorer({win("I", "eat", "it"): -5.00, win("I", "heat", "it"): -5.03})
    (w,) = one([disputed()], {0: "unclear"}, sc)
    assert (w.text, w.tier, w.unclear) == ("eat", "unclear", True)
    assert w.alternatives == ["heat"]
    assert w.confidence == 0.3


def test_scorer_none_uses_picker_choice_unclear():
    (w,) = one([disputed()], {0: "b"}, None)
    assert (w.text, w.tier, w.unclear, w.confidence) == ("heat", "unclear", True, 0.3)


def test_scorer_none_without_pick_uses_default_side():
    (w,) = one([disputed()], None, None, default_side="b")
    assert (w.text, w.unclear) == ("heat", True)
    (w,) = one([disputed()], None, None)
    assert w.text == "eat"


# ---- extras -------------------------------------------------------------------------------


def test_solid_passthrough():
    slot = Slot(0, "agree", W("hello", 0.0, 0.4, 0.9), W("hello", 0.0, 0.5, 0.7))
    span = Span(0, 0, [slot], "solid")
    (w,) = one([span], None, None)
    assert (w.text, w.start, w.end, w.tier, w.source, w.unclear) == ("hello", 0.0, 0.4, "solid", "agree", False)
    assert w.confidence == pytest.approx(0.8)


def test_solid_agree_with_b_none_and_a_none():
    s1 = Span(0, 0, [Slot(0, "agree", W(".", 0.0, 0.0), None)], "solid")
    s2 = Span(1, 0, [Slot(0, "agree", None, W("ok", 1.0, 1.2))], "solid")
    out = one([s1, s2], None, None)
    assert [w.text for w in out] == [".", "ok"]
    assert all(w.source == "agree" and w.tier == "solid" for w in out)


def test_missing_span_empty_winner_drops_words():
    span = disputed("um", None, tier="missing")
    sc = FakeScorer({win("I", "um", "it"): -9.0, win("I", "", "it"): -5.0})
    assert one([span], {0: "b"}, sc) == []


def test_missing_span_nonempty_winner_keeps_words():
    span = Span(0, 0, [Slot(0, "b_only", None, W("um", 1, 1.2)), Slot(0, "b_only", None, W("uh", 1.2, 1.4))],
                "missing", "I", "it")
    sc = FakeScorer({win("I", "", "it"): -9.0, win("I", "um uh", "it"): -5.0})
    out = one([span], {0: "b"}, sc)
    assert [w.text for w in out] == ["um", "uh"]


def test_window_text_identical_context_and_time_union():
    span = Span(0, 0, [Slot(0, "substitute", W("eat", 1.0, 1.4), W("heat", 1.1, 1.8))], "disputed",
                " I   want ", "it  now")
    sc = FakeScorer({})
    one([span], None, sc)
    assert sc.calls[0][:2] == (1.0, 1.8)
    assert [c[2] for c in sc.calls] == ["I want eat it now", "I want heat it now"]


def test_multiword_span_emits_winning_side_words_with_own_times():
    span = Span(0, 0, [
        Slot(0, "substitute", W("a", 1.0, 1.2), W("the", 1.0, 1.1)),
        Slot(0, "substitute", W("lot", 1.2, 1.5), W("lots", 1.1, 1.6)),
    ], "disputed", "", "")
    sc = FakeScorer({"a lot": -9.0, "the lots": -5.0})
    out = one([span], {0: "a"}, sc)
    assert [(w.text, w.start, w.end) for w in out] == [("the", 1.0, 1.1), ("lots", 1.1, 1.6)]


def test_unclear_alternatives_on_first_word_only_and_all_words_unclear():
    span = Span(0, 0, [
        Slot(0, "substitute", W("a", 1.0, 1.2), W("the", 1.0, 1.1)),
        Slot(0, "substitute", W("lot", 1.2, 1.5), W("lots", 1.1, 1.6)),
    ], "disputed", "x", "y")
    sc = FakeScorer({"x a lot y": -5.0, "x the lots y": -5.0})
    out = one([span], None, sc)
    assert [w.text for w in out] == ["a", "lot"]
    assert [w.unclear for w in out] == [True, True]
    assert out[0].alternatives == ["the lots"] and out[1].alternatives == []


def test_neighbour_times_used_when_span_words_are_untimed():
    nan = float("nan")
    left = Span(0, 0, [Slot(0, "agree", W("I", 0.0, 0.5), W("I", 0.0, 0.5))], "solid")
    mid = Span(1, 0, [Slot(0, "substitute", W("eat", nan, nan), W("heat", nan, nan))], "disputed", "I", "it")
    right = Span(2, 0, [Slot(0, "agree", W("it", 2.0, 2.3), W("it", 2.0, 2.3))], "solid")
    sc = FakeScorer({})
    one([left, mid, right], None, sc)
    assert sc.calls[0][:2] == (0.5, 2.0)


# ---- failure handling ---------------------------------------------------------------------


def test_value_error_on_one_span_falls_to_unverified_unclear_and_others_continue():
    s0 = disputed("eat", "heat", 0, "I", "it", 1.0, 1.5)
    s1 = disputed("cat", "hat", 1, "a", "sat", 3.0, 3.5)
    sc = FakeScorer(
        {win("a", "cat", "sat"): -9.0, win("a", "hat", "sat"): -5.0},
        errors={win("I", "eat", "it"): ValueError("degenerate")},
    )
    out = one([s0, s1], {0: "b", 1: "a"}, sc)
    assert (out[0].text, out[0].tier, out[0].unclear) == ("heat", "unclear", True)  # picker choice, unverified
    assert (out[1].text, out[1].source) == ("hat", "verify")


def test_wholesale_scorer_failure_on_first_span_propagates():
    sc = FakeScorer({}, errors={win("I", "eat", "it"): RuntimeError("load failed")})
    with pytest.raises(RuntimeError):
        one([disputed()], None, sc)


def test_later_non_value_error_is_per_span():
    s0 = disputed("eat", "heat", 0, "I", "it", 1.0, 1.5)
    s1 = disputed("cat", "hat", 1, "a", "sat", 3.0, 3.5)
    sc = FakeScorer({}, errors={win("a", "cat", "sat"): RuntimeError("boom")})
    out = one([s0, s1], {1: "a"}, sc)
    assert out[1].text == "cat" and out[1].unclear is True


def test_empty_window_is_not_scored_and_goes_unverified():
    empty = disputed("um", None, 0, tier="missing", left="", right="")
    ok = disputed("cat", "hat", 1, "a", "sat", 3.0, 3.5)  # scores fine, so verify is not a no-op
    sc = FakeScorer({})
    out = one([empty, ok], {0: "a"}, sc)
    assert all("um" not in c[2] for c in sc.calls) and len(sc.calls) == 2
    assert (out[0].text, out[0].unclear) == ("um", True)


# ---- single-engine empty-side guard -------------------------------------------------------


def single(tier="disputed"):
    return disputed("wurd", None, 0, "the", "here", tier=tier)


def test_single_engine_empty_wins_only_with_double_margin():
    keep, drop = win("the", "wurd", "here"), win("the", "", "here")
    # empty better by 1.5 margin: not enough -> word kept
    sc = FakeScorer({keep: -5.0, drop: -5.0 + 1.5 * MARGIN})
    (w,) = one([single()], {0: "b"}, sc)
    assert w.text == "wurd" and w.source == "verify" and w.tier == "resolved"
    # empty better by 2 margin: dropped
    sc = FakeScorer({keep: -5.0, drop: -5.0 + 2.0 * MARGIN + 1e-9})
    assert one([single()], {0: "b"}, sc) == []


def test_single_engine_picker_keeps_word_even_if_empty_better_below_double_margin():
    keep, drop = win("the", "wurd", "here"), win("the", "", "here")
    sc = FakeScorer({keep: -5.0, drop: -5.0 + 1.5 * MARGIN})  # verifier would override under 1x margin
    (w,) = one([single()], {0: "a"}, sc)
    assert (w.text, w.source) == ("wurd", "pick")


def test_single_engine_undecided_empty_higher_below_double_margin_is_unclear_keep():
    keep, drop = win("the", "wurd", "here"), win("the", "", "here")
    sc = FakeScorer({keep: -5.0, drop: -5.0 + 1.5 * MARGIN})
    (w,) = one([single()], None, sc)
    assert (w.text, w.tier, w.unclear, w.alternatives) == ("wurd", "unclear", True, [""])


def test_single_engine_unverified_drop_is_not_honoured():
    (w,) = one([single()], {0: "b"}, None)
    assert w.text == "wurd" and w.unclear is True


# ---- unclear with an empty side (two-engine missing span) ---------------------------------


def ghost_span():
    return disputed("ghost", None, 0, "the", "here", tier="missing")


def test_missing_span_unclear_empty_side_higher_emits_no_words():
    sc = FakeScorer({win("the", "ghost", "here"): -5.03, win("the", "", "here"): -5.00})
    assert one([ghost_span()], {0: "unclear"}, sc) == []


def test_missing_span_unclear_word_side_higher_emitted_unclear_with_empty_alternative():
    sc = FakeScorer({win("the", "ghost", "here"): -5.00, win("the", "", "here"): -5.03})
    (w,) = one([ghost_span()], None, sc)
    assert (w.text, w.tier, w.unclear, w.alternatives) == ("ghost", "unclear", True, [""])


# ---- all-ValueError scorer ----------------------------------------------------------------


def _bad_spans(n):
    return [disputed("eat", "heat", i, "I", "it", 1.0 + i, 1.5 + i) for i in range(n)]


def test_all_value_errors_after_five_spans_reraise():
    sc = FakeScorer({}, errors={win("I", "eat", "it"): ValueError("degenerate")})
    with pytest.raises(ValueError):
        one(_bad_spans(8), None, sc)
    assert len(sc.calls) == 5  # stopped at the fifth failure


def test_all_value_errors_with_fewer_than_five_spans_reraise_at_end():
    sc = FakeScorer({}, errors={win("I", "eat", "it"): ValueError("degenerate")})
    with pytest.raises(ValueError):
        one(_bad_spans(3), None, sc)


def test_value_errors_do_not_reraise_once_a_span_scored():
    spans = _bad_spans(7)
    spans[0] = disputed("cat", "hat", 0, "a", "sat", 1.0, 1.5)
    sc = FakeScorer({}, errors={win("I", "eat", "it"): ValueError("degenerate")})
    out = one(spans, None, sc)
    assert len(out) == 7


# ---- verify window covers the context words ------------------------------------------------


def test_verify_window_spans_first_left_context_word_to_last_right_context_word():
    span = disputed(left="I really", right="it now", start=3.0, end=3.5)
    span.left_context_start = 1.2   # start of "I"
    span.right_context_end = 5.1    # end of "now"
    sc = FakeScorer({win("I really", "eat", "it now"): -5.0, win("I really", "heat", "it now"): -9.0})
    one([span], None, sc)
    assert sc.calls and all((start, end) == (1.2, 5.1) for start, end, _ in sc.calls)


def test_verify_window_falls_back_to_span_times_when_context_is_untimed():
    span = disputed(left="I", right="it", start=3.0, end=3.5)  # no context times set
    sc = FakeScorer({})
    one([span], None, sc)
    assert all((start, end) == (3.0, 3.5) for start, end, _ in sc.calls)
    span.left_context_start = float("nan")
    span.right_context_end = float("inf")
    sc2 = FakeScorer({})
    one([span], None, sc2)
    assert all((start, end) == (3.0, 3.5) for start, end, _ in sc2.calls)


def test_verify_window_never_shrinks_below_the_span():
    span = disputed(start=3.0, end=3.5)
    span.left_context_start = 3.2  # inconsistent context times must not cut the span
    span.right_context_end = 3.3
    sc = FakeScorer({})
    one([span], None, sc)
    assert all((start, end) == (3.0, 3.5) for start, end, _ in sc.calls)

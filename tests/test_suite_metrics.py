"""Tests for benchmarks/suite/metrics.py (pure functions, no models)."""

import pytest

from benchmarks.suite import metrics
from benchmarks.suite.metrics import (bootstrap_ci, der, edit_effects, entity_number_error,
                                      insertions_per_minute, normalize, spearman, wer_cer)
from benchmarks.suite.metrics import unclear_quality as _unclear_quality_full


def unclear_quality(ref, words, lang):
    """Ratio keys only; the count keys are tested in test_unclear_quality_counts."""
    full = _unclear_quality_full(ref, words, lang)
    return {k: full[k] for k in ("precision", "recall", "density")}


def test_unclear_quality_counts():
    words = [{"text": "a", "unclear": False}, {"text": "x", "unclear": True}, {"text": "c", "unclear": True},
             {"text": "d", "unclear": False}]
    out = _unclear_quality_full("a b c e", words, "en")
    assert (out["flagged"], out["flagged_errors"], out["errors"], out["words"]) == (2, 1, 2, 4)
    assert out["precision"] == 0.5 and out["recall"] == 0.5 and out["density"] == 0.5


def test_wer_identical_is_zero():
    r = wer_cer(["The cat sat on the mat."], ["the cat sat on the mat"], "en")
    assert r["wer"] == 0.0 and r["cer"] == 0.0
    assert (r["sub"], r["del"], r["ins"]) == (0, 0, 0)
    assert r["words"] == 6


def test_wer_counts_errors():
    r = wer_cer(["a b c d"], ["a x c"], "en")
    assert (r["sub"], r["del"], r["ins"], r["words"]) == (1, 1, 0, 4)
    assert r["wer"] == pytest.approx(0.5)


def test_wer_empty_ref_does_not_divide_by_zero():
    r = wer_cer([""], ["thanks for watching"], "en")
    assert r["words"] == 0 and r["ins"] == 3 and r["wer"] == 0.0 and r["cer"] == 0.0
    r = wer_cer([""], [""], "none")
    assert r["ins"] == 0 and r["wer"] == 0.0


def test_wer_empty_hyp_counts_deletions():
    r = wer_cer(["a b c"], [""], "en")
    assert r["del"] == 3 and r["wer"] == 1.0


def test_wer_spanish_accents_and_punctuation():
    assert normalize("¿Qué tal, Señor?", "es") == "qué tal señor"
    assert normalize("¿Qué tal, Señor?", "es", strip_accents=True) == "que tal senor"
    assert wer_cer(["¿Qué tal, Señor?"], ["que tal senor"], "es")["wer"] > 0
    r = wer_cer(["¿Qué tal, Señor?"], ["que tal senor"], "es")
    assert r["sub"] == 2


def test_spanish_numbers_compare_equal():
    assert normalize("en dos mil veintiuno", "es") == normalize("en 2021", "es")
    assert normalize("treinta y cinco años", "es") == "35 años"
    assert normalize("1.000 euros", "es") == normalize("mil euros", "es")
    assert normalize("uno", "es") == "uno"  # lone "uno" stays a word


def test_english_numbers_normalized():
    assert normalize("twenty twenty one", "en") == normalize("2021", "en")


def test_entity_number_error():
    ref = "Yesterday Maria paid 300 dollars in Madrid, said Bob."
    out = entity_number_error(ref, "yesterday maria paid 300 dollars in madrid said bob", "en")
    assert out == {"total": 4, "missed": 0}  # Maria, 300, Madrid, Bob
    out = entity_number_error(ref, "yesterday mario paid 400 dollars in madrid said bob", "en")
    assert out == {"total": 4, "missed": 2}


def test_entity_number_words_and_all_caps_ref():
    out = entity_number_error("We sold twenty five units", "we sold 25 units", "en")
    assert out == {"total": 1, "missed": 0}
    # all-caps refs carry no capitalization signal
    assert entity_number_error("HE WENT HOME", "he went home", "en")["total"] == 0


def test_insertions_per_minute_empty_ref():
    assert insertions_per_minute("thanks for watching", 30.0) == 6.0
    assert insertions_per_minute("x", 0.0) == 0.0


def test_edit_effects_counts_help_and_harm():
    assert edit_effects("the cat sat", "the hat sat", "the cat sat", "en") == {"help": 1, "harm": 0, "neutral": 0}
    assert edit_effects("the cat sat", "the cat sat", "the cat sad", "en") == {"help": 0, "harm": 1, "neutral": 0}


def test_edit_effects_neutral_and_regions():
    # wrong -> different wrong word
    assert edit_effects("the cat sat", "the hat sat", "the bat sat", "en") == {"help": 0, "harm": 0, "neutral": 1}
    # two separate regions: one helps, one harms
    res = edit_effects("a b c d e", "a x c d e", "a b c y e", "en")
    assert res == {"help": 1, "harm": 1, "neutral": 0}
    # no change
    assert edit_effects("a b", "a b", "a b", "en") == {"help": 0, "harm": 0, "neutral": 0}
    # inserted junk removed
    assert edit_effects("a b", "a the b", "a b", "en")["help"] == 1
    # empty ref does not crash
    assert edit_effects("", "hi", "", "en")["help"] == 1


def test_unclear_quality():
    words = [{"text": "a", "unclear": False}, {"text": "X", "unclear": True},
             {"text": "c", "unclear": False}, {"text": "d", "unclear": True}]
    out = unclear_quality("a b c d", words, "en")
    assert out == {"precision": 0.5, "recall": 1.0, "density": 0.5}


def test_unclear_quality_degenerate():
    out = unclear_quality("a b", [{"text": "a", "unclear": False}, {"text": "b", "unclear": False}], "en")
    assert out == {"precision": 1.0, "recall": 1.0, "density": 0.0}
    assert unclear_quality("a", [], "en")["density"] == 0.0


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) == 0.0
    assert spearman([1], [1]) == 0.0


def _turns(*spec):
    return [{"start": s, "end": e, "speaker": spk} for s, e, spk in spec]


REF = _turns((0, 5, "A"), (5, 10, "B"), (10, 15, "A"))


def test_der_perfect_and_swapped_labels_zero():
    assert der(REF, _turns((0, 5, "x"), (5, 10, "y"), (10, 15, "x"))) == pytest.approx(0.0)
    assert der(REF, _turns((0, 5, "B"), (5, 10, "A"), (10, 15, "B"))) == pytest.approx(0.0)


def test_der_merges_consecutive_same_speaker_turns():
    ref = _turns((0, 5, "A"), (5, 10, "A"), (10, 15, "B"))
    hyp = _turns((0, 10, "x"), (10, 15, "y"))
    assert der(ref, hyp) == pytest.approx(0.0)


def test_der_detects_errors():
    # hyp misses the middle speaker entirely: 5s of 15s missed (collar disabled)
    hyp = _turns((0, 5, "A"), (10, 15, "A"))
    assert der(REF, hyp, collar=0.0) == pytest.approx(5 / 15, abs=0.01)
    # all one speaker: B region is confusion
    assert der(REF, _turns((0, 15, "A")), collar=0.0) == pytest.approx(5 / 15, abs=0.01)


def test_der_collar_forgives_boundary_drift():
    hyp = _turns((0, 5.2, "A"), (5.2, 10, "B"), (10, 15, "A"))
    assert der(REF, hyp, collar=0.0) > 0.01
    assert der(REF, hyp, collar=0.25) == pytest.approx(0.0)


def test_der_overlap_counts_both_speakers():
    ref = _turns((0, 10, "A"), (0, 10, "B"))
    assert der(ref, _turns((0, 10, "x")), collar=0.0) == pytest.approx(0.5, abs=0.01)  # one speaker missed


def test_der_greedy_fallback(monkeypatch):
    monkeypatch.setattr(metrics, "_linear_sum_assignment", None)
    assert der(REF, _turns((0, 5, "x"), (5, 10, "y"), (10, 15, "x"))) == pytest.approx(0.0)
    assert der(REF, _turns((0, 15, "A")), collar=0.0) == pytest.approx(5 / 15, abs=0.01)


def test_der_empty_inputs():
    assert der([], []) == 0.0
    assert der([], _turns((0, 1, "a"))) == 1.0
    assert der(REF, [], collar=0.0) == pytest.approx(1.0, abs=0.01)


def test_bootstrap_ci_contains_mean():
    values = [0.1, 0.2, 0.15, 0.3, 0.25, 0.18, 0.22, 0.12]
    lo, hi = bootstrap_ci(values)
    assert lo <= sum(values) / len(values) <= hi
    assert bootstrap_ci(values) == bootstrap_ci(values)  # deterministic for a seed
    assert bootstrap_ci([0.5, 0.5, 0.5]) == (0.5, 0.5)
    with pytest.raises(ValueError):
        bootstrap_ci([])


# --- fix round 1 ---


def test_entity_spanish_connector_y_keeps_number_together():
    ref = "Pagué treinta y cinco pesos en México."
    assert entity_number_error(ref, "pagué 35 pesos en méxico", "es") == {"total": 2, "missed": 0}
    assert entity_number_error(ref, "pagué 36 pesos en mexico", "es") == {"total": 2, "missed": 2}
    # "y" elsewhere does not join anything
    assert entity_number_error("Ana y Luis compraron 3 casas.", "ana y luis compraron 3 casas", "es") \
        == {"total": 2, "missed": 0}


def test_unclear_quality_number_words_and_contractions_are_not_errors():
    words = [{"text": "I", "unclear": False}, {"text": "paid", "unclear": False},
             {"text": "twenty", "unclear": False}, {"text": "five", "unclear": False},
             {"text": "don't", "unclear": False}, {"text": "mind", "unclear": False}]
    out = unclear_quality("I paid 25 do not mind", words, "en")
    assert out == {"precision": 1.0, "recall": 1.0, "density": 0.0}
    # flagging one word of a merged number flags the merged token (an error here: ref says 26)
    words[3]["unclear"] = True
    out = unclear_quality("I paid 26 do not mind", words, "en")
    assert out["precision"] == 1.0 and out["recall"] == 1.0
    assert out["density"] == pytest.approx(1 / 6)  # joined transcript: i paid 26 do not mind


def test_unclear_quality_flagged_filler_is_ignored_consistently():
    words = [{"text": "a", "unclear": False}, {"text": "um", "unclear": True}, {"text": "X", "unclear": True},
             {"text": "c", "unclear": False}]
    out = unclear_quality("a b c", words, "en")
    assert out == {"precision": 1.0, "recall": 1.0, "density": pytest.approx(1 / 3)}


def test_unclear_quality_spanish_number_run():
    words = [{"text": t, "unclear": False} for t in "pagué treinta y cinco pesos".split()]
    assert unclear_quality("pagué 35 pesos", words, "es") == {"precision": 1.0, "recall": 1.0, "density": 0.0}


def test_edit_effects_multiword_region_counts_once():
    assert edit_effects("a b c d", "a x y d", "a b c d", "en") == {"help": 1, "harm": 0, "neutral": 0}
    assert edit_effects("a b c d", "a b c d", "a x y d", "en") == {"help": 0, "harm": 1, "neutral": 0}
    assert edit_effects("a b c d", "a x y d", "a q r d", "en") == {"help": 0, "harm": 0, "neutral": 1}
    # an unchanged word still separates regions
    assert edit_effects("a b c d e", "a x c y e", "a b c d e", "en") == {"help": 2, "harm": 0, "neutral": 0}


def test_spanish_decimals_and_digit_by_digit_numbers():
    assert normalize("0,125 litros", "es") == "0 125 litros"  # decimal comma not glued into 125 thousand
    assert normalize("1.000.000 de euros", "es") == "1000000 de euros"
    assert normalize("tres cuatro", "es") == "3 4"
    assert normalize("treinta y cinco", "es") == "35"
    assert normalize("ciento veinte mil", "es") == "120000"
    assert normalize("dos millones trescientos mil", "es") == "2300000"
    assert normalize("tres mil", "es") == "3000"

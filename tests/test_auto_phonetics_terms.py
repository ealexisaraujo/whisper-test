"""Tests for the phonetic distance (CMUdict / Spanish rules) and the metadata term list."""

from __future__ import annotations

import pytest

from transcribe.auto.phonetics import phonemes, phonetic_distance
from transcribe.auto.terms import build_terms


# ---------------------------------------------------------------- phonetics


def test_phonetic_distance_en_close():
    assert phonetic_distance("self tension", "self attention", "en") < 0.35
    assert phonetic_distance("attention", "banana", "en") > 0.6


def test_phonetic_distance_es_seseo():
    assert phonetic_distance("casa", "caza", "es") == 0.0


def test_phonemes_en_strips_stress_and_joins_words():
    out = phonemes("self tension", "en")
    assert out == "S EH L F T EH N SH AH N"
    assert not any(ch.isdigit() for ch in out)


def test_phonemes_en_handles_punctuation_case_and_apostrophe():
    assert phonemes("Don't!", "en") == phonemes("don't", "en")
    assert phonemes("Don't", "en").split() == ["D", "OW", "N", "T"]


def test_phonemes_en_oov_uses_metaphone_symbols():
    out = phonemes("Karpathy", "en")
    assert out and out.replace(" ", "").isalpha() and out == out.upper()


def test_phonetic_distance_en_oov_vs_split_words_is_small():
    assert phonetic_distance("Karpathy", "car pathy", "en") < 0.5


def test_phonetic_distance_identical_and_empty():
    assert phonetic_distance("hello world", "Hello, world!", "en") == 0.0
    assert phonetic_distance("", "", "en") == 0.0
    assert phonetic_distance("", "casa", "es") == 1.0


def test_phonetic_distance_en_digits_kept_as_characters():
    assert phonetic_distance("gpt 4", "gpt 4", "en") == 0.0
    assert phonetic_distance("gpt 4", "gpt 5", "en") > 0.0


@pytest.mark.parametrize(
    "a,b",
    [("vaca", "baca"), ("hola", "ola"), ("llave", "yave"), ("queso", "keso"), ("gente", "jente"), ("cero", "sero")],
)
def test_phonetic_distance_es_rules_equivalences(a, b):
    assert phonetic_distance(a, b, "es") == 0.0


def test_phonetic_distance_es_distinguishes_n_and_enye():
    assert phonetic_distance("cana", "caña", "es") > 0.0


def test_phonetic_distance_es_word_boundary_split_is_close():
    assert phonetic_distance("a plazar", "aplazar", "es") == 0.0
    assert phonetic_distance("aplazar", "aplazar", "es") == 0.0
    assert phonetic_distance("aplazar", "mañana", "es") > 0.6


def test_phonemes_es_accents_stripped_enye_kept_rr_collapsed():
    assert phonemes("Árbol", "es") == phonemes("arbol", "es")
    assert "ñ" in phonemes("mañana", "es")
    assert phonemes("perro", "es") == phonemes("pero", "es")
    assert phonemes("hacha", "es") == phonemes("acha", "es")


def test_phonetic_distance_unsupported_lang_does_not_crash():
    assert phonetic_distance("hallo", "hallo", "de") == 0.0
    assert 0.0 < phonetic_distance("hallo", "hello", "de") <= 1.0


def test_phonetic_distance_is_normalized_0_to_1():
    for a, b in [("a", "zzzzzzzz"), ("self", "tension"), ("casa", "perro")]:
        for lang in ("en", "es"):
            assert 0.0 <= phonetic_distance(a, b, lang) <= 1.0


# ---------------------------------------------------------------- terms


DESC = (
    "In this talk Andrej Karpathy explores how software is evolving. We built MenuGen with LLMs "
    "and compare it to GPT-4. Slides: https://example.com/slides?x=1 #AI @karpathy"
)


def test_build_terms_extracts_names_products():
    terms = build_terms({"title": "Andrej Karpathy: Software Is Changing (Again)", "description": DESC})
    for expected in ("Andrej Karpathy", "MenuGen", "LLMs", "GPT-4"):
        assert expected in terms
    assert not any("http" in t or "example.com" in t for t in terms)
    assert not any(t.startswith("#") or t.startswith("@") for t in terms)
    assert "AI" not in terms  # only appears as a hashtag


def test_build_terms_title_case_title_does_not_make_phrase_names():
    terms = build_terms({"title": "Software Is Changing (Again)"})
    assert "Software Is Changing" not in terms
    assert "Software" not in terms
    assert terms == []


def test_build_terms_accepts_platform_prefixed_keys():
    terms = build_terms({"youtube_title": "Andrej Karpathy: Software Is Changing", "youtube_description": "Using MenuGen."})
    assert "Andrej Karpathy" in terms and "MenuGen" in terms


def test_build_terms_missing_keys_and_none_values():
    assert build_terms({}) == []
    assert build_terms({"title": None, "description": None, "chapters": None, "tags": None}) == []
    assert build_terms({"title": None}, extra=None) == []
    assert build_terms({"chapters": [None, {"title": None}, {}, 5]}) == []


def test_build_terms_chapters_and_tags():
    meta = {
        "title": "Talk",
        "chapters": [{"title": "Intro", "start_time": 0}, {"title": "Using Claude Code", "start_time": 30}, "Demo of OpenAI"],
        "tags": ["Andrej Karpathy", "machine learning", "LLMs", "#shorts"],
    }
    terms = build_terms(meta)
    assert "Claude Code" in terms
    assert "OpenAI" in terms
    assert "Andrej Karpathy" in terms
    assert "LLMs" in terms
    assert "machine learning" not in terms
    assert "#shorts" not in terms


def test_build_terms_excludes_sentence_initial_single_words_and_pure_numbers():
    terms = build_terms({"description": "Today we talk. Welcome back everyone. Version 2024 shipped 12:30 in 2025. Tuesday Update"})
    assert terms == []


def test_build_terms_sentence_initial_multiword_drops_first_word_only_when_long():
    terms = build_terms({"description": "Welcome Andrej Karpathy to the show. Andrej Karpathy spoke."})
    assert "Andrej Karpathy" in terms
    assert "Welcome Andrej Karpathy" not in terms


def test_build_terms_non_initial_names_and_three_word_names():
    terms = build_terms({"description": "We met with Sam Altman and Jean Luc Picard yesterday, plus Open Source Initiative Foundation."})
    assert "Sam Altman" in terms
    assert "Jean Luc Picard" in terms
    assert not any(t == "Open Source Initiative Foundation" for t in terms)


def test_build_terms_stopwords_break_name_runs_spanish():
    terms = build_terms({"description": "Hablamos con Gabriel Garcia de La Torre y El Pais sobre la IA."})
    assert "Gabriel Garcia" in terms
    assert "La Torre" not in terms and "De La Torre" not in terms
    assert "El Pais" not in terms
    assert "IA" in terms


def test_build_terms_digit_tokens_and_unit_like_excluded():
    terms = build_terms({"description": "We use GPT-4o and Claude3 at 5pm, the 3rd of May, 10x faster, v2 and H264."})
    assert "GPT-4o" in terms and "Claude3" in terms and "H264" in terms
    assert "5pm" not in terms and "3rd" not in terms and "10x" not in terms


def test_build_terms_excludes_handles_urls_hashtags_emails():
    terms = build_terms({"description": "Follow @SomeHandle or www.site.org/ABC or mail Foo@Bar.com or #CamelTag https://x.com/AbC"})
    assert terms == []


def test_build_terms_dedupe_case_insensitive_keeps_first_casing_and_order():
    terms = build_terms({"title": "MenuGen and menugen", "description": "MENUGEN llms LLMs LLMs and MenuGen. GPT-4 gpt-4"}, extra=["menugen"])
    lowered = [t.lower() for t in terms]
    assert len(lowered) == len(set(lowered))
    assert terms[0] == "menugen"  # extras first, first-seen casing kept
    assert terms.index("LLMs") < terms.index("GPT-4")


def test_build_terms_deterministic_order_of_first_appearance():
    meta = {"description": "Use Zeta9 then AlphaBeta then NASA."}
    assert build_terms(meta) == ["Zeta9", "AlphaBeta", "NASA"]
    assert build_terms(meta) == build_terms(meta)


def test_build_terms_extras_included_verbatim_and_stripped():
    terms = build_terms({}, extra=["  Qwen3-ASR  ", "", "  ", "karpathy"])
    assert terms == ["Qwen3-ASR", "karpathy"]


def test_build_terms_cap_200():
    desc = " ".join(f"Tool{i}Name" for i in range(500))
    terms = build_terms({"description": desc})
    assert len(terms) == 200
    assert terms[0] == "Tool0Name"
    extras = [f"extra{i}" for i in range(250)]
    assert len(build_terms({"description": desc}, extra=extras)) == 200


def test_build_terms_shouting_text_does_not_flood_acronyms():
    terms = build_terms({"title": "HOW TO BUILD BETTER SOFTWARE", "description": "Uses NASA and HOW TO."})
    assert "NASA" in terms
    assert "BETTER" not in terms and "SOFTWARE" not in terms

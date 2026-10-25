from transcribe.auto.align import (
    Slot,
    Span,
    align_chunk,
    build_spans,
    norm_token,
    single_engine_spans,
    word_confidence,
)
from transcribe.auto.types import EngineOutput, Word


def words(text, prob=None, start=0.0):
    out = []
    for i, tok in enumerate(text.split()):
        out.append(Word(tok, start + i * 0.5, start + i * 0.5 + 0.4, prob))
    return out


def kinds(slots):
    return [s.kind for s in slots]


def test_align_identical_all_agree():
    slots = align_chunk(words("hello big world"), words("hello big world"), "en")
    assert kinds(slots) == ["agree"] * 3
    assert all(s.a is not None and s.b is not None for s in slots)


def test_align_substitution_and_insertion():
    slots = align_chunk(words("the cat sat down"), words("the hat sat"), "en")
    assert kinds(slots) == ["agree", "substitute", "agree", "a_only"]
    assert slots[1].a.text == "cat" and slots[1].b.text == "hat"
    assert slots[3].a.text == "down" and slots[3].b is None


def test_align_ignores_case_punct_accents():
    slots = align_chunk(words("¿Qué tal?"), words("que tal"), "es")
    assert kinds(slots) == ["agree", "agree"]
    assert slots[0].a.text == "¿Qué"  # original characters preserved


def test_align_one_engine_empty_chunk():
    slots = align_chunk(words("a b c"), [], "en")
    assert kinds(slots) == ["a_only"] * 3
    slots = align_chunk([], words("a b"), "en")
    assert kinds(slots) == ["b_only"] * 2
    assert align_chunk([], [], "en") == []


def test_norm_token_numbers_and_punct():
    assert norm_token("Twenty,", "en") == "20"
    assert norm_token("veinte", "es") == "20"
    assert norm_token("Café!", "es") == "cafe"
    assert norm_token("don't", "en") == "dont"
    assert norm_token("¡Hola!", "es") == "hola"
    assert norm_token("7", "en") == "7"


def test_build_spans_merges_consecutive_disputes_and_contexts():
    a = words("one two three four five six seven", prob=None)
    b = words("one two THREEX FOURX five six seven", prob=None)
    slots = {0: align_chunk(a, b, "en")}
    spans = build_spans(slots, context_words=2)
    tiers = [s.tier for s in spans]
    # one two (solid x2), disputed(three,four), five six seven (solid x3)
    assert tiers == ["solid", "solid", "disputed", "solid", "solid", "solid"]
    disputed = spans[2]
    assert len(disputed.slots) == 2
    assert disputed.left_context == "one two"
    assert disputed.right_context == "five six"
    assert [s.id for s in spans] == list(range(6))
    # solid span contexts are filled too
    assert spans[0].left_context == "" and spans[0].right_context == "two three"
    assert spans[5].right_context == ""


def test_build_spans_missing_tier_and_global_ids_across_chunks():
    slots = {
        1: align_chunk(words("x y"), words("x"), "en"),
        0: align_chunk(words("p q"), words("p q"), "en"),
    }
    spans = build_spans(slots)
    assert [s.chunk for s in spans] == [0, 0, 1, 1]
    assert [s.id for s in spans] == [0, 1, 2, 3]
    assert spans[3].tier == "missing"
    assert spans[3].slots[0].kind == "a_only"
    # contexts do not cross chunks
    assert spans[2].left_context == ""


def test_substitute_and_missing_merge_into_one_disputed_span():
    slots = {0: align_chunk(words("a b c"), words("a x"), "en")}
    spans = build_spans(slots)
    non_solid = [s for s in spans if s.tier != "solid"]
    assert len(non_solid) == 1
    assert sorted(s.kind for s in non_solid[0].slots) == ["a_only", "substitute"]
    assert non_solid[0].tier == "disputed"


def test_agree_low_prob_is_not_solid():
    a = [Word("hi", 0, 0.3, 0.95), Word("there", 0.3, 0.6, 0.9)]
    b = [Word("hi", 0, 0.3, 0.95), Word("there", 0.3, 0.6, 0.2)]
    spans = build_spans({0: align_chunk(a, b, "en")})
    assert [s.tier for s in spans] == ["solid", "disputed"]
    assert spans[1].slots[0].kind == "agree"
    # A-side prob None is ignored; only available probs count
    a2 = [Word("hi", 0, 0.3, None)]
    b2 = [Word("hi", 0, 0.3, 0.4)]
    assert build_spans({0: align_chunk(a2, b2, "en")})[0].tier == "disputed"
    b3 = [Word("hi", 0, 0.3, 0.5)]
    assert build_spans({0: align_chunk(a2, b3, "en")})[0].tier == "solid"


def test_word_confidence():
    w = lambda p: Word("x", 0, 1, p)
    assert word_confidence(Slot(0, "agree", w(None), w(None))) == 0.9
    assert word_confidence(Slot(0, "agree", w(None), w(0.6))) == 0.6
    assert abs(word_confidence(Slot(0, "agree", w(0.8), w(0.6))) - 0.7) < 1e-9
    assert abs(word_confidence(Slot(0, "substitute", w(0.5), w(0.9))) - 0.27) < 1e-9
    assert abs(word_confidence(Slot(0, "substitute", w(None), w(None))) - 0.27) < 1e-9
    assert word_confidence(Slot(0, "a_only", w(0.9), None)) == 0.2
    assert word_confidence(Slot(0, "b_only", None, w(0.9))) == 0.2


def test_single_engine_spans_low_prob_disputed():
    out = EngineOutput(
        engine="whisper-large-v3",
        language="en",
        chunks={
            0: [Word("good", 0, 1, 0.9), Word("bad", 1, 2, 0.1), Word("ugly", 2, 3, 0.2), Word("ok", 3, 4, 0.8)],
            1: [],
            2: [Word("tail", 0, 1, None)],
        },
    )
    spans = single_engine_spans(out, "en")
    assert [s.tier for s in spans] == ["solid", "disputed", "solid", "solid"]
    assert [s.id for s in spans] == [0, 1, 2, 3]
    d = spans[1]
    assert [s.a.text for s in d.slots] == ["bad", "ugly"]
    assert all(s.b is None for s in d.slots)
    assert d.left_context == "good" and d.right_context == "ok"
    assert spans[3].chunk == 2


def test_empty_inputs():
    assert build_spans({}) == []
    assert build_spans({0: []}) == []
    assert single_engine_spans(EngineOutput("x", "en"), "en") == []


def test_non_latin_scripts_do_not_collapse():
    cyr = align_chunk(words("привет мир"), words("до свидания"), "ru")
    assert kinds(cyr) == ["substitute", "substitute"]
    assert kinds(align_chunk(words("привет мир"), words("Привет мир!"), "ru")) == ["agree", "agree"]
    cjk = align_chunk(words("你好 世界"), words("再见 朋友"), "zh")
    assert kinds(cjk) == ["substitute", "substitute"]
    assert kinds(align_chunk(words("你好 世界"), words("你好 世界"), "zh")) == ["agree", "agree"]
    assert norm_token("Привет,", "ru") == "привет"
    assert norm_token("Ёлка", "ru") == "елка"  # accents/combining marks still stripped


def test_standalone_punctuation_never_creates_disputes():
    spans = build_spans({0: align_chunk(words("hello — world"), words("hello world"), "en")})
    assert [s.tier for s in spans] == ["solid"] * 3
    # the text is preserved: the dash shows up in the neighbouring context
    assert spans[2].left_context == "hello —"

    both = build_spans({0: align_chunk(words("..."), words("—"), "en")})
    assert [s.tier for s in both] == ["solid"]
    only_a = build_spans({0: align_chunk(words("... hi"), words("hi"), "en")})
    assert [s.tier for s in only_a] == ["solid", "solid"]
    only_b = align_chunk(words("hi"), words("- hi"), "en")
    assert kinds(only_b) == ["agree"]
    # low-prob punctuation is still solid
    a = [Word("hi", 0, 1, 0.9), Word("—", 1, 1.1, 0.01)]
    b = [Word("hi", 0, 1, 0.9)]
    assert [s.tier for s in build_spans({0: align_chunk(a, b, "en")})] == ["solid", "solid"]


def test_single_engine_punctuation_solid_and_context_kwarg():
    out = EngineOutput(
        "w", "en", {0: [Word("a", 0, 1, 0.9), Word("—", 1, 1.1, 0.0), Word("b", 2, 3, 0.1), Word("c", 3, 4, 0.9), Word("d", 4, 5, 0.9)]}
    )
    spans = single_engine_spans(out, "en", context_words=1)
    assert [s.tier for s in spans] == ["solid", "solid", "disputed", "solid", "solid"]
    assert spans[2].left_context == "—" and spans[2].right_context == "c"


def test_spans_carry_the_time_extent_of_their_context_words():
    a = words("one two three four five six")
    b = words("one two three fore five six")
    spans = build_spans({0: align_chunk(a, b, "en")}, context_words=2)
    disputed = next(sp for sp in spans if sp.tier == "disputed")
    assert disputed.left_context == "two three" and disputed.right_context == "five six"
    assert disputed.left_context_start == a[1].start      # start of "two"
    assert disputed.right_context_end == a[5].end         # end of "six"
    first = spans[0]
    assert first.left_context == "" and first.left_context_start is None
    last = spans[-1]
    assert last.right_context == "" and last.right_context_end is None

"""Tests for the local picker (fake `generate`; the real LLM only runs in the slow test)."""

from __future__ import annotations

import json
import os
import random
import re
import time

import pytest

from transcribe.auto.align import Slot, Span
from transcribe.auto.picker import MlxPicker, Picker, parse_choices, span_text
from transcribe.auto.types import Word


def W(text: str) -> Word:
    return Word(text, 0.0, 0.1)


def sub_span(i: int, a: str, b: str, left: str = "left", right: str = "right") -> Span:
    return Span(
        id=i, chunk=0, tier="disputed", left_context=left, right_context=right,
        slots=[Slot(0, "substitute", W(a), W(b))],
    )


LINE = re.compile(r"^(\d+)\. .*?\[(.*?) \| (.*?)\]", re.M)


def shown(prompt: str) -> dict:
    """id -> (option 1, option 2) exactly as the prompt displayed them."""
    return {int(i): (o1, o2) for i, o1, o2 in LINE.findall(prompt)}


class Fake:
    """Records calls and answers with `answer(prompt) -> text`."""

    def __init__(self, answer):
        self.answer = answer
        self.prompts = []
        self.max_tokens = []

    def __call__(self, prompt: str, max_tokens: int) -> str:
        self.prompts.append(prompt)
        self.max_tokens.append(max_tokens)
        return self.answer(prompt)


def reply(prompt_to_choices) -> "callable":
    return lambda prompt: json.dumps({"choices": prompt_to_choices(prompt)})


def test_span_text_one_sided():
    span = Span(
        id=0, chunk=0, tier="disputed",
        slots=[
            Slot(0, "agree", W("the"), W("the")),
            Slot(0, "a_only", W("uh"), None),
            Slot(0, "b_only", None, W("cat")),
            Slot(0, "agree", W(","), None),
        ],
    )
    assert span_text(span, "a") == "the uh ,"
    assert span_text(span, "b") == "the cat"
    only_a = Span(id=1, chunk=0, tier="disputed", slots=[Slot(0, "a_only", W("maybe"), None)])
    assert span_text(only_a, "b") == ""
    assert span_text(only_a, "a") == "maybe"


def test_picker_maps_shuffled_options_back():
    spans = [sub_span(i, f"alpha{i}", f"beta{i}") for i in range(12)]
    fake = Fake(reply(lambda p: [{"id": i, "choice": "1"} for i in shown(p)]))
    result = MlxPicker(generate=fake, seed=7).pick(spans, "en", [])
    assert set(result) == set(range(12))
    sides_seen = set()
    for prompt in fake.prompts:
        for i, (first, _second) in shown(prompt).items():
            expected = "a" if first == f"alpha{i}" else "b"
            assert result[i] == expected
            sides_seen.add(expected)
    assert sides_seen == {"a", "b"}  # shuffling actually happens


def test_picker_shuffle_is_deterministic_per_span_id():
    spans = [sub_span(i, f"alpha{i}", f"beta{i}") for i in range(6)]
    for span in spans:
        flipped = random.Random(3 + span.id).random() < 0.5
        fake = Fake(reply(lambda p: [{"id": i, "choice": "1"} for i in shown(p)]))
        out = MlxPicker(generate=fake, seed=3).pick([span], "en", [])
        assert out[span.id] == ("b" if flipped else "a")


def test_picker_choice_two_maps_to_other_side():
    spans = [sub_span(i, f"alpha{i}", f"beta{i}") for i in range(8)]
    fake = Fake(reply(lambda p: [{"id": i, "choice": "2"} for i in shown(p)]))
    result = MlxPicker(generate=fake).pick(spans, "en", [])
    for prompt in fake.prompts:
        for i, (_first, second) in shown(prompt).items():
            assert result[i] == ("a" if second == f"alpha{i}" else "b")


def test_picker_unclear_choice():
    fake = Fake(reply(lambda p: [{"id": i, "choice": "unclear"} for i in shown(p)]))
    assert MlxPicker(generate=fake).pick([sub_span(4, "x", "y")], "en", []) == {4: "unclear"}


@pytest.mark.parametrize(
    "text",
    ["not json at all", "", '{"choices": [', '{"choices": "oops"}', '{"other": 1}', "[1, 2]"],
)
def test_picker_invalid_json_gives_unclear(text):
    spans = [sub_span(i, f"a{i}", f"b{i}") for i in (3, 9)]
    result = MlxPicker(generate=Fake(lambda p: text)).pick(spans, "en", [])
    assert result == {3: "unclear", 9: "unclear"}


def test_picker_parses_fenced_json_with_prose():
    text = 'Sure!\n```json\n{"choices": [{"id": 5, "choice": "1"}]}\n```\nHope that helps {really}.'
    spans = [sub_span(5, "aa", "bb")]
    flipped = random.Random(0 + 5).random() < 0.5
    out = MlxPicker(generate=Fake(lambda p: text)).pick(spans, "en", [])
    assert out == {5: "b" if flipped else "a"}


def test_picker_missing_invalid_and_duplicate_ids_are_unclear():
    spans = [sub_span(i, f"a{i}", f"b{i}") for i in (1, 2, 3, 4)]
    text = json.dumps({"choices": [
        {"id": 1, "choice": "1"},
        {"id": 2, "choice": "maybe"},   # invalid choice value
        {"id": 3, "choice": "1"},
        {"id": 3, "choice": "2"},       # duplicate id
        {"id": 99, "choice": "1"},      # not in the batch
        # id 4 missing
    ]})
    result = MlxPicker(generate=Fake(lambda p: text)).pick(spans, "en", [])
    assert set(result) == {1, 2, 3, 4}
    assert result[1] in ("a", "b")
    assert result[2] == "unclear"
    assert result[3] == "unclear"
    assert result[4] == "unclear"


def test_picker_batches():
    spans = [sub_span(i, f"a{i}", f"b{i}") for i in range(32)]
    fake = Fake(reply(lambda p: [{"id": i, "choice": "1"} for i in shown(p)]))
    result = MlxPicker(generate=fake, batch_size=15).pick(spans, "en", [])
    assert len(fake.prompts) == 3
    assert [len(shown(p)) for p in fake.prompts] == [15, 15, 2]
    assert set(result) == set(range(32))


def test_picker_max_tokens_scales_with_batch():
    spans = [sub_span(i, f"a{i}", f"b{i}") for i in range(3)]
    fake = Fake(lambda p: "{}")
    MlxPicker(generate=fake).pick(spans, "en", [])
    assert fake.max_tokens == [30 * 3 + 50]


def test_picker_prompt_has_language_terms_context_and_policy():
    terms = [f"term{i}" for i in range(80)]
    fake = Fake(lambda p: "{}")
    MlxPicker(generate=fake).pick([sub_span(1, "eat", "heat", "I want to", "lunch now")], "es", terms)
    prompt = fake.prompts[0]
    assert "Transcript language: es" in prompt
    assert "I want to" in prompt and "lunch now" in prompt
    assert "term49" in prompt and "term50" not in prompt
    assert "unclear" in prompt
    assert "filler" in prompt.lower()


def test_picker_identical_normalized_options_skip_llm():
    spans = [sub_span(1, "Hello,", "hello"), sub_span(2, "ok", "okay")]
    fake = Fake(reply(lambda p: [{"id": i, "choice": "unclear"} for i in shown(p)]))
    result = MlxPicker(generate=fake).pick(spans, "en", [])
    assert result[1] == "a"
    assert result[2] == "unclear"
    assert list(shown(fake.prompts[0])) == [2]


def test_picker_all_identical_makes_no_calls():
    fake = Fake(lambda p: "{}")
    assert MlxPicker(generate=fake).pick([sub_span(1, "Hi.", "hi")], "en", []) == {1: "a"}
    assert fake.prompts == []


def test_picker_one_sided_span_offers_nothing_option():
    span = Span(id=2, chunk=0, tier="disputed", left_context="we", right_context="go",
                slots=[Slot(0, "a_only", W("maybe"), None)])
    fake = Fake(reply(lambda p: [{"id": i, "choice": "1"} for i in shown(p)]))
    result = MlxPicker(generate=fake).pick([span], "en", [])
    options = shown(fake.prompts[0])[2]
    assert "maybe" in options and "" not in options
    assert result[2] == ("a" if options[0] == "maybe" else "b")


def test_picker_empty_input_and_release_without_load():
    fake = Fake(lambda p: "{}")
    picker = MlxPicker(generate=fake)
    assert picker.pick([], "en", []) == {}
    picker.release()  # nothing loaded: must not raise
    assert fake.prompts == []


def test_picker_satisfies_protocol():
    assert isinstance(MlxPicker(generate=Fake(lambda p: "{}")), Picker)


def test_parse_choices_skips_placeholder_and_stray_braces():
    good = '{"choices": [{"id": 1, "choice": "2"}]}'
    assert parse_choices("Format {id}: " + good) == {1: "2"}
    assert parse_choices("a stray { in prose, then " + good) == {1: "2"}
    assert parse_choices('{"note": "x"} then ' + good) == {1: "2"}


def test_parse_choices_truncated_output_keeps_complete_entries():
    text = '```json\n{"choices": [{"id": 1, "choice": "1"}, {"id": 2, "choice": "2"}, {"id": 3, "cho'
    assert parse_choices(text) == {1: "1", 2: "2"}


def test_parse_choices_bare_entries_without_wrapper():
    assert parse_choices('{"choice": "1", "id": 7}\n{"id": 8, "choice": "unclear"}') == {7: "1", 8: "unclear"}


@pytest.mark.parametrize(
    "entries, expected",
    [
        ('[{"id": "2", "choice": "1"}]', {2: "1"}),
        ('[{"id": 2, "choice": 2}]', {2: "2"}),
        ('[{"id": 2, "choice": " 1 "}]', {2: "1"}),
        ('[{"id": 2, "choice": "Unclear"}]', {2: "unclear"}),
        ('[{"id": 9, "choice": "2"}, {"id": 1, "choice": "1"}]', {9: "2", 1: "1"}),
        ('[{"id": true, "choice": "1"}, {"id": 1.5, "choice": "1"}, 3]', {}),
    ],
)
def test_parse_choices_value_variants(entries, expected):
    assert parse_choices('{"choices": ' + entries + "}") == expected


def test_picker_truncated_output_only_loses_cut_off_ids():
    spans = [sub_span(i, f"a{i}", f"b{i}") for i in (1, 2, 3)]
    text = '{"choices": [{"id": 1, "choice": "1"}, {"id": 2, "choice": "unclear"}, {"id": 3, "ch'
    result = MlxPicker(generate=Fake(lambda p: text)).pick(spans, "en", [])
    assert result[1] in ("a", "b") and result[2] == "unclear" and result[3] == "unclear"


def test_picker_batch_isolation():
    spans = [sub_span(i, f"alpha{i}", f"beta{i}") for i in range(6)]
    calls = {"n": 0}

    def answer(prompt):
        calls["n"] += 1
        if calls["n"] == 2:
            return "total garbage"
        return json.dumps({"choices": [{"id": i, "choice": "1"} for i in shown(prompt)]})

    result = MlxPicker(generate=Fake(answer), batch_size=2).pick(spans, "en", [])
    assert calls["n"] == 3
    assert result[2] == result[3] == "unclear"
    for i in (0, 1, 4, 5):
        assert result[i] in ("a", "b")


def test_picker_generation_error_propagates():
    def boom(prompt):
        raise RuntimeError("oom")

    with pytest.raises(RuntimeError):
        MlxPicker(generate=Fake(boom)).pick([sub_span(1, "x", "y")], "en", [])


def test_picker_prompt_collapses_whitespace_and_caps_words():
    long_left = "\n".join(f"L{i}" for i in range(100))
    long_right = " ".join(f"R{i}" for i in range(100))
    long_opt = "\n".join(f"w{i}" for i in range(100))
    span = sub_span(1, long_opt, "short\nopt", long_left, long_right)
    fake = Fake(lambda p: "{}")
    MlxPicker(generate=fake).pick([span], "en", [])
    prompt = fake.prompts[0]
    item_lines = [ln for ln in prompt.splitlines() if re.match(r"^\d+\. ", ln)]
    assert len(item_lines) == 1
    line = item_lines[0]
    assert "L99" in line and "L60" in line and "L59" not in line      # left keeps the end
    assert "R0" in line and "R39" in line and "R40" not in line       # right keeps the start
    assert "w0" in line and "w39" in line and "w40" not in line
    assert "short opt" in line


def test_parse_choices_balanced_braces_and_strings():
    text = 'noise {"choices": [{"id": 1, "choice": "1"}], "note": "a } b"} trailing {"x": 1}'
    assert parse_choices(text) == {1: "1"}
    assert parse_choices("```json\n{bad}\n```") == {}


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="RUN_SLOW=1 required")
def test_real_picker_obvious_spans():
    en_spans = [
        sub_span(1, "eat", "heat", "I want to", "lunch now"),
        sub_span(3, "xqzv glorp", "brnt wuzzle", "the", "was"),
        sub_span(4, "ran", "rang", "the dog", "across the field"),
    ]
    es_span = sub_span(2, "aplazar", "a plazar", "el equipo decidió", "el proyecto la semana pasada")
    picker = MlxPicker()
    try:
        t0 = time.perf_counter()
        first = picker.pick([en_spans[0]], "en", [])  # includes the model load
        load_and_first = time.perf_counter() - t0
        t0 = time.perf_counter()
        es = picker.pick([es_span], "es", [])
        t_es = time.perf_counter() - t0
        t0 = time.perf_counter()
        batch = picker.pick(en_spans, "en", [])  # one batch of 3 same-language spans
        t_batch = time.perf_counter() - t0
    finally:
        picker.release()
    print(f"\nPICKER latency: load+first={load_and_first:.1f}s single={t_es:.2f}s batch3={t_batch:.2f}s")
    print("PICKER results:", first, es, batch)
    assert first[1] == "a"
    assert es[2] == "a"
    assert batch[1] == "a"
    assert batch[3] in ("unclear", "a", "b")  # both options are nonsense: anything but a crash
    assert batch[4] == "a"

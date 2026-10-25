"""Tests for the auto pipeline's local engines (models mocked) and hallucination guards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from transcribe.auto import engines
from transcribe.auto.engines import make_engine, spread_words_by_chars
from transcribe.auto.guards import HALLUCINATION_PHRASES, apply_guards
from transcribe.auto.types import Chunk, EngineOutput, Word
from transcribe.models import CapabilityError

SR = 16000
LOUD = 0.05  # linear RMS, about -26 dBFS


def _out(text: str, chunk: int = 0, start: float = 0.0, step: float = 0.4, prob=None) -> EngineOutput:
    words = [Word(t, start + i * step, start + (i + 1) * step - 0.05, prob) for i, t in enumerate(text.split())]
    return EngineOutput(engine="x", language="en", chunks={chunk: words})


def _texts(out: EngineOutput, chunk: int = 0) -> list:
    return [w.text for w in out.chunks[chunk]]


# ----------------------------------------------------------------------------- guards


def test_phrase_list_has_required_entries():
    for phrase in ("thanks for watching", "thank you for watching", "subscribe", "subtitles by",
                   "gracias por ver", "suscríbete", "amara.org"):
        assert phrase in HALLUCINATION_PHRASES


def test_guard_drops_known_phrase_only_chunk():
    out = apply_guards(_out("Thanks for watching!"), {0: LOUD})
    assert out.chunks[0] == []
    assert "known-phrase" in out.flags[0]


def test_guard_collapses_repetition_loop():
    out = apply_guards(_out("we go we go we go we go we go"), {0: LOUD})
    assert _texts(out) == ["we", "go"]
    assert "repetition" in out.flags[0]


def test_guard_repetition_keeps_surrounding_text_and_first_occurrence_times():
    original = _out("so then we go we go we go we go and stop")
    out = apply_guards(original, {0: LOUD})
    assert _texts(out) == ["so", "then", "we", "go", "and", "stop"]
    assert out.chunks[0][2].start == original.chunks[0][2].start


def test_guard_keeps_normal_text():
    original = _out("this is a perfectly normal sentence about subscribe buttons and more things")
    out = apply_guards(original, {0: LOUD})
    assert _texts(out) == _texts(original)
    assert out.flags.get(0, []) == []


def test_guard_keeps_six_repeats_of_one_word():
    out = apply_guards(_out("no " * 6 + "it is fine"), {0: LOUD})
    assert _texts(out) == ["no"] * 6 + ["it", "is", "fine"]
    assert out.flags.get(0, []) == []


def test_guard_collapses_ten_repeats_of_one_word():
    out = apply_guards(_out("so " + "no " * 10 + "it is fine"), {0: LOUD})
    assert _texts(out) == ["so", "no", "it", "is", "fine"]
    assert "repetition" in out.flags[0]


def test_guard_collapses_alternating_loop_of_five_and_keeps_one_pair():
    out = apply_guards(_out("we go " * 5), {0: LOUD})
    assert _texts(out) == ["we", "go"]


def test_guard_keeps_short_legitimate_repeats():
    out = apply_guards(_out("no no no it is fine"), {0: LOUD})
    assert _texts(out) == ["no", "no", "no", "it", "is", "fine"]


@pytest.mark.parametrize("text", ["I subscribe", "did you subscribe", "so subscribe now",
                                  "I will subscribe to the channel", "thanks for watching it"])
def test_guard_keeps_real_speech_containing_phrase_words(text):
    out = apply_guards(_out(text), {0: LOUD})
    assert _texts(out) == text.split()
    assert out.flags.get(0, []) == []


@pytest.mark.parametrize("text", ["subscribe", "Subscribe.", "Thank you for watching.", "Thanks for watching everyone",
                                  "Gracias por ver", "Suscríbete", "Amara.org"])
def test_guard_drops_exact_phrase_chunks(text):
    out = apply_guards(_out(text), {0: LOUD})
    assert out.chunks[0] == []
    assert "known-phrase" in out.flags[0]


def test_guard_single_word_phrase_gets_no_courtesy_tolerance():
    assert _texts(apply_guards(_out("subscribe everyone"), {0: LOUD})) == ["subscribe", "everyone"]


def test_guard_spanish_phrase():
    out = apply_guards(_out("Gracias por ver el video"), {0: LOUD})
    assert out.chunks[0] == []
    assert "known-phrase" in out.flags[0]


def test_guard_drops_silent_chunk():
    out = apply_guards(_out("hello there friend"), {0: 10 ** (-60 / 20)})
    assert out.chunks[0] == []
    assert "low-energy" in out.flags[0]


def test_guard_keeps_chunk_just_above_threshold():
    out = apply_guards(_out("hello there friend"), {0: 10 ** (-45 / 20)})
    assert _texts(out) == ["hello", "there", "friend"]


def test_guard_missing_rms_entry_is_ignored():
    out = apply_guards(_out("hello there friend"), {})
    assert _texts(out) == ["hello", "there", "friend"]


def test_guard_does_not_mutate_input():
    original = _out("Thanks for watching")
    apply_guards(original, {0: LOUD})
    assert len(original.chunks[0]) == 3 and original.flags == {}


def test_guard_drops_trailing_word_outside_vad_speech():
    # Whisper's "Gracias." invented over trailing silence: VAD speech ends at 1.2 s.
    words = [Word("hola", 0.0, 0.5, 0.9), Word("amigos", 0.6, 1.2, 0.9), Word("Gracias.", 2.0, 3.0, 0.9)]
    out = apply_guards(EngineOutput("w", "es", {0: words}), {0: LOUD}, speech_spans=[(0.0, 1.2)])
    assert _texts(out) == ["hola", "amigos"]
    assert "trailing" in out.flags[0]


def test_guard_keeps_trailing_word_inside_vad_speech():
    words = [Word("hola", 0.0, 0.5, 0.9), Word("amigos", 0.6, 1.2, 0.9)]
    out = apply_guards(EngineOutput("w", "es", {0: words}), {0: LOUD}, speech_spans=[(0.0, 1.25)])
    assert _texts(out) == ["hola", "amigos"]
    assert out.flags.get(0, []) == []


def test_guard_trailing_vad_uses_only_spans_overlapping_the_chunk():
    # Speech elsewhere in the file must not rescue a trailing word in this chunk's silence.
    words = [Word("ok", 10.0, 10.4, 0.9), Word("And.", 12.0, 12.5, 0.9)]
    out = apply_guards(EngineOutput("w", "en", {3: words}), {3: LOUD}, speech_spans=[(10.0, 10.5), (20.0, 25.0)])
    assert _texts(out, 3) == ["ok"]


def test_guard_empty_spans_for_chunk_drops_everything_flagged_outside_speech():
    # VAD speech exists only far from this chunk: every word is outside speech.
    words = [Word("hola", 10.0, 10.4, 0.9), Word("amigos", 10.5, 11.0, 0.9)]
    chunks = [Chunk(3, 10.0, 12.0)]
    out = apply_guards(EngineOutput("w", "es", {3: words}), {3: LOUD}, speech_spans=[(20.0, 25.0)], chunks=chunks)
    assert out.chunks[3] == []
    assert "outside-speech" in out.flags[3]


def test_guard_none_spans_leaves_the_vad_rule_off():
    words = [Word("hola", 10.0, 10.4, 0.9), Word("amigos", 10.5, 11.0, 0.9)]
    out = apply_guards(EngineOutput("w", "es", {3: words}), {3: LOUD}, speech_spans=None, chunks=[Chunk(3, 10.0, 12.0)])
    assert _texts(out, 3) == ["hola", "amigos"]
    assert out.flags.get(3, []) == []


def test_guard_silence_gate_is_per_chunk_across_several_chunks():
    quiet = 10 ** (-55 / 20)
    out_in = EngineOutput("x", "en", {
        0: _out("hello there friend").chunks[0],
        1: _out("phantom words here", start=5.0).chunks[0],
        2: _out("still speaking now", start=10.0).chunks[0],
        3: _out("no rms entry", start=15.0).chunks[0],
    })
    out = apply_guards(out_in, {0: LOUD, 1: quiet, 2: 10 ** (-49 / 20)})
    assert _texts(out, 0) == ["hello", "there", "friend"]
    assert out.chunks[1] == [] and out.flags[1] == ["low-energy"]
    assert _texts(out, 2) == ["still", "speaking", "now"]
    assert _texts(out, 3) == ["no", "rms", "entry"]
    assert 0 not in out.flags and 2 not in out.flags and 3 not in out.flags


def test_guard_drops_low_prob_last_word():
    words = [Word("we", 0.0, 0.3, 0.95), Word("agree", 0.4, 0.9, 0.9), Word("And.", 1.0, 1.4, 0.05)]
    out = apply_guards(EngineOutput("w", "en", {0: words}), {0: LOUD})
    assert _texts(out) == ["we", "agree"]
    assert "trailing" in out.flags[0]


def test_guard_keeps_low_prob_word_in_the_middle_and_none_probs():
    words = [Word("we", 0.0, 0.3, 0.05), Word("agree", 0.4, 0.9, 0.9)]
    assert _texts(apply_guards(EngineOutput("w", "en", {0: words}), {0: LOUD})) == ["we", "agree"]
    unprobed = [Word("we", 0.0, 0.3), Word("agree", 0.4, 0.9)]
    assert _texts(apply_guards(EngineOutput("w", "en", {0: unprobed}), {0: LOUD})) == ["we", "agree"]


def test_guard_never_empties_a_chunk_through_the_low_prob_rule_alone_for_confident_words():
    words = [Word("yes", 0.0, 0.3, 0.9)]
    assert _texts(apply_guards(EngineOutput("w", "en", {0: words}), {0: LOUD})) == ["yes"]


# ----------------------------------------------------------------------------- engines


def test_spread_words_by_chars():
    words = spread_words_by_chars("hi there", 10.0, 12.0)
    assert [w.text for w in words] == ["hi", "there"]
    assert all(10.0 <= w.start < w.end <= 12.0 for w in words)
    assert words[0].end <= words[1].start
    # "there" is 5 characters against 2: it gets the larger share
    assert (words[1].end - words[1].start) > (words[0].end - words[0].start)
    assert all(w.prob is None for w in words)


def test_spread_words_empty_text():
    assert spread_words_by_chars("   ", 0.0, 1.0) == []


def test_make_engine_unknown_raises():
    with pytest.raises(CapabilityError):
        make_engine("nope")


@pytest.mark.parametrize(
    "engine_id,cls",
    [("whisper-large-v3", "WhisperEngine"), ("qwen3-asr-1.7b", "Qwen3AsrEngine"),
     ("parakeet-v3", "ParakeetEngine"), ("eg2", "EG2LocalEngine")],
)
def test_make_engine_maps_ids_without_loading_models(engine_id, cls):
    engine = make_engine(engine_id)
    assert type(engine).__name__ == cls
    assert engine.name == engine_id


def test_eg2_rejects_non_english_language():
    with pytest.raises(CapabilityError):
        engines.EG2LocalEngine().transcribe_chunks(np.zeros(SR, dtype=np.float32), [Chunk(0, 0.0, 1.0)], "es")


def test_eg2_rejection_happens_before_any_model_load(monkeypatch):
    def boom(self):
        raise AssertionError("model must not load for a non-English request")

    monkeypatch.setattr(engines.EG2LocalEngine, "_load", boom)
    with pytest.raises(CapabilityError):
        engines.EG2LocalEngine().transcribe_chunks(np.zeros(SR, dtype=np.float32), [Chunk(0, 0.0, 1.0)], "es")


def test_eg2_happy_path_mocked_gives_absolute_times_and_no_probs():
    from transcribe.eg2 import EG2Word

    class FakeEG2:
        def __init__(self):
            self.lengths = []

        def transcribe(self, audio):
            self.lengths.append(len(audio))
            return [EG2Word("hello", 0.5, 0.54, 3.0), EG2Word("world", 1.0, 1.04, 2.0)]

        def render(self, words):
            return ["Hello", "world."]

    engine = engines.EG2LocalEngine()
    fake = engine._engine = FakeEG2()
    samples = np.zeros(20 * SR, dtype=np.float32)
    for language in ("en", None):
        out = engine.transcribe_chunks(samples, [Chunk(0, 2.0, 4.0), Chunk(1, 10.0, 13.0)], language)
        assert out.engine == "eg2" and out.language == "en"
        assert [w.text for w in out.chunks[1]] == ["Hello", "world."]
        assert out.chunks[1][0].start == pytest.approx(10.5)
        assert out.chunks[1][1].end == pytest.approx(11.04)
        assert all(w.prob is None for ws in out.chunks.values() for w in ws)
    assert fake.lengths[:2] == [2 * SR, 3 * SR]
    engine.release()
    assert engine._engine is None


def test_qwen_engine_mocked_language_padding_and_absolute_times():
    class Out:
        def __init__(self, text, language=None):
            self.text, self.language = text, language

    class FakeQwen:
        def __init__(self, language=None):
            self.calls, self.language = [], language

        def generate(self, audio, language=None):
            self.calls.append((len(audio), language))
            return Out("hi there", self.language)

    engine = engines.Qwen3AsrEngine()
    fake = engine._model = FakeQwen()
    samples = np.zeros(20 * SR, dtype=np.float32)
    chunks = [Chunk(0, 2.0, 2.4), Chunk(1, 10.0, 13.0)]  # first chunk is shorter than 1 s
    out = engine.transcribe_chunks(samples, chunks, "es")
    assert fake.calls == [(SR, "Spanish"), (3 * SR, "Spanish")]  # sub-1 s chunk padded to 1 s
    assert out.language == "es" and out.engine == "qwen3-asr-1.7b"
    # words are spread across the real chunk, not the padded audio, and are absolute
    assert all(2.0 <= w.start <= w.end <= 2.4 for w in out.chunks[0])
    assert out.chunks[1][0].start == pytest.approx(10.0)
    assert out.chunks[1][-1].end == pytest.approx(13.0)
    assert [w.text for w in out.chunks[1]] == ["hi", "there"]
    assert all(w.prob is None for ws in out.chunks.values() for w in ws)

    engine.transcribe_chunks(samples, chunks, "en")
    assert fake.calls[-1][1] == "English"
    engine.transcribe_chunks(samples, chunks, None)
    assert fake.calls[-1][1] is None


def test_qwen_engine_detected_language_is_mapped_to_code_when_none_given():
    class Out:
        text, language = "hola", "Spanish"

    class FakeQwen:
        def generate(self, audio, language=None):
            return Out()

    engine = engines.Qwen3AsrEngine()
    engine._model = FakeQwen()
    out = engine.transcribe_chunks(np.zeros(2 * SR, dtype=np.float32), [Chunk(0, 0.0, 2.0)], None)
    assert out.language == "es"


def test_qwen_release_drops_model():
    engine = engines.Qwen3AsrEngine()
    engine._model = object()
    engine.release()
    assert engine._model is None


def test_whisper_engine_uses_required_params_and_absolute_times(monkeypatch):
    import mlx_whisper

    calls = []

    def fake_transcribe(audio, **kwargs):
        calls.append((len(audio), kwargs))
        return {"language": "en", "segments": [{"words": [
            {"word": " hello", "start": 0.1, "end": 0.5, "probability": 0.9},
            {"word": " world.", "start": 0.6, "end": 1.0, "probability": 0.8},
        ]}]}

    monkeypatch.setattr(mlx_whisper, "transcribe", fake_transcribe)
    samples = np.zeros(10 * SR, dtype=np.float32)
    out = engines.WhisperEngine().transcribe_chunks(samples, [Chunk(0, 0.0, 2.0), Chunk(1, 5.0, 8.0)], None)
    assert [c[0] for c in calls] == [2 * SR, 3 * SR]
    kwargs = calls[0][1]
    assert kwargs["word_timestamps"] is True
    assert kwargs["condition_on_previous_text"] is False
    assert kwargs["compression_ratio_threshold"] == 2.4
    assert kwargs["logprob_threshold"] == -1.0
    assert kwargs["no_speech_threshold"] == 0.6
    assert kwargs["path_or_hf_repo"] == "mlx-community/whisper-large-v3-mlx"
    assert kwargs.get("language") is None
    assert [w.text for w in out.chunks[0]] == ["hello", "world."]
    assert out.chunks[1][0].start == pytest.approx(5.1)
    assert out.chunks[1][1].end == pytest.approx(6.0)
    assert out.chunks[1][1].prob == pytest.approx(0.8)
    assert out.language == "en" and out.engine == "whisper-large-v3"


def test_whisper_engine_forces_language_only_when_given(monkeypatch):
    import mlx_whisper

    seen = []
    monkeypatch.setattr(mlx_whisper, "transcribe", lambda audio, **kw: seen.append(kw.get("language")) or {"segments": []})
    engines.WhisperEngine().transcribe_chunks(np.zeros(SR, dtype=np.float32), [Chunk(0, 0.0, 1.0)], "es")
    assert seen == ["es"]


def test_whisper_release_resets_global_model_holder():
    from mlx_whisper.transcribe import ModelHolder

    ModelHolder.model, ModelHolder.model_path = object(), "x"
    engines.WhisperEngine().release()
    assert ModelHolder.model is None and ModelHolder.model_path is None


def test_parakeet_tokens_group_into_words_with_absolute_times():
    class Tok:
        def __init__(self, text, start, end, confidence):
            self.text, self.start, self.end, self.confidence = text, start, end, confidence

    tokens = [Tok(" Hel", 0.0, 0.2, 0.9), Tok("lo", 0.2, 0.4, 0.6), Tok(" there", 0.5, 0.9, 0.8)]
    words = engines.words_from_tokens(tokens, offset=10.0)
    assert [w.text for w in words] == ["Hello", "there"]
    assert (words[0].start, words[0].end) == (pytest.approx(10.0), pytest.approx(10.4))
    assert words[0].prob == pytest.approx(0.6)  # weakest token of the word


# ----------------------------------------------------------------------------- slow (real models)

SPIKE = Path.home() / ".cache/whisper_translate/suite/spike"


def _wer(ref: str, hyp: str) -> float:
    import jiwer
    from whisper_normalizer.english import EnglishTextNormalizer

    norm = EnglishTextNormalizer()
    return jiwer.wer(norm(ref), norm(hyp))


def _run_real(engine_id: str):
    import time

    import soundfile as sf

    from transcribe.auto.chunks import make_chunks, speech_spans

    wav = SPIKE / "en_30s.wav"
    if not wav.exists():
        pytest.skip(f"missing {wav}")
    samples, sr = sf.read(str(wav), dtype="float32")
    assert sr == SR
    chunks = make_chunks(speech_spans(samples, sr), len(samples) / sr)
    engine = make_engine(engine_id)
    try:
        t0 = time.perf_counter()
        out = engine.transcribe_chunks(samples, chunks, "en")
        elapsed = time.perf_counter() - t0
    finally:
        engine.release()
    print(f"\n{engine_id}: {len(samples) / sr / elapsed:.1f}x RT over {len(chunks)} chunks")
    hyp = " ".join(w.text for i in sorted(out.chunks) for w in out.chunks[i])
    ref = wav.with_suffix(".txt").read_text()
    return out, chunks, hyp, ref


@pytest.mark.slow
def test_whisper_engine_real_clip():
    out, chunks, hyp, ref = _run_real("whisper-large-v3")
    wer = _wer(ref, hyp)
    print(f"whisper WER={wer:.4f} hyp={hyp!r}")
    assert wer < 0.15
    last = max(w.end for ws in out.chunks.values() for w in ws)
    assert last <= chunks[-1].end + 1e-6
    assert all(c.start - 1e-6 <= w.start for c in chunks for w in out.chunks.get(c.index, []))


@pytest.mark.slow
def test_qwen3_engine_real_clip():
    out, chunks, hyp, ref = _run_real("qwen3-asr-1.7b")
    wer = _wer(ref, hyp)
    print(f"qwen3 WER={wer:.4f} hyp={hyp!r}")
    assert wer < 0.15


def test_transcribe_each_clears_mlx_cache_after_every_chunk(monkeypatch):
    calls = []
    monkeypatch.setattr(engines, "_clear_mlx_cache", lambda: calls.append("clear"))
    chunks = [Chunk(0, 0.0, 1.0), Chunk(1, 1.0, 2.0), Chunk(2, 2.0, 3.0)]
    result, lang = engines._transcribe_each("fake", chunks, lambda c: ([Word("hi", c.start, c.end, 0.9)], "en"))
    assert sorted(result) == [0, 1, 2] and lang == "en"
    assert calls == ["clear"] * 3


def test_phrase_list_has_the_full_whisper_hallucination_lines():
    for phrase in ("subtítulos realizados por la comunidad de amara.org", "subtitles by the amara.org community",
                   "transcribed by", "gracias por ver el video", "thank you for watching"):
        assert phrase in HALLUCINATION_PHRASES


def test_guard_drops_amara_line_at_the_end_of_a_chunk():
    out = apply_guards(_out("and that is how it works. Subtítulos realizados por la comunidad de Amara.org"), {0: LOUD})
    assert _texts(out) == ["and", "that", "is", "how", "it", "works."]
    assert "known-phrase-trailing" in out.flags[0]


def test_guard_drops_english_trailing_phrase_and_repeats():
    out = apply_guards(_out("see you next week. Thank you for watching. Thanks for watching."), {0: LOUD})
    assert _texts(out) == ["see", "you", "next", "week."]


def test_guard_drops_chunk_that_is_only_a_long_phrase_with_trailing_punctuation():
    out = apply_guards(_out("Subtitles by the Amara.org community"), {0: LOUD})
    assert out.chunks[0] == []
    assert out.flags[0]


@pytest.mark.parametrize("text", ["I subscribe", "and I always subscribe", "please do transcribed by hand",
                                  "he was transcribed by someone else", "we agreed to subscribe"])
def test_guard_keeps_short_phrases_inside_or_ending_real_speech(text):
    out = apply_guards(_out(text), {0: LOUD})
    assert _texts(out) == text.split()
    assert out.flags.get(0, []) == []

"""Tests for the Whisper teacher-forced audio verifier (fake model; real model only in slow tests)."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from transcribe.auto.verify import AudioScorer, WhisperScorer, compare, window_bounds

SR = 16000
EOT = 0
GOOD, BAD = 20, 30
VOCAB = 64
SPIKE_DIR = Path.home() / ".cache/whisper_translate/suite/spike"


class FakeTokenizer:
    eot = EOT
    sot_sequence_including_notimestamps = (1, 2, 3, 4)

    def __init__(self, language):
        self.language = language

    def encode(self, text):
        return [GOOD if w == "good" else BAD for w in text.split()]


class FakeModel:
    """Fixed distribution: token GOOD is likely, everything else (incl. EOT) is not."""

    dims = SimpleNamespace(n_mels=128)
    is_multilingual = True
    num_languages = 99

    def logits(self, tokens, features):
        import mlx.core as mx

        base = np.zeros((1, tokens.shape[1], VOCAB), dtype=np.float32)
        base[:, :, GOOD] = 6.0
        return mx.array(base)


class FakeScorer(WhisperScorer):
    """Replaces the three mlx-whisper seams; records calls."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.loads = 0
        self.encoded = []
        self.languages = []

    def _load_model(self):
        self.loads += 1
        return FakeModel()

    def _make_tokenizer(self, model, lang):
        self.languages.append(lang)
        return FakeTokenizer(lang)

    def _encode_window(self, model, window):
        self.encoded.append(len(window))
        return object()


def _audio(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def test_compare_returns_score_per_candidate():
    scorer = FakeScorer()
    samples = _audio(10)
    scores = compare(scorer, samples, 2.0, 4.0, ["good good", "bad bad", ""], "en")
    assert len(scores) == 3
    assert all(math.isfinite(s) and s <= 0.0 for s in scores)
    assert scores[0] > scores[1]
    # Empty candidate is scorable: SOT + end-of-text only.
    assert math.isfinite(scores[2])


def test_compare_uses_sums_and_score_is_the_mean():
    scorer = FakeScorer()
    samples = _audio(10)
    texts = ["good", "good good good", ""]
    totals = compare(scorer, samples, 2.0, 4.0, texts, "en")
    for text, total in zip(texts, totals):
        detail_total, count = scorer.score_detail(samples, 2.0, 4.0, text, "en")
        assert total == pytest.approx(detail_total)
        assert count == len(text.split()) + 1  # text tokens + EOT
        assert scorer.score(samples, 2.0, 4.0, text, "en") == pytest.approx(total / count)
    # Each extra "good" token costs log-prob, so totals fall with length; the mean would
    # instead reward the longer text by diluting the expensive EOT. Totals avoid that bias.
    assert totals[1] < totals[0]
    assert scorer.score(samples, 2.0, 4.0, "good good good", "en") > scorer.score(samples, 2.0, 4.0, "good", "en")


def test_degenerate_windows_raise():
    with pytest.raises(ValueError):
        window_bounds(10.0, 11.0, duration=10.0)
    with pytest.raises(ValueError):
        window_bounds(12.0, 13.0, duration=10.0)
    with pytest.raises(ValueError):
        window_bounds(-5.0, -1.0, duration=10.0)
    with pytest.raises(ValueError):
        FakeScorer().score(_audio(5), 6.0, 7.0, "good", "en")
    with pytest.raises(ValueError):
        compare(FakeScorer(), _audio(5), 5.0, 5.5, ["good"], "en")


def test_end_of_text_is_part_of_the_score():
    scorer = FakeScorer()
    samples = _audio(10)
    # EOT is unlikely under the fake distribution, so a longer all-good text is diluted less
    # than a short one only if EOT is counted: mean over (n good tokens + EOT).
    one = scorer.score(samples, 2.0, 4.0, "good", "en")
    three = scorer.score(samples, 2.0, 4.0, "good good good", "en")
    assert three > one


def test_model_loads_lazily_and_once():
    scorer = FakeScorer()
    assert scorer.loads == 0
    scorer.score(_audio(5), 1.0, 2.0, "good", "en")
    scorer.score(_audio(5), 1.0, 2.0, "bad", "en")
    assert scorer.loads == 1


def test_feature_cache_per_window_and_lru_bound():
    scorer = FakeScorer(cache_size=2)
    samples = _audio(20)
    scorer.score(samples, 1.0, 2.0, "good", "en")
    scorer.score(samples, 1.0, 2.0, "bad", "en")
    assert len(scorer.encoded) == 1
    scorer.score(samples, 5.0, 6.0, "good", "en")
    scorer.score(samples, 9.0, 10.0, "good", "en")  # evicts the first window
    assert len(scorer.encoded) == 3
    scorer.score(samples, 1.0, 2.0, "good", "en")
    assert len(scorer.encoded) == 4


def test_cache_not_shared_across_audio_arrays():
    scorer = FakeScorer()
    scorer.score(_audio(5), 1.0, 2.0, "good", "en")
    scorer.score(_audio(5), 1.0, 2.0, "good", "en")
    assert len(scorer.encoded) == 2


def test_language_none_defaults_to_english():
    scorer = FakeScorer()
    scorer.score(_audio(5), 1.0, 2.0, "good", None)
    scorer.score(_audio(5), 1.0, 2.0, "good", "es")
    assert scorer.languages == ["en", "es"]


def test_release_drops_model_and_cache_then_reloads():
    scorer = FakeScorer()
    samples = _audio(5)
    scorer.score(samples, 1.0, 2.0, "good", "en")
    scorer.release()
    assert scorer._model is None and not scorer._features
    scorer.score(samples, 1.0, 2.0, "good", "en")
    assert scorer.loads == 2 and len(scorer.encoded) == 2
    scorer.release()
    scorer.release()  # idempotent


def test_window_padding_and_cap():
    start, end = window_bounds(0.1, 2.0, duration=10.0)
    assert start == 0.0 and end == pytest.approx(2.3)
    start, end = window_bounds(8.0, 9.9, duration=10.0)
    assert start == pytest.approx(7.7) and end == 10.0
    start, end = window_bounds(5.0, 45.0, duration=60.0)
    assert end - start == pytest.approx(30.0)
    assert (start + end) / 2 == pytest.approx(25.0)  # centered on the span
    # A long span that overhangs the file end still yields a 30 s window inside [0, duration].
    start, end = window_bounds(20.0, 70.0, duration=50.0)
    assert end - start == pytest.approx(30.0) and end <= 50.0 and start >= 0.0


def test_scorer_encodes_padded_and_capped_windows():
    scorer = FakeScorer()
    scorer.score(_audio(60), 0.1, 2.0, "good", "en")
    scorer.score(_audio(60), 5.0, 45.0, "good", "en")
    assert scorer.encoded[0] == int(2.3 * SR)
    assert scorer.encoded[1] == 30 * SR


def test_protocol_is_satisfied():
    scorer: AudioScorer = FakeScorer()
    assert callable(scorer.score) and callable(scorer.release)


# ----------------------------------------------------------------------------- slow (real model)


def _load_clip(name: str):
    import soundfile as sf

    wav = SPIKE_DIR / f"{name}.wav"
    if not wav.exists():
        pytest.skip(f"missing {wav}")
    samples, sr = sf.read(str(wav), dtype="float32")
    assert sr == SR
    return samples, wav.with_suffix(".txt").read_text().strip()


def _corrupt(text: str, n: int = 3) -> str:
    words = text.split()
    idx = np.linspace(2, len(words) - 3, n * 2).astype(int)
    for a, b in zip(idx[::2], idx[1::2]):
        words[a], words[b] = words[b], words[a]
    return " ".join(words)


@pytest.fixture(scope="module")
def real_scorer():
    scorer = WhisperScorer()
    yield scorer
    scorer.release()


@pytest.mark.slow
@pytest.mark.parametrize("name,lang", [("en_30s", "en"), ("es_30s", "es")])
def test_reference_beats_corruption_real(real_scorer, name, lang):
    samples, ref = _load_clip(name)
    dur = len(samples) / SR
    good, bad = compare(real_scorer, samples, 0.0, dur, [ref, _corrupt(ref)], lang)
    print(f"\n{name}: reference {good:.3f} vs corrupted {bad:.3f}")
    assert good > bad


@pytest.mark.slow
@pytest.mark.parametrize("name,lang", [("en_30s", "en"), ("es_30s", "es")])
def test_one_word_substitution_ranked_on_real_span(real_scorer, name, lang):
    import mlx_whisper

    samples, _ = _load_clip(name)
    out = mlx_whisper.transcribe(
        samples, path_or_hf_repo=real_scorer.repo, language=lang, word_timestamps=True,
        condition_on_previous_text=False,
    )
    words = [w for seg in out["segments"] for w in seg.get("words", [])]
    assert len(words) > 12
    i = len(words) // 2
    mid = words[i]
    true_word = mid["word"].strip()
    wrong_word = "banana" if lang == "en" else "plátano"
    assert true_word.lower().strip(".,;:!?¿¡") != wrong_word
    left = " ".join(w["word"].strip() for w in words[i - 1 : i])
    right = " ".join(w["word"].strip() for w in words[i + 1 : i + 2])
    # Span = left + true word + right, with the exact Whisper timestamps for those 3 words.
    start, end = words[i - 1]["start"], words[i + 1]["end"]
    true_text = f"{left} {true_word} {right}"
    wrong_text = f"{left} {wrong_word} {right}"
    good, bad = compare(real_scorer, samples, start, end, [true_text, wrong_text], lang)
    print(f"\n{name}: '{true_text}' {good:.3f} vs '{wrong_text}' {bad:.3f}")
    assert good > bad


def _real_words(real_scorer, name, lang):
    import mlx_whisper

    samples, _ = _load_clip(name)
    out = mlx_whisper.transcribe(
        samples, path_or_hf_repo=real_scorer.repo, language=lang, word_timestamps=True,
        condition_on_previous_text=False,
    )
    words = [w for seg in out["segments"] for w in seg.get("words", [])]
    assert len(words) > 12
    return samples, words


@pytest.mark.slow
@pytest.mark.parametrize("name,lang,filler", [("en_30s", "en", "uh"), ("es_30s", "es", "eh")])
def test_deletion_and_insertion_ranked_with_shared_context(real_scorer, name, lang, filler):
    samples, words = _real_words(real_scorer, name, lang)
    i = len(words) // 2
    left = " ".join(w["word"].strip() for w in words[i - 2 : i])
    gone = words[i]["word"].strip()
    right = " ".join(w["word"].strip() for w in words[i + 1 : i + 3])
    start, end = words[i - 2]["start"], words[i + 2]["end"]
    true_text = f"{left} {gone} {right}"
    deleted = f"{left} {right}"  # deletion candidate, scored with the SAME context
    inserted = f"{left} {gone} {filler} {right}"
    t, d, ins = compare(real_scorer, samples, start, end, [true_text, deleted, inserted], lang)
    print(f"\n{name}: true {t:.3f} | deleted '{deleted}' {d:.3f} | inserted '{inserted}' {ins:.3f}")
    assert t > d
    assert t > ins

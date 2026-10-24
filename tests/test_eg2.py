from pathlib import Path

import numpy as np
import pytest

from transcribe.backends import EG2Backend
from transcribe.chunking import should_chunk_audio
from transcribe.eg2 import (
    STEP_SAMPLES,
    WINDOW_SAMPLES,
    EG2Word,
    LogMel,
    _Collapse,
    window_starts,
    words_to_segments,
)
from transcribe.models import CapabilityError


def test_window_starts_matches_app_layout():
    assert window_starts(WINDOW_SAMPLES) == [0]
    # 30 s: second window starts at 18 s and reaches the end
    assert window_starts(30 * 16000) == [0, STEP_SAMPLES]
    starts = window_starts(3600 * 16000)
    assert starts[1] - starts[0] == STEP_SAMPLES
    assert starts[-1] + WINDOW_SAMPLES >= 3600 * 16000
    assert starts[-2] + WINDOW_SAMPLES < 3600 * 16000


def test_collapse_merges_repeats_blanks_and_rep_token():
    vocab = ["the", "cat", "<rep>"]
    c = _Collapse(vocab, rep_id=2)
    ids = np.array([-1, 0, 0, -1, 1, 2, -1, 0])
    c.feed(ids, np.ones(len(ids)), t0=1.0)
    c.close(2.0)
    assert [w.text for w in c.words] == ["the", "cat", "cat", "the"]
    assert c.words[0].start == pytest.approx(1.04)
    assert c.words[0].end == pytest.approx(1.12)
    assert c.words[-1].end == pytest.approx(2.0)


def test_collapse_keeps_state_across_windows():
    c = _Collapse(["hello"], rep_id=99)
    c.feed(np.array([0, 0]), np.ones(2), t0=0.0)
    c.feed(np.array([0, -1]), np.ones(2), t0=0.08)  # same word continues into next window
    c.close(0.2)
    assert [w.text for w in c.words] == ["hello"]


def test_words_to_segments_splits_on_sentence_end_and_pause():
    words = [EG2Word(t, s, s + 0.04, 5.0) for t, s in [("hi", 0.0), ("there", 0.3), ("ok", 3.0), ("bye", 3.2)]]
    segs = words_to_segments(words, ["Hi", "there.", "Ok", "bye."])
    assert [s["text"] for s in segs] == ["Hi there.", "Ok bye."]
    assert segs[0]["start"] == 0.0 and segs[0]["end"] <= 3.0
    assert segs[1]["end"] == pytest.approx(3.7)


def test_logmel_frame_count_and_floor(tmp_path: Path):
    np.hanning(320).astype("<f4").tofile(tmp_path / "hann_320.f32")
    np.zeros((257, 128), dtype="<f4").tofile(tmp_path / "mel_filters_257x128.f32")
    feats = LogMel(tmp_path)(np.zeros(16000, dtype=np.float32))
    assert feats.shape == (99, 128)  # frames whose hop start (i+1)*160 < 16000
    assert np.allclose(feats, np.log(1e-3))


def test_eg2_rejects_non_english_without_fallback(tmp_path: Path):
    with pytest.raises(CapabilityError):
        EG2Backend("eg2demo-v5").transcribe(tmp_path / "a.wav", "es", None, 60, {})


def test_eg2_skips_auto_chunking(tmp_path: Path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"0")
    kwargs = dict(duration_sec=7200, audio_path=audio, chunk_threshold_minutes=20, openai_max_file_mb=24)
    assert should_chunk_audio(chunk_mode="auto", backend="eg2", **kwargs) is False
    assert should_chunk_audio(chunk_mode="force", backend="eg2", **kwargs) is True

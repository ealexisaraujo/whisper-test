"""Tests for transcribe.auto core types, VAD chunking and the audio probe (VAD/LID injected)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from transcribe.auto.chunks import make_chunks
from transcribe.auto.probe import build_profile
from transcribe.auto.types import AudioProfile, Chunk, EngineOutput, StageStatus, Word

SR = 16000


def _noise(seconds: float, amp: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * SR)) * amp).astype(np.float32)


def _lid_sequence(*answers):
    calls = iter(answers)
    seen = []

    def lid(window: np.ndarray):
        seen.append(len(window))
        return next(calls)

    lid.seen = seen
    return lid


def test_types_are_json_serialisable():
    out = EngineOutput(
        engine="x",
        language="en",
        chunks={0: [Word("hi", 0.0, 0.5, 0.9)]},
        flags={0: ["repeat"]},
    )
    profile = AudioProfile(
        duration=1.0, speech_seconds=0.5, speech_ratio=0.5, language="en",
        language_windows=[(0.0, "en", 0.9)], code_switching=False, snr_db=None,
        phone_band=False, key="en",
    )
    for obj in (out, profile, Chunk(0, 0.0, 1.0), StageStatus("probe", "ok")):
        json.dumps(dataclasses.asdict(obj))
    assert Word("a", 0.0, 1.0).prob is None
    assert StageStatus("probe", "ok").seconds == 0.0


def test_make_chunks_merges_and_caps():
    samples = _noise(80, 0.1)
    chunks = make_chunks([(0, 10), (10.3, 25), (26, 70)], samples, SR)
    assert chunks
    assert all(c.end - c.start <= 30.0 + 1e-6 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    for a, b in zip(chunks, chunks[1:]):
        assert a.end <= b.start + 1e-9
    # the first two spans merge across the 0.3 s gap
    assert chunks[0].start == pytest.approx(0.0)
    assert chunks[0].end == pytest.approx(25.0)
    # all speech is covered
    for s, e in [(0, 10), (10.3, 25), (26, 70)]:
        covered = sum(
            max(0.0, min(e, c.end) - max(s, c.start)) for c in chunks
        )
        assert covered == pytest.approx(e - s, abs=1e-6)


def test_make_chunks_splits_at_lowest_energy_frame_in_last_ten_seconds():
    samples = _noise(60, 0.1)
    # carve a quiet 100 ms hole at 22.0-22.1 s (inside the last 10 s of a 30 s window from 0)
    samples[int(22.0 * SR):int(22.1 * SR)] = 0.0
    chunks = make_chunks([(0, 60)], samples, SR)
    assert chunks[0].end == pytest.approx(22.05, abs=0.06)
    assert chunks[1].start == pytest.approx(chunks[0].end)
    assert all(c.end - c.start <= 30.0 + 1e-6 for c in chunks)
    assert chunks[-1].end == pytest.approx(60.0)


def test_make_chunks_span_of_exactly_max_seconds_is_one_chunk():
    chunks = make_chunks([(5.0, 35.0)], _noise(40, 0.1), SR)
    assert [(c.index, c.start, c.end) for c in chunks] == [(0, 5.0, 35.0)]


def test_make_chunks_tiny_span_is_kept():
    chunks = make_chunks([(1.0, 1.2)], _noise(5, 0.1), SR)
    assert len(chunks) == 1
    assert chunks[0].start == pytest.approx(1.0) and chunks[0].end == pytest.approx(1.2)


def test_make_chunks_span_past_audio_end_still_splits_and_covers():
    samples = _noise(40, 0.1)  # audio ends at 40 s, span claims 75 s
    chunks = make_chunks([(0.0, 75.0)], samples, SR)
    assert all(c.end - c.start <= 30.0 + 1e-6 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        assert a.end == pytest.approx(b.start)
    assert chunks[0].start == 0.0
    assert chunks[-1].end == pytest.approx(75.0)


def test_make_chunks_empty_spans_returns_empty():
    assert make_chunks([], _noise(5, 0.1), SR) == []


def test_build_profile_one_dissenting_window_is_not_code_switching():
    lid = _lid_sequence(("en", 0.9), ("en", 0.9), ("es", 0.9))
    # 90 s of speech -> three 30 s windows
    samples = _noise(90, 0.05)
    spans = [(0.0, 90.0)]
    profile = build_profile(samples, SR, spans, lid, original_sample_rate=16000)
    assert profile.language == "en"
    assert profile.code_switching is False  # one confident dissenter is LID noise
    assert profile.key == "en"
    assert len(profile.language_windows) == len(lid.seen) == 3
    assert all(n <= 30 * SR for n in lid.seen)


def test_build_profile_two_dissenting_windows_are_code_switching():
    lid = _lid_sequence(("en", 0.9), ("en", 0.9), ("en", 0.9), ("es", 0.9), ("es", 0.9))
    profile = build_profile(_noise(150, 0.05), SR, [(0.0, 150.0)], lid, original_sample_rate=16000)
    assert profile.language == "en"
    assert profile.code_switching is True
    assert profile.key == "other"


def test_build_profile_low_confidence_dissenters_do_not_count():
    lid = _lid_sequence(("en", 0.9), ("en", 0.9), ("en", 0.9), ("es", 0.5), ("es", 0.6))
    profile = build_profile(_noise(150, 0.05), SR, [(0.0, 150.0)], lid, original_sample_rate=16000)
    assert profile.code_switching is False and profile.key == "en"


def test_build_profile_caps_windows_at_five():
    samples = _noise(600, 0.05)
    lid = _lid_sequence(*[("en", 0.95)] * 10)
    profile = build_profile(samples, SR, [(0.0, 600.0)], lid, original_sample_rate=None)
    assert len(profile.language_windows) == 5
    assert profile.code_switching is False
    assert profile.language == "en"


def test_build_profile_key_phone_noisy():
    rng = np.random.default_rng(1)
    speech_amp, noise_amp = 0.1, 0.1 / (10 ** (5 / 20))  # 5 dB SNR
    samples = (rng.standard_normal(40 * SR) * noise_amp).astype(np.float32)
    samples[10 * SR:30 * SR] = (rng.standard_normal(20 * SR) * speech_amp).astype(np.float32)
    lid = _lid_sequence(("en", 0.95), ("en", 0.95))
    profile = build_profile(samples, SR, [(10.0, 30.0)], lid, original_sample_rate=8000)
    assert profile.snr_db == pytest.approx(5.0, abs=0.5)
    assert profile.phone_band is True
    assert profile.language == "en"
    assert profile.key == "en-phone-noisy"


def test_build_profile_wideband_clean_en():
    rng = np.random.default_rng(2)
    samples = (rng.standard_normal(40 * SR) * 0.001).astype(np.float32)
    samples[10 * SR:30 * SR] = (rng.standard_normal(20 * SR) * 0.1).astype(np.float32)
    profile = build_profile(samples, SR, [(10.0, 30.0)], _lid_sequence(("es", 0.9)), 44100)
    assert profile.snr_db > 30
    assert profile.phone_band is False
    assert profile.key == "es"
    assert profile.speech_seconds == pytest.approx(20.0)
    assert profile.speech_ratio == pytest.approx(0.5)
    assert profile.duration == pytest.approx(40.0)


def test_build_profile_phone_band_from_spectrum():
    t = np.arange(20 * SR) / SR
    # energy only below 3 kHz even though the container sample rate is 16 kHz
    samples = (0.1 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    profile = build_profile(samples, SR, [(0.0, 20.0)], _lid_sequence(("en", 0.9)), 16000)
    assert profile.snr_db is None  # no non-speech
    assert profile.phone_band is True
    assert profile.key == "en-phone"


def _shaped_noise(seconds: float, gain) -> np.ndarray:
    """White noise filtered in the frequency domain by `gain(freqs)` (amplitude response)."""
    rng = np.random.default_rng(7)
    n = int(seconds * SR)
    spec = np.fft.rfft(rng.standard_normal(n))
    spec *= gain(np.fft.rfftfreq(n, d=1.0 / SR))
    out = np.fft.irfft(spec, n)
    return (out / np.max(np.abs(out)) * 0.3).astype(np.float32)


def _wideband_speech_like(seconds: float, high_fraction: float = 0.005) -> np.ndarray:
    """1/f power spectrum up to 4 kHz, then a quiet shelf to 8 kHz holding `high_fraction` of the energy."""

    def power(f):
        f = np.maximum(f, 50.0)
        low = np.where(f <= 4000, 1.0 / f, 0.0)
        high_mask = f > 4000
        low_total = low.sum()
        high_shape = np.where(high_mask, 1.0, 0.0)
        high = high_shape * (high_fraction / (1 - high_fraction)) * low_total / max(high_shape.sum(), 1)
        return low + high

    return _shaped_noise(seconds, lambda f: np.sqrt(power(f)))


def test_phone_band_boundary_with_realistic_spectra():
    from transcribe.auto.probe import _high_band_fraction

    phone = _shaped_noise(20, lambda f: (f < 3400).astype(float))
    wide = _wideband_speech_like(20)
    assert _high_band_fraction(phone, SR) < 0.001
    assert 0.003 < _high_band_fraction(wide, SR) < 0.008  # real wideband speech: 0.0045-0.016

    spans = [(0.0, 20.0)]
    p_phone = build_profile(phone, SR, spans, _lid_sequence(("en", 0.9)), 16000)
    p_wide = build_profile(wide, SR, spans, _lid_sequence(("en", 0.9)), 16000)
    assert p_phone.phone_band is True and p_phone.key == "en-phone"
    assert p_wide.phone_band is False and p_wide.key == "en"


def test_build_profile_other_language_key():
    samples = _noise(20, 0.1)
    profile = build_profile(samples, SR, [(0.0, 20.0)], _lid_sequence(("fr", 0.9)), 16000)
    assert profile.language == "fr"
    assert profile.key == "other"


def test_build_profile_no_speech():
    def lid(_window):  # must not be called
        raise AssertionError("LID called without speech")

    profile = build_profile(np.zeros(10 * SR, dtype=np.float32), SR, [], lid, 16000)
    assert profile.speech_ratio == 0.0
    assert profile.speech_seconds == 0.0
    assert profile.language == "other"
    assert profile.language_windows == []
    assert profile.code_switching is False
    assert profile.snr_db is None
    assert profile.key == "other"


@pytest.mark.slow
@pytest.mark.parametrize("name,expected", [("en_30s.wav", "en"), ("es_30s.wav", "es")])
def test_real_vad_and_lid_on_spike_clips(name, expected):
    import soundfile as sf

    from transcribe.auto.chunks import speech_spans
    from transcribe.auto.probe import whisper_lid

    path = Path.home() / ".cache/whisper_translate/suite/spike" / name
    if not path.exists():
        pytest.skip(f"missing {path}")
    samples, sr = sf.read(str(path), dtype="float32")
    assert sr == 16000
    spans = speech_spans(samples, sr)
    assert spans
    profile = build_profile(
        samples, sr, spans, whisper_lid("mlx-community/whisper-large-v3-mlx"), sr
    )
    assert profile.language == expected


@pytest.mark.slow
@pytest.mark.parametrize(
    "slice_name,expected_phone", [("en_phone", True), ("es_dialects", False)]
)
def test_real_suite_items_phone_band(slice_name, expected_phone):
    import soundfile as sf

    manifest = Path(__file__).resolve().parents[1] / "benchmarks/suite/manifests" / f"{slice_name}.jsonl"
    item = json.loads(manifest.read_text().splitlines()[0])
    audio = Path(item["audio"])
    if not audio.exists():
        pytest.skip(f"missing {audio}")
    samples, sr = sf.read(str(audio), dtype="float32")
    profile = build_profile(
        samples, sr, [(0.0, len(samples) / sr)], lambda w: ("en", 0.9), sr
    )
    assert profile.phone_band is expected_phone

"""Audio probe: language windows, code-switching, SNR and phone-band detection."""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Callable, List, Optional, Tuple

import numpy as np

from .types import AudioProfile

LOGGER = logging.getLogger("transcribe_youtube")

LID_WINDOW_SECONDS = 30.0
LID_MAX_WINDOWS = 5
CODE_SWITCH_CONFIDENCE = 0.6
CODE_SWITCH_MIN_WINDOWS = 2  # windows disagreeing with the majority language (above the confidence) to call it
PHONE_CUTOFF_HZ = 4000
# Fraction of spectral energy above 4 kHz below which audio counts as phone-band. Measured on the
# eval suite: en_phone 0.0000-0.0002; wideband es_dialects 0.0045-0.0064, en_meeting 0.0055-0.0162,
# two en_talks files < 0.01 and the es_30s spike clip 0.0045. A 0.01 cut misrouted real wideband
# speech as phone, so the threshold sits between the two populations.
PHONE_ENERGY_FRACTION = 0.001
NOISY_SNR_DB = 10.0
ENERGY_FLOOR = 1e-12
SPECTRUM_FRAME = 1024
SPECTRUM_MAX_FRAMES = 2000

LidFn = Callable[[np.ndarray], Tuple[str, float]]


def probe_config(lid_repo: str) -> dict:
    """Cache fingerprint of the probe: the LID model and every threshold that shapes the profile."""
    return {
        "lid_repo": lid_repo,
        "window_seconds": LID_WINDOW_SECONDS,
        "max_windows": LID_MAX_WINDOWS,
        "code_switch_confidence": CODE_SWITCH_CONFIDENCE,
        "code_switch_min_windows": CODE_SWITCH_MIN_WINDOWS,
        "phone_cutoff_hz": PHONE_CUTOFF_HZ,
        "phone_energy_fraction": PHONE_ENERGY_FRACTION,
        "noisy_snr_db": NOISY_SNR_DB,
    }


def _speech_mask(n: int, sr: int, spans: List[Tuple[float, float]]) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for start, end in spans:
        lo = max(0, int(round(start * sr)))
        hi = min(n, int(round(end * sr)))
        if hi > lo:
            mask[lo:hi] = True
    return mask


def _mean_power(x: np.ndarray) -> float:
    return float(np.mean(x.astype(np.float64) ** 2)) if len(x) else 0.0


def _snr_db(samples: np.ndarray, mask: np.ndarray) -> Optional[float]:
    speech, other = samples[mask], samples[~mask]
    if len(speech) == 0 or len(other) == 0:
        return None
    return float(
        10.0 * math.log10(max(_mean_power(speech), ENERGY_FLOOR) / max(_mean_power(other), ENERGY_FLOOR))
    )


def _high_band_fraction(x: np.ndarray, sr: int) -> float:
    """Fraction of spectral energy above 4 kHz (0.0 when the signal is silent or sr <= 8 kHz)."""
    if sr <= 2 * PHONE_CUTOFF_HZ or len(x) < SPECTRUM_FRAME:
        return 0.0
    count = len(x) // SPECTRUM_FRAME
    picks = np.unique(np.linspace(0, count - 1, min(count, SPECTRUM_MAX_FRAMES)).astype(int))
    frames = x[: count * SPECTRUM_FRAME].astype(np.float64).reshape(count, SPECTRUM_FRAME)[picks]
    frames = frames * np.hanning(SPECTRUM_FRAME)
    power = np.mean(np.abs(np.fft.rfft(frames, axis=1)) ** 2, axis=0)
    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    freqs = np.fft.rfftfreq(SPECTRUM_FRAME, d=1.0 / sr)
    return float(power[freqs > PHONE_CUTOFF_HZ].sum() / total)


def _lid_windows(
    samples: np.ndarray, sr: int, spans: List[Tuple[float, float]]
) -> List[Tuple[float, np.ndarray]]:
    """Up to 5 windows of 30 s of pure speech, spread evenly; start is mapped to file time."""
    pieces = []
    for start, end in spans:
        lo, hi = max(0, int(round(start * sr))), min(len(samples), int(round(end * sr)))
        if hi > lo:
            pieces.append((start, lo, hi))
    if not pieces:
        return []
    speech = np.concatenate([samples[lo:hi] for _, lo, hi in pieces])
    win = int(LID_WINDOW_SECONDS * sr)
    if len(speech) <= win:
        offsets = [0]
    else:
        n = min(LID_MAX_WINDOWS, math.ceil(len(speech) / win))
        offsets = [int(round(o)) for o in np.linspace(0, len(speech) - win, n)]

    def file_time(offset: int) -> float:
        remaining = offset
        for start, lo, hi in pieces:
            if remaining < hi - lo:
                return start + remaining / sr
            remaining -= hi - lo
        return pieces[-1][0] + (pieces[-1][2] - pieces[-1][1]) / sr

    return [(file_time(o), speech[o:o + win]) for o in offsets]


def build_profile(
    samples: np.ndarray,
    sr: int,
    spans: List[Tuple[float, float]],
    lid: LidFn,
    original_sample_rate: Optional[int],
) -> AudioProfile:
    """Probe the audio: LID on up to 5 speech windows, SNR, phone band and the routing key."""
    duration = len(samples) / sr if sr else 0.0
    mask = _speech_mask(len(samples), sr, spans)
    speech_seconds = float(mask.sum()) / sr if sr else 0.0
    speech_ratio = speech_seconds / duration if duration > 0 else 0.0

    windows: List[Tuple[float, str, float]] = []
    for start, window in _lid_windows(samples, sr, spans):
        language, confidence = lid(window)
        windows.append((float(start), str(language), float(confidence)))

    if windows:
        votes = Counter(lang for _, lang, _ in windows)
        score = {lang: sum(c for _, l, c in windows if l == lang) for lang in votes}
        language = max(votes, key=lambda lang: (votes[lang], score[lang]))
        # one confident dissenting window is LID noise; code-switching needs at least two
        dissenting = sum(1 for _, lang, conf in windows if lang != language and conf > CODE_SWITCH_CONFIDENCE)
        code_switching = dissenting >= CODE_SWITCH_MIN_WINDOWS
    else:
        language, code_switching = "other", False

    snr_db = _snr_db(samples, mask)
    reference = samples[mask] if mask.any() else samples
    low_rate = original_sample_rate is not None and original_sample_rate <= 8000
    if len(reference) == 0 or _mean_power(reference) <= 0.0:
        phone_band = low_rate  # silent input has no spectrum to judge
    else:
        phone_band = low_rate or _high_band_fraction(reference, sr) < PHONE_ENERGY_FRACTION

    lang_key = language if language in ("en", "es") and not code_switching else "other"
    key = lang_key + ("-phone" if phone_band else "") + ("-noisy" if snr_db is not None and snr_db < NOISY_SNR_DB else "")

    LOGGER.info(
        "Probe: language=%s windows=%d code_switching=%s snr_db=%s phone_band=%s key=%s",
        language, len(windows), code_switching,
        None if snr_db is None else round(snr_db, 1), phone_band, key,
    )
    return AudioProfile(
        duration=float(duration),
        speech_seconds=speech_seconds,
        speech_ratio=float(speech_ratio),
        language=language,
        language_windows=windows,
        code_switching=code_switching,
        snr_db=snr_db,
        phone_band=phone_band,
        key=key,
    )


def whisper_lid(model_repo: str) -> LidFn:
    """Build an LID callable using mlx-whisper `detect_language` on a 30 s mel window.

    The model loads lazily on the first call through mlx-whisper's shared model holder, so the
    same repo used by an ASR engine is not downloaded or loaded twice.
    """

    def detect(window: np.ndarray) -> Tuple[str, float]:
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.decoding import detect_language
        from mlx_whisper.transcribe import ModelHolder

        model = ModelHolder.get_model(model_repo, mx.float16)
        audio = np.zeros(N_SAMPLES, dtype=np.float32)
        trimmed = np.asarray(window[:N_SAMPLES], dtype=np.float32)
        audio[: len(trimmed)] = trimmed
        mel = pad_or_trim(log_mel_spectrogram(audio, n_mels=model.dims.n_mels), N_FRAMES, axis=-2)
        _, probs = detect_language(model, mel.astype(mx.float16))
        language = max(probs, key=probs.get)
        return str(language), float(probs[language])

    return detect

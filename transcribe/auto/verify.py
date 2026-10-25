"""Audio verifier: which candidate text better matches this audio window?

`WhisperScorer` teacher-forces a candidate text through the Whisper decoder and returns the
mean log-probability of its tokens plus the end-of-text token. Higher (closer to 0) means the
audio supports the text better. Scores are only comparable between candidates for the same
window, which is how the picker and the refine gate use them (`compare`).
"""

from __future__ import annotations

import gc
import os
from collections import OrderedDict
from typing import Any, List, Optional, Protocol

import numpy as np

SAMPLE_RATE = 16000
WINDOW_PAD_S = 0.3
MAX_WINDOW_S = 30.0
DEFAULT_REPO = "mlx-community/whisper-large-v3-mlx"
# Whisper's SOT sequence needs a language token in multilingual mode; used when `lang` is None.
DEFAULT_LANG = "en"
# Whisper decoder context is 448 tokens; leave room for the SOT prefix and EOT.
_MAX_TEXT_TOKENS = 448 - 8


class AudioScorer(Protocol):
    def score(self, samples: np.ndarray, start: float, end: float, text: str, lang: Optional[str]) -> float: ...

    def score_detail(
        self, samples: np.ndarray, start: float, end: float, text: str, lang: Optional[str]
    ) -> tuple[float, int]: ...

    def release(self) -> None: ...


def window_bounds(start: float, end: float, duration: float) -> tuple[float, float]:
    """[start - 0.3, end + 0.3] clamped to [0, duration], capped at 30 s (centered on the span).

    Raises ValueError for a degenerate window (start at or past the end of the audio, or an
    empty window after clamping); callers treat that span as unverifiable.
    """
    if start >= duration:
        raise ValueError(f"span start {start:.3f}s is at or past the audio end {duration:.3f}s")
    lo = max(0.0, start - WINDOW_PAD_S)
    hi = min(duration, end + WINDOW_PAD_S)
    if hi <= lo:
        raise ValueError(f"empty window for span [{start:.3f}, {end:.3f}] in {duration:.3f}s of audio")
    if hi - lo > MAX_WINDOW_S:
        mid = (start + end) / 2.0
        lo = mid - MAX_WINDOW_S / 2.0
        lo = min(max(0.0, lo), max(0.0, duration - MAX_WINDOW_S))
        hi = lo + MAX_WINDOW_S
    return lo, max(lo, hi)


class WhisperScorer:
    """Teacher-forced Whisper scoring. The model loads lazily on the first `score`.

    Audio features are cached per (audio array, window) in a bounded LRU (default 256 windows,
    about 4 MB each for large-v3, so roughly 1 GB at the cap). `lang=None` scores with the
    English language token. Uses `mlx_whisper.load_models.load_model` directly, so
    mlx-whisper's global ModelHolder is never touched.

    The cache holds a reference to the samples array and assumes it is not mutated in place
    while cached; call `release()` per file (it clears the cache).
    """

    def __init__(self, repo: str = DEFAULT_REPO, cache_size: int = 256) -> None:
        self.repo = repo
        self.cache_size = max(1, int(cache_size))
        self._model: Any = None
        self._tokenizers: dict[str, Any] = {}
        # key -> (samples reference, features). The reference keeps the array alive so an
        # `id()` can never be reused by a different array while its entry is cached.
        self._features: "OrderedDict[tuple, tuple[np.ndarray, Any]]" = OrderedDict()

    # --- mlx-whisper seams (overridden in fast tests) ---

    def _load_model(self) -> Any:
        import mlx.core as mx
        from mlx_whisper.load_models import load_model

        path = self.repo
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download

            path = snapshot_download(self.repo)
        return load_model(path, dtype=mx.float16)

    def _make_tokenizer(self, model: Any, lang: str) -> Any:
        from mlx_whisper.tokenizer import get_tokenizer

        return get_tokenizer(
            model.is_multilingual, num_languages=model.num_languages, language=lang, task="transcribe"
        )

    def _encode_window(self, model: Any, window: np.ndarray) -> Any:
        import mlx.core as mx
        from mlx_whisper.audio import N_SAMPLES, log_mel_spectrogram

        audio = np.zeros(N_SAMPLES, dtype=np.float32)
        audio[: len(window)] = window[:N_SAMPLES]
        mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels).astype(mx.float16)
        features = model.embed_audio(mel[None])
        mx.eval(features)
        return features

    # --- public API ---

    def score(self, samples: np.ndarray, start: float, end: float, text: str, lang: Optional[str]) -> float:
        """Mean log-prob per token (text tokens + EOT)."""
        total, count = self.score_detail(samples, start, end, text, lang)
        return total / count

    def score_detail(
        self, samples: np.ndarray, start: float, end: float, text: str, lang: Optional[str]
    ) -> tuple[float, int]:
        """(total log-prob over text tokens + EOT, number of those tokens)."""
        import mlx.core as mx

        model = self._ensure_model()
        tokenizer = self._tokenizer(model, lang or DEFAULT_LANG)
        features = self._window_features(model, samples, start, end)

        prefix = list(tokenizer.sot_sequence_including_notimestamps)
        body = text.strip()
        ids = list(tokenizer.encode(" " + body)) if body else []
        ids = ids[:_MAX_TEXT_TOKENS]
        tokens = mx.array([prefix + ids + [tokenizer.eot]])
        logits = model.logits(tokens[:, :-1], features)[0].astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        target = tokens[0, 1:]
        picked = logp[mx.arange(target.shape[0]), target][len(prefix) - 1 :]
        return float(picked.sum()), int(picked.shape[0])

    def release(self) -> None:
        """Drop the model and cached features and hand MLX's buffer cache back to the OS."""
        self._model = None
        self._tokenizers.clear()
        self._features.clear()
        gc.collect()
        try:
            import mlx.core as mx

            clear = getattr(mx, "clear_cache", None) or getattr(mx.metal, "clear_cache", None)
            if clear is not None:
                clear()
        except ImportError:  # pragma: no cover - mlx is a hard dependency on target hardware
            pass

    # --- internals ---

    def _ensure_model(self) -> Any:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _tokenizer(self, model: Any, lang: str) -> Any:
        if lang not in self._tokenizers:
            self._tokenizers[lang] = self._make_tokenizer(model, lang)
        return self._tokenizers[lang]

    def _window_features(self, model: Any, samples: np.ndarray, start: float, end: float) -> Any:
        lo, hi = window_bounds(start, end, len(samples) / SAMPLE_RATE)
        key = (id(samples), len(samples), round(lo, 4), round(hi, 4))
        hit = self._features.get(key)
        if hit is not None and hit[0] is samples:
            self._features.move_to_end(key)
            return hit[1]
        window = np.asarray(samples[int(round(lo * SAMPLE_RATE)) : int(round(hi * SAMPLE_RATE))], dtype=np.float32)
        features = self._encode_window(model, window)
        self._features[key] = (samples, features)
        while len(self._features) > self.cache_size:
            self._features.popitem(last=False)
        return features


def compare(
    scorer: AudioScorer,
    samples: np.ndarray,
    start: float,
    end: float,
    candidates: List[str],
    lang: Optional[str],
) -> List[float]:
    """Total log-likelihood of each candidate against the same audio window; higher is better.

    Totals (not means) make candidates of different lengths comparable: a deletion pays for the
    words it drops and an insertion for the words it adds. Callers must pass FULL window texts
    with identical left/right context around the disputed words. An empty string is only a
    valid deletion candidate when it is the whole window text; "" standing for a deletion
    without the surrounding context is not comparable and must not be scored this way.
    """
    return [scorer.score_detail(samples, start, end, text, lang)[0] for text in candidates]

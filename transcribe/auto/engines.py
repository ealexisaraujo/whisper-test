"""Local ASR engines for `--backend auto`: one `transcribe_chunks` call per engine over VAD chunks.

Every engine receives the whole 16 kHz mono float32 array plus the chunks, slices each chunk itself
and returns `EngineOutput` with word times in absolute file seconds. Models load lazily on the
first `transcribe_chunks` call and `release()` really frees them (one stage's models at a time).
Engines without word timestamps spread each chunk's words by character length; engines without
probabilities set `prob=None`.
"""

from __future__ import annotations

import gc
import importlib.util
import logging
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np

from ..constants import DEFAULT_MODELS
from ..models import CapabilityError, TranscriptionError
from .types import Chunk, EngineOutput, Word

LOGGER = logging.getLogger("transcribe_youtube")

SAMPLE_RATE = 16000
WHISPER_REPO = "mlx-community/whisper-large-v3-mlx"
QWEN3_ASR_REPO = "mlx-community/Qwen3-ASR-1.7B-bf16"
PARAKEET_REPO = "mlx-community/parakeet-tdt-0.6b-v3"
QWEN_MIN_SECONDS = 1.0  # mlx-audio's Qwen3-ASR drops shorter audio; pad with silence
QWEN_LANGUAGES = {"en": "English", "es": "Spanish"}
QWEN_LANGUAGE_CODES = {"english": "en", "spanish": "es"}
WHISPER_DECODE_PARAMS: Dict[str, Any] = {
    "word_timestamps": True,
    "condition_on_previous_text": False,
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
}


class LocalEngine(Protocol):
    name: str

    def transcribe_chunks(
        self, samples: np.ndarray, chunks: List[Chunk], language: Optional[str]
    ) -> EngineOutput: ...

    def release(self) -> None: ...


# ----------------------------------------------------------------------------- helpers


def spread_words_by_chars(text: str, start: float, end: float) -> List[Word]:
    """Split `text` on whitespace and spread the words over [start, end] by character length.

    Words are contiguous (each ends where the next begins), monotonic and inside the interval.
    """
    tokens = text.split()
    if not tokens:
        return []
    total = sum(len(t) for t in tokens)
    span = max(0.0, end - start)
    words: List[Word] = []
    consumed = 0
    for token in tokens:
        lo = start + span * consumed / total
        consumed += len(token)
        hi = start + span * consumed / total
        words.append(Word(token, lo, hi, None))
    words[-1].end = max(words[-1].start, end)
    return words


def _slice(samples: np.ndarray, chunk: Chunk) -> np.ndarray:
    lo = max(0, int(round(chunk.start * SAMPLE_RATE)))
    hi = min(len(samples), int(round(chunk.end * SAMPLE_RATE)))
    return np.ascontiguousarray(samples[lo:hi], dtype=np.float32)


def _clamp(word: Word, chunk: Chunk) -> Word:
    start = min(max(word.start, chunk.start), chunk.end)
    end = min(max(word.end, start), chunk.end)
    return Word(word.text, start, end, word.prob)


def _free_memory() -> None:
    gc.collect()
    _clear_mlx_cache()
    gc.collect()


def _clear_mlx_cache() -> None:
    """Hand MLX's buffer cache back between chunks.

    Freed Metal buffers stay cached for reuse, and per-chunk shapes differ (word-timestamp alignment, varying
    lengths), so without this the cache grows with audio length: a 30-minute file reached a 22 GB footprint in
    Whisper large-v3 before any model release.
    """
    try:
        import mlx.core as mx

        clear = getattr(mx, "clear_cache", None) or getattr(mx.metal, "clear_cache", None)
        if clear is not None:
            clear()
    except Exception:  # pragma: no cover - mlx missing or no Metal device
        pass


def _majority_language(languages: Sequence[Optional[str]]) -> Optional[str]:
    seen = [lang for lang in languages if lang]
    return Counter(seen).most_common(1)[0][0] if seen else None


def _transcribe_each(
    engine_name: str,
    chunks: Sequence[Chunk],
    transcribe_one: Callable[[Chunk], tuple],
) -> tuple:
    """Run `transcribe_one(chunk) -> (words, detected_language)` per chunk; core-stage errors raise."""
    result: Dict[int, List[Word]] = {}
    detected: List[Optional[str]] = []
    for chunk in chunks:
        try:
            words, lang = transcribe_one(chunk)
        except (CapabilityError, TranscriptionError):
            raise
        except Exception as exc:
            raise TranscriptionError(f"{engine_name} failed on chunk {chunk.index}: {exc}") from exc
        result[chunk.index] = [_clamp(w, chunk) for w in words]
        detected.append(lang)
        _clear_mlx_cache()
    return result, _majority_language(detected)


# ----------------------------------------------------------------------------- Whisper


class WhisperEngine:
    """mlx-whisper large-v3: word timestamps and word probabilities, hallucination thresholds on."""

    name = "whisper-large-v3"

    def __init__(self, repo: str = WHISPER_REPO) -> None:
        self.repo = repo

    @classmethod
    def cache_config(cls) -> Dict[str, Any]:
        return {"repo": WHISPER_REPO, "decode": dict(WHISPER_DECODE_PARAMS)}

    def transcribe_chunks(
        self, samples: np.ndarray, chunks: List[Chunk], language: Optional[str]
    ) -> EngineOutput:
        import mlx_whisper

        def one(chunk: Chunk):
            kwargs: Dict[str, Any] = dict(path_or_hf_repo=self.repo, verbose=None, **WHISPER_DECODE_PARAMS)
            if language:
                kwargs["language"] = language
            out = mlx_whisper.transcribe(_slice(samples, chunk), **kwargs)
            words = [
                Word(
                    str(w["word"]).strip(),
                    chunk.start + float(w["start"]),
                    chunk.start + float(w["end"]),
                    float(w["probability"]) if w.get("probability") is not None else None,
                )
                for segment in out.get("segments", [])
                for w in segment.get("words", [])
                if str(w["word"]).strip()
            ]
            return words, out.get("language")

        result, detected = _transcribe_each(self.name, chunks, one)
        return EngineOutput(self.name, language or detected, result, {})

    def release(self) -> None:
        # mlx-whisper keeps the loaded model in a process-global singleton.
        try:
            from mlx_whisper.transcribe import ModelHolder

            ModelHolder.model = None
            ModelHolder.model_path = None
        except ImportError:  # pragma: no cover
            pass
        _free_memory()


# ----------------------------------------------------------------------------- Qwen3-ASR


class Qwen3AsrEngine:
    """Qwen3-ASR 1.7B through mlx-audio: text only (no word times, no probabilities)."""

    name = "qwen3-asr-1.7b"

    def __init__(self, repo: str = QWEN3_ASR_REPO) -> None:
        self.repo = repo
        self._model: Any = None

    @classmethod
    def cache_config(cls) -> Dict[str, Any]:
        return {"repo": QWEN3_ASR_REPO, "min_seconds": QWEN_MIN_SECONDS, "languages": dict(QWEN_LANGUAGES)}

    def _load(self) -> Any:
        if self._model is None:
            from mlx_audio.stt.utils import load_model

            self._model = load_model(self.repo)
        return self._model

    def transcribe_chunks(
        self, samples: np.ndarray, chunks: List[Chunk], language: Optional[str]
    ) -> EngineOutput:
        model = self._load()
        name = QWEN_LANGUAGES.get(language.lower(), language) if language else None

        def one(chunk: Chunk):
            audio = _slice(samples, chunk)
            if len(audio) < int(QWEN_MIN_SECONDS * SAMPLE_RATE):
                audio = np.pad(audio, (0, int(QWEN_MIN_SECONDS * SAMPLE_RATE) - len(audio)))
            out = model.generate(audio, language=name)
            detected = getattr(out, "language", None)
            if isinstance(detected, list):
                detected = detected[0] if detected else None
            code = QWEN_LANGUAGE_CODES.get(str(detected).lower(), str(detected).lower()) if detected else None
            return spread_words_by_chars(getattr(out, "text", "") or "", chunk.start, chunk.end), code

        result, detected = _transcribe_each(self.name, chunks, one)
        return EngineOutput(self.name, language or detected, result, {})

    def release(self) -> None:
        self._model = None
        _free_memory()


# ----------------------------------------------------------------------------- Parakeet


def words_from_tokens(tokens: Sequence[Any], offset: float = 0.0) -> List[Word]:
    """Group SentencePiece tokens into words (a leading space starts a word); prob is the weakest token."""
    words: List[Word] = []
    pieces: List[Any] = []

    def flush() -> None:
        if not pieces:
            return
        text = "".join(p.text for p in pieces).strip()
        if text:
            confidences = [float(p.confidence) for p in pieces if getattr(p, "confidence", None) is not None]
            words.append(
                Word(
                    text,
                    offset + float(pieces[0].start),
                    offset + float(pieces[-1].end),
                    min(confidences) if confidences else None,
                )
            )
        pieces.clear()

    for token in tokens:
        if token.text.startswith(" ") and pieces:
            flush()
        pieces.append(token)
    flush()
    return words


class ParakeetEngine:
    """Parakeet TDT 0.6B v3 (parakeet-mlx): token timestamps and token confidences."""

    name = "parakeet-v3"

    def __init__(self, repo: str = PARAKEET_REPO) -> None:
        self.repo = repo
        self._model: Any = None

    @classmethod
    def cache_config(cls) -> Dict[str, Any]:
        return {"repo": PARAKEET_REPO}

    def _load(self) -> Any:
        if self._model is None:
            from parakeet_mlx import from_pretrained

            self._model = from_pretrained(self.repo)
        return self._model

    def transcribe_chunks(
        self, samples: np.ndarray, chunks: List[Chunk], language: Optional[str]
    ) -> EngineOutput:
        import soundfile as sf

        model = self._load()

        with tempfile.TemporaryDirectory(prefix="parakeet_") as tmp:
            path = Path(tmp) / "chunk.wav"  # parakeet-mlx's public API takes a file path

            def one(chunk: Chunk):
                sf.write(path, _slice(samples, chunk), SAMPLE_RATE, subtype="PCM_16")
                result = model.transcribe(path)
                return words_from_tokens(result.tokens, offset=chunk.start), None

            result, _ = _transcribe_each(self.name, chunks, one)
        return EngineOutput(self.name, language, result, {})

    def release(self) -> None:
        self._model = None
        _free_memory()


# ----------------------------------------------------------------------------- EG2


class EG2LocalEngine:
    """EmbeddingGemma 2 on Core ML / ANE (English only); word probabilities are not calibrated, so None."""

    name = "eg2"

    def __init__(self, workers: int = 2) -> None:
        self.workers = workers
        self._engine: Any = None

    @classmethod
    def cache_config(cls) -> Dict[str, Any]:
        return {"model": DEFAULT_MODELS["eg2"]}  # `workers` only affects speed, not the words

    def _load(self) -> Any:
        if self._engine is None:
            from ..eg2 import EG2Engine

            self._engine = EG2Engine(workers=self.workers)
        return self._engine

    def transcribe_chunks(
        self, samples: np.ndarray, chunks: List[Chunk], language: Optional[str]
    ) -> EngineOutput:
        if language and language.lower() != "en":
            raise CapabilityError("The eg2 engine is English only; pick another engine for this language")
        engine = self._load()

        def one(chunk: Chunk):
            decoded = engine.transcribe(_slice(samples, chunk))
            tokens = engine.render(decoded)
            return [
                Word(str(token), chunk.start + w.start, chunk.start + w.end, None)
                for w, token in zip(decoded, tokens)
            ], "en"

        result, _ = _transcribe_each(self.name, chunks, one)
        return EngineOutput(self.name, "en", result, {})

    def release(self) -> None:
        self._engine = None
        _free_memory()


# ----------------------------------------------------------------------------- factory

_ENGINES: Dict[str, tuple] = {
    "whisper-large-v3": (WhisperEngine, ("mlx_whisper",)),
    "qwen3-asr-1.7b": (Qwen3AsrEngine, ("mlx_audio",)),
    "parakeet-v3": (ParakeetEngine, ("parakeet_mlx",)),
    "eg2": (EG2LocalEngine, ("coremltools",)),
}


def make_engine(engine_id: str) -> LocalEngine:
    """Build an engine by id without loading its model; unknown or uninstalled ids raise CapabilityError."""
    if engine_id not in _ENGINES:
        raise CapabilityError(f"Unknown auto engine {engine_id!r}; choose one of {', '.join(_ENGINES)}")
    cls, modules = _ENGINES[engine_id]
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        raise CapabilityError(f"Engine {engine_id!r} needs {', '.join(missing)}, which is not installed")
    return cls()


# Engines that return no per-word probabilities: with one of them alone the trust features are constant.
ENGINES_WITHOUT_WORD_PROBS = frozenset({"qwen3-asr-1.7b", "parakeet-v3", "eg2"})
# Engines without native word times: their words are spread over each chunk by character length (approximate).
# Parakeet (token times), Whisper (word times) and eg2 (decoder frame times) have native times.
ENGINES_WITH_APPROX_TIMES = frozenset({"qwen3-asr-1.7b"})


def known_engine_ids() -> List[str]:
    return list(_ENGINES)


def engine_cache_config(engine_id: str) -> Dict[str, Any]:
    """Cache fingerprint of an engine: id, model repo and decoding parameters (no model is loaded)."""
    config: Dict[str, Any] = {"id": engine_id}
    if engine_id in _ENGINES:
        config.update(_ENGINES[engine_id][0].cache_config())
    return config

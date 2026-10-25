"""VAD speech spans (silero-vad) and non-overlapping chunking for the auto pipeline."""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from .types import Chunk

LOGGER = logging.getLogger("transcribe_youtube")

MAX_CHUNK_SECONDS = 30.0
MERGE_GAP_SECONDS = 0.5
# silero-vad parameters this module relies on (its defaults, spelled out so that they are part of
# the cache fingerprint and a library default change cannot silently change cached spans).
VAD_KWARGS = {
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 100,
    "speech_pad_ms": 30,
}
ENERGY_FRAME_SECONDS = 0.1
SPLIT_SEARCH_SECONDS = 10.0
MIN_CHUNK_SECONDS = 1e-3

_VAD_MODEL = None


def _load_vad():
    """Load silero-vad on first use (never at import)."""
    global _VAD_MODEL
    if _VAD_MODEL is None:
        from silero_vad import load_silero_vad

        _VAD_MODEL = load_silero_vad()
    return _VAD_MODEL


def vad_config() -> dict:
    """Cache fingerprint of the VAD + chunking settings (read at call time)."""
    try:
        from importlib.metadata import version

        silero = version("silero-vad")
    except Exception:  # pragma: no cover - package metadata missing
        silero = None
    return {
        "model": "silero-vad",
        "version": silero,
        "vad": dict(VAD_KWARGS),
        "max_chunk_seconds": MAX_CHUNK_SECONDS,
        "merge_gap_seconds": MERGE_GAP_SECONDS,
        "split_search_seconds": SPLIT_SEARCH_SECONDS,
        "energy_frame_seconds": ENERGY_FRAME_SECONDS,
    }


def speech_spans(samples: np.ndarray, sr: int = 16000) -> List[Tuple[float, float]]:
    """Return speech (start, end) spans in seconds for mono float samples using silero-vad."""
    import torch
    from silero_vad import get_speech_timestamps

    if len(samples) == 0:
        return []
    audio = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))
    stamps = get_speech_timestamps(audio, _load_vad(), sampling_rate=sr, return_seconds=True, **VAD_KWARGS)
    return [(float(item["start"]), float(item["end"])) for item in stamps]


def _merge_spans(
    spans: List[Tuple[float, float]], merge_gap: float
) -> List[Tuple[float, float]]:
    merged: List[Tuple[float, float]] = []
    for start, end in sorted((float(s), float(e)) for s, e in spans if e > s):
        if merged and start - merged[-1][1] < merge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _quietest_cut(
    samples: np.ndarray, sr: int, window_start: float, window_end: float, search_seconds: float
) -> float:
    """Midpoint of the lowest-energy 100 ms frame inside the last `search_seconds` of the window."""
    frame = max(1, int(round(ENERGY_FRAME_SECONDS * sr)))
    lo = max(window_start, window_end - search_seconds)
    first = int(np.ceil(lo * sr))
    last = min(int(np.floor(window_end * sr)), len(samples))
    count = (last - first) // frame
    if count < 1:
        return window_end
    block = samples[first:first + count * frame].astype(np.float64).reshape(count, frame)
    energy = np.mean(block * block, axis=1)
    best = int(np.argmin(energy))  # first minimum on ties
    cut = (first + best * frame + frame / 2) / sr
    return float(min(max(cut, window_start + MIN_CHUNK_SECONDS), window_end))


def make_chunks(
    spans: List[Tuple[float, float]],
    samples: np.ndarray,
    sr: int = 16000,
    max_seconds: float = MAX_CHUNK_SECONDS,
    merge_gap: float = MERGE_GAP_SECONDS,
) -> List[Chunk]:
    """Merge spans closer than `merge_gap`, then split anything longer than `max_seconds`.

    A long span is cut at the lowest-energy 100 ms frame inside the last 10 s of the window, so
    chunks never overlap and cover all speech.
    """
    chunks: List[Chunk] = []
    for start, end in _merge_spans(spans, merge_gap):
        cursor = start
        while end - cursor > max_seconds:
            window_end = cursor + max_seconds
            cut = _quietest_cut(samples, sr, cursor, window_end, SPLIT_SEARCH_SECONDS)
            chunks.append(Chunk(index=len(chunks), start=cursor, end=cut))
            cursor = cut
        if end - cursor > MIN_CHUNK_SECONDS:
            chunks.append(Chunk(index=len(chunks), start=cursor, end=end))
    LOGGER.debug("make_chunks: %d spans -> %d chunks", len(spans), len(chunks))
    return chunks

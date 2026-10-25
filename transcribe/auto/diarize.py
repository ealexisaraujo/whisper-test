"""Diarization stage: pyannote community-1 speaker turns (an enhancement stage: failures skip).

Needs a Hugging Face token in `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) with the model terms
accepted at https://huggingface.co/pyannote/speaker-diarization-community-1. Audio is decoded
with soundfile and passed as a waveform dict, because torchcodec (pyannote's default decoder)
does not load against FFmpeg 9.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


def _hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def _skipped(exc: BaseException) -> Tuple[List[dict], str]:
    return [], f"skipped: {type(exc).__name__}: {exc}"


def _best_device():
    import torch

    mps = getattr(torch.backends, "mps", None)
    return torch.device("mps" if mps is not None and mps.is_available() else "cpu")


def _renumber(annotation) -> List[dict]:
    mapping: Dict[str, str] = {}
    turns: List[dict] = []
    for segment, _track, label in annotation.itertracks(yield_label=True):
        key = str(label)
        if key not in mapping:
            mapping[key] = str(len(mapping) + 1)
        turns.append({"start": float(segment.start), "end": float(segment.end), "speaker": mapping[key]})
    turns.sort(key=lambda t: (t["start"], t["end"]))
    return turns


def diarize(audio_path: Path, num_speakers: Optional[int] = None) -> Tuple[List[dict], str]:
    """Return `(turns, status)`; turns are `{"start","end","speaker"}` with speakers "1", "2", ..."""
    token = _hf_token()
    if not token:
        return [], (
            "skipped: no HF_TOKEN (accept the model terms at "
            "https://huggingface.co/pyannote/speaker-diarization-community-1 and set HF_TOKEN)"
        )
    pipeline = None
    try:
        import soundfile as sf
        import torch
        from pyannote.audio import Pipeline

        data, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T.copy())
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)
        if pipeline is None:  # pyannote returns None (no exception) when the gated model terms are not accepted
            return [], (
                f"skipped: could not load {DIARIZATION_MODEL}: accept the model terms at "
                f"https://huggingface.co/{DIARIZATION_MODEL} with the account of HF_TOKEN"
            )
        pipeline.to(_best_device())
        kwargs = {"num_speakers": num_speakers} if num_speakers is not None else {}
        output = pipeline({"waveform": waveform, "sample_rate": int(sample_rate)}, **kwargs)
        annotation = getattr(output, "speaker_diarization", output)
        return _renumber(annotation), "ok"
    except Exception as exc:  # enhancement stage: never abort the pipeline
        logger.warning("diarization skipped: %s: %s", type(exc).__name__, exc)
        return _skipped(exc)
    finally:
        del pipeline
        try:
            import gc

            gc.collect()
            import torch

            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

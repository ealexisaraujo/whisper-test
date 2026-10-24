"""EmbeddingGemma 2 transcription engine (Core ML, Apple Neural Engine).

Python port of the pipeline reverse-engineered from Embedding Gemma Transcribe 1.6.
See docs/agents/eg2-engine-analysis.md for the evidence behind every constant here.

audio 16 kHz -> log-mel [2000x128] per 20 s window -> EG2 audio encoder (layer 11) [500x1024]
-> reader: per-frame vector in EG2 text space [500x768] + blank logit
-> block scorer (top-16 blocks of 4 words) -> exact re-rank against 229k word vectors
-> word if (bias + scale * dot) > blank, else blank -> word-level CTC collapse.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .models import TranscriptionError

LOGGER = logging.getLogger("transcribe_youtube")

SAMPLE_RATE = 16000
FRAME_SECONDS = 0.04  # one encoder/reader frame
WINDOW_SAMPLES = 320_000  # 20 s
STEP_SAMPLES = 288_000  # 18 s -> 2 s overlap
MEL_FRAMES = 2000
ENC_FRAMES = 500
KEEP_START = 25  # frames dropped at the start of every window but the first (1 s)
KEEP_END = 475  # frames kept at the end of every window but the last (19 s)
HOP = 160
WIN = 320
N_FFT = 512
LOG_FLOOR = 1e-3
REP_TOKEN = "<rep>"
PUNCT_BLOCK = 96

DEFAULT_MODEL_BASE_URL = "https://haku-models.tslfiles.org/eg2demo/"
DEFAULT_MANIFEST = "v5/manifest.json"


def default_model_dir() -> Path:
    override = os.getenv("EG2_MODEL_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "whisper_translate" / "eg2" / "v5"


@dataclass
class EG2Word:
    text: str
    start: float
    end: float
    score: float  # calibrated word logit minus blank logit, averaged over the word's frames


# --------------------------------------------------------------------------- assets


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_models(
    model_dir: Optional[Path] = None,
    *,
    base_url: Optional[str] = None,
    manifest_path: str = DEFAULT_MANIFEST,
) -> Path:
    """Download (once) and SHA-256 verify every file listed in the EG2 manifest."""
    model_dir = Path(model_dir or default_model_dir())
    base_url = base_url or os.getenv("EG2_MODEL_BASE_URL", DEFAULT_MODEL_BASE_URL)
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = model_dir / "manifest.json"
    if not manifest_file.exists():
        LOGGER.info("Fetching EG2 manifest %s%s", base_url, manifest_path)
        manifest_file.write_bytes(_http_get(base_url + manifest_path))
    manifest = json.loads(manifest_file.read_text())
    stamp = model_dir / ".verified"
    if stamp.exists() and stamp.read_text().strip() == str(manifest.get("version")):
        return model_dir

    for entry in manifest["files"]:
        target = model_dir / entry["path"]
        if target.exists() and target.stat().st_size == entry["size"] and _sha256(target) == entry["sha256"]:
            continue
        LOGGER.info("Downloading EG2 model file %s (%d bytes)", entry["path"], entry["size"])
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        partial.write_bytes(_http_get(base_url + entry["url"]))
        if _sha256(partial) != entry["sha256"]:
            partial.unlink(missing_ok=True)
            raise TranscriptionError(f"EG2 model file failed SHA-256 check: {entry['path']}")
        partial.replace(target)
    stamp.write_text(str(manifest.get("version")))
    return model_dir


def _http_get(url: str) -> bytes:
    # The model host rejects Python's default User-Agent.
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8.7.1"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return response.read()


# --------------------------------------------------------------------------- front end


class LogMel:
    """Exact port of LogMel.callAsFunction (vDSP magnitude spectrum, ln(mel + 1e-3))."""

    def __init__(self, model_dir: Path) -> None:
        self.hann = np.fromfile(model_dir / "hann_320.f32", dtype="<f4")
        self.filters = np.fromfile(model_dir / "mel_filters_257x128.f32", dtype="<f4").reshape(N_FFT // 2 + 1, 128)
        if self.hann.shape != (WIN,):
            raise TranscriptionError("EG2 mel or window file has the wrong size")

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        n = int(samples.shape[0])
        padded = np.zeros(((n + 127) // 128) * 128 + HOP, dtype=np.float32)
        padded[HOP : HOP + n] = samples
        total = (padded.shape[0] - (WIN + 1)) // HOP + 1
        valid = sum(1 for i in range(total) if HOP * (i + 1) < n)
        if valid <= 0:
            return np.zeros((0, 128), dtype=np.float32)
        idx = np.arange(WIN)[None, :] + HOP * np.arange(valid)[:, None]
        frames = padded[idx] * self.hann
        spectrum = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)).astype(np.float32)
        return np.log(spectrum @ self.filters + LOG_FLOOR)


def window_starts(num_samples: int) -> List[int]:
    """Window start offsets exactly as EG2Transcriber.transcribeShared lays them out."""
    starts = [0]
    if num_samples > WINDOW_SAMPLES:
        offset = 0
        while True:
            starts.append(offset + STEP_SAMPLES)
            reach = offset + STEP_SAMPLES + WINDOW_SAMPLES
            offset += STEP_SAMPLES
            if reach >= num_samples:
                break
    return starts


# --------------------------------------------------------------------------- engine


class EG2Engine:
    def __init__(
        self,
        model_dir: Optional[Path] = None,
        *,
        compute_units: str = "CPU_AND_NE",
        workers: int = 2,
        gpu_encoder: bool = False,
        punctuate: bool = True,
    ) -> None:
        try:
            import coremltools as ct
        except ImportError as exc:  # pragma: no cover - depends on platform
            raise TranscriptionError("The eg2 backend needs coremltools (pip install coremltools) on macOS") from exc

        self.model_dir = ensure_models(model_dir)
        units = getattr(ct.ComputeUnit, compute_units)
        self.mel = LogMel(self.model_dir)
        reader_cfg = json.loads((self.model_dir / "reader_P6.json").read_text())
        self.scale = float(reader_cfg["scale"])
        self.bias = float(reader_cfg["bias"])
        self.words: List[str] = json.loads((self.model_dir / "reader_words.json").read_text())
        self.units = np.fromfile(self.model_dir / "word_unit_f16.bin", dtype="<f2").reshape(-1, 768)
        if self.units.shape[0] != len(self.words):
            raise TranscriptionError("EG2 word_unit_f16.bin does not match reader_words.json")
        self.rep_id = self.words.index(REP_TOKEN)

        load = lambda name, cu=units: ct.models.CompiledMLModel(str(self.model_dir / name), compute_units=cu)
        # One encoder+reader pair per worker so CPU/GPU work of one window overlaps the ANE work of
        # the next. gpu_encoder puts extra workers' encoders on the GPU (as the app's Front does);
        # the reader always stays on the ANE: on CPU/GPU it emits NaN for speech frames.
        ane = ct.ComputeUnit.CPU_AND_NE
        gpu = ct.ComputeUnit.CPU_AND_GPU
        encoder_units = [units] + [gpu if gpu_encoder else units] * max(0, workers - 1)
        self._pairs = [(load("eg2_audio_l11.mlmodelc", cu), load("reader_P6.mlmodelc", ane)) for cu in encoder_units]
        # The block scorer is a 500x128x229k matmul: 19 ms on the GPU vs 47 ms on the ANE (M1 Pro),
        # and the exact re-rank absorbs the few top-block differences.
        self.scorer = load("scorer_blk_d128_k16_b4_t500.mlmodelc", gpu)
        self.punct = None
        if punctuate:
            self.punct = HeadOnlyPunct(self.model_dir, self.words, self.units, load("punct_headonly.mlmodelc"))
        self._scorer_lock = threading.Lock()

    # -- per window ----------------------------------------------------------

    def _front(self, pair_index: int, chunk: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        encoder, reader = self._pairs[pair_index]
        feats = self.mel(chunk)
        valid_mel = min(feats.shape[0], MEL_FRAMES)
        x = np.zeros((1, MEL_FRAMES, 128), dtype=np.float32)
        x[0, :valid_mel] = feats[:valid_mel]
        mask = np.zeros((1, MEL_FRAMES), dtype=np.float32)
        mask[0, :valid_mel] = 1.0
        l11 = encoder.predict({"input_features": x, "feature_mask": mask})["l11"]
        valid = min(ENC_FRAMES, (valid_mel + 3) // 4)
        enc_mask = np.zeros((1, ENC_FRAMES), dtype=np.float32)
        enc_mask[0, :valid] = 1.0
        out = reader.predict({"l11": l11.astype(np.float32), "mask": enc_mask})
        return out["vec"][0], out["blank"][0, :, 0].astype(np.float32), valid

    def _back(self, vec: np.ndarray, blank: np.ndarray, valid: int) -> Tuple[np.ndarray, np.ndarray]:
        """Per-frame word id (-1 = blank) and margin, as EG2Transcriber.back + Rerank.run."""
        with self._scorer_lock:
            blocks = self.scorer.predict({"vec": vec[None].astype(np.float16)})["bidx"][0, :valid]
        cand = (blocks[:, :, None] * 4 + np.arange(4)).reshape(valid, -1)
        cand = np.minimum(cand, len(self.words) - 1)
        dots = np.einsum("td,tkd->tk", vec[:valid].astype(np.float32), self.units[cand].astype(np.float32))
        best = dots.argmax(axis=1)
        word_ids = cand[np.arange(valid), best]
        margin = self.bias + self.scale * dots[np.arange(valid), best] - blank[:valid]
        word_ids = np.where(margin > 0, word_ids, -1)
        return word_ids, margin

    # -- full file -----------------------------------------------------------

    def transcribe(
        self,
        samples: np.ndarray,
        *,
        on_window: Optional[Callable[[int, int], None]] = None,
    ) -> List[EG2Word]:
        samples = np.asarray(samples, dtype=np.float32)
        starts = window_starts(samples.shape[0])
        last = len(starts) - 1

        def run(i: int):
            chunk = samples[starts[i] : starts[i] + WINDOW_SAMPLES]
            vec, blank, valid = self._front(i % len(self._pairs), chunk)
            ids, margin = self._back(vec, blank, valid)
            return ids, margin, valid

        collapse = _Collapse(self.words, self.rep_id)
        with ThreadPoolExecutor(max_workers=len(self._pairs)) as pool:
            for i, (ids, margin, valid) in enumerate(pool.map(run, range(len(starts)))):
                t0 = starts[i] / SAMPLE_RATE
                keep_from = 0 if i == 0 else min(valid, KEEP_START)
                keep_to = valid if i == last else min(valid, KEEP_END)
                collapse.feed(ids[keep_from:keep_to], margin[keep_from:keep_to], t0 + keep_from * FRAME_SECONDS)
                if i == last:
                    collapse.close(t0 + valid * FRAME_SECONDS)
                if on_window:
                    on_window(i + 1, len(starts))
        return collapse.words

    def render(self, words: Sequence[EG2Word]) -> List[str]:
        """Apply punctuation/case to decoded words and return display tokens."""
        tokens = [w.text for w in words]
        if self.punct is None or not tokens:
            return tokens
        return self.punct.apply(tokens)


def words_to_segments(
    words: Sequence[EG2Word],
    tokens: Sequence[str],
    *,
    max_words: int = 14,
    max_gap: float = 1.0,
    tail: float = 0.5,
) -> List[dict]:
    """Group decoded words into subtitle-sized segments.

    CTC fires one 40 ms spike per word, so a segment ends at the next segment's start when
    speech is continuous, or `tail` seconds after its last spike when a pause follows.
    """
    segments: List[dict] = []
    current: List[int] = []

    def close(next_start: Optional[float]) -> None:
        if not current:
            return
        start = words[current[0]].start
        last = words[current[-1]].start
        end = last + tail if next_start is None else min(next_start, last + tail + max_gap)
        segments.append({"start": round(start, 3), "end": round(max(end, last + FRAME_SECONDS), 3),
                         "speaker": None, "text": " ".join(tokens[i] for i in current)})
        current.clear()

    for i, (word, token) in enumerate(zip(words, tokens)):
        if current and word.start - words[current[-1]].start > max_gap:
            close(word.start)
        current.append(i)
        if token[-1:] in ".?" or len(current) >= max_words:
            close(words[i + 1].start if i + 1 < len(words) else None)
    close(None)
    return segments


class _Collapse:
    """Word-level CTC collapse from Stream.decode / Stream.closeRun."""

    def __init__(self, words: List[str], rep_id: int) -> None:
        self.vocab = words
        self.rep_id = rep_id
        self.prev = -2
        self.pending: Optional[List] = None  # [text, start, margin_sum, frames]
        self.last_text: Optional[str] = None
        self.words: List[EG2Word] = []

    def _flush(self, t: float) -> None:
        if self.pending is not None:
            text, start, total, count = self.pending
            self.words.append(EG2Word(text, start, t, total / max(count, 1)))
            self.pending = None

    def feed(self, ids: np.ndarray, margin: np.ndarray, t0: float) -> None:
        for j, wid in enumerate(ids.tolist()):
            t = t0 + j * FRAME_SECONDS
            if wid != self.prev:
                self._flush(t)
                if wid >= 0:
                    if wid == self.rep_id:
                        if self.last_text is not None:
                            self.pending = [self.last_text, t, 0.0, 0]
                    else:
                        text = self.vocab[wid]
                        self.pending = [text, t, 0.0, 0]
                        self.last_text = text
            if wid >= 0 and self.pending is not None:
                self.pending[2] += float(margin[j])
                self.pending[3] += 1
            self.prev = wid

    def close(self, t: float) -> None:
        self._flush(t)


# --------------------------------------------------------------------------- punctuation


class HeadOnlyPunct:
    """Punctuation + casing from the decoded words' text vectors (punct_headonly.mlmodelc)."""

    PUNCT = ("", ",", ".", "?")

    def __init__(self, model_dir: Path, words: List[str], units: np.ndarray, model) -> None:
        self.index = {w: i for i, w in enumerate(words)}
        self.units = units
        self.model = model
        self.case_of = json.loads((model_dir / "punct_case.json").read_text())

    def _run(self, tokens: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        n = len(tokens)
        vec = np.zeros((1, PUNCT_BLOCK, 768), dtype=np.float16)
        mask = np.zeros((1, PUNCT_BLOCK), dtype=np.float16)
        for i, tok in enumerate(tokens):
            idx = self.index.get(tok)
            if idx is not None:
                vec[0, i] = self.units[idx]
            mask[0, i] = 1.0
        out = self.model.predict({"vec": vec, "mask": mask})
        return out["punct"][0, :n].argmax(-1), out["case"][0, :n].argmax(-1)

    def decide(self, tokens: Sequence[str]) -> Tuple[List[int], List[int]]:
        punct: List[int] = []
        case: List[int] = []
        for start in range(0, len(tokens), PUNCT_BLOCK):
            p, c = self._run(tokens[start : start + PUNCT_BLOCK])
            punct.extend(p.tolist())
            case.extend(c.tolist())
        return punct, case

    def apply(self, tokens: Sequence[str]) -> List[str]:
        punct, case = self.decide(tokens)
        out = []
        for tok, p, c in zip(tokens, punct, case):
            # HeadOnlyPunct.run: lexicon 2 (acronym) always wins; lexicon 1 only lifts a lowercase guess.
            forced = self.case_of.get(tok)
            if forced == 2 or (forced == 1 and c == 0):
                c = forced
            if c == 1:
                tok = tok[:1].upper() + tok[1:]
            elif c == 2:
                tok = tok.upper()
            out.append(tok + self.PUNCT[p])
        return out

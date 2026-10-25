"""Eval datasets for the self-grading suite: public corpora with human references, written as JSONL manifests.

    .venv/bin/python -m benchmarks.suite.datasets prepare [--smoke-only] [--slices a,b] [--force]

Audio (16 kHz mono PCM WAV) goes to ~/.cache/whisper_translate/suite/audio/<slice>/, never into the repo. One manifest
per slice lands in benchmarks/suite/manifests/<slice>.jsonl plus manifests/smoke.jsonl (about 30 min, one short item
per slice). The manifests embed the reference transcripts (TED-LIUM is CC BY-NC-ND) so they are git-ignored and local-only;
audio paths in them are relative to the data root (absolute paths from older manifests still read fine). Slices, sources and minutes are defined in slices.json; a source that cannot be read is skipped and
recorded under "gaps" there, and the rest of prepare carries on.

Sources are read straight from the Hub's parquet files with row-group reads (no whole-corpus downloads, no
`datasets`/torchcodec: audio is decoded with soundfile + librosa).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

SR = 16000
HERE = Path(__file__).resolve().parent
SLICES_JSON = HERE / "slices.json"
MANIFEST_DIR = HERE / "manifests"
DATA_ROOT = Path.home() / ".cache" / "whisper_translate" / "suite"
AUDIO_DIR = DATA_ROOT / "audio"
SLICE_ORDER = ["en_clean", "en_talks", "en_accents", "en_business", "en_meeting", "es_fleurs", "es_mls",
               "es_dialects", "en_noise", "es_noise", "en_phone", "es_phone", "nospeech"]

Item = Dict[str, Any]


def log(msg: str) -> None:
    print(f"[suite] {msg}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------- pure audio helpers


def assign_split(item_id: str) -> str:
    """Deterministic 50/50 dev/test split from a hash of the item id."""
    return "dev" if int(hashlib.sha256(item_id.encode("utf-8")).hexdigest(), 16) % 2 == 0 else "test"


MARKUP = re.compile(r"<[^<>\s][^<>]*>|IGNORE_TIME_SEGMENT_IN_SCORING")


def has_markup(text: str) -> bool:
    """True for annotation tags (`<unk>`, `<inaudible>`, `<laugh>`, ...) and EDACC's ignore marker."""
    return bool(MARKUP.search(text or ""))


def clean_text(text: str) -> str:
    """Whitespace-normalise a reference. Segments with markup are dropped by the builders (`has_markup`), never
    silently edited, so the audio and the reference always cover the same speech."""
    return re.sub(r"\s+", " ", text or "").strip()


def spread_order(count: int) -> List[int]:
    """A visiting order over `count` shards that jumps around (stride of about count/3, coprime with count)."""
    if count <= 1:
        return list(range(count))
    stride = next(s for s in range(max(1, count // 3), count + 2) if math.gcd(s, count) == 1)
    return [(i * stride) % count for i in range(count)]


def build_longform(utterances: Iterable[Dict[str, Any]], target_seconds: float,
                   gap_seconds: float = 0.6) -> Tuple[np.ndarray, str, List[Dict[str, Any]]]:
    """Concatenate utterances (dicts with `audio`, `text`, optional `speaker`) with silent gaps until the file is
    at least `target_seconds` long. Returns (audio, reference, speaker turns), one turn per utterance."""
    target = int(round(target_seconds * SR))
    gap = np.zeros(int(round(gap_seconds * SR)), dtype=np.float32)
    parts: List[np.ndarray] = []
    words: List[str] = []
    turns: List[Dict[str, Any]] = []
    total = 0
    for utt in utterances:
        if parts:
            parts.append(gap)
            total += len(gap)
        audio = np.asarray(utt["audio"], dtype=np.float32)
        start = total
        parts.append(audio)
        total += len(audio)
        words.append(clean_text(utt["text"]))
        turns.append({"start": round(start / SR, 3), "end": round(total / SR, 3), "speaker": utt.get("speaker")})
        if total >= target:
            break
    if not parts:
        raise ValueError("build_longform got no utterances")
    return np.concatenate(parts), " ".join(w for w in words if w), turns


def assemble_timeline(segments: Sequence[Dict[str, Any]], max_gap: float = 1.0) -> Tuple[np.ndarray, str, List[Dict[str, Any]]]:
    """Lay segments (`start`, `end` in source seconds, `audio`, `text`, `speaker`) back on their real timeline.
    Overlapping segments overwrite each other (they come from one microphone, so the overlap is the same signal);
    silence longer than `max_gap` seconds between segments is shortened to `max_gap`."""
    segs = sorted(segments, key=lambda s: (s["start"], s["end"]))
    if not segs:
        raise ValueError("assemble_timeline got no segments")
    shift = segs[0]["start"]
    covered_until = segs[0]["start"]
    placed: List[Tuple[float, Dict[str, Any]]] = []
    for seg in segs:
        gap = seg["start"] - covered_until
        if gap > max_gap:
            shift += gap - max_gap
        placed.append((seg["start"] - shift, seg))
        covered_until = max(covered_until, seg["start"] + len(seg["audio"]) / SR)
    length = max(int(round(offset * SR)) + len(seg["audio"]) for offset, seg in placed)
    out = np.zeros(length, dtype=np.float32)
    words: List[str] = []
    turns: List[Dict[str, Any]] = []
    for offset, seg in placed:
        begin = int(round(offset * SR))
        out[begin:begin + len(seg["audio"])] = seg["audio"]
        words.append(clean_text(seg["text"]))
        turns.append({"start": round(begin / SR, 3), "end": round((begin + len(seg["audio"])) / SR, 3),
                      "speaker": seg.get("speaker")})
    return out, " ".join(w for w in words if w), turns


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add `noise` (tiled or cut to the speech length) at the given speech-to-noise power ratio in dB."""
    speech = np.asarray(speech, dtype=np.float32)
    noise = np.resize(np.asarray(noise, dtype=np.float32), len(speech))
    p_speech = float(np.mean(speech.astype(np.float64) ** 2))
    p_noise = float(np.mean(noise.astype(np.float64) ** 2))
    if p_speech == 0.0 or p_noise == 0.0:
        return speech.copy()
    scale = np.sqrt(p_speech / (p_noise * 10 ** (snr_db / 10.0)))
    mixed = (speech + noise * scale).astype(np.float32)
    peak = float(np.max(np.abs(mixed)))
    return mixed * (0.99 / peak) if peak > 0.99 else mixed  # a global gain keeps the SNR


def phone_band(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Telephone channel: 300-3400 Hz band limit, 8 kHz mu-law round trip through ffmpeg, back at `sr`."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH")
    audio = np.asarray(audio, dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        src, mid, dst = (Path(tmp) / n for n in ("src.wav", "mid.wav", "dst.wav"))
        sf.write(src, audio, sr, subtype="FLOAT")
        base = [ffmpeg, "-y", "-loglevel", "error", "-i"]
        subprocess.run(base + [str(src), "-af", "highpass=f=300,lowpass=f=3400", "-ar", "8000", "-ac", "1",
                               "-c:a", "pcm_mulaw", str(mid)], check=True)
        subprocess.run(base + [str(mid), "-ar", str(sr), "-ac", "1", "-c:a", "pcm_f32le", str(dst)], check=True)
        out, _ = sf.read(dst, dtype="float32")
    if len(out) < len(audio):
        out = np.pad(out, (0, len(audio) - len(out)))
    return out[:len(audio)].astype(np.float32)


def make_babble(sources: Sequence[np.ndarray], n_samples: int, rng: np.random.Generator, n_voices: int = 6) -> np.ndarray:
    """Babble noise: `n_voices` random crops of other speech, each at unit RMS, summed."""
    out = np.zeros(n_samples, dtype=np.float64)
    picks = rng.choice(len(sources), size=min(n_voices, len(sources)), replace=False)
    for index in picks:
        src = np.asarray(sources[int(index)], dtype=np.float64)
        if len(src) < n_samples:
            src = np.resize(src, n_samples)
        begin = int(rng.integers(0, len(src) - n_samples + 1))
        crop = src[begin:begin + n_samples]
        rms = np.sqrt(np.mean(crop ** 2))
        out += crop / rms if rms > 0 else crop
    return out.astype(np.float32)


def make_pink(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise by shaping white noise in the frequency domain."""
    spectrum = np.fft.rfft(rng.standard_normal(n_samples))
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    pink = np.fft.irfft(spectrum / np.sqrt(freqs), n=n_samples)
    return (pink / np.max(np.abs(pink))).astype(np.float32)


def plan_targets(minutes: float, rng: np.random.Generator, first_seconds: float = 120.0,
                 low: float = 200.0, high: float = 420.0) -> List[float]:
    """Item lengths for a slice: one short item (it feeds the smoke set), then 3-7 minute items up to the cap."""
    budget = minutes * 60.0
    targets = [min(first_seconds, budget)]
    used = targets[0]
    while budget - used >= 100.0:
        length = min(float(rng.uniform(low, high)), budget - used)
        targets.append(length)
        used += length
    return targets


def slice_rng(name: str, salt: str = "") -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(f"{name}:{salt}".encode()).hexdigest()[:8], 16))


def decode_audio(blob: bytes) -> np.ndarray:
    """Decode an audio blob to 16 kHz mono float32 with soundfile (torchcodec cannot load with FFmpeg 9)."""
    data, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if rate != SR:
        import librosa

        data = librosa.resample(data, orig_sr=rate, target_sr=SR)
    return np.ascontiguousarray(data, dtype=np.float32)


# ----------------------------------------------------------------------------- remote parquet access

_FS = None


def hf_fs():
    global _FS
    if _FS is None:
        from huggingface_hub import HfFileSystem

        _FS = HfFileSystem()
    return _FS


def _retry(fn: Callable[[], Any], attempts: int = 4) -> Any:
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - network flakiness: back off and retry, then re-raise
            if attempt == attempts - 1:
                raise
            log(f"retry after {type(exc).__name__}: {str(exc)[:120]}")
            time.sleep(2 * (attempt + 1))


class RemoteParquet:
    """Row-group reads of one parquet file on the Hub (`datasets/<repo>/<path>`)."""

    def __init__(self, path: str):
        import pyarrow.parquet as pq

        self.path = path
        self._file = _retry(lambda: hf_fs().open(path, "rb"))
        self.pf = pq.ParquetFile(self._file)
        meta = self.pf.metadata
        sizes = [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]
        self.bounds = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
        self.num_rows = int(self.bounds[-1])

    def columns(self, names: Sequence[str]) -> Dict[str, list]:
        return _retry(lambda: self.pf.read(columns=list(names)).to_pydict())

    def rows(self, indices: Sequence[int], columns: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        """Rows at the given global indices (in the given order), reading only the row groups they live in."""
        wanted: Dict[int, List[int]] = {}
        for index in indices:
            group = int(np.searchsorted(self.bounds, index, side="right") - 1)
            wanted.setdefault(group, []).append(int(index) - int(self.bounds[group]))
        found: Dict[int, Dict[str, Any]] = {}
        for group, local in wanted.items():
            table = _retry(lambda g=group: self.pf.read_row_group(g, columns=list(columns) if columns else None))
            for pos, row in zip(sorted(set(local)), table.take(sorted(set(local))).to_pylist()):
                found[int(self.bounds[group]) + pos] = row
        return [found[int(i)] for i in indices]

    def iter_rows(self, start: int, columns: Optional[Sequence[str]] = None) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """(global index, row) from `start` on, one row group at a time."""
        for group in range(int(np.searchsorted(self.bounds, start, side="right") - 1), len(self.bounds) - 1):
            table = _retry(lambda g=group: self.pf.read_row_group(g, columns=list(columns) if columns else None))
            base = int(self.bounds[group])
            for offset, row in enumerate(table.to_pylist()):
                if base + offset >= start:
                    yield base + offset, row


def glob_parquet(pattern: str) -> List[str]:
    return sorted(_retry(lambda: hf_fs().glob(pattern)))


# ----------------------------------------------------------------------------- utterance pools


class UttPool:
    """Utterances grouped by speaker; `stream` hands them out round-robin in short consecutive runs, decoding lazily."""

    def __init__(self, utts: Sequence[Dict[str, Any]], dropped: int = 0):
        self.dropped = dropped  # utterances left out because their transcript had markup or was empty
        self.by_speaker: "OrderedDict[str, Deque[Dict[str, Any]]]" = OrderedDict()
        for utt in utts:
            self.by_speaker.setdefault(str(utt.get("speaker")), deque()).append(utt)

    def minutes_left(self) -> float:
        return sum(len(u["raw"]) for q in self.by_speaker.values() for u in q) / 1e6  # rough, only for logging

    def stream(self, rng: np.random.Generator, chunk: int = 3) -> Iterator[Dict[str, Any]]:
        speakers = list(self.by_speaker)
        offset = int(rng.integers(len(speakers)))
        order = speakers[offset:] + speakers[:offset]
        while any(self.by_speaker[s] for s in speakers):
            for spk in order:
                queue = self.by_speaker[spk]
                for _ in range(min(chunk, len(queue))):
                    utt = queue.popleft()
                    yield {"audio": decode_audio(utt["raw"]), "text": utt["text"], "speaker": utt["speaker"]}


def load_pool(path: str, text_col: str, speaker_col: Optional[str], max_speakers: int, per_speaker_seconds: float,
              chars_per_second: float, label: str, audio_col: str = "audio") -> UttPool:
    """Pick evenly spaced speakers and the first utterances of each (about `per_speaker_seconds`, estimated from
    the text length), then read only the row groups that hold them."""
    src = RemoteParquet(path)
    cols = [text_col] + ([speaker_col] if speaker_col else [])
    meta = src.columns(cols)
    texts = meta[text_col]
    speakers = [str(s) for s in meta[speaker_col]] if speaker_col else ["-"] * len(texts)
    unique = sorted(set(speakers), key=lambda s: (len(s), s))
    stride = max(1, len(unique) // max_speakers)
    chosen = unique[::stride][:max_speakers]
    picked: List[int] = []
    for spk in chosen:
        seconds = 0.0
        for index, who in enumerate(speakers):
            if who == spk:
                picked.append(index)
                seconds += len(texts[index]) / chars_per_second
                if seconds >= per_speaker_seconds:
                    break
    rows = src.rows(picked, [audio_col, text_col])
    keep = [(i, r) for i, r in zip(picked, rows) if clean_text(r[text_col]) and not has_markup(r[text_col])]
    utts = [{"speaker": None if speaker_col is None else speakers[i], "text": clean_text(r[text_col]),
             "raw": r[audio_col]["bytes"]} for i, r in keep]
    log(f"{label}: pool of {len(utts)} utterances from {len(chosen)} speakers ({len(rows) - len(keep)} dropped for markup)")
    return UttPool(utts, dropped=len(rows) - len(keep))


# ----------------------------------------------------------------------------- manifest plumbing


class Ctx:
    def __init__(self, specs: Dict[str, Any], smoke_only: bool):
        self.specs = specs
        self.smoke_only = smoke_only
        self.built: Dict[str, List[Item]] = {}
        self.gaps: Dict[str, str] = {}
        self.dropped: Dict[str, int] = {}  # segments left out per slice because of markup

    def drop(self, name: str, count: int) -> None:
        self.dropped[name] = self.dropped.get(name, 0) + count

    def targets(self, name: str, minutes: Optional[float] = None) -> List[float]:
        full = plan_targets(minutes if minutes is not None else self.specs["slices"][name]["minutes"], slice_rng(name, "plan"),
                            first_seconds=float(self.specs.get("smoke_item_seconds", 120)))
        return full[:1] if self.smoke_only else full


def emit(ctx: Ctx, name: str, key: str, audio: np.ndarray, reference: str, speakers: Optional[List[Dict[str, Any]]],
         meta: Optional[Dict[str, Any]] = None, split: Optional[str] = None) -> Item:
    spec = ctx.specs["slices"][name]
    item_id = f"{name}-{key}"
    path = AUDIO_DIR / name / f"{item_id}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    sf.write(path, audio, SR, subtype="PCM_16")
    if speakers is not None and all(t.get("speaker") is None for t in speakers):
        speakers = None
    return {"id": item_id, "slice": name, "split": split or assign_split(item_id), "lang": spec["lang"],
            "audio": str(path), "duration": round(len(audio) / SR, 2), "reference": reference, "speakers": speakers,
            "condition": spec["condition"], "meta": meta or {}}


def resolve_audio(audio: Any, root: Optional[Path] = None) -> Path:
    """Audio path of a manifest item: absolute paths are kept, relative ones resolve against the data root."""
    path = Path(audio)
    return path if path.is_absolute() else Path(root if root is not None else DATA_ROOT) / path


def relative_audio(audio: Any, root: Optional[Path] = None) -> str:
    """Manifest form of an audio path: relative to the data root when it lives under it, else unchanged."""
    path = Path(audio)
    try:
        return path.relative_to(root if root is not None else DATA_ROOT).as_posix()
    except ValueError:
        return str(audio)


def read_manifest(name: str) -> List[Item]:
    """Items of a manifest with `audio` resolved to an absolute path ([] when missing or any audio is absent)."""
    path = MANIFEST_DIR / f"{name}.jsonl"
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    for item in items:
        item["audio"] = str(resolve_audio(item["audio"]))
    return items if items and all(Path(i["audio"]).exists() for i in items) else []


def write_manifest(name: str, items: Sequence[Item]) -> None:
    """Write the manifest with audio paths relative to the data root (portable; the manifests are git-ignored)."""
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_DIR / f"{name}.jsonl", "w", encoding="utf-8") as fh:
        for item in sorted(items, key=lambda i: i["id"]):
            fh.write(json.dumps({**item, "audio": relative_audio(item["audio"])}, ensure_ascii=False) + "\n")


def minutes(items: Sequence[Item]) -> float:
    return sum(i["duration"] for i in items) / 60.0


def pool_items(ctx: Ctx, name: str, pool: UttPool, targets: Sequence[float], key_prefix: str, chunk: int = 3,
               meta: Optional[Dict[str, Any]] = None, gap_seconds: float = 0.6) -> List[Item]:
    items = []
    ctx.drop(name, pool.dropped)
    for n, target in enumerate(targets):
        rng = slice_rng(name, f"{key_prefix}{n}")
        audio, reference, turns = build_longform(pool.stream(rng, chunk), target, gap_seconds)
        items.append(emit(ctx, name, f"{key_prefix}{n:02d}", audio, reference, turns, meta))
        log(f"{name}: {items[-1]['id']} {items[-1]['duration']:.0f}s")
    return items


# ----------------------------------------------------------------------------- slice builders


def build_en_clean(ctx: Ctx) -> List[Item]:
    minutes_total = ctx.specs["slices"]["en_clean"]["minutes"]
    clean_minutes = round(minutes_total * 0.75)
    items = pool_items(ctx, "en_clean", load_pool(
        "datasets/openslr/librispeech_asr/clean/test/0000.parquet", "text", "speaker_id", 24, 150, 14.0,
        "librispeech test-clean"), ctx.targets("en_clean", clean_minutes), "c", meta={"source": "test-clean"})
    if not ctx.smoke_only:
        other = plan_targets(minutes_total - clean_minutes, slice_rng("en_clean", "other"), first_seconds=240)
        items += pool_items(ctx, "en_clean", load_pool(
            "datasets/openslr/librispeech_asr/other/test/0000.parquet", "text", "speaker_id", 16, 120, 14.0,
            "librispeech test-other"), other, "o", meta={"source": "test-other"})
    return items


def build_en_talks(ctx: Ctx) -> List[Item]:
    spec_minutes = ctx.specs["slices"]["en_talks"]["minutes"]
    candidates: List[Tuple[str, int, int]] = []
    for path in glob_parquet("datasets/distil-whisper/tedlium-long-form/data/*.parquet"):
        src = RemoteParquet(path)
        meta = src.columns(["text", "speaker_id"])
        for i, (text, speaker) in enumerate(zip(meta["text"], meta["speaker_id"])):
            words = len(text.split())
            if 300 <= words <= 1700:  # about 2-10 minutes of TED speech; verified on the real duration below
                candidates.append((path, i, words))
    items: List[Item] = []
    seen_minutes = 0.0
    cache: Dict[str, RemoteParquet] = {}
    for path, index, _ in sorted(candidates, key=lambda c: (c[2], c[0], c[1])):
        if seen_minutes >= spec_minutes or (ctx.smoke_only and items):
            break
        src = cache.setdefault(path, RemoteParquet(path))
        row = src.rows([index])[0]
        if has_markup(row["text"]):  # a whole talk cannot be cut around markup: leave it out
            ctx.drop("en_talks", 1)
            continue
        audio = decode_audio(row["audio"]["bytes"])
        if len(audio) / SR > 600.0:
            log(f"en_talks: {row['speaker_id']} is {len(audio) / SR / 60:.1f} min, over the 10 min cut: skipped")
            continue
        items.append(emit(ctx, "en_talks", re.sub(r"[^A-Za-z0-9]+", "-", row["speaker_id"]).strip("-").lower(),
                          audio, clean_text(row["text"]), None, {"talk": row["speaker_id"]}))
        seen_minutes = minutes(items)
        log(f"en_talks: {items[-1]['id']} {items[-1]['duration']:.0f}s")
    return items


def _pick_conversation_start(meta: Dict[str, list], target: float, rng: np.random.Generator) -> int:
    """A start row inside one conversation with enough speech after it (estimated from the text length) to fill `target`."""
    conv = [spk[:9] for spk in meta["speaker"]]
    blocks: List[Tuple[int, int]] = []
    begin = 0
    for i in range(1, len(conv) + 1):
        if i == len(conv) or conv[i] != conv[begin]:
            blocks.append((begin, i))
            begin = i
    for block in rng.permutation(len(blocks)):
        lo, hi = blocks[int(block)]
        start = lo + int(rng.integers(0, min(60, hi - lo)))
        seconds = sum(len(t) / 14.0 + 0.4 for t in meta["text"][start:hi] if "IGNORE_TIME_SEGMENT" not in t)
        if seconds >= target * 1.3:
            return start
    raise RuntimeError("no EDACC conversation is long enough for the requested item")


def build_en_accents(ctx: Ctx) -> List[Item]:
    shards = glob_parquet("datasets/edinburghcstr/edacc/data/test-*.parquet")
    order = spread_order(len(shards))  # spreads over the accent mix of the shards
    items: List[Item] = []
    for n, target in enumerate(ctx.targets("en_accents")):
        src = RemoteParquet(shards[order[n % len(order)]])
        rng = slice_rng("en_accents", f"item{n}")
        meta = src.columns(["speaker", "text", "accent"])
        start = _pick_conversation_start(meta, target, rng)
        conversation = meta["speaker"][start][:9]
        accent = meta["accent"][start]

        dropped = [0]

        def utterances() -> Iterator[Dict[str, Any]]:
            for index, row in src.iter_rows(start, ["audio", "text", "speaker"]):
                if row["speaker"][:9] != conversation:
                    return
                text = clean_text(row["text"])
                if not text or has_markup(text):
                    dropped[0] += 1
                    continue
                yield {"audio": decode_audio(row["audio"]["bytes"]), "text": text, "speaker": row["speaker"]}

        audio, reference, turns = build_longform(utterances(), target, 0.4)
        ctx.drop("en_accents", dropped[0])
        items.append(emit(ctx, "en_accents", f"{n:02d}", audio, reference, turns,
                          {"accent": accent, "conversation": conversation, "dropped_segments": dropped[0]}))
        log(f"en_accents: {items[-1]['id']} {accent} {items[-1]['duration']:.0f}s")
    return items


def _window_item(ctx: Ctx, name: str, key: str, segments: List[Dict[str, Any]], meta: Dict[str, Any],
                 max_gap: float) -> Item:
    audio, reference, turns = assemble_timeline(segments, max_gap=max_gap)
    item = emit(ctx, name, key, audio, reference, turns, meta)
    log(f"{name}: {item['id']} {item['duration']:.0f}s from {len(segments)} segments")
    return item


def build_en_business(ctx: Ctx) -> List[Item]:
    shards = glob_parquet("datasets/distil-whisper/earnings22/chunked/test-*.parquet")
    items: List[Item] = []
    n_shard = 0
    for n, target in enumerate(ctx.targets("en_business")):
        for _ in range(len(shards)):
            path = shards[(3 + 7 * n_shard) % len(shards)]
            n_shard += 1
            src = RemoteParquet(path)
            meta = src.columns(["file_id", "start_ts", "end_ts", "transcription"])
            counts: Dict[str, int] = {}
            for file_id in meta["file_id"]:
                counts[file_id] = counts.get(file_id, 0) + 1
            file_id = max(counts, key=lambda f: (counts[f], f))
            rows = [i for i, f in enumerate(meta["file_id"]) if f == file_id]
            begin = min(meta["start_ts"][i] for i in rows)
            t0 = begin + 480.0
            in_window = [i for i in rows if meta["start_ts"][i] >= t0 and meta["end_ts"][i] <= t0 + target]
            window = [i for i in in_window if meta["transcription"][i].strip() and not has_markup(meta["transcription"][i])]
            if len(window) < 8:
                log(f"en_business: shard {Path(path).name} file {file_id} has too few segments in the window")
                continue
            fetched = src.rows(window, ["audio", "transcription"])
            segments = [{"start": float(meta["start_ts"][i]), "end": float(meta["end_ts"][i]),
                         "audio": decode_audio(r["audio"]["bytes"]), "text": r["transcription"], "speaker": None}
                        for i, r in zip(window, fetched)]
            ctx.drop("en_business", len(in_window) - len(window))
            items.append(_window_item(ctx, "en_business", f"{n:02d}", segments,
                                      {"call": file_id, "dropped_segments": len(in_window) - len(window)}, 1.0))
            break
        else:
            raise RuntimeError("no usable earnings call window found")
    return items


def build_en_meeting(ctx: Ctx) -> List[Item]:
    shards = glob_parquet("datasets/edinburghcstr/ami/sdm/test-*.parquet")
    sources = [RemoteParquet(p) for p in shards]
    per_meeting: Dict[str, List[Tuple[int, int, float, float]]] = {}
    for s, src in enumerate(sources):
        meta = src.columns(["meeting_id", "begin_time", "end_time", "text"])
        for i, (m, b, e, t) in enumerate(zip(meta["meeting_id"], meta["begin_time"], meta["end_time"], meta["text"])):
            per_meeting.setdefault(m, []).append((s, i, float(b), float(e), t))
    families = sorted({m[:-1] for m in per_meeting})  # meeting types interleaved: EN..a, ES..a, IS..a, TS..a, EN..b, ...
    order = [f"{f}{letter}" for letter in sorted({m[-1] for m in per_meeting}) for f in families if f"{f}{letter}" in per_meeting]
    items: List[Item] = []
    targets = ctx.targets("en_meeting")
    for meeting in [m for m in order if m in per_meeting]:
        if len(items) == len(targets):
            break
        target = targets[len(items)]
        t0 = 120.0
        in_window = [x for x in per_meeting[meeting] if x[2] >= t0 and x[3] <= t0 + target]
        segs = [(s, i, b, e) for s, i, b, e, t in in_window if clean_text(t) and not has_markup(t)]
        dropped = len(in_window) - len(segs)
        if len(segs) < 15:
            log(f"en_meeting: {meeting} has {len(segs)} segments in its window: skipped")
            continue
        segments = []
        for s, src in enumerate(sources):
            mine = [(i, b, e) for sh, i, b, e in segs if sh == s]
            if not mine:
                continue
            fetched = src.rows([i for i, _, _ in mine], ["audio", "text", "speaker_id"])
            for (i, b, e), row in zip(mine, fetched):
                segments.append({"start": b, "end": e, "audio": decode_audio(row["audio"]["bytes"]),
                                 "text": clean_text(row["text"]), "speaker": row["speaker_id"]})
        ctx.drop("en_meeting", dropped)
        items.append(_window_item(ctx, "en_meeting", f"{len(items):02d}", segments,
                                  {"meeting": meeting, "dropped_segments": dropped}, 1.0))
    if len(items) < len(targets):
        log(f"en_meeting: built {len(items)} of {len(targets)} planned items")
    return items


def build_es_fleurs(ctx: Ctx) -> List[Item]:
    src = RemoteParquet("datasets/google/fleurs/parquet-data/es_419/test-00000-of-00001.parquet")
    texts = src.columns(["raw_transcription"])["raw_transcription"]
    targets = ctx.targets("es_fleurs")
    need = sum(targets) * 1.15
    picked, seconds = [], 0.0
    for i, text in enumerate(texts):  # about 15 characters per second of read speech
        picked.append(i)
        seconds += len(text) / 15.0 + 0.6
        if seconds >= need:
            break
    rows = src.rows(picked, ["audio", "raw_transcription"])
    utts = [{"speaker": None, "text": clean_text(r["raw_transcription"]), "raw": r["audio"]["bytes"]} for r in rows
            if clean_text(r["raw_transcription"]) and not has_markup(r["raw_transcription"])]
    log(f"es_fleurs: pool of {len(utts)} utterances ({len(rows) - len(utts)} dropped for markup)")
    return pool_items(ctx, "es_fleurs", UttPool(utts, dropped=len(rows) - len(utts)), targets, "f", chunk=1)


def build_es_mls(ctx: Ctx) -> List[Item]:
    pool = load_pool("datasets/facebook/multilingual_librispeech/spanish/test-00000-of-00001.parquet", "transcript",
                     "speaker_id", 28, 170, 15.0, "mls spanish test")
    return pool_items(ctx, "es_mls", pool, ctx.targets("es_mls"), "m")


DIALECTS = [("chilean", "ylacombe/google-chilean-spanish", "female"), ("colombian", "ylacombe/google-colombian-spanish", "male"),
            ("argentinian", "ylacombe/google-argentinian-spanish", "female"),
            ("chilean", "ylacombe/google-chilean-spanish", "male"), ("colombian", "ylacombe/google-colombian-spanish", "female"),
            ("argentinian", "ylacombe/google-argentinian-spanish", "male")]
GATED_DIALECTS = [("peruvian", "rcastrovexler/openslr-slr73-peruvian-spanish"),
                  ("puerto_rican", "rcastrovexler/openslr-slr74-puertorican-spanish"),
                  ("venezuelan", "rcastrovexler/openslr-slr75-venezuelan-spanish")]


def build_es_dialects(ctx: Ctx) -> List[Item]:
    from huggingface_hub import HfApi

    for dialect, repo in GATED_DIALECTS:  # OpenSLR 73/74/75 mirrors
        try:
            gated = HfApi().dataset_info(repo).gated
        except Exception as exc:  # noqa: BLE001
            ctx.gaps[f"es_dialects:{dialect}"] = f"{repo} is not reachable: {type(exc).__name__}"
            continue
        if gated:
            ctx.gaps[f"es_dialects:{dialect}"] = f"{repo} is gated ({gated} approval) and HF_TOKEN is not set"
    items: List[Item] = []
    for n, target in enumerate(ctx.targets("es_dialects")):
        dialect, repo, config = DIALECTS[n % len(DIALECTS)]
        files = glob_parquet(f"datasets/{repo}/{config}/train-*.parquet")
        src = RemoteParquet(files[0])
        start = int(slice_rng("es_dialects", f"start{n}").integers(0, 400))

        dropped = [0]

        def utterances() -> Iterator[Dict[str, Any]]:
            for _, row in src.iter_rows(start, ["audio", "text", "speaker_id"]):
                text = clean_text(row["text"])
                if not text or has_markup(text):
                    dropped[0] += 1
                else:
                    yield {"audio": decode_audio(row["audio"]["bytes"]), "text": text, "speaker": str(row["speaker_id"])}

        audio, reference, turns = build_longform(utterances(), target)
        ctx.drop("es_dialects", dropped[0])
        items.append(emit(ctx, "es_dialects", f"{dialect}-{config[0]}{n:02d}", audio, reference, turns,
                          {"dialect": dialect, "repo": repo, "dropped_segments": dropped[0]}))
        log(f"es_dialects: {items[-1]['id']} {items[-1]['duration']:.0f}s")
    return items


def source_items(ctx: Ctx, source: str, only_source_key: Optional[str] = None) -> List[Item]:
    items = ctx.built.get(source) or read_manifest(source)
    if only_source_key:
        items = [i for i in items if i["meta"].get("source") == only_source_key]
    if not items:
        raise RuntimeError(f"source slice {source} has no items (it failed or was not built)")
    return items


NOISE_COMBOS = [("white", 5), ("pink", 10), ("babble", 0), ("white", 0), ("pink", 5), ("babble", 10),
                ("white", 10), ("pink", 0), ("babble", 5)]


def _cap(items: Sequence[Item], cap_minutes: float) -> List[Item]:
    """Source items for a derived slice. The derived items inherit the source split, so pick the subset that reaches
    `cap_minutes` with the least overshoot while keeping >= 2 items and >= 35 % of the minutes on each side (exhaustive
    over the few candidates; if no subset qualifies, all candidates are used)."""
    ordered = sorted(items, key=lambda i: (i["duration"], i["id"]))
    if len(ordered) > 16:
        return ordered[:max(1, int(len(ordered) * cap_minutes * 60 / max(1.0, sum(i["duration"] for i in ordered))))]
    best: Optional[Tuple[float, List[Item]]] = None
    for mask in range(1, 2 ** len(ordered)):
        chosen = [ordered[k] for k in range(len(ordered)) if mask >> k & 1]
        mins = {k: sum(i["duration"] for i in chosen if i["split"] == k) / 60.0 for k in ("dev", "test")}
        total = mins["dev"] + mins["test"]
        counts = {k: sum(1 for i in chosen if i["split"] == k) for k in mins}
        if total < cap_minutes or min(counts.values()) < MIN_SIDE_ITEMS or min(mins.values()) / total < MIN_SIDE_FRACTION:
            continue
        if best is None or total < best[0] - 1e-9:
            best = (total, chosen)
    return best[1] if best else ordered


def build_noise(ctx: Ctx, name: str, source: str, source_key: Optional[str]) -> List[Item]:
    sources = source_items(ctx, source, source_key)
    everything = source_items(ctx, source)
    chosen = _cap(sources[:1] if ctx.smoke_only else sources, ctx.specs["slices"][name]["minutes"])
    items = []
    seen = {"dev": 0, "test": 0}
    for n, src in enumerate(chosen):
        kind, snr = NOISE_COMBOS[seen[src["split"]] % len(NOISE_COMBOS)]  # every noise type shows up on both sides
        seen[src["split"]] += 1
        speech, _ = sf.read(src["audio"], dtype="float32")
        rng = slice_rng(name, f"mix{n}")
        if kind == "white":
            noise = rng.standard_normal(len(speech)).astype(np.float32)
        elif kind == "pink":
            noise = make_pink(len(speech), rng)
        else:
            others = [sf.read(o["audio"], dtype="float32")[0] for o in everything if o["id"] != src["id"]][:8]
            noise = make_babble(others, len(speech), rng)
        key = f"{src['id'].split('-', 1)[1]}-{kind}{snr}"
        items.append(emit(ctx, name, key, mix_at_snr(speech, noise, snr), src["reference"], src["speakers"],
                          {"source_id": src["id"], "noise": kind, "snr_db": snr}, split=src["split"]))
        log(f"{name}: {items[-1]['id']} {kind} {snr} dB")
    return items


def build_phone(ctx: Ctx, name: str, source: str, source_key: Optional[str]) -> List[Item]:
    sources = source_items(ctx, source, source_key)
    chosen = _cap(sources[:1] if ctx.smoke_only else sources, ctx.specs["slices"][name]["minutes"])
    items = []
    for src in chosen:
        speech, _ = sf.read(src["audio"], dtype="float32")
        items.append(emit(ctx, name, f"{src['id'].split('-', 1)[1]}-phone", phone_band(speech), src["reference"],
                          src["speakers"], {"source_id": src["id"]}, split=src["split"]))
        log(f"{name}: {items[-1]['id']}")
    return items


AMBIENT = ["rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds", "wind", "thunderstorm", "water_drops",
           "pouring_water", "vacuum_cleaner", "washing_machine", "engine", "train", "airplane", "helicopter", "clock_tick"]


def build_nospeech(ctx: Ctx) -> List[Item]:
    clip = 20 * SR
    items: List[Item] = []

    def add(key: str, audio: np.ndarray, kind: str, **extra: Any) -> None:
        items.append(emit(ctx, "nospeech", key, audio, "", None, {"kind": kind, **extra}))

    n_per_kind = 2 if ctx.smoke_only else 10
    for n in range(n_per_kind):
        rng = slice_rng("nospeech", f"quiet{n}")
        if n < n_per_kind // 2:
            add(f"silence{n:02d}", np.zeros(clip, dtype=np.float32), "silence")
        else:
            room = make_pink(clip, rng) * 10 ** (-55 / 20) / 0.3  # pink noise peaks near 1; ~ -55 dBFS RMS
            add(f"room{n:02d}", room.astype(np.float32), "room_tone")
    for n in range(n_per_kind):
        level_db = -40 + 20 * n / max(1, n_per_kind - 1)
        white = slice_rng("nospeech", f"white{n}").standard_normal(clip).astype(np.float32)
        add(f"white{n:02d}", white * 10 ** (level_db / 20), "white_noise", level_dbfs=round(level_db, 1))
    src = RemoteParquet(glob_parquet("datasets/ashraq/esc50/data/train-*.parquet")[0])
    meta = src.columns(["category"])["category"]
    for n in range(n_per_kind):
        category = AMBIENT[n % len(AMBIENT)]
        picks = [i for i, c in enumerate(meta) if c == category][:4]
        rows = src.rows(picks, ["audio"])
        audio = np.concatenate([decode_audio(r["audio"]["bytes"]) for r in rows])
        audio = np.resize(audio, clip) if len(audio) < clip else audio[:clip]
        add(f"ambient{n:02d}-{category}", audio, "ambient", category=category)
    for item in items:
        log(f"nospeech: {item['id']}")
    return items


BUILDERS: Dict[str, Callable[[Ctx], List[Item]]] = {
    "en_clean": build_en_clean, "en_talks": build_en_talks, "en_accents": build_en_accents,
    "en_business": build_en_business, "en_meeting": build_en_meeting, "es_fleurs": build_es_fleurs,
    "es_mls": build_es_mls, "es_dialects": build_es_dialects,
    "en_noise": lambda c: build_noise(c, "en_noise", "en_clean", "test-clean"),
    "es_noise": lambda c: build_noise(c, "es_noise", "es_fleurs", None),
    "en_phone": lambda c: build_phone(c, "en_phone", "en_clean", "test-clean"),
    "es_phone": lambda c: build_phone(c, "es_phone", "es_fleurs", None),
    "nospeech": build_nospeech,
}

KNOWN_GAPS = {
    "code_switching": ("no es-en code-switching slice: BrunoHays/Bangor-Miami-Spanish-English-Corpus is the only ungated corpus "
                       "found, but it ships whole 30 min conversations with CHAT-annotated transcripts (xxx, www, &=laugh, "
                       "overlap codes) and no timestamps, so there is no trustworthy reference that can be cut to item length"),
    "nospeech_music": "ESC-50 has no music category, so the nospeech slice has silence, room tone, white noise and ambient scenes only",
}


# ----------------------------------------------------------------------------- split balancing and validation

MIN_SIDE_ITEMS = 2
MIN_SIDE_FRACTION = 0.35
DERIVED = {"en_noise", "es_noise", "en_phone", "es_phone"}  # these inherit the split of their source item
STRATA: Dict[str, Callable[[Item], str]] = {
    "en_clean": lambda i: str(i["meta"].get("source")), "es_dialects": lambda i: str(i["meta"].get("dialect")),
    "nospeech": lambda i: str(i["meta"].get("kind")),
}


def _split_score(items: Sequence[Item], side: Dict[str, str], stratum: Callable[[Item], str]) -> float:
    """Lower is better: heavy penalties for too few items / too few minutes on a side or a stratum missing from a side."""
    mins = {"dev": 0.0, "test": 0.0}
    count = {"dev": 0, "test": 0}
    strata: Dict[str, set] = {}
    sizes: Dict[str, int] = {}
    for item in items:
        mins[side[item["id"]]] += item["duration"]
        count[side[item["id"]]] += 1
        strata.setdefault(stratum(item), set()).add(side[item["id"]])
        sizes[stratum(item)] = sizes.get(stratum(item), 0) + 1
    total = max(mins["dev"] + mins["test"], 1e-9)
    score = 0.0
    if len(items) >= 4:
        score += 1000.0 * sum(max(0, MIN_SIDE_ITEMS - count[k]) for k in count)
        score += 10000.0 * max(0.0, MIN_SIDE_FRACTION - min(mins.values()) / total)
    score += 500.0 * sum(1 for k, v in strata.items() if sizes[k] >= 2 and len(v) < 2)
    return score + abs(mins["dev"] - mins["test"]) / total


def balance_splits(items: Sequence[Item], stratum: Optional[Callable[[Item], str]] = None) -> None:
    """Re-assign dev/test inside one slice (in place). Each stratum is split by greedy minute balancing (longest item
    first; ties by `assign_split`), then single moves and same-stratum swaps repair any slice-level violation of
    >= 2 items and >= 35 % of the minutes per side (slices with >= 4 items)."""
    stratum = stratum or (lambda item: "")
    groups: Dict[str, List[Item]] = {}
    for item in items:
        groups.setdefault(stratum(item), []).append(item)
    side: Dict[str, str] = {}
    total = {"dev": 0.0, "test": 0.0}
    for key in sorted(groups):
        local = {"dev": 0.0, "test": 0.0}
        for item in sorted(groups[key], key=lambda i: (-i["duration"], i["id"])):
            if local["dev"] != local["test"]:
                pick = "dev" if local["dev"] < local["test"] else "test"
            elif total["dev"] != total["test"]:
                pick = "dev" if total["dev"] < total["test"] else "test"
            else:
                pick = assign_split(item["id"])
            side[item["id"]] = pick
            local[pick] += item["duration"]
            total[pick] += item["duration"]
    best = _split_score(items, side, stratum)
    for _ in range(100):
        candidate, candidate_score = None, best
        for a in items:
            trial = dict(side)
            trial[a["id"]] = "test" if side[a["id"]] == "dev" else "dev"
            if (sc := _split_score(items, trial, stratum)) < candidate_score - 1e-9:
                candidate, candidate_score = trial, sc
            for b in items:
                if side[a["id"]] == "dev" and side[b["id"]] == "test" and stratum(a) == stratum(b):
                    trial = dict(side)
                    trial[a["id"]], trial[b["id"]] = "test", "dev"
                    if (sc := _split_score(items, trial, stratum)) < candidate_score - 1e-9:
                        candidate, candidate_score = trial, sc
        if candidate is None:
            break
        side, best = candidate, candidate_score
    for item in items:
        item["split"] = side[item["id"]]


def validate_manifests(manifest_dir: Path) -> List[str]:
    """Problems found in the manifests of `manifest_dir` (empty list = all good)."""
    problems: List[str] = []
    manifest_dir = Path(manifest_dir)
    slice_ids: Dict[str, set] = {}
    for path in sorted(manifest_dir.glob("*.jsonl")):
        name = path.stem
        items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        seen: set = set()
        for item in items:
            tag = f"{name}/{item.get('id')}"
            if item["id"] in seen:
                problems.append(f"{tag}: duplicate id")
            seen.add(item["id"])
            if item.get("split") not in ("dev", "test"):
                problems.append(f"{tag}: bad split {item.get('split')!r}")
            if item.get("duration", 0) > 600:
                problems.append(f"{tag}: duration {item['duration']} s is over 600 s")
            if not item.get("reference", "").strip() and item.get("slice") != "nospeech":
                problems.append(f"{tag}: empty reference")
            audio = resolve_audio(item["audio"])
            if not audio.exists():
                problems.append(f"{tag}: audio file missing")
                continue
            info = sf.info(audio)
            if info.samplerate != SR or info.channels != 1:
                problems.append(f"{tag}: audio is {info.samplerate} Hz x {info.channels}, expected {SR} Hz mono")
            if abs(info.duration - item["duration"]) > 0.05:
                problems.append(f"{tag}: manifest duration {item['duration']} s but the audio is {info.duration:.2f} s")
            for turn in item.get("speakers") or []:
                if not (0 <= turn["start"] <= turn["end"] <= item["duration"] + 0.05):
                    problems.append(f"{tag}: speaker turn {turn} outside the {item['duration']} s of audio")
                    break
        if name == "smoke":
            for item in items:
                if item["slice"] in slice_ids and item["id"] not in slice_ids[item["slice"]]:
                    problems.append(f"smoke/{item['id']}: not in the {item['slice']} manifest")
            continue
        slice_ids[name] = seen
        if len(items) >= 4:
            mins = {k: sum(i["duration"] for i in items if i["split"] == k) for k in ("dev", "test")}
            counts = {k: sum(1 for i in items if i["split"] == k) for k in ("dev", "test")}
            total = max(sum(mins.values()), 1e-9)
            for k in ("dev", "test"):
                if counts[k] < MIN_SIDE_ITEMS or mins[k] / total < MIN_SIDE_FRACTION:
                    problems.append(f"{name}: unbalanced split, {k} has {counts[k]} items and {mins[k] / total:.0%} of the minutes")
    return problems


# ----------------------------------------------------------------------------- smoke set, summary, CLI


def choose_smoke(by_slice: Dict[str, List[Item]], smoke_seconds: float) -> List[Item]:
    """One short item per slice (the shortest); for nospeech two clips of each kind."""
    smoke: List[Item] = []
    for name, items in by_slice.items():
        if not items:
            continue
        if name == "nospeech":
            kinds: Dict[str, List[Item]] = {}
            for item in sorted(items, key=lambda i: hashlib.sha256(i["id"].encode()).hexdigest()):
                kinds.setdefault(item["meta"].get("kind", "?"), []).append(item)
            smoke += [i for group in kinds.values() for i in group[:2]]
        else:
            smoke.append(min(items, key=lambda i: (i["duration"], i["id"])))
    return smoke


def summary_table(by_slice: Dict[str, List[Item]], smoke: Sequence[Item], gaps: Dict[str, str]) -> str:
    rows = [f"{'slice':<13}{'items':>6}{'minutes':>9}{'dev':>8}{'test':>8}{'smoke':>8}  status"]
    totals = [0, 0.0, 0.0, 0.0, 0.0]
    for name in SLICE_ORDER:
        if name not in by_slice:
            continue
        items = by_slice[name]
        dev = minutes([i for i in items if i["split"] == "dev"])
        test = minutes([i for i in items if i["split"] == "test"])
        sm = minutes([i for i in smoke if i["slice"] == name])
        status = "ok" if items else "MISSING"
        note = [v for k, v in gaps.items() if k == name or k.startswith(name + ":")]
        rows.append(f"{name:<13}{len(items):>6}{minutes(items):>9.1f}{dev:>8.1f}{test:>8.1f}{sm:>8.1f}  {status}"
                    + (f" ({len(note)} gap note{'s' if len(note) != 1 else ''})" if note else ""))
        totals = [totals[0] + len(items), totals[1] + minutes(items), totals[2] + dev, totals[3] + test, totals[4] + sm]
    rows.append(f"{'TOTAL':<13}{totals[0]:>6}{totals[1]:>9.1f}{totals[2]:>8.1f}{totals[3]:>8.1f}{totals[4]:>8.1f}"
                f"   ({totals[1] / 60:.2f} h)")
    return "\n".join(rows)


def prepare(smoke_only: bool = False, only: Optional[Sequence[str]] = None, force: bool = False) -> Dict[str, List[Item]]:
    specs = json.loads(SLICES_JSON.read_text())
    ctx = Ctx(specs, smoke_only)
    all_names = [n for n in SLICE_ORDER if n in specs["slices"]]
    names = [n for n in all_names if not only or n in only]
    cached_names: List[str] = []
    for name in names:
        cached = read_manifest(name)
        if cached and not force:
            cached_names.append(name)
            log(f"{name}: reusing {len(cached)} items from the manifest (use --force to rebuild)")
            ctx.built[name] = cached
            continue
        t0 = time.time()
        if not smoke_only:
            shutil.rmtree(AUDIO_DIR / name, ignore_errors=True)  # no orphans from an earlier build
        try:
            items = BUILDERS[name](ctx)
            if items and not smoke_only and name not in DERIVED:
                balance_splits(items, STRATA.get(name))
        except Exception as exc:  # noqa: BLE001 - one broken source must not stop the whole prepare
            log(f"{name}: SKIPPED ({type(exc).__name__}: {exc})\n{traceback.format_exc(limit=4)}")
            ctx.gaps[name] = f"{type(exc).__name__}: {str(exc)[:300]}"
            items = []
        ctx.built[name] = items
        if not smoke_only:
            if items:
                write_manifest(name, items)
            else:
                (MANIFEST_DIR / f"{name}.jsonl").unlink(missing_ok=True)  # a stale manifest would hide the failure
        log(f"{name}: {len(items)} items, {minutes(items):.1f} min in {time.time() - t0:.0f}s")
    # smoke set and summary always cover every slice: untouched slices come from their existing manifests
    by_slice = {n: ctx.built[n] if n in ctx.built else read_manifest(n) for n in all_names}
    smoke = choose_smoke(by_slice, specs.get("smoke_item_seconds", 120))
    write_manifest("smoke", smoke)
    rebuilt = [n for n in names if n not in cached_names]
    stays = lambda key, n: key == n or key.startswith(n + ":")  # noqa: E731
    kept = {k: v for k, v in specs.get("gaps", {}).items() if not any(stays(k, n) for n in rebuilt)}
    gaps = {**KNOWN_GAPS, **kept, **ctx.gaps}
    dropped = {k: v for k, v in specs.get("dropped_segments", {}).items() if k not in rebuilt}
    dropped.update({k: v for k, v in ctx.dropped.items() if k in rebuilt})
    if not smoke_only:
        specs["gaps"] = gaps
        specs["dropped_segments"] = dropped
        SLICES_JSON.write_text(json.dumps(specs, indent=2, ensure_ascii=False) + "\n")
        summary = {"minutes": {n: round(minutes(v), 2) for n, v in by_slice.items()}, "items": {n: len(v) for n, v in by_slice.items()},
                   "smoke_minutes": round(minutes(smoke), 2), "gaps": gaps, "dropped_segments": dropped}
        (MANIFEST_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(summary_table(by_slice, smoke, gaps), flush=True)
    for key, why in gaps.items():
        print(f"gap {key}: {why}", flush=True)
    if dropped:
        print("segments dropped for markup: " + ", ".join(f"{k} {v}" for k, v in dropped.items()), flush=True)
    problems = validate_manifests(MANIFEST_DIR)
    print("manifest problems:" if problems else "manifests valid", flush=True)
    for problem in problems:
        print(f"  {problem}", flush=True)
    return by_slice


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--smoke-only", action="store_true", help="build only the short smoke items and manifests/smoke.jsonl")
    prep.add_argument("--slices", default="", help="comma-separated slice names (default: all)")
    prep.add_argument("--force", action="store_true", help="rebuild slices that already have a manifest")
    args = parser.parse_args(argv)
    prepare(args.smoke_only, [s for s in args.slices.split(",") if s] or None, args.force)


if __name__ == "__main__":
    main()

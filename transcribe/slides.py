"""Opt-in `--slides` stage: turn the slides shown in a talk video into a markdown deck.

Pipeline: find the slide box (auto-detected or `--slides-box`), sample 1 fps grayscale thumbs of it, split them
into stable shots, keep one frame per distinct slide (a later identical shot is a repeat), collapse progressive
reveals into their final state, read each slide with Claude through the headless Claude Code CLI (same isolation
flags as the refine stage, answers cached per image), drop frames Claude says are not slides, and attach to each
slide the transcript segments spoken while it was on screen.

Imported lazily by `cli.py` only when `--slides` is set, so other runs never load numpy from here.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from logger_utils import log_stage

from .constants import DEFAULT_SLIDES_MODEL
from .models import CapabilityError, SlidesError
from .refine import resolve_claude_bin

LOGGER = logging.getLogger("transcribe_youtube")

Box = Tuple[int, int, int, int]  # x, y, w, h in source pixels

# Analysis grid (not the source resolution): thresholds below were measured on 192x108 thumbs.
THUMB_W, THUMB_H = 192, 108
PIX_DELTA = 40  # a thumb pixel "changed" if its gray level moved more than this
SHOT_CHANGE = 0.002  # fraction of changed pixels that marks a different picture
SAME_SLIDE = 0.0003  # stricter: a later shot this close is the same slide shown again (a 1-line diff measured 0.00106)
MIN_SHOT_S = 2  # drop shots shorter than this (transitions, animations)

# Slide-box detection on full-frame thumbs.
SPEAKER_MOTION_RATE = 0.15  # pixels changing in more than this fraction of seconds are live video (the speaker);
# measured: slide pixels <= 0.04, speaker 0.1-0.3 on a real talk, and its box is the same for 0.05-0.2
DETECT_TIME_BINS = 20  # the video is split into this many equal periods ...
SLIDE_MIN_BIN_FRACTION = 0.2  # ... and slide pixels change in at least this fraction of them (chrome only in bursts)
FULL_FRAME_AREA = 0.8  # a detected region covering this much of the frame means full-screen slides
MIN_REGION_AREA = 0.02  # a largest region smaller than this means no slide region was found

# Progressive reveals.
INK_W, INK_H = 384, 216
INK_LEVEL = 200
MAX_REVEAL_INK = 0.5  # reveal check applies only to light slides (ink on at most half the canvas)

# Claude.
SLIDES_PROMPT_VERSION = 1  # bump when PROMPT changes: it is part of the cache key
SLIDES_MAX_WORKERS = 4
SLIDES_ATTEMPTS = 2  # one retry per frame, then fail loudly
SLIDES_CALL_TIMEOUT_SECONDS = 300
NOT_A_SLIDE = "NOT_A_SLIDE"
_NOT_A_SLIDE_RE = re.compile(r"[\s`*_\"'.]*NOT_A_SLIDE[\s`*_\"'.]*")

PROMPT = (
    "This image is one frame from the talk \"{title}\". If it is not a presentation slide or a screen share "
    f"(for example a camera shot of the speaker, the stage or the audience), reply with exactly {NOT_A_SLIDE}. "
    "Otherwise transcribe the slide to markdown.\n"
    "- Start with the slide title as '## <title>' (no separate title: use the most prominent heading).\n"
    "- Keep every visible text verbatim, including lowercase styling.\n"
    "- Code goes in fenced blocks with a language tag.\n"
    "- Charts: one line naming the chart type and axes, then the labelled values as a list.\n"
    "- Diagrams: reproduce the structure as nested lists or arrows (A -> B).\n"
    "- Illustrations or photos: one italic line describing what they show.\n"
    "- Leave out the speaker handle or logo in the footer.\n"
    "Reply with only the markdown."
)


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "whisper_translate" / "slides"


# --------------------------------------------------------------------------- video helpers


def parse_box(raw: str) -> Box:
    """Parse `x,y,w,h` (source pixels) for `--slides-box`."""
    parts = [p.strip() for p in str(raw).split(",")]
    try:
        values = [int(p) for p in parts]
    except ValueError:
        values = []
    if len(values) != 4 or values[0] < 0 or values[1] < 0 or values[2] <= 0 or values[3] <= 0:
        raise SlidesError(f"--slides-box must be x,y,w,h in pixels (w,h > 0), got {raw!r}")
    return values[0], values[1], values[2], values[3]


def probe_video_size(path: Path) -> Optional[Tuple[int, int]]:
    """(width, height) of the first video stream, or None when the file has none (cover art does not count)."""
    command = ["ffprobe", "-v", "error", "-select_streams", "V:0", "-show_entries", "stream=width,height",
               "-of", "csv=p=0:s=x", str(path)]
    try:
        out = subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
        width, height = (int(v) for v in out.splitlines()[0].split("x")[:2])
    except (subprocess.CalledProcessError, OSError, ValueError, IndexError):
        return None
    return (width, height) if width > 0 and height > 0 else None


def _crop_filter(box: Optional[Box]) -> str:
    return "" if box is None else "crop={2}:{3}:{0}:{1}".format(*box)


def sample_thumbs(video: Path, box: Optional[Box] = None) -> np.ndarray:
    """1 fps grayscale THUMB_W x THUMB_H thumbs (optionally of the box only): uint8 array (seconds, H, W)."""
    filters = ",".join(f for f in ("fps=1", _crop_filter(box), f"scale={THUMB_W}:{THUMB_H}", "format=gray") if f)
    try:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", filters, "-f", "rawvideo", "-"],
                             check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as exc:
        raise SlidesError(f"ffmpeg could not sample frames: {exc.stderr.decode(errors='replace')[-300:]}") from exc
    return np.frombuffer(raw, np.uint8).reshape(-1, THUMB_H, THUMB_W)


def extract_frame(video: Path, second: int, box: Optional[Box], out: Path) -> None:
    crop = _crop_filter(box)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(second), "-i", str(video),
                    *(["-vf", crop] if crop else []), "-frames:v", "1", "-q:v", "3", str(out)],
                   check=True, capture_output=True)


def load_ink(image: Path) -> np.ndarray:
    """Dark-pixel mask of an image on a fixed INK_W x INK_H grid (for build-step detection)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(image), "-vf", f"scale={INK_W}:{INK_H},format=gray",
                          "-f", "rawvideo", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(INK_H, INK_W) < INK_LEVEL


# --------------------------------------------------------------------------- slide box


def _changed_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a.astype(np.int16) - b.astype(np.int16)) > PIX_DELTA


def _largest_component(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Pixels of the largest 4-connected True region."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best: List[Tuple[int, int]] = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        queue, region = deque([(int(y0), int(x0))]), []
        while queue:
            y, x = queue.popleft()
            region.append((y, x))
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        if len(region) > len(best):
            best = region
    return best


def _even_down(value: int) -> int:
    return value - value % 2


def _even_up(value: int) -> int:
    return value + value % 2


def detect_slide_box(thumbs: np.ndarray, width: int, height: int) -> Tuple[Box, str]:
    """Find the slide area in full-frame thumbs; returns (box in source pixels, "detected" | "full-frame").

    Per pixel: motion rate = fraction of seconds it changed; spread = in how many of DETECT_TIME_BINS equal
    periods it changed at all. The speaker (live video) changes most seconds; static conference chrome changes
    only in short bursts (intro, outro); slide pixels step at slide changes spread over the whole talk.
    """
    full: Box = (0, 0, width, height)
    count = len(thumbs)
    if count < 3:
        return full, "full-frame"
    _, th, tw = thumbs.shape
    bins = min(DETECT_TIME_BINS, count - 1)
    changes = np.zeros((th, tw), np.int32)
    spread = np.zeros((bins, th, tw), bool)
    for i in range(1, count):
        diff = _changed_mask(thumbs[i], thumbs[i - 1])
        changes += diff
        spread[(i - 1) * bins // (count - 1)] |= diff
    speaker = changes / (count - 1) > SPEAKER_MOTION_RATE
    slide = ~speaker & (spread.sum(axis=0) >= max(1, round(bins * SLIDE_MIN_BIN_FRACTION)))
    region = _largest_component(slide)
    if len(region) < MIN_REGION_AREA * th * tw:
        return full, "full-frame"
    ys = [p[0] for p in region]
    xs = [p[1] for p in region]
    x0, x1, y0, y1 = min(xs), max(xs) + 1, min(ys), max(ys) + 1
    if (x1 - x0) * (y1 - y0) >= FULL_FRAME_AREA * th * tw:
        return full, "full-frame"
    sx, sy = width / tw, height / th
    # Even coordinates keep ffmpeg's crop exact on chroma-subsampled video.
    left, top = _even_down(int(x0 * sx)), _even_down(int(y0 * sy))
    right, bottom = min(width, _even_up(int(np.ceil(x1 * sx)))), min(height, _even_up(int(np.ceil(y1 * sy))))
    return (left, top, right - left, bottom - top), "detected"


# --------------------------------------------------------------------------- shots


def changed_fraction(a: np.ndarray, b: np.ndarray) -> float:
    return float(_changed_mask(a, b).mean())


def segment_shots(thumbs: np.ndarray) -> List[Tuple[int, int]]:
    """Stable shots as [start, end) seconds; shots shorter than MIN_SHOT_S are dropped."""
    shots: List[Tuple[int, int]] = []
    start = 0
    for i in range(1, len(thumbs) + 1):
        if i == len(thumbs) or changed_fraction(thumbs[i], thumbs[i - 1]) > SHOT_CHANGE:
            if i - start >= MIN_SHOT_S:
                shots.append((start, i))
            start = i
    return shots


def dedupe_shots(thumbs: np.ndarray, shots: Sequence[Tuple[int, int]]) -> List[Dict[str, Any]]:
    """One entry per distinct picture: {start, end, second, repeats}; a later identical shot becomes a repeat span."""
    kept: List[Dict[str, Any]] = []
    for start, end in shots:
        rep = thumbs[(start + end) // 2]
        dup = next((k for k in kept if changed_fraction(k["thumb"], rep) <= SAME_SLIDE), None)
        if dup is not None:
            dup["repeats"].append([start, end])
            continue
        kept.append({"start": start, "end": end, "second": (start + end) // 2, "thumb": rep, "repeats": []})
    for k in kept:
        del k["thumb"]
    return kept


def is_build_step(ink_a: np.ndarray, ink_b: np.ndarray) -> bool:
    """True when slide b only adds content to slide a (a progressive reveal).

    Only light, mostly blank slides qualify: a full-bleed picture or a dark slide has ink almost
    everywhere, so it "covers" any earlier slide (measured: 0.86-0.98 ink) and must not swallow it.
    """
    a_sum = int(ink_a.sum())
    if ink_a.mean() > MAX_REVEAL_INK or ink_b.mean() > MAX_REVEAL_INK:
        return False
    if a_sum == 0 or int(ink_b.sum()) <= a_sum * 1.05:
        return False
    return int((ink_a & ink_b).sum()) / a_sum >= 0.95


def collapse_builds(frames: Sequence[Mapping[str, Any]], inks: Sequence[np.ndarray]) -> List[Dict[str, Any]]:
    """Merge progressive reveals into their final state: [{frame (index of the final frame), spans}]."""
    slides: List[Dict[str, Any]] = []
    for idx, (frame, ink) in enumerate(zip(frames, inks)):
        spans = [[frame["start"], frame["end"]], *[list(s) for s in frame.get("repeats", [])]]
        if slides and is_build_step(inks[slides[-1]["frame"]], ink):
            slides[-1]["frame"] = idx
            slides[-1]["spans"] += spans
        else:
            slides.append({"frame": idx, "spans": spans})
    return slides


def assign_segments(spans_per_slide: Sequence[Sequence[Sequence[float]]],
                    segments: Sequence[Mapping[str, Any]]) -> List[List[str]]:
    """Segment texts per slide: the slide on screen at the segment midpoint, else the last one shown before it."""
    said: List[List[str]] = [[] for _ in spans_per_slide]
    timeline = sorted((float(span[0]), i) for i, spans in enumerate(spans_per_slide) for span in spans)
    if not timeline:
        return said
    for seg in segments:
        start, end = seg.get("start"), seg.get("end")
        text = str(seg.get("text") or "").strip()
        if start is None or end is None or not text:
            continue
        mid = (float(start) + float(end)) / 2
        idx = next((i for s0, i in reversed(timeline) if s0 <= mid), timeline[0][1])
        said[idx].append(text)
    return said


def format_mmss(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"


def is_not_a_slide(answer: str) -> bool:
    return bool(_NOT_A_SLIDE_RE.fullmatch(answer or ""))


def render_deck(title: str, model: str, slides: Sequence[Mapping[str, Any]]) -> str:
    """slides: [{image (relative path), md, spans, said}] in order."""
    lines = [f"# {title}", "", f"{len(slides)} slides read by Claude ({model}).", ""]
    for n, slide in enumerate(slides, 1):
        when = ", ".join(f"{format_mmss(s)}-{format_mmss(e)}" for s, e in sorted(slide["spans"]))
        lines += ["---", "", f"<!-- slide {n} -->", "", f"![slide {n}]({slide['image']})", "",
                  f"*On screen: {when}*", "", str(slide["md"]).strip(), ""]
        if slide["said"]:
            lines += ["**What was said:**", "", "> " + " ".join(slide["said"]), ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- Claude


class SlideReader:
    """Reads one slide image with Claude through the headless Claude Code CLI, with an on-disk answer cache."""

    def __init__(self, model: Optional[str] = None, *, claude_bin: Optional[str] = None,
                 cache_dir: Optional[Path] = None, timeout_seconds: int = SLIDES_CALL_TIMEOUT_SECONDS):
        self.model = model or os.getenv("TRANSCRIBE_SLIDES_MODEL") or DEFAULT_SLIDES_MODEL
        self.claude_bin = resolve_claude_bin(claude_bin, feature="--slides")
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
        self.timeout_seconds = timeout_seconds

    def build_command(self) -> List[str]:
        # Never drop --safe-mode/--strict-mcp-config/--disable-slash-commands/--tools "": without them user
        # plugins, MCP servers and skills load into every call (orders of magnitude more expensive).
        return [self.claude_bin, "-p", "--safe-mode", "--strict-mcp-config", "--disable-slash-commands",
                "--tools", "", "--no-session-persistence", "--model", self.model,
                "--input-format", "stream-json", "--output-format", "stream-json", "--verbose"]

    def cache_path(self, image: bytes) -> Path:
        key = hashlib.sha256(
            f"{hashlib.sha256(image).hexdigest()}|{self.model}|{SLIDES_PROMPT_VERSION}".encode()).hexdigest()
        return self.cache_dir / f"{key}.md"

    def read(self, image: bytes, title: str) -> Tuple[str, float, bool]:
        """(markdown or NOT_A_SLIDE, cost in USD of this call, served from cache)."""
        cache = self.cache_path(image)
        if cache.exists():
            return cache.read_text(encoding="utf-8"), 0.0, True
        message = {"type": "user", "message": {"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.b64encode(image).decode()}},
            {"type": "text", "text": PROMPT.format(title=title)}]}}
        last_error = ""
        for attempt in range(1, SLIDES_ATTEMPTS + 1):
            try:
                answer, cost = self._call(json.dumps(message) + "\n")
            except SlidesError as exc:
                last_error = str(exc)
                LOGGER.warning("slides: claude attempt %s/%s failed: %s", attempt, SLIDES_ATTEMPTS, last_error)
                continue
            self._store(cache, answer)
            return answer, cost, False
        raise SlidesError(f"claude failed to read a slide after {SLIDES_ATTEMPTS} attempts: {last_error}")

    def _call(self, stdin: str) -> Tuple[str, float]:
        try:
            proc = subprocess.run(self.build_command(), input=stdin, capture_output=True, text=True,
                                  timeout=self.timeout_seconds or None)
        except subprocess.TimeoutExpired as exc:
            raise SlidesError(f"claude CLI timed out after {self.timeout_seconds}s") from exc
        except OSError as exc:
            raise SlidesError(f"claude CLI could not be started: {exc}") from exc
        for line in (proc.stdout or "").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("type") != "result":
                continue
            answer = str(event.get("result") or "").strip()
            if event.get("is_error") or not answer:
                raise SlidesError(f"claude CLI reported an error (subtype={event.get('subtype')}): {answer[:300]}")
            return answer + "\n", float(event.get("total_cost_usd") or 0.0)
        detail = (proc.stderr or proc.stdout or "").strip()[-500:]
        raise SlidesError(f"claude CLI returned no result (exit code {proc.returncode}): {detail}")

    @staticmethod
    def _store(path: Path, answer: str) -> None:
        try:  # a cache that cannot write must never fail the run
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(answer, encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            LOGGER.warning("slides cache: could not write %s (%s)", path, exc)


# --------------------------------------------------------------------------- deck


class SlideDeckBuilder:
    """Video + transcript segments -> `<out_dir>/slides.md` + `<out_dir>/img/*.jpg`."""

    def __init__(self, model: Optional[str] = None, *, box: Optional[Box] = None, claude_bin: Optional[str] = None,
                 cache_dir: Optional[Path] = None, max_workers: int = SLIDES_MAX_WORKERS):
        self.reader = SlideReader(model, claude_bin=claude_bin, cache_dir=cache_dir)  # fails fast without the CLI
        self.box = box
        self.max_workers = max(1, max_workers)

    @property
    def model(self) -> str:
        return self.reader.model

    def check_box(self, size: Tuple[int, int]) -> None:
        """A manual `--slides-box` must fit inside the video frame."""
        if self.box is not None:
            x, y, w, h = self.box
            if x + w > size[0] or y + h > size[1]:
                raise SlidesError(f"--slides-box {x},{y},{w},{h} does not fit the {size[0]}x{size[1]} video")

    def resolve_box(self, video: Path, size: Tuple[int, int]) -> Tuple[Box, str, Optional[np.ndarray]]:
        """(box, how, full-frame thumbs when they double as the slide thumbs)."""
        width, height = size
        if self.box is not None:
            self.check_box(size)
            return self.box, "manual", None
        with log_stage(LOGGER, "slides_detect_box"):
            thumbs = sample_thumbs(video)
            box, how = detect_slide_box(thumbs, width, height)
        LOGGER.info("slides: box %s (%s) on %sx%s video", ",".join(map(str, box)), how, width, height)
        return box, how, thumbs if how == "full-frame" else None

    def build(self, video: Path, segments: Sequence[Mapping[str, Any]], out_dir: Path, title: str,
              work_dir: Path) -> Dict[str, Any]:
        size = probe_video_size(video)
        if size is None:
            raise CapabilityError(f"--slides needs a video stream, but {video.name} has none.")
        box, how, thumbs = self.resolve_box(video, size)
        crop = None if box == (0, 0, size[0], size[1]) else box
        with log_stage(LOGGER, "slides_shots"):
            if thumbs is None:
                thumbs = sample_thumbs(video, crop)
            shots = segment_shots(thumbs)
            frames = dedupe_shots(thumbs, shots)
        LOGGER.info("slides: %s s sampled, %s stable shots, %s distinct frames", len(thumbs), len(shots), len(frames))
        if not frames:
            LOGGER.warning("slides: no picture stayed on screen for %s s or more; the deck will be empty", MIN_SHOT_S)

        work_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for frame in frames:
            path = work_dir / f"frame_{frame['second']:05d}.jpg"
            extract_frame(video, frame["second"], crop, path)
            paths.append(path)
        candidates = collapse_builds(frames, [load_ink(p) for p in paths])

        with log_stage(LOGGER, "slides_read", model=self.model, frames=len(candidates)):
            results = self._read_all([paths[c["frame"]] for c in candidates], title)

        slides = [dict(c, md=md) for c, (md, _, _) in zip(candidates, results) if not is_not_a_slide(md)]
        said = assign_segments([s["spans"] for s in slides], segments)
        img_dir = out_dir / "img"
        img_dir.mkdir(parents=True, exist_ok=True)
        for n, (slide, text) in enumerate(zip(slides, said), 1):
            name = f"slide_{n:03d}.jpg"
            shutil.copyfile(paths[slide["frame"]], img_dir / name)
            slide.update(image=f"img/{name}", said=text)
        deck = out_dir / "slides.md"
        deck.write_text(render_deck(title, self.model, slides), encoding="utf-8")

        cost = round(sum(cost for _, cost, _ in results), 6)
        meta = {
            "deck": str(deck),
            "model": self.model,
            "box": list(box),
            "box_source": how,
            "video_size": list(size),
            "shots": len(shots),
            "frames": len(frames),
            "frames_read": len(candidates),
            "not_a_slide": len(candidates) - len(slides),
            "slides": len(slides),
            "cached": sum(1 for _, _, cached in results if cached),
            "cost_usd": cost,
        }
        LOGGER.info("slides: %s frames -> %s slides (%s not slides, %s cached); claude cost $%.3f; deck %s",
                    len(frames), len(slides), meta["not_a_slide"], meta["cached"], cost, deck)
        return meta

    def _read_all(self, images: Sequence[Path], title: str) -> List[Tuple[str, float, bool]]:
        if not images:
            return []
        pool = ThreadPoolExecutor(max_workers=min(self.max_workers, len(images)))
        try:
            futures = [pool.submit(self.reader.read, p.read_bytes(), title) for p in images]
            return [f.result() for f in futures]
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

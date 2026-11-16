"""Opt-in `--slides` stage: turn the slides shown in a talk video into a markdown deck.

Pipeline: find the layout (auto-detected or `--slides-box`): a slide box beside the speaker video ("composite"),
full-screen slides with a picture-in-picture speaker overlay ("full-frame-pip", the overlay becomes an ignore rect)
or plain full-screen slides ("full-frame"). Sample 1 fps grayscale thumbs of the box (ignore rect blanked), split
them into stable shots, keep one frame per distinct slide (a later identical shot is a repeat), collapse progressive
reveals into their final state, read each slide with Claude through the headless Claude Code CLI (same isolation
flags as the refine stage, answers cached per image), drop frames Claude says are not slides, attach to each slide
the transcript segments spoken while it was on screen, and write `slides.md` plus a self-contained `slides.html`.

Imported lazily by `cli.py` only when `--slides` is set, so other runs never load numpy from here.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import logging
import os
import re
import shutil
import subprocess
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from markdown_it import MarkdownIt

from logger_utils import log_stage

from .constants import DEFAULT_SLIDES_MODEL
from .downloader import _extract_youtube_id
from .models import CapabilityError, SlidesError
from .refine import resolve_claude_bin

LOGGER = logging.getLogger("transcribe_youtube")

Box = Tuple[int, int, int, int]  # x, y, w, h in source pixels


class Layout(NamedTuple):
    kind: str  # "composite" | "full-frame-pip" | "full-frame" | "manual"
    box: Box  # area cropped for analysis and saved images
    ignore: Optional[Box] = None  # speaker overlay inside the box: blanked for shots, repeats and reveals only

# Analysis grid (not the source resolution): thresholds below were measured on 192x108 thumbs.
THUMB_W, THUMB_H = 192, 108
PIX_DELTA = 40  # a thumb pixel "changed" if its gray level moved more than this
SHOT_CHANGE = 0.002  # fraction of changed pixels that marks a different picture
SAME_SLIDE = 0.0003  # stricter: a later shot this close is the same slide shown again (a 1-line diff measured 0.00106)
MIN_SHOT_S = 2  # drop shots shorter than this (transitions, animations)

# Layout detection on full-frame thumbs.
SPEAKER_MOTION_RATE = 0.15  # pixels changing in more than this fraction of seconds are live video (the speaker);
# measured: slide pixels <= 0.04, speaker 0.1-0.3 on a real talk, and its box is the same for 0.05-0.2
SPEAKER_SOFT_DELTA = PIX_DELTA // 4  # smaller gray-level moves trace the speaker video's full extent (a PiP's core
# alone measured 96 thumb px, about a fifth of the overlay)
MIN_SPEAKER_AREA = 0.001  # a moving cluster smaller than this fraction of the frame is noise, not a speaker
MAX_PIP_AREA = 0.25  # a moving region larger than this is not an overlay (e.g. cuts to a camera): nothing to ignore
PIP_PAD = 3  # thumb pixels added around the speaker region to make the ignore rect
DETECT_TIME_BINS = 20  # the video is split into this many equal periods ...
SLIDE_MIN_BIN_FRACTION = 0.2  # ... and slide pixels change in at least this fraction of them (chrome only in bursts)
FULL_FRAME_AREA = 0.8  # a detected region covering this much of the frame means full-screen slides
MIN_REGION_AREA = 0.02  # a largest region smaller than this means no slide region was found
EDGE_BAND = 2  # thumb pixels sampled on each side of a region border
MIN_BOX_EDGE = 30  # median gray step across the region border that makes it a slide box (composite layout);
# measured: a composite box 180-237, the content bbox of a full-screen dark deck 2-9

# Progressive reveals: "content" = pixels that differ from the slide background (median gray), dark or light slides.
INK_W, INK_H = 384, 216
MAX_REVEAL_INK = 0.5  # reveal check applies only when content covers at most half the canvas (not full-bleed pictures)

# HTML deck.
HTML_IMAGE_W = 1100  # embedded slide images are scaled down to this width
HTML_PICTURE_HEADING = 60  # characters kept from a picture-only slide's description when it becomes the heading
_MD = MarkdownIt("commonmark", {"html": False}).enable("table")  # raw HTML is escaped
# A markdown image would load a remote URL: render its alt text instead, so the page never touches the network.
_MD.add_render_rule("image", lambda self, tokens, idx, options, env: html.escape(f"[image: {tokens[idx].content}]"))

# Claude.
SLIDES_PROMPT_VERSION = 2  # bump when PROMPT changes: it is part of the cache key
SLIDES_MAX_WORKERS = 4
SLIDES_ATTEMPTS = 2  # one retry per frame, then fail loudly
SLIDES_CALL_TIMEOUT_SECONDS = 300
NOT_A_SLIDE = "NOT_A_SLIDE"
_NOT_A_SLIDE_RE = re.compile(r"[\s`*_\"'.]*NOT_A_SLIDE[\s`*_\"'.]*")

PROMPT = (
    "This image is one frame from the talk \"{title}\". If it is not a presentation slide or a screen share "
    f"(for example a camera shot of the speaker, the stage or the audience), reply with exactly {NOT_A_SLIDE}. "
    "A slide with a small video of the speaker overlaid on it is still a slide. Otherwise transcribe the slide to "
    "markdown.\n"
    "- Start with the slide title as '## <title>' (no separate title: use the most prominent heading).\n"
    "- Keep every visible text verbatim, including lowercase styling.\n"
    "- Code goes in fenced blocks with a language tag.\n"
    "- Charts: one line naming the chart type and axes, then the labelled values as a list.\n"
    "- Diagrams: reproduce the structure as nested lists or arrows (A -> B).\n"
    "- Illustrations or photos: one italic line describing what they show.\n"
    "- Leave out the speaker handle or logo in the footer.\n"
    "- Ignore any video of the speaker overlaid on the slide (picture-in-picture): do not describe it. If it hides "
    "part of the slide content, say so in one italic line.\n"
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


def load_content(image: Path, ignore: Optional[np.ndarray] = None) -> np.ndarray:
    """Content mask on a fixed INK_W x INK_H grid (for build-step detection): pixels whose gray level differs from
    the slide background (the median gray) by more than PIX_DELTA, so dark and light slides both work.
    `ignore` (bool, same grid) blanks the speaker overlay, which is left out of the background too."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(image), "-vf", f"scale={INK_W}:{INK_H},format=gray",
                          "-f", "rawvideo", "-"], check=True, capture_output=True).stdout
    gray = np.frombuffer(raw, np.uint8).reshape(INK_H, INK_W).astype(np.int16)
    keep = np.ones_like(gray, bool) if ignore is None else ~ignore
    background = int(np.median(gray[keep])) if keep.any() else 0
    return (np.abs(gray - background) > PIX_DELTA) & keep


def html_jpeg(image: Path) -> bytes:
    """The image as a JPEG at most HTML_IMAGE_W wide, for embedding in slides.html."""
    try:
        return subprocess.run(["ffmpeg", "-v", "error", "-i", str(image), "-vf", f"scale='min({HTML_IMAGE_W},iw)':-2",
                               "-q:v", "5", "-f", "image2pipe", "-c:v", "mjpeg", "-"],
                              check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode(errors="replace")[-300:]
        raise SlidesError(f"ffmpeg could not resize {image.name}: {detail}") from exc


def rect_mask(shape: Tuple[int, int], rect: Box, box: Box) -> np.ndarray:
    """Bool mask on a (height, width) grid that samples `box`, True inside `rect` (source pixels, rounded outwards)."""
    height, width = shape
    bx, by, bw, bh = box
    x, y, w, h = rect
    mask = np.zeros(shape, bool)
    x0, x1 = int(np.floor((x - bx) * width / bw)), int(np.ceil((x + w - bx) * width / bw))
    y0, y1 = int(np.floor((y - by) * height / bh)), int(np.ceil((y + h - by) * height / bh))
    mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
    return mask


def blank(thumbs: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Copy of the thumbs with the masked pixels set to 0 (they then never count as changed)."""
    if mask is None or not mask.any():
        return thumbs
    out = thumbs.copy()
    out[:, mask] = 0
    return out


# --------------------------------------------------------------------------- slide box


def _changed_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a.astype(np.int16) - b.astype(np.int16)) > PIX_DELTA


def _flood(mask: np.ndarray, seeds: Sequence[Tuple[int, int]], seen: np.ndarray) -> List[Tuple[int, int]]:
    """Pixels of `mask` 4-connected to the seeds (marks them in `seen`)."""
    height, width = mask.shape
    queue, region = deque(), []
    for y, x in seeds:
        if mask[y, x] and not seen[y, x]:
            seen[y, x] = True
            queue.append((int(y), int(x)))
    while queue:
        y, x = queue.popleft()
        region.append((y, x))
        for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                queue.append((ny, nx))
    return region


def _largest_component(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Pixels of the largest 4-connected True region."""
    seen = np.zeros_like(mask, dtype=bool)
    best: List[Tuple[int, int]] = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if not seen[y0, x0]:
            region = _flood(mask, [(y0, x0)], seen)
            if len(region) > len(best):
                best = region
    return best


def _bbox(pixels: Sequence[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """(x0, y0, x1, y1), end-exclusive, of (y, x) pixels."""
    ys = [p[0] for p in pixels]
    xs = [p[1] for p in pixels]
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def _even_down(value: int) -> int:
    return value - value % 2


def _even_up(value: int) -> int:
    return value + value % 2


def _to_source(rect: Tuple[int, int, int, int], width: int, height: int, tw: int, th: int) -> Box:
    """Thumb-grid (x0, y0, x1, y1) -> source-pixel box; even coordinates keep ffmpeg's crop exact on 4:2:0 video."""
    x0, y0, x1, y1 = rect
    sx, sy = width / tw, height / th
    left, top = _even_down(int(x0 * sx)), _even_down(int(y0 * sy))
    right, bottom = min(width, _even_up(int(np.ceil(x1 * sx)))), min(height, _even_up(int(np.ceil(y1 * sy))))
    return left, top, right - left, bottom - top


def _speaker_rect(strong_rate: np.ndarray, soft_rate: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Thumb-grid (x0, y0, x1, y1) of the live speaker video, or None.

    Core = the largest cluster of pixels changing (by > PIX_DELTA) in more than SPEAKER_MOTION_RATE of the seconds;
    it is grown through pixels that change that often by the smaller SPEAKER_SOFT_DELTA, which fills in the parts of
    the video that move less (background, clothes), so the rect covers the whole overlay.
    """
    core = _largest_component(strong_rate > SPEAKER_MOTION_RATE)
    if len(core) < MIN_SPEAKER_AREA * strong_rate.size:
        return None
    soft = (soft_rate > SPEAKER_MOTION_RATE) | (strong_rate > SPEAKER_MOTION_RATE)
    return _bbox(_flood(soft, core, np.zeros_like(soft)))


def _border_step(thumbs: np.ndarray, rect: Tuple[int, int, int, int]) -> float:
    """Median gray step across the borders of rect (sides away from the frame edge), median over time.

    A composite slide box sits in static chrome, so its border is a sharp edge in every frame; the bounding box of
    the changing content of a full-screen deck lies inside a uniform slide background, so its border is flat.
    """
    _, th, tw = thumbs.shape
    x0, y0, x1, y1 = rect
    b = EDGE_BAND
    pairs = []  # (inside band, outside band) per side, each (seconds, side length)
    if x0 >= b + 1:
        pairs.append((thumbs[:, y0:y1, x0 + 1:x0 + 1 + b].mean(2), thumbs[:, y0:y1, x0 - 1 - b:x0 - 1].mean(2)))
    if tw - x1 >= b + 1:
        pairs.append((thumbs[:, y0:y1, x1 - 1 - b:x1 - 1].mean(2), thumbs[:, y0:y1, x1 + 1:x1 + 1 + b].mean(2)))
    if y0 >= b + 1:
        pairs.append((thumbs[:, y0 + 1:y0 + 1 + b, x0:x1].mean(1), thumbs[:, y0 - 1 - b:y0 - 1, x0:x1].mean(1)))
    if th - y1 >= b + 1:
        pairs.append((thumbs[:, y1 - 1 - b:y1 - 1, x0:x1].mean(1), thumbs[:, y1 + 1:y1 + 1 + b, x0:x1].mean(1)))
    if not pairs:
        return 0.0
    return float(np.median([np.median(np.abs(inside - outside).mean(1)) for inside, outside in pairs]))


def detect_layout(thumbs: np.ndarray, width: int, height: int) -> Layout:
    """Classify the video layout from full-frame thumbs; find the slide box and the speaker overlay to ignore.

    Per pixel: motion rate = fraction of seconds it changed; spread = in how many of DETECT_TIME_BINS equal
    periods it changed at all. The speaker (live video) changes most seconds; static conference chrome changes
    only in short bursts (intro, outro); slide pixels step at slide changes spread over the whole talk.
    - composite: the largest slide region outside the speaker is bounded by a sharp static edge -> crop to it.
    - full-frame-pip: no such box, and the speaker video is a small overlay -> full frame, overlay ignored.
    - full-frame: no such box and no small overlay -> full frame.
    """
    full: Box = (0, 0, width, height)
    count = len(thumbs)
    if count < 3:
        return Layout("full-frame", full)
    _, th, tw = thumbs.shape
    bins = min(DETECT_TIME_BINS, count - 1)
    strong = np.zeros((th, tw), np.int32)
    soft = np.zeros((th, tw), np.int32)
    spread = np.zeros((bins, th, tw), bool)
    for i in range(1, count):
        diff = np.abs(thumbs[i].astype(np.int16) - thumbs[i - 1].astype(np.int16))
        changed = diff > PIX_DELTA
        strong += changed
        soft += diff > SPEAKER_SOFT_DELTA
        spread[(i - 1) * bins // (count - 1)] |= changed
    speaker = _speaker_rect(strong / (count - 1), soft / (count - 1))
    outside_speaker = np.ones((th, tw), bool)
    if speaker is not None:
        outside_speaker[speaker[1]:speaker[3], speaker[0]:speaker[2]] = False
    slide = outside_speaker & (spread.sum(axis=0) >= max(1, round(bins * SLIDE_MIN_BIN_FRACTION)))
    region = _largest_component(slide)
    if len(region) >= MIN_REGION_AREA * th * tw:
        rect = _bbox(region)
        area = (rect[2] - rect[0]) * (rect[3] - rect[1])
        if area < FULL_FRAME_AREA * th * tw and _border_step(thumbs, rect) >= MIN_BOX_EDGE:
            return Layout("composite", _to_source(rect, width, height, tw, th))
    if speaker is not None:
        x0, y0, x1, y1 = speaker
        x0, y0, x1, y1 = max(0, x0 - PIP_PAD), max(0, y0 - PIP_PAD), min(tw, x1 + PIP_PAD), min(th, y1 + PIP_PAD)
        if (x1 - x0) * (y1 - y0) <= MAX_PIP_AREA * th * tw:
            return Layout("full-frame-pip", full, _to_source((x0, y0, x1, y1), width, height, tw, th))
    return Layout("full-frame", full)


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
    """True when slide b only adds content to slide a (a progressive reveal); masks come from `load_content`.

    Only mostly blank slides (dark or light) qualify: a full-bleed picture has content almost everywhere,
    so it "covers" any earlier slide and must not swallow it (MAX_REVEAL_INK, on the content fraction).
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


_HTML_CSS = """
:root{--bg:#faf9f7;--card:#fff;--fg:#1d1d1f;--muted:#6b6b70;--line:#e6e4e0;--accent:#c2410c;--quote:#f3f1ed;--code:#f1efea}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--card:#1e1e1f;--fg:#ececec;--muted:#a0a0a6;--line:#2e2e30;--accent:#fb923c;--quote:#262628;--code:#2a2a2c}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:17px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:760px;margin:0 auto;padding:20px 16px 64px}
header h1{font-size:1.45rem;line-height:1.3;margin:0 0 6px}
header p{margin:0 0 4px;color:var(--muted);font-size:.92rem;overflow-wrap:anywhere}
a{color:var(--accent)}
nav{margin:18px 0 26px;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:4px 14px}
nav summary{cursor:pointer;font-weight:600;padding:10px 0}
nav ol{margin:0 0 12px;padding-left:1.6em;font-size:.95rem}
nav li{margin:4px 0}
nav li span{color:var(--muted);font-variant-numeric:tabular-nums;font-size:.85rem;margin-left:6px}
section{background:var(--card);border:1px solid var(--line);border-radius:14px;margin:0 0 22px;overflow:hidden;scroll-margin-top:8px}
section>img{display:block;width:100%;height:auto;border-bottom:1px solid var(--line);background:#fff}
.body{padding:12px 16px 16px}
.meta{display:flex;flex-wrap:wrap;justify-content:space-between;gap:4px 8px;color:var(--muted);font-size:.85rem;font-variant-numeric:tabular-nums}
h2{font-size:1.15rem;margin:6px 0 10px;line-height:1.35;overflow-wrap:anywhere}
.said{background:var(--quote);border-left:3px solid var(--accent);border-radius:6px;padding:10px 12px;margin:10px 0 0}
.said p{margin:0}
.said b{display:block;font-size:.8rem;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin-bottom:4px}
details.text{margin-top:12px;font-size:.93rem;overflow-wrap:anywhere}
details.text summary{cursor:pointer;color:var(--muted)}
details.text h2,details.text h3{font-size:1rem}
table{border-collapse:collapse;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:4px 8px}
pre{background:var(--code);padding:10px;border-radius:8px;overflow-x:auto;font-size:.82rem;line-height:1.45}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.88em}
ul,ol{padding-left:1.3em}
.top{display:inline-block;margin-top:10px;font-size:.85rem}
"""
_HEADING_RE = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t#]*$", re.M)
_ITALIC_LINE_RE = re.compile(r"^[ \t]*([*_])(?![*_\s])(.+?)\1[ \t]*$", re.M)
_TAG_RE = re.compile(r"<[^>]+>")


def split_heading(md: str) -> Tuple[str, str]:
    """(heading markdown, body markdown) of a slide answer. Picture-only slides (no markdown heading) use their
    italic description, shortened to HTML_PICTURE_HEADING characters, as the heading and keep the full body."""
    md = str(md).strip()
    match = _HEADING_RE.search(md)
    if match:
        return match.group(1).strip(), (md[:match.start()] + md[match.end():]).strip()
    match = _ITALIC_LINE_RE.search(md)
    text = match.group(2).strip() if match else next((line.strip() for line in md.splitlines() if line.strip()), "")
    if len(text) > HTML_PICTURE_HEADING:
        text = text[:HTML_PICTURE_HEADING].rsplit(" ", 1)[0].rstrip(" ,.;:") + "…"
    return text or "(picture)", md


def youtube_time_url(source: Optional[str], seconds: float) -> Optional[str]:
    video_id = _extract_youtube_id(source) if source and re.match(r"https?://", source) else None
    return f"https://www.youtube.com/watch?v={video_id}&t={int(seconds)}s" if video_id else None


def render_html(title: str, model: str, slides: Sequence[Mapping[str, Any]], images: Sequence[bytes],
                source: Optional[str] = None) -> str:
    """One self-contained page (no JavaScript, no network): slides [{md, spans, said}] with their JPEG bytes."""
    esc = html.escape
    total = len(slides)
    toc, cards = [], []
    for n, (slide, image) in enumerate(zip(slides, images), 1):
        heading, body = split_heading(slide["md"])
        heading_html = _MD.renderInline(heading)
        spans = sorted(slide["spans"])
        start = spans[0][0] if spans else 0
        toc.append(f'<li><a href="#s{n}">{_TAG_RE.sub("", heading_html)}</a><span>{format_mmss(start)}</span></li>')
        times = []
        for s, e in spans:
            label = f"{format_mmss(s)}-{format_mmss(e)}"
            url = youtube_time_url(source, s)
            times.append(f'<a href="{esc(url)}">▶ {label}</a>' if url else label)
        said = " ".join(slide.get("said") or [])
        said_html = f'<div class="said"><b>What was said</b><p>{esc(said)}</p></div>' if said else ""
        text_html = (f'<details class="text"><summary>Slide text</summary>{_MD.render(body)}</details>'
                     if body else "")
        data = base64.b64encode(image).decode()
        cards.append(
            f'<section id="s{n}"><img loading="lazy" alt="Slide {n}" src="data:image/jpeg;base64,{data}">'
            f'<div class="body"><div class="meta"><span>Slide {n} of {total}</span><span>{", ".join(times)}</span>'
            f'</div><h2>{heading_html}</h2>{said_html}{text_html}<a class="top" href="#top">↑ Contents</a></div>'
            f'</section>')
    if source and re.match(r"https?://", source):
        source_html = f'<p>Video: <a href="{esc(source)}">{esc(source)}</a></p>'
    elif source:
        source_html = f"<p>Video: {esc(Path(source).name)}</p>"
    else:
        source_html = ""
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{esc(title)}</title><style>{_HTML_CSS}</style></head><body><main id=\"top\">"
            f"<header><h1>{esc(title)}</h1>{source_html}<p>{total} slides read by Claude ({esc(model)}), each with "
            f"what the speaker said while it was on screen.</p></header>"
            f'<nav><details open><summary>Contents</summary><ol>{"".join(toc)}</ol></details></nav>'
            + "".join(cards) + "</main></body></html>\n")


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
    """Video + transcript segments -> `<out_dir>/slides.md`, `<out_dir>/slides.html` and `<out_dir>/img/*.jpg`."""

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

    def resolve_layout(self, video: Path, size: Tuple[int, int]) -> Tuple[Layout, Optional[np.ndarray]]:
        """(layout, full-frame thumbs when they double as the slide thumbs)."""
        width, height = size
        if self.box is not None:
            self.check_box(size)
            return Layout("manual", self.box), None
        with log_stage(LOGGER, "slides_detect_layout"):
            thumbs = sample_thumbs(video)
            layout = detect_layout(thumbs, width, height)
        LOGGER.info("slides: layout %s, box %s, ignore %s on %sx%s video", layout.kind, ",".join(map(str, layout.box)),
                    ",".join(map(str, layout.ignore)) if layout.ignore else "none", width, height)
        return layout, thumbs if layout.box == (0, 0, width, height) else None

    def select_frames(self, video: Path, size: Tuple[int, int], work_dir: Path) -> Dict[str, Any]:
        """Everything before Claude: layout, shots, distinct frames (extracted to work_dir), collapsed candidates."""
        layout, thumbs = self.resolve_layout(video, size)
        crop = None if layout.box == (0, 0, size[0], size[1]) else layout.box
        with log_stage(LOGGER, "slides_shots"):
            if thumbs is None:
                thumbs = sample_thumbs(video, crop)
            if layout.ignore is not None:  # the speaker overlay must not split shots or hide repeats
                thumbs = blank(thumbs, rect_mask(thumbs.shape[1:], layout.ignore, layout.box))
            shots = segment_shots(thumbs)
            frames = dedupe_shots(thumbs, shots)
        LOGGER.info("slides: %s s sampled, %s stable shots, %s distinct frames", len(thumbs), len(shots), len(frames))
        if not frames:
            LOGGER.warning("slides: no picture stayed on screen for %s s or more; the deck will be empty", MIN_SHOT_S)

        work_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for frame in frames:
            path = work_dir / f"frame_{frame['second']:05d}.jpg"
            extract_frame(video, frame["second"], crop, path)  # saved images keep the overlay
            paths.append(path)
        ignore = None if layout.ignore is None else rect_mask((INK_H, INK_W), layout.ignore, layout.box)
        candidates = collapse_builds(frames, [load_content(p, ignore) for p in paths])
        return {"layout": layout, "seconds": len(thumbs), "shots": shots, "frames": frames, "paths": paths,
                "candidates": candidates}

    def build(self, video: Path, segments: Sequence[Mapping[str, Any]], out_dir: Path, title: str,
              work_dir: Path, source: Optional[str] = None) -> Dict[str, Any]:
        """`source` (URL or path of the input) is linked from slides.html; YouTube sources get timestamp links."""
        size = probe_video_size(video)
        if size is None:
            raise CapabilityError(f"--slides needs a video stream, but {video.name} has none.")
        picked = self.select_frames(video, size, work_dir)
        layout, frames, paths, candidates = picked["layout"], picked["frames"], picked["paths"], picked["candidates"]

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
        page = out_dir / "slides.html"
        images = [html_jpeg(paths[s["frame"]]) for s in slides]
        page.write_text(render_html(title, self.model, slides, images, source), encoding="utf-8")

        cost = round(sum(cost for _, cost, _ in results), 6)
        meta = {
            "deck": str(deck),
            "html": str(page),
            "model": self.model,
            "layout": layout.kind,
            "box": list(layout.box),
            "box_source": "manual" if layout.kind == "manual" else "detected",
            "ignore_rect": list(layout.ignore) if layout.ignore is not None else None,
            "video_size": list(size),
            "shots": len(picked["shots"]),
            "frames": len(frames),
            "frames_read": len(candidates),
            "not_a_slide": len(candidates) - len(slides),
            "slides": len(slides),
            "cached": sum(1 for _, _, cached in results if cached),
            "cost_usd": cost,
        }
        LOGGER.info("slides: %s layout, %s frames -> %s slides (%s not slides, %s cached); claude cost $%.3f; "
                    "deck %s; html %s", layout.kind, len(frames), len(slides), meta["not_a_slide"], meta["cached"],
                    cost, deck, page)
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

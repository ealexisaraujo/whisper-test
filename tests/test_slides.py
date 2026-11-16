import base64
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

import transcribe.slides as slides
from transcribe.models import BackendResult, CapabilityError, SlidesError, TranscriptionError

H, W = slides.THUMB_H, slides.THUMB_W


def _pattern(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (H, W), dtype=np.uint8)


def _thumbs(*runs) -> np.ndarray:
    """runs: (picture, seconds) pairs -> stacked 1 fps thumbs."""
    return np.stack([pic for pic, secs in runs for _ in range(secs)])


# --------------------------------------------------------------------------- shots


def test_segment_shots_drops_short_shots_and_dedupe_records_repeats():
    a, b, c = _pattern(1), _pattern(2), _pattern(3)
    thumbs = _thumbs((a, 5), (b, 4), (c, 1), (a, 3))
    shots = slides.segment_shots(thumbs)
    assert shots == [(0, 5), (5, 9), (10, 13)]  # the 1 s shot of c is a transition
    frames = slides.dedupe_shots(thumbs, shots)
    assert [(f["start"], f["end"], f["second"]) for f in frames] == [(0, 5, 2), (5, 9, 7)]
    assert frames[0]["repeats"] == [[10, 13]]
    assert "thumb" not in frames[0]


def test_repeat_threshold_is_stricter_than_shot_threshold():
    a = np.full((H, W), 255, np.uint8)
    b = a.copy()
    b[100:102, 10:40] = 0  # one short subtitle-like line: a different slide, not a repeat
    frac = slides.changed_fraction(a, b)
    assert slides.SAME_SLIDE < frac
    frames = slides.dedupe_shots(_thumbs((a, 3), (b, 3)), [(0, 3), (3, 6)])
    assert len(frames) == 2


# --------------------------------------------------------------------------- slide box


def _composite_video(seconds: int = 400) -> np.ndarray:
    """Static chrome (with an animated intro), speaker video on the left, slides stepping every 10 s on the right."""
    rng = np.random.default_rng(0)
    frames = []
    slide = None
    for t in range(seconds):
        frame = np.full((H, W), 20, np.uint8)
        if t < 6:  # intro animation over the whole frame
            frame[:] = rng.integers(0, 256, (H, W), dtype=np.uint8)
            frames.append(frame)
            continue
        if slide is None or t % 10 == 0:
            slide = rng.integers(0, 256, (80, 150), dtype=np.uint8)
        frame[2:82, 40:190] = slide
        frame[60:106, 2:38] = rng.integers(0, 256, (46, 36), dtype=np.uint8)  # the speaker moves every second
        frames.append(frame)
    return np.stack(frames)


def test_detect_slide_box_composite_layout():
    box, how = slides.detect_slide_box(_composite_video(), 1920, 1080)
    assert how == "detected"
    x, y, w, h = box
    assert abs(x - 400) <= 10 and abs(y - 20) <= 10
    assert abs(x + w - 1900) <= 10 and abs(y + h - 820) <= 10
    assert all(v % 2 == 0 for v in box)


def test_detect_slide_box_scales_to_any_resolution():
    box, how = slides.detect_slide_box(_composite_video(), 1280, 720)
    assert how == "detected"
    x, y, w, h = box
    assert abs(x - 266) <= 8 and abs(x + w - 1266) <= 8 and abs(y + h - 546) <= 8


def test_detect_slide_box_full_screen_slides_and_static_video():
    pictures = [_pattern(i) for i in range(30)]
    full_screen = np.stack([pictures[t // 10] for t in range(300)])
    assert slides.detect_slide_box(full_screen, 1920, 1080) == ((0, 0, 1920, 1080), "full-frame")
    static = np.zeros((50, H, W), np.uint8)
    assert slides.detect_slide_box(static, 1280, 720) == ((0, 0, 1280, 720), "full-frame")


def test_parse_box():
    assert slides.parse_box("432,16,1472,827") == (432, 16, 1472, 827)
    for bad in ("1,2,3", "a,b,c,d", "0,0,0,10", "-1,0,10,10"):
        with pytest.raises(SlidesError):
            slides.parse_box(bad)


# --------------------------------------------------------------------------- builds, timeline


def _ink(rows: int) -> np.ndarray:
    ink = np.zeros((slides.INK_H, slides.INK_W), bool)
    ink[10:10 + rows * 10, 20:200] = True
    return ink


def test_collapse_builds_keeps_final_state_and_all_spans():
    frames = [
        {"start": 0, "end": 5, "repeats": []},
        {"start": 5, "end": 9, "repeats": [[30, 32]]},
        {"start": 9, "end": 14, "repeats": []},
        {"start": 14, "end": 20, "repeats": []},
    ]
    other = np.zeros_like(_ink(1))
    other[150:200, 250:380] = True
    merged = slides.collapse_builds(frames, [_ink(1), _ink(2), _ink(3), other])
    assert merged == [
        {"frame": 2, "spans": [[0, 5], [5, 9], [30, 32], [9, 14]]},
        {"frame": 3, "spans": [[14, 20]]},
    ]
    assert not slides.is_build_step(_ink(2), _ink(1))  # removing content is not a reveal
    assert not slides.is_build_step(np.zeros_like(other), other)


def test_full_bleed_picture_or_dark_slide_is_not_a_reveal():
    # A text slide followed by a photo/dark slide: the picture's ink covers the text, but it's a new slide.
    text = _ink(2)
    picture = np.ones_like(text)
    picture[200:, :] = False  # ~93 % ink, like an illustration or a dark-themed slide
    assert not slides.is_build_step(text, picture)
    dark_a = np.ones_like(text)
    dark_a[100:, 300:] = False
    assert not slides.is_build_step(dark_a, np.ones_like(text))


def test_assign_segments_by_midpoint_else_last_shown():
    spans = [[[0, 10], [40, 50]], [[12, 30]]]
    segs = [
        {"start": 0.0, "end": 4.0, "text": " intro "},
        {"start": 9.0, "end": 13.0, "text": "gap"},  # midpoint 11: between shots -> last shown (slide 0)
        {"start": 20.0, "end": 22.0, "text": "second"},
        {"start": 33.0, "end": 35.0, "text": "after second"},
        {"start": 44.0, "end": 46.0, "text": "back"},
        {"start": 1.0, "end": 2.0, "text": ""},
    ]
    assert slides.assign_segments(spans, segs) == [["intro", "gap", "back"], ["second", "after second"]]
    assert slides.assign_segments([[[5, 9]]], [{"start": 0.0, "end": 1.0, "text": "early"}]) == [["early"]]


def test_is_not_a_slide():
    assert slides.is_not_a_slide("NOT_A_SLIDE")
    assert slides.is_not_a_slide("  `NOT_A_SLIDE`.\n")
    assert not slides.is_not_a_slide("## NOT_A_SLIDE handling\n- text")
    assert slides.format_mmss(754.9) == "12:34"


# --------------------------------------------------------------------------- Claude


class FakeRun:
    """Records subprocess.run calls; `responder(call_no) -> (stdout, stderr)`."""

    def __init__(self, responder):
        self.responder = responder
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, cmd, input=None, capture_output=None, text=None, timeout=None, **kw):
        self.calls.append({"cmd": cmd, "input": input})
        stdout, stderr = self.responder(len(self.calls))
        return subprocess.CompletedProcess(cmd, 0 if stdout else 1, stdout=stdout, stderr=stderr)


def _stream(result: str, cost: float = 0.014, is_error: bool = False) -> str:
    events = [
        {"type": "system", "subtype": "init"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": result}]}},
        {"type": "result", "subtype": "success", "is_error": is_error, "result": result, "total_cost_usd": cost},
    ]
    return "\n".join(json.dumps(e) for e in events) + "\n"


@pytest.fixture
def reader(monkeypatch, tmp_path):
    monkeypatch.setattr(slides.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.delenv("CLAUDE_BIN", raising=False)
    monkeypatch.delenv("TRANSCRIBE_SLIDES_MODEL", raising=False)
    return slides.SlideReader(claude_bin=None, cache_dir=tmp_path / "cache")


def test_reader_command_keeps_isolation_flags_and_sends_image(monkeypatch, reader):
    fake = FakeRun(lambda n: (_stream("## hello"), ""))
    monkeypatch.setattr(slides.subprocess, "run", fake)
    md, cost, cached = reader.read(b"jpeg-bytes", "My Talk")
    assert (md, cost, cached) == ("## hello\n", 0.014, False)
    cmd = fake.calls[0]["cmd"]
    for flag in ("-p", "--safe-mode", "--strict-mcp-config", "--disable-slash-commands"):
        assert flag in cmd
    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--model") + 1] == "sonnet"
    assert cmd[cmd.index("--input-format") + 1] == "stream-json"
    message = json.loads(fake.calls[0]["input"])
    image, prompt = message["message"]["content"]
    assert image["type"] == "image" and image["source"]["media_type"] == "image/jpeg"
    assert image["source"]["data"] == "anBlZy1ieXRlcw=="
    assert "My Talk" in prompt["text"] and "NOT_A_SLIDE" in prompt["text"]


def test_reader_retries_once_then_succeeds(monkeypatch, reader):
    fake = FakeRun(lambda n: ("", "boom") if n == 1 else (_stream("## ok"), ""))
    monkeypatch.setattr(slides.subprocess, "run", fake)
    assert reader.read(b"img", "t")[0] == "## ok\n"
    assert len(fake.calls) == 2


def test_reader_raises_after_two_failures(monkeypatch, reader):
    fake = FakeRun(lambda n: (_stream("rate limited", is_error=True), ""))
    monkeypatch.setattr(slides.subprocess, "run", fake)
    with pytest.raises(SlidesError, match="2 attempts"):
        reader.read(b"img", "t")
    assert len(fake.calls) == 2
    assert not reader.cache_path(b"img").exists()


def test_reader_cache_hit_skips_subprocess(monkeypatch, reader):
    fake = FakeRun(lambda n: (_stream("## cached"), ""))
    monkeypatch.setattr(slides.subprocess, "run", fake)
    reader.read(b"img", "t")
    assert reader.read(b"img", "another title") == ("## cached\n", 0.0, True)
    assert len(fake.calls) == 1
    reader.model = "haiku"  # the model is part of the key
    reader.read(b"img", "t")
    assert len(fake.calls) == 2


def test_missing_claude_binary_raises_capability_error(monkeypatch, tmp_path):
    monkeypatch.setattr(slides.shutil, "which", lambda name: None)
    with pytest.raises(CapabilityError, match="--slides requires"):
        slides.SlideReader(cache_dir=tmp_path)


def test_model_env_override(monkeypatch, tmp_path):
    monkeypatch.setattr(slides.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setenv("TRANSCRIBE_SLIDES_MODEL", "haiku")
    assert slides.SlideReader(cache_dir=tmp_path).model == "haiku"
    assert slides.SlideReader("opus", cache_dir=tmp_path).model == "opus"


# --------------------------------------------------------------------------- builder


def _fake_video_tools(monkeypatch, thumbs: np.ndarray, inks: Dict[int, np.ndarray]):
    monkeypatch.setattr(slides, "probe_video_size", lambda path: (1920, 1080))
    sampled: List[Any] = []

    def sample(video, box=None):
        sampled.append(box)
        return thumbs

    def extract(video, second, box, out):
        out.write_bytes(f"frame-{second}".encode())

    monkeypatch.setattr(slides, "sample_thumbs", sample)
    monkeypatch.setattr(slides, "extract_frame", extract)
    monkeypatch.setattr(slides, "load_ink", lambda path: inks[int(path.stem.split("_")[1])])
    return sampled


def test_builder_drops_not_a_slide_frames_and_writes_deck(monkeypatch, tmp_path):
    a, cam, b = _pattern(1), _pattern(2), _pattern(3)
    thumbs = _thumbs((a, 10), (cam, 6), (b, 8))  # frames at seconds 5, 13, 20
    sampled = _fake_video_tools(monkeypatch, thumbs, {5: _ink(1), 13: _ink(1)[::-1, ::-1], 20: np.zeros_like(_ink(1))})
    monkeypatch.setattr(slides.shutil, "which", lambda name: f"/bin/{name}")
    answers = {"frame-5": "## first", "frame-13": "NOT_A_SLIDE", "frame-20": "## second\n- point"}
    calls: List[Any] = []

    def run(cmd, input=None, **kw):
        data = json.loads(input)["message"]["content"][0]["source"]["data"]
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=_stream(answers[base64.b64decode(data).decode()]), stderr="")

    monkeypatch.setattr(slides.subprocess, "run", run)
    builder = slides.SlideDeckBuilder("sonnet", box=(432, 16, 1472, 827), cache_dir=tmp_path / "cache")
    segs = [
        {"start": 1.0, "end": 3.0, "text": "hello"},
        {"start": 12.0, "end": 14.0, "text": "on camera"},
        {"start": 18.0, "end": 21.0, "text": "next"},
    ]
    meta = builder.build(Path("talk.mp4"), segs, tmp_path / "out_slides", "My Talk", tmp_path / "work")

    assert sampled == [(432, 16, 1472, 827)]  # manual box: no detection pass
    assert meta["box_source"] == "manual" and meta["box"] == [432, 16, 1472, 827]
    assert (meta["frames"], meta["frames_read"], meta["not_a_slide"], meta["slides"]) == (3, 3, 1, 2)
    assert meta["cost_usd"] == pytest.approx(0.042)
    deck = (tmp_path / "out_slides" / "slides.md").read_text()
    assert deck.startswith("# My Talk")
    assert "## first" in deck and "## second" in deck and "NOT_A_SLIDE" not in deck
    assert "*On screen: 00:00-00:10*" in deck
    assert "> hello on camera" in deck  # speech over the camera shot stays with the slide shown before it
    assert "> next" in deck
    assert sorted(p.name for p in (tmp_path / "out_slides" / "img").iterdir()) == ["slide_001.jpg", "slide_002.jpg"]
    assert (tmp_path / "out_slides" / "img" / "slide_002.jpg").read_bytes() == b"frame-20"

    # Rerun: every answer comes from the cache.
    count = len(calls)
    meta = builder.build(Path("talk.mp4"), segs, tmp_path / "out_slides", "My Talk", tmp_path / "work")
    assert len(calls) == count and meta["cached"] == 3 and meta["cost_usd"] == 0.0


def test_builder_detects_box_and_rejects_bad_manual_box(monkeypatch, tmp_path):
    monkeypatch.setattr(slides.shutil, "which", lambda name: f"/bin/{name}")
    sampled = _fake_video_tools(monkeypatch, _composite_video(60), {})
    builder = slides.SlideDeckBuilder(cache_dir=tmp_path)
    box, how, reuse = builder.resolve_box(Path("v.mp4"), (1920, 1080))
    assert how == "detected" and reuse is None and sampled == [None]

    builder.box = (1000, 0, 1000, 100)
    with pytest.raises(SlidesError, match="does not fit"):
        builder.resolve_box(Path("v.mp4"), (1920, 1080))

    monkeypatch.setattr(slides, "probe_video_size", lambda path: None)
    with pytest.raises(CapabilityError, match="video stream"):
        builder.build(Path("audio.m4a"), [], tmp_path / "o", "t", tmp_path / "w")


# --------------------------------------------------------------------------- CLI


def test_cli_slides_flags(monkeypatch):
    from transcribe.cli import parse_args

    monkeypatch.delenv("TRANSCRIBE_SLIDES_MODEL", raising=False)
    args = parse_args(["talk.mp4"])
    assert args.slides is False and args.slides_box is None and args.slides_model == "sonnet"
    args = parse_args(["talk.mp4", "--slides", "--slides-box", "1,2,3,4", "--slides-model", "opus"])
    assert args.slides and args.slides_box == "1,2,3,4" and args.slides_model == "opus"
    monkeypatch.setenv("TRANSCRIBE_SLIDES_MODEL", "haiku")
    assert parse_args(["talk.mp4"]).slides_model == "haiku"


def test_cli_slides_validation(monkeypatch, tmp_path):
    import transcribe.cli as cli

    media = tmp_path / "talk.mp4"
    media.write_bytes(b"x")
    assert cli.main([str(media), "--slides-box", "1,2,3,4"]) == 1  # needs --slides
    with pytest.raises(CapabilityError):
        cli.validate_slides_constraints("openai", "gpt-4o-transcribe")
    cli.validate_slides_constraints("openai", "whisper-1")
    cli.validate_slides_constraints("auto", "auto")


def _patch_pipeline(monkeypatch, tmp_path):
    import transcribe.cli as cli
    from transcribe.models import NormalizedMedia

    media = tmp_path / "My Talk.mp4"
    media.write_bytes(b"video")
    monkeypatch.setattr(cli.AudioNormalizer, "normalize", lambda self, path, backend, temp: NormalizedMedia(path, 2.0))
    monkeypatch.setattr(cli.BackendFactory, "create", staticmethod(lambda args: object()))
    result = BackendResult(text="hi", segments=[{"start": 0.0, "end": 1.0, "speaker": None, "text": "hi"}],
                           language="en", duration_sec=2.0, raw={})
    monkeypatch.setattr(cli, "transcribe_with_chunking", lambda **kw: result)
    monkeypatch.setattr(slides, "probe_video_size", lambda path: (1920, 1080))
    return cli, media


class FakeBuilder:
    instances: List["FakeBuilder"] = []

    def __init__(self, model=None, *, box=None, fail=False):
        self.model, self.box, self.calls = model, box, []
        FakeBuilder.instances.append(self)

    def check_box(self, size):
        self.size = size

    def build(self, video, segments, out_dir, title, work_dir):
        self.calls.append({"video": video, "segments": segments, "out_dir": out_dir, "title": title})
        # Transcript outputs must already be on disk when slides run.
        assert (out_dir.parent / "out.txt").exists()
        if self.box == (9, 9, 9, 9):
            raise RuntimeError("claude exploded")
        return {"deck": str(out_dir / "slides.md"), "slides": 3, "cost_usd": 0.04}


def test_cli_main_writes_transcript_then_slides(monkeypatch, tmp_path):
    cli, media = _patch_pipeline(monkeypatch, tmp_path)
    FakeBuilder.instances.clear()
    monkeypatch.setattr(slides, "SlideDeckBuilder", FakeBuilder)
    code = cli.main([str(media), "--backend", "eg2", "--slides", "--slides-box", "432,16,1472,827",
                     "--slides-model", "opus", "--output-dir", str(tmp_path), "--output-basename", "out",
                     "--output-formats", "txt,json"])
    assert code == 0
    builder = FakeBuilder.instances[0]
    assert builder.model == "opus" and builder.box == (432, 16, 1472, 827)
    call = builder.calls[0]
    assert call["video"] == media.resolve() and call["title"] == "My Talk"
    assert call["out_dir"] == tmp_path.resolve() / "out_slides"
    assert call["segments"][0]["text"] == "hi"
    data = json.loads((tmp_path / "out.json").read_text())
    assert data["metadata"]["slides"]["slides"] == 3


def test_cli_slides_failure_keeps_valid_transcript(monkeypatch, tmp_path):
    cli, media = _patch_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(slides, "SlideDeckBuilder", FakeBuilder)
    code = cli.main([str(media), "--backend", "eg2", "--slides", "--slides-box", "9,9,9,9",
                     "--output-dir", str(tmp_path), "--output-basename", "out", "--output-formats", "txt,json"])
    assert code == 1
    assert (tmp_path / "out.txt").read_text().strip().endswith("hi")
    data = json.loads((tmp_path / "out.json").read_text())
    assert data["segments"][0]["text"] == "hi" and "slides" not in data["metadata"]


def test_cli_slides_rejects_media_without_video(monkeypatch, tmp_path):
    cli, media = _patch_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(slides, "SlideDeckBuilder", FakeBuilder)
    monkeypatch.setattr(slides, "probe_video_size", lambda path: None)
    called = []
    monkeypatch.setattr(cli, "transcribe_with_chunking", lambda **kw: called.append(1))
    code = cli.main([str(media), "--backend", "eg2", "--slides", "--output-dir", str(tmp_path)])
    assert code == 1 and not called  # fails before transcribing


def test_media_downloader_video_prefers_mp4_video_only(monkeypatch, tmp_path):
    import types

    from transcribe.downloader import MediaDownloader

    seen: Dict[str, Any] = {}

    class FakeYDL:
        def __init__(self, opts):
            seen.update(opts)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            Path(seen["outtmpl"].replace("%(id)s", "abc").replace("%(ext)s", "mp4")).write_bytes(b"v")
            return {"id": "abc"}

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYDL))
    path = MediaDownloader().download_video("https://youtu.be/abc", tmp_path / "video")
    assert path == tmp_path / "video" / "abc.video.mp4"
    assert seen["format"].startswith("bestvideo[ext=mp4][height<=1080]")
    assert "postprocessors" not in seen


def test_runs_without_slides_do_not_import_the_slides_module():
    code = ("import sys, transcribe.cli, transcribe, transcribe_youtube; "
            "from transcribe.cli import parse_args; parse_args(['x.mp4', '--backend', 'mlx-whisper']); "
            "assert 'transcribe.slides' not in sys.modules, 'transcribe.slides imported'; "
            "transcribe.cli._slides_module(); assert 'transcribe.slides' in sys.modules")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          cwd=str(Path(__file__).resolve().parent.parent))
    assert done.returncode == 0, done.stderr


def test_slides_error_is_a_transcription_error():
    assert issubclass(SlidesError, TranscriptionError)

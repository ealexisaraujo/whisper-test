import sys
import types
from pathlib import Path

import pytest

from transcribe_youtube import (
    BackendResult,
    CapabilityError,
    InputResolver,
    MediaDownloader,
    OutputWriter,
    TranscriptionError,
    build_context_info,
    format_srt_timestamp,
    has_transcription_content,
    offset_segments,
    parse_output_formats,
    run_selected_backend,
    should_chunk_audio,
    validate_openai_output_constraints,
)


def test_format_srt_timestamp_uses_comma_milliseconds() -> None:
    assert format_srt_timestamp(3661.125) == "01:01:01,125"


def test_build_context_info_merges_hotwords() -> None:
    context = build_context_info("Meeting about platform migration", "Alice, Bob ,  API")
    assert context == "Meeting about platform migration\nHotwords: Alice, Bob, API"


def test_parse_output_formats_dedupes_and_validates() -> None:
    assert parse_output_formats("txt, srt,txt,json") == ["txt", "srt", "json"]
    with pytest.raises(TranscriptionError):
        parse_output_formats("txt,doc")


def test_openai_srt_constraint_rejected_for_non_timestamp_model() -> None:
    with pytest.raises(CapabilityError):
        validate_openai_output_constraints("gpt-4o-transcribe", ["txt", "srt", "json"])

    validate_openai_output_constraints("whisper-1", ["txt", "srt", "json"])


def test_manual_switch_policy_calls_only_selected_backend(tmp_path: Path) -> None:
    class FailingBackend:
        called = False

        def transcribe(self, **kwargs):
            self.called = True
            raise TranscriptionError("selected backend failed")

    class OtherBackend:
        called = False

        def transcribe(self, **kwargs):
            self.called = True
            return None

    failing = FailingBackend()
    other = OtherBackend()

    with pytest.raises(TranscriptionError):
        run_selected_backend(
            "mlx-whisper",
            {"mlx-whisper": failing, "openai": other},
            audio_path=tmp_path / "audio.wav",
            language=None,
            context_info=None,
            timeout_seconds=0,
            generation={},
        )

    assert failing.called is True
    assert other.called is False


def test_input_resolver_handles_url_and_local_file(tmp_path: Path) -> None:
    url_req = InputResolver.resolve("https://www.youtube.com/watch?v=abc123")
    assert url_req.is_youtube is True

    local_file = tmp_path / "meeting.mp4"
    local_file.write_text("x", encoding="utf-8")
    local_req = InputResolver.resolve(str(local_file))
    assert local_req.is_youtube is False
    assert local_req.local_path == local_file.resolve()


def test_input_resolver_normalizes_escaped_youtube_url() -> None:
    req = InputResolver.resolve("https://www.youtube.com/watch\\?v\\=fcZMmP5dsl4")
    assert req.is_youtube is True
    assert req.source == "https://www.youtube.com/watch?v=fcZMmP5dsl4"


def test_media_downloader_rejects_youtube_id_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download=True):
            del url, download
            return {"id": "wrong_video_id", "title": "Wrong Video"}

    fake_module = types.SimpleNamespace(YoutubeDL=FakeYDL)
    monkeypatch.setitem(sys.modules, "yt_dlp", fake_module)

    with pytest.raises(TranscriptionError, match="YouTube ID mismatch detected"):
        MediaDownloader().download(
            "https://www.youtube.com/watch?v=fcZMmP5dsl4",
            output_dir=tmp_path,
        )


def test_output_writer_generates_txt_srt_json(tmp_path: Path) -> None:
    payload = {
        "backend": "mlx-whisper",
        "model": "mlx-community/whisper-turbo",
        "source": "meeting.mp4",
        "language": "en",
        "duration_sec": 10.0,
        "context_info": "Hotwords: Alice",
        "segments": [
            {"start": 0.0, "end": 1.5, "speaker": "A", "text": "Hello"},
            {"start": 1.5, "end": 3.0, "speaker": "B", "text": "Hi"},
        ],
        "text": "Hello Hi",
        "created_at": "2026-02-13T00:00:00+00:00",
        "metadata": {},
    }

    written = OutputWriter.write(payload, ["txt", "srt", "json"], tmp_path, "sample")

    assert written["txt"].exists()
    assert written["srt"].exists()
    assert written["json"].exists()

    srt_content = written["srt"].read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:01,500" in srt_content
    assert "Speaker A: Hello" in srt_content


def test_offset_segments_applies_running_offset() -> None:
    segments = [{"start": 1.0, "end": 2.5, "speaker": "A", "text": "hola"}]
    out = offset_segments(segments, 30.0)
    assert out[0]["start"] == 31.0
    assert out[0]["end"] == 32.5


def test_should_chunk_audio_auto_rules(tmp_path: Path) -> None:
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"x" * (26 * 1024 * 1024))

    # Auto chunk by duration
    assert should_chunk_audio(
        chunk_mode="auto",
        backend="mlx-whisper",
        duration_sec=4000.0,
        audio_path=audio_file,
        chunk_threshold_minutes=45,
        openai_max_file_mb=24,
    )

    # Auto chunk by OpenAI file size
    assert should_chunk_audio(
        chunk_mode="auto",
        backend="openai",
        duration_sec=60.0,
        audio_path=audio_file,
        chunk_threshold_minutes=45,
        openai_max_file_mb=24,
    )

    # Explicit off disables chunking
    assert not should_chunk_audio(
        chunk_mode="off",
        backend="openai",
        duration_sec=9999.0,
        audio_path=audio_file,
        chunk_threshold_minutes=45,
        openai_max_file_mb=1,
    )


def test_should_chunk_audio_mlx_whisper_uses_generic_threshold(tmp_path: Path) -> None:
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"x")

    # Should chunk when duration exceeds generic threshold
    assert should_chunk_audio(
        chunk_mode="auto",
        backend="mlx-whisper",
        duration_sec=4000.0,
        audio_path=audio_file,
        chunk_threshold_minutes=45,
        openai_max_file_mb=24,
    )

    # Should NOT chunk when duration is under generic threshold
    assert not should_chunk_audio(
        chunk_mode="auto",
        backend="mlx-whisper",
        duration_sec=600.0,
        audio_path=audio_file,
        chunk_threshold_minutes=45,
        openai_max_file_mb=24,
    )


def test_has_transcription_content_detects_empty_result() -> None:
    empty = BackendResult(text="  ", segments=[], language=None, duration_sec=1.0, raw={})
    non_empty = BackendResult(
        text="",
        segments=[{"start": 0.0, "end": 1.0, "speaker": None, "text": "hola"}],
        language="es",
        duration_sec=1.0,
        raw={},
    )

    assert has_transcription_content(empty) is False
    assert has_transcription_content(non_empty) is True

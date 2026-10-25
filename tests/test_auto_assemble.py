"""Tests for diarization, segment assembly, trust score and output rendering (no models)."""

from __future__ import annotations

import sys
import types
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from transcribe.auto.assemble import (
    UNCLEAR_NONE,
    assign_speakers,
    build_segments,
    trust_score,
)
from transcribe.auto.diarize import diarize
from transcribe.auto.resolve import FusedWord
from transcribe.auto.types import StageStatus
from transcribe.output import OutputWriter


def fw(text, start, end, speaker="1", unclear=False, alternatives=None, source="agree", conf=0.7):
    return FusedWord(
        text=text,
        start=start,
        end=end,
        tier="unclear" if unclear else "solid",
        confidence=0.3 if unclear else conf,
        source=source,
        alternatives=list(alternatives or []),
        unclear=unclear,
        speaker=speaker,
    )


def seq(texts, start=0.0, step=0.3, speaker="1"):
    return [fw(t, start + i * step, start + i * step + 0.25, speaker) for i, t in enumerate(texts)]


# ----------------------------------------------------------------------------- build_segments


def test_build_segments_speaker_change_splits():
    words = seq(["hello", "there"], speaker="1") + seq(["hi", "you"], start=0.6, speaker="2")
    segs = build_segments(words)
    assert [s["speaker"] for s in segs] == ["1", "2"]
    assert [s["text"] for s in segs] == ["hello there", "hi you"]
    assert segs[0]["start"] == 0.0 and segs[0]["end"] == pytest.approx(0.55)


def test_segment_dict_shape_and_words_are_dicts():
    words = seq(["a", "b"])
    seg = build_segments(words)[0]
    assert set(seg) == {"start", "end", "speaker", "text", "words"}
    assert seg["words"] == [asdict(w) for w in words]


def test_build_segments_empty():
    assert build_segments([]) == []


def test_build_segments_splits_on_pause():
    words = seq(["one", "two"]) + seq(["three"], start=3.0)
    segs = build_segments(words, pause=0.8)
    assert [s["text"] for s in segs] == ["one two", "three"]


def test_build_segments_no_split_below_pause():
    words = seq(["one", "two"]) + seq(["three"], start=1.0)
    assert len(build_segments(words, pause=0.8)) == 1


def test_build_segments_sentence_end_after_min_duration():
    words = [fw(f"w{i}", i * 0.5, i * 0.5 + 0.4) for i in range(8)]
    words[6] = fw("done.", 3.0, 3.4)
    segs = build_segments(words)
    assert [s["text"] for s in segs] == ["w0 w1 w2 w3 w4 w5 done.", "w7"]


def test_build_segments_sentence_end_before_min_duration_keeps_going():
    words = [fw("hi.", 0.0, 0.3), fw("there", 0.4, 0.7)]
    assert len(build_segments(words)) == 1


def test_build_segments_max_seconds():
    words = [fw(f"w{i}", i * 0.5, i * 0.5 + 0.45) for i in range(30)]
    segs = build_segments(words, max_seconds=5.0)
    assert len(segs) >= 3
    assert all(s["end"] - s["start"] <= 5.0 + 1e-9 for s in segs)
    assert " ".join(s["text"] for s in segs) == " ".join(w.text for w in words)


def test_unclear_marker_groups_consecutive():
    # alternatives live on the first word of a span only
    words = [
        fw("a", 0.0, 0.2),
        fw("b", 0.3, 0.5, unclear=True, alternatives=["x"]),
        fw("c", 0.6, 0.8, unclear=True),
        fw("d", 0.9, 1.1),
    ]
    assert build_segments(words)[0]["text"] == "a [unclear: b c | x] d"


def test_unclear_marker_empty_alternative_renders_em_dash():
    words = [
        fw("a", 0.0, 0.2),
        fw("b", 0.3, 0.5, unclear=True, alternatives=[""]),
        fw("d", 0.9, 1.1),
    ]
    assert build_segments(words)[0]["text"] == f"a [unclear: b | {UNCLEAR_NONE}] d"
    assert UNCLEAR_NONE == "—"


def test_unclear_marker_without_alternatives_has_no_pipe():
    words = [fw("a", 0.0, 0.2), fw("b", 0.3, 0.5, unclear=True), fw("d", 0.6, 0.8)]
    assert build_segments(words)[0]["text"] == "a [unclear: b] d"


def test_unclear_marker_splits_at_each_span_with_alternatives():
    words = [
        fw("a", 0.0, 0.2),
        fw("b", 0.3, 0.5, unclear=True, alternatives=["x"]),
        fw("c", 0.6, 0.8, unclear=True, alternatives=["y"]),
        fw("d", 0.9, 1.1),
    ]
    assert build_segments(words)[0]["text"] == "a [unclear: b | x] [unclear: c | y] d"


def test_unclear_marker_multiword_spans_split_deterministically():
    words = [
        fw("b1", 0.0, 0.2, unclear=True, alternatives=["x1 x2"]),
        fw("b2", 0.3, 0.5, unclear=True),
        fw("c1", 0.6, 0.8, unclear=True, alternatives=[""]),
        fw("c2", 0.9, 1.1, unclear=True),
        fw("e", 1.2, 1.4),
    ]
    text = build_segments(words)[0]["text"]
    assert text == f"[unclear: b1 b2 | x1 x2] [unclear: c1 c2 | {UNCLEAR_NONE}] e"


def test_no_segment_split_inside_unclear_marker():
    # pause, speaker change and max_seconds all land inside the marker group
    words = [
        fw("a", 0.0, 0.2),
        fw("b", 0.3, 0.5, unclear=True, alternatives=["x"]),
        fw("c", 5.0, 5.2, unclear=True, speaker="2"),
        fw("d", 11.9, 12.5, unclear=True),
        fw("e", 13.0, 13.2),
    ]
    segs = build_segments(words, max_seconds=4.0, pause=0.8)
    for seg in segs:
        assert seg["text"].count("[") == seg["text"].count("]")
    marker_segs = [s for s in segs if "[unclear:" in s["text"]]
    assert len(marker_segs) == 1
    assert "[unclear: b c d | x]" in marker_segs[0]["text"]


def test_spanish_text_preserved():
    words = seq(["¿Qué", "tal,", "señor?"])
    seg = build_segments(words)[0]
    assert seg["text"] == "¿Qué tal, señor?"
    assert [w["text"] for w in seg["words"]] == ["¿Qué", "tal,", "señor?"]


def test_segment_renders_through_output_writer(tmp_path):
    words = [
        fw("¿Qué", 0.0, 0.3, speaker="1"),
        fw("tal?", 0.4, 0.7, speaker="1", unclear=True, alternatives=["tan"]),
        fw("Bien", 1.0, 1.3, speaker="2", unclear=True, alternatives=[""]),
    ]
    segs = build_segments(words)
    payload = {"segments": segs, "text": " ".join(s["text"] for s in segs)}
    written = OutputWriter.write(payload, ["txt", "srt", "json"], tmp_path, "out")
    txt = written["txt"].read_text(encoding="utf-8")
    srt = written["srt"].read_text(encoding="utf-8")
    assert "Speaker 1: ¿Qué [unclear: tal? | tan]" in txt
    assert f"Speaker 2: [unclear: Bien | {UNCLEAR_NONE}]" in txt
    assert "Speaker 1: ¿Qué [unclear: tal? | tan]" in srt
    assert "00:00:00,000 --> 00:00:00,700" in srt
    assert written["json"].read_text(encoding="utf-8").count('"speaker"') >= 2


# ----------------------------------------------------------------------------- assign_speakers


def test_assign_speakers_overlap():
    words = [fw("a", 0.0, 1.0, speaker=None), fw("b", 1.8, 3.0, speaker=None)]
    turns = [
        {"start": 0.0, "end": 1.5, "speaker": "1"},
        {"start": 1.5, "end": 4.0, "speaker": "2"},
    ]
    assign_speakers(words, turns)
    assert [w.speaker for w in words] == ["1", "2"]


def test_assign_speakers_largest_overlap_wins():
    words = [fw("a", 1.0, 2.0, speaker=None)]
    turns = [
        {"start": 0.0, "end": 1.2, "speaker": "1"},
        {"start": 1.2, "end": 3.0, "speaker": "2"},
    ]
    assign_speakers(words, turns)
    assert words[0].speaker == "2"


def test_assign_speakers_nearest_turn_when_no_overlap():
    words = [fw("a", 5.0, 5.2, speaker=None), fw("b", 9.0, 9.2, speaker=None)]
    turns = [
        {"start": 0.0, "end": 4.0, "speaker": "1"},
        {"start": 10.0, "end": 12.0, "speaker": "2"},
    ]
    assign_speakers(words, turns)
    assert [w.speaker for w in words] == ["1", "2"]


def test_assign_speakers_no_turns_defaults_to_one():
    words = [fw("a", 0.0, 1.0, speaker=None)]
    assign_speakers(words, [])
    assert words[0].speaker == "1"


# ----------------------------------------------------------------------------- diarize


def test_diarize_without_token_skips(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    turns, status = diarize(tmp_path / "x.wav")
    assert turns == []
    assert status.startswith("skipped: no HF_TOKEN")
    assert "pyannote/speaker-diarization-community-1" in status


class FakeAnnotation:
    def __init__(self, items):
        self.items = items

    def itertracks(self, yield_label=False):
        for start, end, label in self.items:
            yield types.SimpleNamespace(start=start, end=end), None, label


class FakePipeline:
    def __init__(self, result, error=None):
        self.result = result
        self.error = error
        self.calls = []
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def __call__(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        if self.error:
            raise self.error
        return self.result


def install_fake_pyannote(monkeypatch, pipeline=None, load_error=None):
    loads = []

    class Pipeline:
        @staticmethod
        def from_pretrained(name, **kwargs):
            loads.append((name, kwargs))
            if load_error:
                raise load_error
            return pipeline

    module = types.ModuleType("pyannote.audio")
    module.Pipeline = Pipeline
    parent = types.ModuleType("pyannote")
    parent.audio = module
    monkeypatch.setitem(sys.modules, "pyannote", parent)
    monkeypatch.setitem(sys.modules, "pyannote.audio", module)
    return loads


@pytest.fixture
def wav(tmp_path):
    import soundfile as sf

    path = tmp_path / "a.wav"
    sf.write(str(path), np.zeros(16000, dtype=np.float32), 16000)
    return path


def test_diarize_renumbers_speakers_by_first_appearance(monkeypatch, wav):
    monkeypatch.setenv("HF_TOKEN", "tok")
    annotation = FakeAnnotation(
        [(0.0, 1.0, "SPEAKER_07"), (1.0, 2.0, "SPEAKER_02"), (2.0, 3.0, "SPEAKER_07")]
    )
    pipeline = FakePipeline(types.SimpleNamespace(speaker_diarization=annotation))
    loads = install_fake_pyannote(monkeypatch, pipeline)
    turns, status = diarize(wav, num_speakers=2)
    assert status == "ok"
    assert [(t["start"], t["end"], t["speaker"]) for t in turns] == [
        (0.0, 1.0, "1"),
        (1.0, 2.0, "2"),
        (2.0, 3.0, "1"),
    ]
    assert loads[0][0] == "pyannote/speaker-diarization-community-1"
    audio, kwargs = pipeline.calls[0]
    assert audio["sample_rate"] == 16000
    assert tuple(audio["waveform"].shape) == (1, 16000)
    assert kwargs == {"num_speakers": 2}


def test_diarize_accepts_plain_annotation_and_uses_fallback_token(monkeypatch, wav):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "tok2")
    pipeline = FakePipeline(FakeAnnotation([(0.5, 1.5, "A")]))
    install_fake_pyannote(monkeypatch, pipeline)
    turns, status = diarize(wav)
    assert status == "ok" and turns[0]["speaker"] == "1"
    assert pipeline.calls[0][1] == {}


def test_diarize_load_failure_skips(monkeypatch, wav):
    monkeypatch.setenv("HF_TOKEN", "tok")
    install_fake_pyannote(monkeypatch, load_error=OSError("gated repo"))
    turns, status = diarize(wav)
    assert turns == [] and status == "skipped: OSError: gated repo"


def test_diarize_pipeline_none_means_terms_not_accepted(monkeypatch, wav):
    monkeypatch.setenv("HF_TOKEN", "tok")
    loads = install_fake_pyannote(monkeypatch, pipeline=None)  # pyannote returns None for a gated, unaccepted model
    turns, status = diarize(wav)
    assert loads and turns == []
    assert status == (
        "skipped: could not load pyannote/speaker-diarization-community-1: accept the model terms at "
        "https://huggingface.co/pyannote/speaker-diarization-community-1 with the account of HF_TOKEN"
    )


def test_diarize_run_failure_skips(monkeypatch, wav):
    monkeypatch.setenv("HF_TOKEN", "tok")
    pipeline = FakePipeline(None, error=RuntimeError("boom"))
    install_fake_pyannote(monkeypatch, pipeline)
    turns, status = diarize(wav)
    assert turns == [] and status == "skipped: RuntimeError: boom"


def test_diarize_unreadable_audio_skips(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "tok")
    install_fake_pyannote(monkeypatch, FakePipeline(FakeAnnotation([])))
    turns, status = diarize(tmp_path / "missing.wav")
    assert turns == [] and status.startswith("skipped: ")


# ----------------------------------------------------------------------------- trust_score


def many(n_agree, n_other=0, n_unclear=0, conf=0.7):
    words = [fw(f"w{i}", i, i + 0.5, source="agree", conf=conf) for i in range(n_agree)]
    words += [fw(f"o{i}", 100 + i, 100.5 + i, source="verify", conf=conf) for i in range(n_other)]
    words += [fw(f"u{i}", 200 + i, 200.5 + i, unclear=True, source="verify") for i in range(n_unclear)]
    return words


def test_trust_score_levels():
    high = trust_score(many(100), [StageStatus("diarize", "ok")], None)
    assert high["level"] == "high" and high["estimated_wer"] == pytest.approx(0.0)
    edge = trust_score(many(85, n_other=15), [], None)  # 0.5*0.15 = 0.075
    assert edge["level"] == "high"
    mid = trust_score(many(70, n_other=30), [], None)  # 0.5*0.3 = 0.15
    assert mid["level"] == "medium" and mid["estimated_wer"] == pytest.approx(0.15)
    low = trust_score(many(40, n_other=40, n_unclear=20), [], None)  # 0.5*0.6+0.2
    assert low["level"] == "low" and low["estimated_wer"] == pytest.approx(0.5)


def test_trust_score_features():
    words = many(6, n_other=2, n_unclear=2, conf=0.8)
    statuses = [StageStatus("a", "ok"), StageStatus("b", "skipped: no HF_TOKEN"), StageStatus("c", "failed")]
    result = trust_score(words, statuses, None, flag_rate=0.25)
    feats = result["features"]
    assert feats["agreement_rate"] == pytest.approx(0.6)
    assert feats["unclear_rate"] == pytest.approx(0.2)
    assert feats["mean_confidence"] == pytest.approx((8 * 0.8 + 2 * 0.3) / 10)
    assert feats["hallucination_flag_rate"] == 0.25
    assert feats["skipped_stages"] == 1
    assert result["estimated_wer"] == pytest.approx(0.5 * 0.4 + 0.2 + 0.05)  # + 0.05 for the one skipped stage
    assert result["level"] == "low"
    assert len(result["warnings"]) == 1 and "'b'" in result["warnings"][0]


def test_trust_score_empty_words_is_safe():
    result = trust_score([], [], None)
    assert result["features"]["agreement_rate"] == 0.0
    assert result["level"] == "low"


def test_trust_score_with_coef():
    words = many(8, n_other=2)
    coef = {"intercept": 0.01, "weights": {"agreement_rate": -0.02, "skipped_stages": 0.1, "unknown": 5.0}}
    result = trust_score(words, [StageStatus("x", "skipped: y")], coef)
    # the fixed per-stage penalty replaces a fitted `skipped_stages` weight: 0 (clipped 0.01 - 0.016) + 0.05
    assert result["estimated_wer"] == pytest.approx(0.05)
    assert result["level"] == "medium"  # "high" lowered one step for the skipped stage


def test_trust_score_coef_clipped_to_unit_interval():
    words = many(10)
    hi = trust_score(words, [], {"intercept": 5.0, "weights": {}})
    lo = trust_score(words, [], {"intercept": -5.0, "weights": {}})
    assert hi["estimated_wer"] == 1.0 and hi["level"] == "low"
    assert lo["estimated_wer"] == 0.0 and lo["level"] == "high"


def test_trust_score_threshold_boundaries():
    # 0.08 is medium, 0.2 is low
    at_high = trust_score(many(10), [], {"intercept": 0.08, "weights": {}})
    at_low = trust_score(many(10), [], {"intercept": 0.2, "weights": {}})
    assert at_high["level"] == "medium"
    assert at_low["level"] == "low"


# ----------------------------------------------------------------------------- trust honesty (uncalibrated)

PAIR = {"engines": ["qwen3-asr-1.7b", "whisper-large-v3"], "trust_spearman": 0.8}


def test_trust_single_qwen3_is_uncalibrated_even_with_a_good_spearman():
    entry = {"engines": ["qwen3-asr-1.7b"], "trust_spearman": 0.95}
    result = trust_score(many(100), [], None, entry=entry)
    assert result["level"] == "uncalibrated" and result["estimated_wer"] is None
    assert "constant" in result["reason"] and "qwen3-asr-1.7b" in result["reason"]
    for engine in ("parakeet-v3", "eg2"):
        assert trust_score(many(100), [], None, entry={"engines": [engine], "trust_spearman": 0.95})["level"] == (
            "uncalibrated")


def test_trust_single_engine_with_word_probs_is_judged_by_its_spearman():
    whisper = {"engines": ["whisper-large-v3"]}
    assert trust_score(many(100), [], None, entry={**whisper, "trust_spearman": 0.8})["level"] == "high"
    assert trust_score(many(100), [], None, entry={**whisper, "trust_spearman": 0.63})["level"] == "uncalibrated"


@pytest.mark.parametrize("entry", [
    {"engines": ["qwen3-asr-1.7b", "whisper-large-v3"]},  # no validated fit stored
    {"engines": ["qwen3-asr-1.7b", "whisper-large-v3"], "trust_spearman": None},
    {"engines": ["qwen3-asr-1.7b", "whisper-large-v3"], "trust_spearman": 0.69},
    {"engines": ["qwen3-asr-1.7b", "whisper-large-v3"], "trust_spearman": 0.24},
])
def test_trust_without_a_validated_fit_or_below_0_7_is_uncalibrated(entry):
    result = trust_score(many(100), [], None, entry=entry)
    assert result["level"] == "uncalibrated" and result["estimated_wer"] is None and result["reason"]
    assert result["features"]["agreement_rate"] == pytest.approx(1.0)  # the features are still reported


def test_trust_spearman_at_or_above_0_7_is_calibrated():
    for value in (0.7, 0.81):
        result = trust_score(many(100), [], None, entry={**PAIR, "trust_spearman": value})
        assert result["level"] == "high" and result["estimated_wer"] == pytest.approx(0.0)
        assert "reason" not in result


def test_trust_skipped_stage_adds_penalty_and_lowers_the_level_one_step():
    words = many(70, n_other=30)  # 0.5 * 0.3 = 0.15 -> medium
    base = trust_score(words, [StageStatus("verify", "ok")], None, entry=PAIR)
    assert base["level"] == "medium" and base["estimated_wer"] == pytest.approx(0.15) and base["warnings"] == []
    one = trust_score(words, [StageStatus("verify", "skipped: boom")], None, entry=PAIR)
    assert one["estimated_wer"] == pytest.approx(0.20) and one["level"] == "low"
    assert one["warnings"] == ["stage 'verify' was skipped: skipped: boom"]
    high = many(100)
    assert trust_score(high, [], None, entry=PAIR)["level"] == "high"
    assert trust_score(high, [StageStatus("a", "skipped: x")], None, entry=PAIR)["level"] == "medium"
    two = trust_score(high, [StageStatus("a", "skipped: x"), StageStatus("b", "skipped: y")], None, entry=PAIR)
    assert two["level"] == "low" and two["estimated_wer"] == pytest.approx(0.10)
    assert two["features"]["skipped_stages"] == 2 and len(two["warnings"]) == 2


def test_trust_uncalibrated_still_lists_skipped_stages():
    entry = {"engines": ["qwen3-asr-1.7b"]}
    result = trust_score(many(10), [StageStatus("diarize", "skipped: no HF_TOKEN")], None, entry=entry)
    assert result["level"] == "uncalibrated" and result["estimated_wer"] is None
    assert result["warnings"] == ["stage 'diarize' was skipped: skipped: no HF_TOKEN"]

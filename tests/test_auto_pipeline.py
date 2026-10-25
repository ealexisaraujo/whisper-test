"""Pipeline orchestration, cache, profiles and the `--backend auto` CLI branch (fakes only)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

import numpy as np
import pytest
import soundfile as sf

from transcribe import cli
from transcribe.auto.cache import StageCache, engine_output_from_dict, profile_from_dict
from transcribe.auto.pipeline import run_auto
from transcribe.auto.profiles import load_profiles, select
from transcribe.auto.refine_gated import GatedRefiner
from transcribe.auto.types import EngineOutput, Word
from transcribe.cli import parse_args
from transcribe.models import BackendResult, NormalizedMedia, TranscriptionError

SR = 16000


# ----------------------------------------------------------------------------- fakes


def words_of(texts, start=0.5, step=0.5, prob=None):
    out = []
    for i, text in enumerate(texts):
        s = start + i * step
        out.append(Word(text, s, s + 0.4, prob))
    return out


class FakeEngine:
    def __init__(self, name, texts, events, prob=None, fail=None, languages=None):
        self.name, self.texts, self.events, self.prob, self.fail = name, texts, events, prob, fail
        self.languages = languages if languages is not None else []

    def transcribe_chunks(self, samples, chunks, language):
        self.events.append(f"{self.name}.transcribe")
        self.languages.append(language)
        if self.fail is not None:
            raise self.fail
        return EngineOutput(
            self.name,
            language,
            {c.index: words_of(self.texts, c.start, prob=self.prob) for c in chunks},
            {},
        )

    def release(self):
        self.events.append(f"{self.name}.release")


class FakeScorer:
    """Favours texts containing 'hat' unless `constant`; optionally fails on score."""

    def __init__(self, events, constant=False, fail=None):
        self.events, self.constant, self.fail = events, constant, fail

    def score_detail(self, samples, start, end, text, lang):
        self.events.append("scorer.score")
        if self.fail is not None:
            raise self.fail
        if self.constant:
            return (-5.0, 5)
        return (-1.0 if "hat" in text.split() else -9.0, 5)

    def score(self, samples, start, end, text, lang):
        total, count = self.score_detail(samples, start, end, text, lang)
        return total / count

    def release(self):
        self.events.append("scorer.release")


class FakePicker:
    def __init__(self, events, answer="b", fail=None):
        self.events, self.answer, self.fail = events, answer, fail

    def pick(self, spans, lang, terms):
        self.events.append("picker.pick")
        if self.fail is not None:
            raise self.fail
        return {s.id: self.answer for s in spans}

    def release(self):
        self.events.append("picker.release")


class FakeRefiner:
    def __init__(self, events, status="ok"):
        self.events, self.status = events, status
        self.calls = 0

    def refine(self, words, targets, terms, lang, scorer, samples):
        self.calls += 1
        self.events.append("refiner.refine")
        return list(words), {"status": self.status, "targets": len(targets)}


class Rig:
    """Bundle of fakes plus the factories dict `run_auto` consumes."""

    def __init__(self, tmp_path, *, a=None, b=None, picker=None, scorer=None, refiner=None,
                 spans=((0.5, 4.0),), lang=("en", 0.95), turns=None, a_prob=None, b_prob=None):
        self.events = []
        self.a_langs, self.b_langs = [], []
        self.engine_calls = []
        self.vad_calls = 0
        self.lid_calls = 0
        self.scorer_calls = 0
        self.diarize_calls = []
        self.a = a or FakeEngine("qwen3-asr-1.7b", "the cat sat on the mat".split(), self.events,
                                 prob=a_prob, languages=self.a_langs)
        self.b = b or FakeEngine("whisper-large-v3", "the hat sat on the mat".split(), self.events,
                                 prob=b_prob, languages=self.b_langs)
        self.picker = picker or FakePicker(self.events)
        self.scorer = scorer or FakeScorer(self.events)
        self.refiner = refiner or FakeRefiner(self.events)
        self.spans = list(spans)
        self.lang = lang
        self.turns = turns if turns is not None else [{"start": 0.0, "end": 6.0, "speaker": "1"}]
        self.cache = StageCache(tmp_path / "cache")

    def factories(self):
        engines = {"qwen3-asr-1.7b": self.a, "whisper-large-v3": self.b}

        def engine(engine_id):
            self.engine_calls.append(engine_id)
            return engines[engine_id]

        def scorer():
            self.scorer_calls += 1
            return self.scorer

        def vad(samples, sr):
            self.vad_calls += 1
            return list(self.spans)

        def lid():
            def detect(window):
                self.lid_calls += 1
                return self.lang
            return detect

        def diarizer(path, num_speakers):
            self.diarize_calls.append(num_speakers)
            return list(self.turns), "ok"

        return {
            "engine": engine,
            "scorer": scorer,
            "picker": lambda: self.picker,
            "refiner": lambda model: self.refiner,
            "diarizer": diarizer,
            "lid": lid,
            "vad": vad,
            "cache": self.cache,
        }


@pytest.fixture
def audio(tmp_path):
    t = np.arange(int(6 * SR)) / SR
    path = tmp_path / "talk.wav"
    sf.write(str(path), (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), SR)
    return path


def run(audio, rig, **kw):
    return run_auto(audio, kw.pop("metadata", {}), factories=rig.factories(), **kw)


def stage(meta, name):
    return next(s for s in meta["stages"] if s["name"] == name)


# ----------------------------------------------------------------------------- pipeline


def test_run_auto_end_to_end_fakes(tmp_path, audio):
    rig = Rig(tmp_path)
    result, meta = run(audio, rig)

    assert isinstance(result, BackendResult)
    assert "the hat sat on the mat" in result.text
    assert result.segments[0]["speaker"] == "1"
    assert result.language == "en"
    assert result.duration_sec == pytest.approx(6.0)
    assert result.raw is meta
    names = [s["name"] for s in meta["stages"]]
    assert names == ["load", "vad", "probe", "diarize", "engine_a", "engine_b", "guards", "align",
                     "pick", "verify", "refine", "assemble", "trust"]
    assert all(s["status"] == "ok" for s in meta["stages"]), meta["stages"]
    assert meta["engines"] == ["qwen3-asr-1.7b", "whisper-large-v3"]
    assert meta["profile"]["language"] == "en"
    assert meta["profile_entry"]["margin"] == 0.05
    assert meta["trust"]["level"] in {"high", "medium", "low"}
    assert [w["text"] for w in meta["words"]] == "the hat sat on the mat".split()
    json.dumps(meta)  # fully serialisable
    # no refine targets -> no refiner call and the verify scorer is not reloaded
    assert rig.refiner.calls == 0
    assert rig.scorer_calls == 1


def test_engine_exception_is_core_failure(tmp_path, audio):
    rig = Rig(tmp_path, a=FakeEngine("qwen3-asr-1.7b", [], [], fail=TranscriptionError("boom")))
    with pytest.raises(TranscriptionError, match="boom"):
        run(audio, rig)

    rig = Rig(tmp_path, b=FakeEngine("whisper-large-v3", [], [], fail=RuntimeError("oom")))
    with pytest.raises(TranscriptionError, match="engine_b"):
        run(audio, rig)


def test_picker_failure_skips_and_marks_unclear(tmp_path, audio):
    rig = Rig(tmp_path, picker=FakePicker([], fail=RuntimeError("no model")),
              scorer=FakeScorer([], constant=True))
    result, meta = run(audio, rig)

    pick = stage(meta, "pick")
    assert pick["status"].startswith("skipped: RuntimeError: no model")
    unclear = [w for w in meta["words"] if w["unclear"]]
    assert unclear and unclear[0]["text"] == "cat"  # engine A is the default side
    assert "[unclear:" in result.text
    assert meta["trust"]["features"]["skipped_stages"] == 1


def test_no_speech_no_engine_calls(tmp_path, audio):
    rig = Rig(tmp_path, spans=())
    result, meta = run(audio, rig)

    assert rig.engine_calls == []
    assert rig.scorer_calls == 0
    assert result.segments == [] and result.text == ""
    assert "no speech detected" in meta["notes"]
    assert meta["no_speech"] is True
    assert meta["words"] == []
    assert [s["name"] for s in meta["stages"]][:3] == ["load", "vad", "probe"]


def test_releases_called_between_stages(tmp_path, audio):
    rig = Rig(tmp_path)
    run(audio, rig)
    ev = rig.events
    order = ["qwen3-asr-1.7b.transcribe", "qwen3-asr-1.7b.release", "whisper-large-v3.transcribe",
             "whisper-large-v3.release", "picker.pick", "picker.release", "scorer.score",
             "scorer.release"]
    positions = [ev.index(name) for name in order]
    assert positions == sorted(positions)
    assert ev.count("qwen3-asr-1.7b.release") == 1
    assert ev.count("picker.release") == 1
    assert ev.count("scorer.release") == 1
    assert ev.index("scorer.release") > max(i for i, e in enumerate(ev) if e == "scorer.score")


def test_release_happens_even_when_stage_fails(tmp_path, audio):
    rig = Rig(tmp_path, picker=FakePicker([], fail=RuntimeError("x")))
    rig.picker.events = rig.events
    run(audio, rig)
    assert "picker.release" in rig.events

    rig = Rig(tmp_path / "second", b=FakeEngine("whisper-large-v3", [], [], fail=RuntimeError("oom")))
    rig.b.events = rig.events
    with pytest.raises(TranscriptionError):
        run(audio, rig)
    assert "whisper-large-v3.release" in rig.events


def test_cache_hit_skips_engine_second_run(tmp_path, audio):
    rig = Rig(tmp_path)
    first, _ = run(audio, rig)
    assert sorted(rig.engine_calls) == ["qwen3-asr-1.7b", "whisper-large-v3"]
    assert (rig.vad_calls, rig.lid_calls) == (1, 1)

    rig.engine_calls.clear()
    second, meta = run(audio, rig)
    assert rig.engine_calls == []
    assert (rig.vad_calls, rig.lid_calls) == (1, 1)
    assert second.text == first.text
    assert "cache" in stage(meta, "engine_a")["detail"]

    # a different language override is a different engine key
    run(audio, rig, language="es")
    assert sorted(rig.engine_calls) == ["qwen3-asr-1.7b", "whisper-large-v3"]


def test_profiles_select_fallbacks_and_unvalidated():
    profiles = load_profiles()
    en = select(profiles, "en")
    assert en["engines"] == ["qwen3-asr-1.7b", "whisper-large-v3"]
    assert "unvalidated" not in en
    noisy = select(profiles, "en-phone-noisy")
    assert noisy["engines"] == en["engines"] and noisy["unvalidated"] is True
    assert select(profiles, "es")["engines"] == ["qwen3-asr-1.7b", "whisper-large-v3"]
    other = select(profiles, "other-phone")
    assert other["engines"] == ["whisper-large-v3", "qwen3-asr-1.7b"] and other["unvalidated"] is True
    fr = select(profiles, "fr")
    assert fr["engines"] == profiles["default"]["engines"] and fr["unvalidated"] is True
    # selection never mutates the table
    assert "unvalidated" not in profiles["profiles"]["en"]
    assert profiles["version"] == 1 and profiles["generated_from"] is None and profiles["trust"] is None
    for entry in [*profiles["profiles"].values(), profiles["default"]]:
        assert entry["pick"] and entry["verify"] and entry["refine"]
        assert entry["margin"] == 0.05 and entry["solid_min_prob"] == 0.5


def test_single_engine_profile(tmp_path, audio):
    profiles = load_profiles()
    profiles["profiles"]["en"]["engines"] = ["qwen3-asr-1.7b"]
    rig = Rig(tmp_path, scorer=FakeScorer([], constant=True), a_prob=0.9)
    rig.a = FakeEngine("qwen3-asr-1.7b", "the cat sat on the mat".split(), rig.events, prob=0.2)
    rig.picker = FakePicker(rig.events, fail=AssertionError("picker must not be called"))
    result, meta = run(audio, rig, profiles=profiles)

    names = [s["name"] for s in meta["stages"]]
    assert "engine_b" not in names
    assert "picker.pick" not in rig.events
    assert stage(meta, "pick")["status"] == "off"
    assert rig.engine_calls == ["qwen3-asr-1.7b"]
    assert any(w["unclear"] for w in meta["words"])
    assert meta["engines"] == ["qwen3-asr-1.7b"]


def test_resolve_scorer_failure_marks_verify_skipped_and_resolves_unverified(tmp_path, audio):
    rig = Rig(tmp_path, scorer=FakeScorer([], fail=RuntimeError("scorer load")))
    rig.scorer.events = rig.events
    result, meta = run(audio, rig)

    verify = stage(meta, "verify")
    assert verify["status"].startswith("skipped: RuntimeError: scorer load")
    assert "scorer.release" in rig.events
    hat = next(w for w in meta["words"] if w["text"] == "hat")  # picker said b
    assert hat["unclear"] is True
    assert meta["trust"]["features"]["skipped_stages"] == 1


@pytest.mark.parametrize(
    "refine_status, expected",
    [("ok", "ok"), ("skipped: claude CLI failed", "skipped: claude CLI failed"),
     ("partial: 1 batches failed", "partial: 1 batches failed")],
)
def test_refine_status_mapping(tmp_path, audio, refine_status, expected):
    rig = Rig(tmp_path, picker=FakePicker([], answer="unclear"), scorer=FakeScorer([], constant=True),
              refiner=FakeRefiner([], status=refine_status))
    result, meta = run(audio, rig)
    assert rig.refiner.calls == 1
    assert stage(meta, "refine")["status"] == expected
    assert meta["refine"]["status"] == refine_status
    assert rig.scorer_calls == 2  # reloaded for refine because there are targets


def test_refine_disabled_and_unsupported_language(tmp_path, audio):
    rig = Rig(tmp_path, picker=FakePicker([], answer="unclear"), scorer=FakeScorer([], constant=True))
    _, meta = run(audio, rig, refine=False)
    assert rig.refiner.calls == 0
    assert stage(meta, "refine")["status"] == "off"
    assert meta["trust"]["features"]["skipped_stages"] == 0


def test_language_override_and_other_language(tmp_path, audio):
    rig = Rig(tmp_path)
    result, meta = run(audio, rig, language="es")
    assert rig.a_langs == ["es"] and rig.b_langs == ["es"]
    assert result.language == "es"
    assert meta["language"] == "es"
    assert meta["profile_key"].startswith("es")

    rig = Rig(tmp_path / "other", lang=("fr", 0.9))
    result, meta = run(audio, rig)
    assert rig.a_langs == [None] and rig.b_langs == [None]  # whisper (A) detects per chunk
    assert meta["engines"] == ["whisper-large-v3", "qwen3-asr-1.7b"]
    assert result.language == "fr"
    assert stage(meta, "refine")["status"] == "off"
    assert any("best effort" in note for note in meta["notes"])


def test_diarize_skipped_for_single_speaker(tmp_path, audio):
    rig = Rig(tmp_path)
    _, meta = run(audio, rig, num_speakers=1)
    assert rig.diarize_calls == []
    assert stage(meta, "diarize")["status"] == "off"

    rig = Rig(tmp_path / "two")
    run(audio, rig, num_speakers=2)
    assert rig.diarize_calls == [2]


def test_diarize_skipped_status_is_kept(tmp_path, audio):
    rig = Rig(tmp_path)
    fac = rig.factories()
    fac["diarizer"] = lambda path, n: ([], "skipped: no HF_TOKEN")
    result, meta = run_auto(audio, {}, factories=fac)
    assert stage(meta, "diarize")["status"] == "skipped: no HF_TOKEN"
    assert result.segments[0]["speaker"] == "1"
    assert meta["trust"]["features"]["skipped_stages"] == 1


def test_language_windows_round_trip(tmp_path, audio):
    rig = Rig(tmp_path)
    _, meta = run(audio, rig)
    from transcribe.auto.types import AudioProfile

    profile = AudioProfile(**{**meta["profile"], "language_windows": [tuple(w) for w in meta["profile"]["language_windows"]]})
    assert profile.language_windows and isinstance(profile.language_windows[0], tuple)
    again = profile_from_dict(json.loads(json.dumps(asdict(profile))))
    assert again == profile
    assert all(isinstance(w, tuple) for w in again.language_windows)


def test_engine_output_round_trip():
    out = EngineOutput("e", "en", {0: words_of(["a", "b"], prob=0.5), 3: []}, {0: ["repetition"]})
    assert engine_output_from_dict(json.loads(json.dumps(asdict(out)))) == out


def test_stage_cache_get_put(tmp_path):
    cache = StageCache(tmp_path / "c")
    key = {"audio": "abc", "engine": "x"}
    assert cache.get("engine", key) is None
    cache.put("engine", key, {"k": [1, 2]})
    assert cache.get("engine", key) == {"k": [1, 2]}
    assert cache.get("engine", {**key, "engine": "y"}) is None
    assert cache.get("probe", key) is None
    path = cache._path("engine", key)
    path.write_text("{not json")
    assert cache.get("engine", key) is None  # corrupt entry is a miss


# ----------------------------------------------------------------------------- CLI


def test_existing_backends_unchanged(monkeypatch):
    monkeypatch.delenv("TRANSCRIBE_REFINE", raising=False)
    args = parse_args(["x.wav"])
    assert args.backend == "mlx-whisper"
    assert args.refine is None and args.terms is None and args.num_speakers is None
    assert parse_args(["x.wav", "--backend", "auto", "--terms", "A,B", "--num-speakers", "2"]).num_speakers == 2
    assert parse_args(["x.wav", "--no-refine"]).refine is False


class _Records(logging.Handler):
    """Capture `transcribe_youtube` records; `configure_logging` clears the root handlers caplog uses."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def caplog():
    handler = _Records()
    logger = logging.getLogger("transcribe_youtube")
    logger.addHandler(handler)
    yield handler
    logger.removeHandler(handler)


def _patch_cli(monkeypatch, rig):
    monkeypatch.setattr(cli.AudioNormalizer, "normalize", lambda self, path, backend, temp: NormalizedMedia(path, 6.0))
    monkeypatch.setattr(cli, "_probe_sample_rate", lambda path: SR)
    real = cli.run_auto
    monkeypatch.setattr(cli, "run_auto", lambda *a, **k: real(*a, factories=rig.factories(), **k))


def test_refine_cli_missing_still_writes(tmp_path, audio, monkeypatch, caplog):
    rig = Rig(tmp_path, picker=FakePicker([], answer="unclear"), scorer=FakeScorer([], constant=True))
    rig.refiner = GatedRefiner(model="sonnet")
    monkeypatch.setenv("CLAUDE_BIN", "/nonexistent/claude")
    _patch_cli(monkeypatch, rig)
    out = tmp_path / "out"

    code = cli.main([str(audio), "--backend", "auto", "--output-dir", str(out), "--output-basename", "r"])

    assert code == 0
    for ext in ("txt", "srt", "json"):
        assert (out / f"r.{ext}").exists()
    data = json.loads((out / "r.json").read_text())
    assert data["backend"] == "auto"
    refine = next(s for s in data["metadata"]["auto"]["stages"] if s["name"] == "refine")
    assert refine["status"].startswith("skipped")
    assert any("refine" in rec.getMessage() and rec.levelno == logging.WARNING for rec in caplog.records)
    assert "[unclear:" in data["text"]


def test_cli_low_trust_logs_warning(tmp_path, audio, monkeypatch, caplog):
    rig = Rig(tmp_path, picker=FakePicker([], answer="unclear"), scorer=FakeScorer([], constant=True))
    _patch_cli(monkeypatch, rig)
    code = cli.main([str(audio), "--backend", "auto", "--no-refine", "--output-dir", str(tmp_path / "o"),
                     "--output-formats", "json"])
    assert code == 0
    data = json.loads(next((tmp_path / "o").glob("*.json")).read_text())
    assert data["metadata"]["auto"]["trust"]["level"] == "low"
    assert any("low trust" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_cli_no_speech_writes_outputs(tmp_path, audio, monkeypatch, caplog):
    rig = Rig(tmp_path, spans=())
    _patch_cli(monkeypatch, rig)
    out = tmp_path / "ns"
    code = cli.main([str(audio), "--backend", "auto", "--output-dir", str(out), "--output-basename", "n"])
    assert code == 0
    assert (out / "n.txt").exists() and (out / "n.srt").read_text() == ""
    data = json.loads((out / "n.json").read_text())
    assert data["segments"] == [] and data["metadata"]["no_speech"] is True
    assert "no speech detected" in data["metadata"]["auto"]["notes"]
    assert rig.engine_calls == []


def test_cli_core_failure_message_for_auto(tmp_path, audio, monkeypatch, caplog):
    rig = Rig(tmp_path, a=FakeEngine("qwen3-asr-1.7b", [], [], fail=TranscriptionError("boom")))
    _patch_cli(monkeypatch, rig)
    code = cli.main([str(audio), "--backend", "auto", "--output-dir", str(tmp_path / "e")])
    assert code == 1
    messages = [r.getMessage() for r in caplog.records]
    assert "auto pipeline: core stage failed (no engine switching)" in messages
    assert not any("No automatic fallback" in m for m in messages)


def test_cli_existing_backend_keeps_fallback_message(tmp_path, audio, monkeypatch, caplog):
    monkeypatch.setattr(cli.AudioNormalizer, "normalize", lambda self, path, backend, temp: NormalizedMedia(path, 6.0))
    monkeypatch.setattr(cli.BackendFactory, "create", staticmethod(lambda args: object()))

    def boom(**kw):
        raise TranscriptionError("nope")

    monkeypatch.setattr(cli, "transcribe_with_chunking", boom)
    assert cli.main([str(audio), "--backend", "eg2", "--output-dir", str(tmp_path)]) == 1
    assert any("No automatic fallback was applied (manual switch policy)." in r.getMessage() for r in caplog.records)


def test_normalizer_loudnorm_only_for_auto(tmp_path, monkeypatch):
    from transcribe import normalizer

    commands = []

    def fake_run(command, **kw):
        commands.append(command)
        (tmp_path / "x").write_text("")

    monkeypatch.setattr(normalizer.subprocess, "run", fake_run)
    monkeypatch.setattr(normalizer, "_probe_duration_seconds", lambda p: 1.0)
    norm = normalizer.AudioNormalizer()
    assert norm.SAMPLE_RATES["auto"] == 16000
    norm.normalize(tmp_path / "in.wav", "auto", tmp_path)
    norm.normalize(tmp_path / "in.wav", "mlx-whisper", tmp_path)
    auto_cmd, other_cmd = commands
    assert "-af" in auto_cmd and auto_cmd[auto_cmd.index("-af") + 1].startswith("loudnorm=")
    assert "16000" in auto_cmd
    assert "-af" not in other_cmd


# ----------------------------------------------------------------------------- fix round 1


def _second_run_calls(rig, audio, monkeypatch, target, name, value):
    """Run once, change one config constant, run again; returns (vad, lid, engine ids) of run two."""
    run(audio, rig)
    rig.engine_calls.clear()
    before = (rig.vad_calls, rig.lid_calls)
    monkeypatch.setattr(target, name, value)
    run(audio, rig)
    return rig.vad_calls - before[0], rig.lid_calls - before[1], sorted(rig.engine_calls)


def test_cache_unchanged_config_is_a_hit(tmp_path, audio):
    rig = Rig(tmp_path)
    run(audio, rig)
    rig.engine_calls.clear()
    run(audio, rig)
    assert (rig.vad_calls, rig.lid_calls, rig.engine_calls) == (1, 1, [])


def test_cache_vad_param_change_misses(tmp_path, audio, monkeypatch):
    from transcribe.auto import chunks as chunk_mod

    rig = Rig(tmp_path)
    vad, lid, engines = _second_run_calls(
        rig, audio, monkeypatch, chunk_mod, "VAD_KWARGS", {**chunk_mod.VAD_KWARGS, "threshold": 0.7})
    assert vad == 1 and lid == 1  # spans changed, so the probe (which consumes them) reruns too


def test_cache_chunk_setting_change_misses(tmp_path, audio, monkeypatch):
    from transcribe.auto import chunks as chunk_mod

    rig = Rig(tmp_path)
    vad, lid, engines = _second_run_calls(rig, audio, monkeypatch, chunk_mod, "MERGE_GAP_SECONDS", 0.2)
    assert vad == 1


def test_cache_probe_config_change_misses_only_probe(tmp_path, audio, monkeypatch):
    from transcribe.auto import probe as probe_mod

    rig = Rig(tmp_path)
    vad, lid, engines = _second_run_calls(rig, audio, monkeypatch, probe_mod, "PHONE_ENERGY_FRACTION", 0.5)
    assert (vad, lid, engines) == (0, 1, [])


def test_cache_lid_repo_change_misses_probe(tmp_path, audio):
    rig = Rig(tmp_path)
    run(audio, rig)
    fac = rig.factories()
    fac["lid_repo"] = "other/lid-model"
    run_auto(audio, {}, factories=fac)
    assert rig.lid_calls == 2 and rig.vad_calls == 1


def test_cache_engine_decode_param_change_misses_only_that_engine(tmp_path, audio, monkeypatch):
    from transcribe.auto import engines as engine_mod

    rig = Rig(tmp_path)
    vad, lid, engines = _second_run_calls(
        rig, audio, monkeypatch, engine_mod, "WHISPER_DECODE_PARAMS",
        {**engine_mod.WHISPER_DECODE_PARAMS, "no_speech_threshold": 0.9})
    assert (vad, lid) == (0, 0) and engines == ["whisper-large-v3"]


def test_cache_engine_repo_change_misses(tmp_path, audio, monkeypatch):
    from transcribe.auto import engines as engine_mod

    rig = Rig(tmp_path)
    vad, lid, engines = _second_run_calls(rig, audio, monkeypatch, engine_mod, "QWEN3_ASR_REPO", "x/other-qwen")
    assert engines == ["qwen3-asr-1.7b"]


def test_engine_cache_config_contents():
    from transcribe.auto.engines import engine_cache_config

    cfg = engine_cache_config("whisper-large-v3")
    assert cfg["id"] == "whisper-large-v3" and cfg["repo"] and cfg["decode"]["no_speech_threshold"] == 0.6
    assert engine_cache_config("qwen3-asr-1.7b")["repo"]
    assert engine_cache_config("parakeet-v3")["repo"] and engine_cache_config("eg2")["model"]
    assert engine_cache_config("unknown") == {"id": "unknown"}


def test_chunks_with_no_words_write_all_outputs_through_cli(tmp_path, audio, monkeypatch, caplog):
    empty_a = FakeEngine("qwen3-asr-1.7b", [], [])
    empty_b = FakeEngine("whisper-large-v3", [], [])
    rig = Rig(tmp_path, a=empty_a, b=empty_b)
    _patch_cli(monkeypatch, rig)
    out = tmp_path / "nw"
    code = cli.main([str(audio), "--backend", "auto", "--output-dir", str(out), "--output-basename", "w"])
    assert code == 0
    assert (out / "w.txt").exists() and (out / "w.srt").read_text() == "" and (out / "w.json").exists()
    data = json.loads((out / "w.json").read_text())
    assert data["segments"] == [] and data["metadata"]["no_speech"] is True
    auto = data["metadata"]["auto"]
    assert auto["no_speech_reason"] == "no words recognized" and "no words recognized" in auto["notes"]
    assert auto["trust"]["level"] == "n/a"
    assert not any("low trust" in r.getMessage() for r in caplog.records)


def test_no_speech_reasons(tmp_path, audio):
    _, meta = run(audio, Rig(tmp_path / "a", spans=()))
    assert meta["no_speech_reason"] == "no speech detected"
    _, meta = run(audio, Rig(tmp_path / "b", a=FakeEngine("qwen3-asr-1.7b", [], []),
                             b=FakeEngine("whisper-large-v3", [], [])))
    assert meta["no_speech_reason"] == "no words recognized" and meta["no_speech"] is True


def test_resolve_fallback_is_a_core_stage(tmp_path, audio, monkeypatch):
    from transcribe.auto import pipeline

    profiles = load_profiles()
    profiles["profiles"]["en"]["verify"] = False

    def boom(*a, **k):
        raise ValueError("resolve bug")

    monkeypatch.setattr(pipeline, "resolve", boom)
    with pytest.raises(TranscriptionError, match="stage 'resolve' failed: ValueError: resolve bug"):
        run(audio, Rig(tmp_path), profiles=profiles)


def test_resolve_fallback_stage_recorded_when_verify_off(tmp_path, audio):
    profiles = load_profiles()
    profiles["profiles"]["en"]["verify"] = False
    _, meta = run(audio, Rig(tmp_path), profiles=profiles)
    names = [s["name"] for s in meta["stages"]]
    assert stage(meta, "verify")["status"] == "off" and "resolve" in names
    assert names.index("resolve") > names.index("verify")


def test_degraded_stage_warns_once(tmp_path, audio, monkeypatch, caplog):
    rig = Rig(tmp_path)
    fac = rig.factories()
    fac["diarizer"] = lambda path, n: ([], "skipped: no HF_TOKEN")
    monkeypatch.setattr(cli.AudioNormalizer, "normalize", lambda self, path, backend, temp: NormalizedMedia(path, 6.0))
    monkeypatch.setattr(cli, "_probe_sample_rate", lambda path: SR)
    real = cli.run_auto
    monkeypatch.setattr(cli, "run_auto", lambda *a, **k: real(*a, factories=fac, **k))
    cli.main([str(audio), "--backend", "auto", "--output-dir", str(tmp_path / "w"), "--output-formats", "json"])
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "diarize" in r.getMessage()]
    assert len(warnings) == 1


def test_profile_validation(tmp_path):
    import copy

    from transcribe.auto.profiles import validate_profiles

    good = load_profiles()
    validate_profiles(good)

    def broken(mutate):
        bad = copy.deepcopy(good)
        mutate(bad)
        return bad

    cases = [
        broken(lambda p: p.pop("default")),
        broken(lambda p: p["profiles"]["en"].update(engines=[])),
        broken(lambda p: p["profiles"]["en"].update(engines="qwen3-asr-1.7b")),
        broken(lambda p: p["profiles"]["en"].update(engines=["nope"])),
        broken(lambda p: p["profiles"]["en"].update(pick="yes")),
        broken(lambda p: p["default"].update(margin="0.05")),
        broken(lambda p: p.update(trust={"weights": {}})),
        {"version": 1},
    ]
    for bad in cases:
        with pytest.raises(TranscriptionError, match="profiles"):
            validate_profiles(bad)
    # select validates the entry it returns, also for tables that were not loaded from disk
    with pytest.raises(TranscriptionError, match="profiles"):
        select(broken(lambda p: p["profiles"]["en"].update(engines=[])), "en")
    with pytest.raises(TranscriptionError, match="profiles"):
        select(broken(lambda p: p["default"].update(engines=["nope"])), "fr")
    path = tmp_path / "p.json"
    path.write_text(json.dumps(broken(lambda p: p.pop("default"))))
    with pytest.raises(TranscriptionError, match="profiles"):
        load_profiles(path)
    path.write_text("{oops")
    with pytest.raises(TranscriptionError, match="profiles"):
        load_profiles(path)


def test_no_speech_reports_profile_key_adjusted_by_language(tmp_path, audio):
    _, meta = run(audio, Rig(tmp_path, spans=()), language="es")
    assert meta["profile_key"].partition("-")[0] == "es"
    assert meta["language"] == "es"
    _, meta = run(audio, Rig(tmp_path / "fr", spans=()), language="fr")
    assert meta["profile_key"].partition("-")[0] == "other"


def test_cli_merges_hotwords_into_terms_and_logs_ignored_options(tmp_path, audio, monkeypatch, caplog):
    rig = Rig(tmp_path)
    captured = {}
    monkeypatch.setattr(cli.AudioNormalizer, "normalize", lambda self, path, backend, temp: NormalizedMedia(path, 6.0))
    monkeypatch.setattr(cli, "_probe_sample_rate", lambda path: SR)
    real = cli.run_auto

    def spy(*a, **k):
        captured.update(k)
        return real(*a, factories=rig.factories(), **k)

    monkeypatch.setattr(cli, "run_auto", spy)
    code = cli.main([str(audio), "--backend", "auto", "--terms", "Alpha, Beta", "--hotwords", "Beta,Gamma",
                     "--model", "foo", "--chunk-mode", "force", "--timeout-seconds", "5",
                     "--context-info", "ctx", "--output-dir", str(tmp_path / "h"), "--output-formats", "json"])
    assert code == 0
    assert captured["terms"] == ["Alpha", "Beta", "Gamma"]
    infos = [r.getMessage() for r in caplog.records if "ignores these options" in r.getMessage()]
    assert len(infos) == 1
    for flag in ("--model", "--chunk-mode", "--timeout-seconds", "--context-info"):
        assert flag in infos[0]
    assert "--chunk-seconds" not in infos[0]

    caplog.records.clear()
    cli.main([str(audio), "--backend", "auto", "--output-dir", str(tmp_path / "h2"), "--output-formats", "json"])
    assert not any("ignores these options" in r.getMessage() for r in caplog.records)


def test_live_profiles_file_is_valid():
    from pathlib import Path

    import transcribe.auto.profiles as profiles_mod

    live = Path(profiles_mod.__file__).resolve().parent.parent / "profiles.json"
    table = load_profiles(live)
    assert "default" in table and len(table["default"]["engines"]) == 2  # the suite's ablation pair


# ----------------------------------------------------------------------------- final-review fixes


def _live_profiles_path():
    from pathlib import Path

    import transcribe.auto.profiles as profiles_mod

    return Path(profiles_mod.__file__).resolve().parent.parent / "profiles.json"


def test_live_profiles_ship_the_calibrated_behaviour():
    table = load_profiles(_live_profiles_path())
    en, es, other = (table["profiles"][k] for k in ("en", "es", "other"))
    assert en["engines"] == ["qwen3-asr-1.7b"] and es["engines"] == ["whisper-large-v3"]
    assert other["engines"] == ["whisper-large-v3"]
    for entry in (en, es, other, *(v for k, v in table["profiles"].items() if k not in ("en", "es", "other"))):
        assert (entry["pick"], entry["verify"], entry["refine"]) == (False, False, False)
    assert other["evidence"].startswith("manual: unvalidated")
    for key, entry in table["profiles"].items():
        if key != "other":  # recorded from the results doc: below the 0.7 bar, so trust stays uncalibrated
            assert entry["trust_spearman"] < 0.7, key
    assert "trust_spearman" not in other


def test_other_language_runs_whisper_alone_with_no_stages(tmp_path, audio):
    rig = Rig(tmp_path, lang=("fr", 0.9))
    _, meta = run(audio, rig, profiles=load_profiles(_live_profiles_path()))
    assert meta["engines"] == ["whisper-large-v3"]
    assert stage(meta, "pick")["status"] == "off" and stage(meta, "verify")["status"] == "off"
    assert stage(meta, "refine")["status"] == "off"
    assert meta["trust"]["level"] == "uncalibrated"


def test_single_qwen3_profile_reports_uncalibrated_trust_and_approximate_timestamps(tmp_path, audio):
    rig = Rig(tmp_path)
    result, meta = run(audio, rig, profiles=load_profiles(_live_profiles_path()))
    assert meta["engines"] == ["qwen3-asr-1.7b"]
    assert meta["trust"]["level"] == "uncalibrated" and meta["trust"]["estimated_wer"] is None
    assert meta["trust"]["reason"]
    assert meta["timestamps"] == "approximate"
    assert result.segments  # still a normal transcript


def test_whisper_first_profile_has_engine_timestamps_and_spanish_trust_is_uncalibrated(tmp_path, audio):
    rig = Rig(tmp_path, lang=("es", 0.95))
    _, meta = run(audio, rig, profiles=load_profiles(_live_profiles_path()))
    assert meta["engines"] == ["whisper-large-v3"] and meta["timestamps"] == "engine"
    assert meta["trust"]["level"] == "uncalibrated"  # Spearman 0.63 < 0.7


def test_timestamps_follow_the_engine_that_supplies_the_words(tmp_path, audio):
    _, meta = run(audio, Rig(tmp_path))  # frozen table: qwen3 (no word times) is engine A
    assert meta["timestamps"] == "approximate"
    from transcribe.auto.engines import ENGINES_WITH_APPROX_TIMES, ENGINES_WITHOUT_WORD_PROBS

    assert ENGINES_WITH_APPROX_TIMES == {"qwen3-asr-1.7b"}  # parakeet (token times), whisper and eg2 are native
    assert ENGINES_WITHOUT_WORD_PROBS == {"qwen3-asr-1.7b", "parakeet-v3", "eg2"}
    _, meta = run(audio, Rig(tmp_path / "none", spans=()))
    assert meta["timestamps"] == "n/a"


def test_skipped_stage_lowers_trust_in_the_pipeline(tmp_path, audio):
    rig = Rig(tmp_path)
    fac = rig.factories()
    fac["diarizer"] = lambda path, n: ([], "skipped: no HF_TOKEN")
    _, ok_meta = run(audio, Rig(tmp_path / "ok"))
    _, meta = run_auto(audio, {}, factories=fac)
    assert meta["trust"]["features"]["skipped_stages"] == 1
    assert meta["trust"]["warnings"] and "diarize" in meta["trust"]["warnings"][0]
    assert meta["trust"]["estimated_wer"] == pytest.approx(ok_meta["trust"]["estimated_wer"] + 0.05)


def _live_cli(monkeypatch, rig):
    import transcribe.auto.profiles as profiles_mod

    monkeypatch.setattr(profiles_mod, "PROFILES_PATH", _live_profiles_path())
    _patch_cli(monkeypatch, rig)


def test_cli_uncalibrated_trust_is_info_not_warning(tmp_path, audio, monkeypatch, caplog):
    _live_cli(monkeypatch, Rig(tmp_path))
    code = cli.main([str(audio), "--backend", "auto", "--output-dir", str(tmp_path / "o"), "--output-formats", "json"])
    assert code == 0
    data = json.loads(next((tmp_path / "o").glob("*.json")).read_text())
    assert data["metadata"]["auto"]["trust"]["level"] == "uncalibrated"
    notes = [r for r in caplog.records if "trust score not calibrated for this profile" in r.getMessage()]
    assert len(notes) == 1 and notes[0].levelno == logging.INFO
    assert not any("low trust" in r.getMessage() for r in caplog.records)


def test_cli_explicit_refine_on_a_profile_with_refine_off_is_ignored_with_an_info(tmp_path, audio, monkeypatch, caplog):
    _live_cli(monkeypatch, Rig(tmp_path))
    code = cli.main([str(audio), "--backend", "auto", "--refine", "--output-dir", str(tmp_path / "o"),
                     "--output-formats", "json"])
    assert code == 0
    messages = [(r.levelno, r.getMessage()) for r in caplog.records]
    assert any(lvl == logging.INFO and "refine is off for profile" in m and "--refine ignored" in m
               for lvl, m in messages)
    assert any("refine: off (profile" in m for _, m in messages)
    data = json.loads(next((tmp_path / "o").glob("*.json")).read_text())
    refine = next(s for s in data["metadata"]["auto"]["stages"] if s["name"] == "refine")
    assert refine["status"] == "off"


def test_cli_default_refine_on_a_profile_with_refine_off_logs_off_profile_without_ignored_note(
        tmp_path, audio, monkeypatch, caplog):
    _live_cli(monkeypatch, Rig(tmp_path))
    cli.main([str(audio), "--backend", "auto", "--output-dir", str(tmp_path / "o"), "--output-formats", "json"])
    messages = [r.getMessage() for r in caplog.records]
    assert any("refine: off (profile" in m for m in messages)
    assert not any("--refine ignored" in m for m in messages)


def test_refine_env_tristate(monkeypatch):
    monkeypatch.delenv("TRANSCRIBE_REFINE", raising=False)
    assert parse_args(["x.wav"]).refine is None
    for value, expected in (("1", True), ("true", True), ("0", False), ("false", False), ("off", False)):
        monkeypatch.setenv("TRANSCRIBE_REFINE", value)
        assert parse_args(["x.wav"]).refine is expected, value
    monkeypatch.setenv("TRANSCRIBE_REFINE", "0")
    assert parse_args(["x.wav", "--refine"]).refine is True  # the flag wins over the env


def test_env_refine_off_turns_auto_refine_off(tmp_path, audio, monkeypatch):
    rig = Rig(tmp_path)
    _patch_cli(monkeypatch, rig)
    monkeypatch.setenv("TRANSCRIBE_REFINE", "0")
    captured = {}
    real = cli.run_auto
    monkeypatch.setattr(cli, "run_auto", lambda *a, **k: (captured.update(k), real(*a, **k))[1])
    assert cli.main([str(audio), "--backend", "auto", "--output-dir", str(tmp_path / "o"),
                     "--output-formats", "json"]) == 0
    assert captured["refine"] is False
    monkeypatch.delenv("TRANSCRIBE_REFINE")
    cli.main([str(audio), "--backend", "auto", "--output-dir", str(tmp_path / "o2"), "--output-formats", "json"])
    assert captured["refine"] is True  # unset: auto defers to the profile


def test_non_auto_backends_do_not_import_the_auto_pipeline():
    import subprocess
    import sys

    code = ("import sys, transcribe.cli, transcribe, transcribe_youtube; "
            "assert 'transcribe.auto' not in sys.modules, 'transcribe.auto imported'; "
            "assert 'jellyfish' not in sys.modules and 'pronouncing' not in sys.modules; "
            "from transcribe.cli import parse_args; parse_args(['x.wav', '--backend', 'mlx-whisper']); "
            "assert 'transcribe.auto' not in sys.modules; "
            "assert callable(transcribe.run_auto) and 'transcribe.auto.pipeline' in sys.modules")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          cwd=str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    assert done.returncode == 0, done.stderr

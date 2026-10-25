"""Pure-function tests for the eval-suite dataset builders (no network, no real datasets)."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from benchmarks.suite import datasets as ds
from benchmarks.suite.datasets import (assemble_timeline, assign_split, build_longform, make_babble, mix_at_snr,
                                       phone_band, plan_targets)

SR = 16000


def _utt(text, seconds=1.0, speaker="s1"):
    return {"audio": np.ones(int(seconds * SR), dtype=np.float32) * 0.1, "text": text, "speaker": speaker}


def test_build_longform_concatenates_with_gaps():
    audio, reference, speakers = build_longform([_utt("a", speaker="x"), _utt("b", speaker="y")], target_seconds=2.6)
    assert abs(len(audio) - int(2.6 * SR)) <= 1
    assert reference == "a b"
    assert [round(s["start"], 3) for s in speakers] == [0.0, 1.6]
    assert [round(s["end"], 3) for s in speakers] == [1.0, 2.6]
    assert [s["speaker"] for s in speakers] == ["x", "y"]


def test_build_longform_stops_at_target():
    utts = [_utt(f"w{i}") for i in range(100)]
    audio, reference, speakers = build_longform(utts, target_seconds=30.0)
    assert 28.0 <= len(audio) / SR <= 32.0
    assert len(reference.split()) == len(speakers)


def test_mix_at_snr_hits_target():
    rng = np.random.default_rng(0)
    speech = (rng.standard_normal(SR * 5) * 0.1).astype(np.float32)
    noise = (rng.standard_normal(SR * 2) * 0.3).astype(np.float32)  # shorter: must be tiled
    mixed = mix_at_snr(speech, noise, 5.0)
    assert len(mixed) == len(speech)
    residual = mixed - speech
    snr = 10 * np.log10(np.mean(speech.astype(np.float64) ** 2) / np.mean(residual.astype(np.float64) ** 2))
    assert snr == pytest.approx(5.0, abs=0.1)


def test_assign_split_is_deterministic_and_balanced():
    assert assign_split("abc") == assign_split("abc")
    splits = [assign_split(f"item-{i}") for i in range(1000)]
    assert set(splits) == {"dev", "test"}
    assert 450 <= splits.count("dev") <= 550


def test_phone_band_removes_energy_above_4khz():
    t = np.arange(SR * 2) / SR
    tone = (0.5 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)
    out = phone_band(tone)
    assert len(out) == len(tone)
    drop_db = 10 * np.log10(np.mean(tone**2) / max(np.mean(out[800:-800] ** 2), 1e-20))
    assert drop_db >= 30


def test_assemble_timeline_keeps_overlaps_and_caps_long_gaps():
    one = np.ones(SR, dtype=np.float32) * 0.2
    segs = [
        {"start": 100.0, "end": 101.0, "audio": one, "text": "first", "speaker": "a"},
        {"start": 100.5, "end": 101.5, "audio": one * 2, "text": "second", "speaker": "b"},  # overlaps the first
        {"start": 160.0, "end": 161.0, "audio": one, "text": "third", "speaker": "a"},  # 58 s of silence before it
    ]
    audio, reference, turns = assemble_timeline(segs, max_gap=1.0)
    assert reference == "first second third"
    assert turns[0]["start"] == 0.0 and turns[1]["start"] == 0.5
    assert turns[2]["start"] == pytest.approx(2.5, abs=0.01)  # 1.5 s of speech + the 1 s capped gap
    assert len(audio) == int(round(3.5 * SR))
    assert audio[int(0.75 * SR)] == pytest.approx(0.4)  # the later segment overwrote the overlap


def test_make_babble_has_requested_length_and_is_deterministic():
    rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(1)
    sources = [np.random.default_rng(i).standard_normal(SR) .astype(np.float32) for i in range(8)]
    a, b = make_babble(sources, SR * 3, rng_a), make_babble(sources, SR * 3, rng_b)
    assert len(a) == SR * 3 and np.array_equal(a, b)


def test_plan_targets_starts_short_and_respects_the_cap():
    targets = plan_targets(30, np.random.default_rng(0))
    assert targets[0] == 120.0
    assert 28 * 60 <= sum(targets) <= 30 * 60 + 1
    assert all(t >= 100 for t in targets)


# ----------------------------------------------------------------------------- text helpers, splits, smoke, manifests


def test_has_markup_and_clean_text():
    for tagged in ("we met <unk> there", "<inaudible> Lee", "ok <laugh>", "IGNORE_TIME_SEGMENT_IN_SCORING"):
        assert ds.has_markup(tagged)
    assert not ds.has_markup("a < b and c > d? no tags here")
    assert ds.clean_text("  two   words\n here ") == "two words here"
    assert ds.clean_text("keep <unk> visible") == "keep <unk> visible"  # never edited silently
    assert ds.clean_text(None) == ""


def test_spread_order_is_a_permutation_for_any_shard_count():
    for count in (1, 2, 4, 6, 9, 10, 12):
        assert sorted(ds.spread_order(count)) == list(range(count))


def _fake_item(item_id, minutes, **meta):
    return {"id": item_id, "slice": item_id.split("-")[0], "split": ds.assign_split(item_id), "duration": minutes * 60.0,
            "meta": meta, "reference": "x", "speakers": None}


def _side_stats(items):
    mins = {k: sum(i["duration"] for i in items if i["split"] == k) for k in ("dev", "test")}
    counts = {k: sum(1 for i in items if i["split"] == k) for k in ("dev", "test")}
    return counts, {k: v / sum(mins.values()) for k, v in mins.items()}


def test_balance_splits_gives_each_side_two_items_and_35_percent_in_every_stratum():
    durations = [2.0, 7.0, 6.5, 4.0, 5.0, 6.0, 3.5, 7.0, 3.0, 2.5]
    items = [_fake_item(f"s-{n:02d}", m, source="clean" if n < 7 else "other") for n, m in enumerate(durations)]
    ds.balance_splits(items, lambda i: i["meta"]["source"])
    counts, fractions = _side_stats(items)
    assert min(counts.values()) >= 2 and min(fractions.values()) >= 0.35
    for source in ("clean", "other"):
        assert {i["split"] for i in items if i["meta"]["source"] == source} == {"dev", "test"}
    again = [dict(i, split="x") for i in items]
    ds.balance_splits(again, lambda i: i["meta"]["source"])
    assert [i["split"] for i in again] == [i["split"] for i in items]  # deterministic


def test_balance_splits_handles_the_smallest_slices():
    items = [_fake_item(f"t-{n}", m) for n, m in enumerate([3.1, 5.6, 7.7, 5.7])]
    ds.balance_splits(items)
    counts, fractions = _side_stats(items)
    assert counts == {"dev": 2, "test": 2} and min(fractions.values()) >= 0.35


def test_derived_sources_are_taken_alternately_from_both_sides():
    items = [_fake_item(f"c-{n:02d}", 3.0 + n) for n in range(8)]
    for n, item in enumerate(items):
        item["split"] = "dev" if n % 2 else "test"
    chosen = ds._cap(items, 12.0)
    counts, fractions = _side_stats(chosen)
    assert min(counts.values()) >= 2 and min(fractions.values()) >= 0.35
    assert sum(i["duration"] for i in chosen) / 60 >= 12.0
    assert len(chosen) < len(items)  # the cap is honoured, not everything is taken


def test_choose_smoke_takes_the_shortest_item_per_slice_and_two_clips_per_nospeech_kind():
    by_slice = {"a": [_fake_item("a-1", 5), _fake_item("a-2", 2), _fake_item("a-3", 3)], "b": [],
                "nospeech": [dict(_fake_item(f"nospeech-{k}{n}", 0.3), meta={"kind": k})
                             for k in ("silence", "white_noise", "ambient") for n in range(4)]}
    smoke = ds.choose_smoke(by_slice, 120)
    assert [i["id"] for i in smoke if i["slice"] == "a"] == ["a-2"]
    kinds = [i["meta"]["kind"] for i in smoke if i["slice"] == "nospeech"]
    assert sorted(kinds) == ["ambient"] * 2 + ["silence"] * 2 + ["white_noise"] * 2


def _write_item(directory, item_id, seconds=12.0, sr=SR, **overrides):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{item_id}.wav"
    sf.write(path, np.full(int(seconds * sr), 0.01, dtype=np.float32), sr, subtype="PCM_16")
    item = {"id": item_id, "slice": item_id.split("-")[0], "split": "dev", "lang": "en", "audio": str(path),
            "duration": seconds, "reference": "some words", "speakers": None, "condition": "clean", "meta": {}}
    item.update(overrides)
    return item


def test_read_and_write_manifest_roundtrip_and_missing_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "MANIFEST_DIR", tmp_path / "manifests")
    items = [_write_item(tmp_path / "audio", "x-02"), _write_item(tmp_path / "audio", "x-01")]
    ds.write_manifest("x", items)
    assert [i["id"] for i in ds.read_manifest("x")] == ["x-01", "x-02"]  # sorted by id
    (tmp_path / "audio" / "x-01.wav").unlink()
    assert ds.read_manifest("x") == []  # a manifest with a missing wav is not trusted
    assert ds.read_manifest("never-written") == []


def test_validate_manifests_flags_each_kind_of_problem(tmp_path):
    audio, manifests = tmp_path / "audio", tmp_path / "manifests"
    good = [_write_item(audio, f"ok-{n}", split="dev" if n % 2 else "test") for n in range(4)]
    ds.MANIFEST_DIR, saved = manifests, ds.MANIFEST_DIR
    try:
        ds.write_manifest("ok", good)
    finally:
        ds.MANIFEST_DIR = saved
    assert ds.validate_manifests(manifests) == []
    bad = [
        _write_item(audio, "bad-rate", sr=8000),
        _write_item(audio, "bad-duration", duration=99.0),
        _write_item(audio, "bad-long", seconds=601.0, duration=601.0),
        _write_item(audio, "bad-ref", reference="  "),
        _write_item(audio, "bad-turn", speakers=[{"start": 1.0, "end": 50.0, "speaker": "a"}]),
        {**_write_item(audio, "bad-missing"), "audio": str(audio / "gone.wav")},
    ]
    ds.MANIFEST_DIR = manifests
    try:
        ds.write_manifest("bad", bad)
    finally:
        ds.MANIFEST_DIR = saved
    text = "\n".join(ds.validate_manifests(manifests))
    for expected in ("bad-rate: audio is 8000 Hz", "bad-duration: manifest duration", "bad-long: duration 601",
                     "bad-ref: empty reference", "bad-turn: speaker turn", "bad-missing: audio file missing"):
        assert expected in text, expected
    assert "unbalanced split" in text  # all six bad items are dev
    nospeech = [_write_item(audio, "nospeech-0", reference="", slice="nospeech", condition="nospeech")]
    ds.MANIFEST_DIR = manifests
    try:
        ds.write_manifest("nospeech", nospeech)
    finally:
        ds.MANIFEST_DIR = saved
    assert "nospeech-0: empty reference" not in "\n".join(ds.validate_manifests(manifests))


def _stub_env(tmp_path, monkeypatch, failing=("c",)):
    specs = {"smoke_item_seconds": 120, "slices": {n: {"lang": "en", "condition": "clean", "minutes": 1} for n in "abc"},
             "gaps": {}}
    (tmp_path / "slices.json").write_text(json.dumps(specs))
    monkeypatch.setattr(ds, "SLICES_JSON", tmp_path / "slices.json")
    monkeypatch.setattr(ds, "MANIFEST_DIR", tmp_path / "manifests")
    monkeypatch.setattr(ds, "AUDIO_DIR", tmp_path / "audio")
    monkeypatch.setattr(ds, "SLICE_ORDER", ["a", "b", "c"])
    monkeypatch.setattr(ds, "KNOWN_GAPS", {"known": "documented gap"})

    def make(name):
        def build(ctx):
            if name in failing:
                raise RuntimeError(f"{name} source is unavailable")
            return [ds.emit(ctx, name, f"{k:02d}", np.full(SR * (10 + 2 * k), 0.01, dtype=np.float32),
                            f"{name} words {k}", None) for k in range(4)]
        return build

    monkeypatch.setattr(ds, "BUILDERS", {n: make(n) for n in "abc"})
    return tmp_path


def test_prepare_skips_a_failing_slice_records_the_gap_and_finishes_the_rest(tmp_path, monkeypatch, capsys):
    _stub_env(tmp_path, monkeypatch)
    by_slice = ds.prepare()
    out = capsys.readouterr().out
    assert len(by_slice["a"]) == 4 and len(by_slice["b"]) == 4 and by_slice["c"] == []
    specs = json.loads((tmp_path / "slices.json").read_text())
    assert "c source is unavailable" in specs["gaps"]["c"] and specs["gaps"]["known"] == "documented gap"
    assert (tmp_path / "manifests" / "a.jsonl").exists() and not (tmp_path / "manifests" / "c.jsonl").exists()
    smoke = [json.loads(line)["slice"] for line in (tmp_path / "manifests" / "smoke.jsonl").read_text().splitlines()]
    assert sorted(smoke) == ["a", "b"]
    assert "MISSING" in out and "manifests valid" in out
    counts, fractions = _side_stats(by_slice["a"])  # prepare balanced the split
    assert counts == {"dev": 2, "test": 2}


def test_partial_prepare_keeps_smoke_and_summary_complete(tmp_path, monkeypatch, capsys):
    _stub_env(tmp_path, monkeypatch)
    ds.prepare()
    ds.prepare(only=["a"], force=True)
    smoke = [json.loads(line)["slice"] for line in (tmp_path / "manifests" / "smoke.jsonl").read_text().splitlines()]
    assert sorted(smoke) == ["a", "b"]  # b was untouched but is still in the smoke set
    summary = json.loads((tmp_path / "manifests" / "summary.json").read_text())
    assert summary["items"]["a"] == 4 and summary["items"]["b"] == 4
    ds.prepare(smoke_only=True, only=["b"])  # smoke-only must not shrink anything either
    assert (tmp_path / "manifests" / "a.jsonl").exists()
    assert sorted(json.loads(l)["slice"] for l in (tmp_path / "manifests" / "smoke.jsonl").read_text().splitlines()) == ["a", "b"]
    assert json.loads((tmp_path / "manifests" / "summary.json").read_text())["items"]["a"] == 4
    assert "c" in json.loads((tmp_path / "slices.json").read_text())["gaps"]  # the earlier gap survives


def test_write_manifest_stores_audio_paths_relative_to_the_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "MANIFEST_DIR", tmp_path / "manifests")
    monkeypatch.setattr(ds, "DATA_ROOT", tmp_path / "root")
    items = [_write_item(tmp_path / "root" / "audio" / "x", "x-01"),
             _write_item(tmp_path / "elsewhere", "x-02")]  # outside the root: kept as given
    ds.write_manifest("x", items)
    lines = [json.loads(line) for line in (tmp_path / "manifests" / "x.jsonl").read_text().splitlines()]
    assert lines[0]["audio"] == "audio/x/x-01.wav"
    assert lines[1]["audio"] == str(tmp_path / "elsewhere" / "x-02.wav")
    assert str(tmp_path / "root") not in (tmp_path / "manifests" / "x.jsonl").read_text().splitlines()[0]
    read = ds.read_manifest("x")  # readers resolve relative paths against the data root
    assert read[0]["audio"] == str(tmp_path / "root" / "audio" / "x" / "x-01.wav")
    assert read[1]["audio"] == str(tmp_path / "elsewhere" / "x-02.wav")


def test_read_manifest_accepts_absolute_and_relative_audio_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "MANIFEST_DIR", tmp_path / "manifests")
    monkeypatch.setattr(ds, "DATA_ROOT", tmp_path / "root")
    rel = _write_item(tmp_path / "root" / "audio", "x-01")
    absolute = _write_item(tmp_path / "root" / "audio", "x-02")
    (tmp_path / "manifests").mkdir()
    rel_line = {**rel, "audio": "audio/x-01.wav"}
    (tmp_path / "manifests" / "x.jsonl").write_text(json.dumps(rel_line) + "\n" + json.dumps(absolute) + "\n")
    items = ds.read_manifest("x")
    assert [Path(i["audio"]).is_absolute() and Path(i["audio"]).exists() for i in items] == [True, True]
    assert not any("audio file missing" in p for p in ds.validate_manifests(tmp_path / "manifests"))
    assert ds.resolve_audio("audio/x-01.wav") == tmp_path / "root" / "audio" / "x-01.wav"
    assert ds.resolve_audio(str(absolute["audio"])) == Path(absolute["audio"])

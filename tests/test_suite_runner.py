"""Tests for benchmarks/suite/run_suite.py (fake systems only; no models, no audio reads)."""

import json

import pytest

from benchmarks.suite import run_suite as rs

REF_EN = "the quick brown fox jumps over the lazy dog near Paris in 2021"
REF_ES = "el rapido zorro marron salta sobre el perro perezoso cerca de Madrid"


def _item(item_id, slice_, lang, ref, split="dev", duration=30.0, speakers=None):
    return {"id": item_id, "slice": slice_, "split": split, "lang": lang, "audio": f"/nonexistent/{item_id}.wav",
            "duration": duration, "reference": ref, "speakers": speakers, "condition": "clean", "meta": {}}


class FakeSystem:
    """Returns the reference with its second word dropped (one deletion per item)."""

    languages = None

    def __init__(self, name="fake"):
        self.name = name
        self.calls = []
        self.warmups = 0

    def warmup(self):
        self.warmups += 1

    def run(self, item):
        self.calls.append(item["id"])
        words = item["reference"].split()
        del words[1]
        return {"text": " ".join(words), "words": None, "segments": [], "meta": {}}


@pytest.fixture
def manifest():
    return [_item("en_x-00", "en_x", "en", REF_EN), _item("es_x-00", "es_x", "es", REF_ES, split="test")]


def test_run_suite_writes_per_slice_metrics(tmp_path, manifest):
    out = tmp_path / "res" / "out.json"
    result = rs.run_suite([FakeSystem()], manifest, out, tmp_path / "cache")
    on_disk = json.loads(out.read_text())
    for key in ("config", "created_at", "machine", "per_item", "per_slice"):
        assert key in on_disk
    assert on_disk["config"]["systems"] == ["fake"]
    assert set(on_disk["per_slice"]) == {"en_x", "es_x"}
    en = on_disk["per_slice"]["en_x"]["fake"]
    assert en["wer"] > 0 and en["cer"] > 0
    assert en["wer"] == pytest.approx(1 / 13)  # 13 words, one deleted
    assert en["xrt"] > 0
    assert set(en["item_wer"]) == {"en_x-00"}
    assert en["splits"]["dev"]["wer"] == pytest.approx(1 / 13)
    assert on_disk["per_slice"]["es_x"]["fake"]["splits"]["test"]["n"] == 1
    assert result["per_slice"] == on_disk["per_slice"]
    rec = {r["id"]: r for r in on_disk["per_item"]}["en_x-00"]
    for key in ("system", "slice", "split", "lang", "seconds", "xrt", "peak_rss_mb", "wer", "text", "cached"):
        assert key in rec
    assert rec["xrt"] == pytest.approx(rec["duration"] / rec["seconds"])
    assert "entity_err" in en and en["entity_err"] is not None
    assert "ins_per_min" not in en and "der" not in en and "unclear" not in en  # absent for plain backends


def test_run_suite_uses_cache(tmp_path, manifest):
    out, cache = tmp_path / "out.json", tmp_path / "cache"
    first = FakeSystem()
    rs.run_suite([first], manifest, out, cache)
    assert len(first.calls) == 2 and first.warmups == 1
    assert (cache / "fake" / "en_x-00.json").exists()
    second = FakeSystem()
    result = rs.run_suite([second], manifest, out, cache)
    assert second.calls == [] and second.warmups == 0
    assert all(r["cached"] for r in result["per_item"])
    assert result["per_slice"]["en_x"]["fake"]["wer"] == pytest.approx(1 / 13)
    assert result["per_slice"]["en_x"]["fake"]["xrt"] > 0  # timing survives the cache


def test_language_filter_skips_eg2_for_es(tmp_path, manifest):
    eg2 = FakeSystem("eg2")
    eg2.languages = {"en", "none"}
    result = rs.run_suite([eg2], manifest, tmp_path / "out.json", tmp_path / "cache")
    assert eg2.calls == ["en_x-00"]
    skipped = [r for r in result["per_item"] if r.get("skipped")]
    assert [(r["id"], r["skipped"]) for r in skipped] == [("es_x-00", "language")]
    assert result["per_slice"]["es_x"]["eg2"]["n_skipped"] == 1
    assert result["per_slice"]["es_x"]["eg2"]["wer"] is None


def test_builtin_eg2_is_english_only():
    eg2 = rs.build_system("eg2")
    assert eg2.name == "eg2" and "es" not in eg2.languages and "en" in eg2.languages
    assert rs.build_system("mlx-whisper-turbo").languages is None
    with pytest.raises(ValueError):
        rs.build_system("nope")


def test_errors_are_recorded_not_fatal(tmp_path, manifest):
    class Boom(FakeSystem):
        def run(self, item):
            if item["lang"] == "es":
                raise RuntimeError("boom")
            return super().run(item)

    result = rs.run_suite([Boom()], manifest, tmp_path / "o.json", tmp_path / "c")
    errs = [r for r in result["per_item"] if r.get("error")]
    assert len(errs) == 1 and "boom" in errs[0]["error"]
    assert result["per_slice"]["es_x"]["fake"]["n_errors"] == 1
    assert not (tmp_path / "c" / "fake" / "es_x-00.json").exists()  # failures are retried next run


def test_nospeech_reports_insertions_per_minute(tmp_path):
    class Halluc(FakeSystem):
        def run(self, item):
            return {"text": "thanks for watching everybody", "words": None, "segments": [], "meta": {}}

    items = [_item("ns-0", "nospeech", "none", "", duration=60.0), _item("ns-1", "nospeech", "none", "", duration=60.0)]
    result = rs.run_suite([Halluc()], items, tmp_path / "o.json", tmp_path / "c")
    ns = result["per_slice"]["nospeech"]["fake"]
    assert ns["ins_per_min"] == pytest.approx(4.0)
    assert ns["wer"] is None and ns["item_wer"] == {}


def test_optional_fields_are_carried_and_aggregated(tmp_path):
    ref = "alpha beta gamma delta"
    speakers = [{"start": 0.0, "end": 5.0, "speaker": "A"}, {"start": 5.0, "end": 10.0, "speaker": "B"}]

    class Rich(FakeSystem):
        def run(self, item):
            words = [{"text": "alpha", "start": 0, "end": 1, "unclear": False, "speaker": "Speaker 1"},
                     {"text": "bet", "start": 1, "end": 2, "unclear": True, "speaker": "Speaker 1"},
                     {"text": "gamma", "start": 5, "end": 6, "unclear": False, "speaker": "Speaker 2"},
                     {"text": "delta", "start": 6, "end": 7, "unclear": False, "speaker": "Speaker 2"}]
            return {"text": "alpha bet gamma delta", "words": words,
                    "segments": [{"start": 0.0, "end": 5.0, "speaker": "Speaker 1", "text": "alpha bet"},
                                 {"start": 5.0, "end": 10.0, "speaker": "Speaker 2", "text": "gamma delta"}],
                    "meta": {"stages": {"refine": {"before": "alpha bet gamma delta", "after": "alpha beta gamma delta"},
                                        "pick": {"before": "alpha beta gamma delta", "after": "alpha bet gamma delta"}}}}

    item = _item("m-0", "en_meeting", "en", ref, duration=10.0, speakers=speakers)
    result = rs.run_suite([Rich("auto:x")], [item], tmp_path / "o.json", tmp_path / "c")
    rec = result["per_item"][0]
    assert rec["edits"]["refine"]["help"] == 1 and rec["edits"]["pick"]["harm"] == 1
    assert rec["unclear"]["precision"] == 1.0 and rec["unclear"]["recall"] == 1.0
    assert rec["der"] == pytest.approx(0.0, abs=0.01)
    m = result["per_slice"]["en_meeting"]["auto:x"]
    assert m["edits"]["refine"] == {"help": 1, "harm": 0, "neutral": 0}
    assert m["edits"]["pick"]["harm"] == 1
    assert m["unclear"]["density"] == pytest.approx(0.25)
    assert m["der"] == pytest.approx(0.0, abs=0.01)
    # carried through the cache too
    again = rs.run_suite([Rich("auto:x")], [item], tmp_path / "o.json", tmp_path / "c")
    assert again["per_slice"]["en_meeting"]["auto:x"]["edits"]["refine"]["help"] == 1


def test_select_items_filters_split_and_slices(manifest):
    assert [i["id"] for i in rs.select_items(manifest, "dev")] == ["en_x-00"]
    assert [i["id"] for i in rs.select_items(manifest, "test")] == ["es_x-00"]
    assert len(rs.select_items(manifest, "all")) == 2
    assert [i["id"] for i in rs.select_items(manifest, "all", slices=["es_x"])] == ["es_x-00"]


def test_format_table_lists_each_slice_and_system(tmp_path, manifest):
    result = rs.run_suite([FakeSystem()], manifest, tmp_path / "o.json", tmp_path / "c")
    table = rs.format_table(result)
    assert "en_x" in table and "es_x" in table and "fake" in table and "WER" in table


# --------------------------------------------------------------------------
# Fix round 1
# --------------------------------------------------------------------------


class Scripted(FakeSystem):
    """Returns a scripted hypothesis per item id."""

    def __init__(self, outputs, name="fake"):
        super().__init__(name)
        self.outputs = outputs

    def run(self, item):
        self.calls.append(item["id"])
        return {"text": self.outputs[item["id"]], "words": None, "segments": [], "meta": {}}


def test_slice_wer_is_corpus_level_not_mean_of_items(tmp_path):
    short, long_ = "a b", " ".join("apple pear plum fig lime kiwi date nut oat rye corn bean pea yam leek kale dill mint".split())
    items = [_item("s", "sl", "en", short), _item("l", "sl", "en", long_)]
    # short: both words wrong (2 errors / 2 words); long: one deletion (1 error / 18 words)
    sysm = Scripted({"s": "x y", "l": " ".join(long_.split()[1:])})
    m = rs.run_suite([sysm], items, tmp_path / "o.json", tmp_path / "c")["per_slice"]["sl"]["fake"]
    assert m["wer"] == pytest.approx(3 / 20)
    assert m["wer"] != pytest.approx((1.0 + 1 / 18) / 2)
    assert m["item_wer"] == {"s": pytest.approx(1.0), "l": pytest.approx(1 / 18)}
    assert (m["sub"], m["del"], m["ins"], m["words"]) == (2, 1, 0, 20)


def test_mixed_speech_and_nospeech_slice(tmp_path):
    items = [_item("sp", "mix", "en", "red green blue pink"), _item("ns", "mix", "none", "", duration=120.0)]
    sysm = Scripted({"sp": "red green blue pink", "ns": "ghost words appear here now"})
    m = rs.run_suite([sysm], items, tmp_path / "o.json", tmp_path / "c")["per_slice"]["mix"]["fake"]
    assert m["wer"] == 0.0 and m["words"] == 4  # the no-speech hypothesis never leaks into WER
    assert m["ins_per_min"] == pytest.approx(5 / 2.0)  # 5 words over the 2 no-speech minutes only
    assert m["item_wer"] == {"sp": 0.0}
    assert m["n"] == 2


def test_per_system_and_splits_with_several_items(tmp_path):
    items = [_item("d1", "a", "en", "a b c d", split="dev"), _item("d2", "a", "en", "a b c d", split="dev"),
             _item("t1", "b", "en", "a b c d", split="test", duration=60.0)]
    sysm = Scripted({"d1": "a b c d", "d2": "a b c", "t1": "x b c d"})
    res = rs.run_suite([sysm], items, tmp_path / "o.json", tmp_path / "c")
    a = res["per_slice"]["a"]["fake"]
    assert a["splits"]["dev"]["n"] == 2 and a["splits"]["dev"]["wer"] == pytest.approx(1 / 8)
    assert "test" not in a["splits"]
    total = res["per_system"]["fake"]
    assert total["n"] == 3 and total["wer"] == pytest.approx(2 / 12)
    assert total["splits"]["dev"]["wer"] == pytest.approx(1 / 8)
    assert total["splits"]["test"]["wer"] == pytest.approx(1 / 4)
    assert total["audio_seconds"] == pytest.approx(120.0)
    assert set(total["item_wer"]) == {"d1", "d2", "t1"}


def test_cache_invalidated_by_fingerprint_change(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"0" * 100)
    item = _item("x", "sl", "en", REF_EN)
    item["audio"] = str(audio)
    cache = tmp_path / "c"

    class Keyed(FakeSystem):
        key = "v1"

        def cache_key(self):
            return self.key

    first = Keyed()
    rs.run_suite([first], [item], tmp_path / "o.json", cache)
    assert first.calls == ["x"]
    again = Keyed()
    rs.run_suite([again], [item], tmp_path / "o.json", cache)
    assert again.calls == []  # same fingerprint
    bumped = Keyed()
    bumped.key = "v2"
    rs.run_suite([bumped], [item], tmp_path / "o.json", cache)
    assert bumped.calls == ["x"]  # config changed
    audio.write_bytes(b"1" * 150)  # audio replaced (size changes)
    changed = Keyed()
    changed.key = "v2"
    rs.run_suite([changed], [item], tmp_path / "o.json", cache)
    assert changed.calls == ["x"]
    item["lang"] = "es"  # language changed
    relang = Keyed()
    relang.key = "v2"
    rs.run_suite([relang], [item], tmp_path / "o.json", cache)
    assert relang.calls == ["x"]
    refreshed = Keyed()
    refreshed.key = "v2"
    rs.run_suite([refreshed], [item], tmp_path / "o.json", cache, refresh=True)
    assert refreshed.calls == ["x"]  # --refresh ignores a valid cache


def test_legacy_cache_entries_without_fingerprint(tmp_path, manifest):
    cache = tmp_path / "c"
    (cache / "fake").mkdir(parents=True)
    legacy = {"text": "the brown fox", "words": None, "segments": [], "meta": {}, "seconds": 3.0,
              "peak_rss_mb": 100.0, "duration": 30.0}
    (cache / "fake" / "en_x-00.json").write_text(json.dumps(legacy))
    builtin_like = FakeSystem()
    res = rs.run_suite([builtin_like], manifest[:1], tmp_path / "o.json", cache)
    assert builtin_like.calls == [] and res["per_item"][0]["cached"] and res["per_item"][0]["text"] == "the brown fox"

    class Keyed(FakeSystem):
        def cache_key(self):
            return "abc"

    keyed = Keyed()
    rs.run_suite([keyed], manifest[:1], tmp_path / "o.json", cache)
    assert keyed.calls == ["en_x-00"]  # legacy entry is not trusted for a keyed system


def test_cache_file_names_are_collision_free():
    names = ["auto:fast", "auto__fast", "auto%3Afast", "auto/fast", "a b"]
    assert len({rs._safe(n) for n in names}) == len(names)
    assert rs._safe("mlx-whisper-turbo") == "mlx-whisper-turbo" and rs._safe("eg2") == "eg2"
    assert rs._safe("auto:fast") == "auto%3Afast"


def test_scoring_exception_is_an_error_not_fatal(tmp_path, manifest, monkeypatch):
    real = rs.score_item

    def flaky(item, system, out):
        if item["id"] == "es_x-00":
            raise ValueError("bad score")
        return real(item, system, out)

    monkeypatch.setattr(rs, "score_item", flaky)
    sysm = FakeSystem()
    res = rs.run_suite([sysm], manifest, tmp_path / "o.json", tmp_path / "c")
    errs = [r for r in res["per_item"] if r.get("error")]
    assert [r["id"] for r in errs] == ["es_x-00"] and "bad score" in errs[0]["error"]
    assert res["per_slice"]["en_x"]["fake"]["n"] == 1
    assert not (tmp_path / "c" / "fake" / "es_x-00.json").exists()  # not cached, so the rerun retries it


def test_warmup_and_close_failures_do_not_crash(tmp_path, manifest):
    class Bad(FakeSystem):
        def warmup(self):
            raise RuntimeError("no model")

        def close(self):
            raise RuntimeError("stuck")

    res = rs.run_suite([Bad()], manifest, tmp_path / "o.json", tmp_path / "c")
    assert all("warmup failed" in r["error"] for r in res["per_item"])
    assert res["per_slice"]["en_x"]["fake"]["n_errors"] == 1


def test_numpy_values_in_output_serialize(tmp_path, manifest):
    import numpy as np

    class Np(FakeSystem):
        def run(self, item):
            return {"text": item["reference"], "segments": [{"start": np.float32(0), "end": np.float64(1.5),
                                                              "speaker": None, "text": "x"}],
                    "words": [{"text": "x", "start": np.float32(0.0), "end": np.float32(1.0),
                               "unclear": np.bool_(False), "speaker": None}],
                    "meta": {"scores": np.arange(3), "n": np.int64(7), "stages": {}}}

    res = rs.run_suite([Np()], manifest[:1], tmp_path / "o.json", tmp_path / "c")
    assert not res["per_item"][0].get("error")
    cached = json.loads((tmp_path / "c" / "fake" / "en_x-00.json").read_text())
    assert cached["meta"]["scores"] == [0, 1, 2] and cached["meta"]["n"] == 7


def test_unclear_is_pooled_not_averaged(tmp_path):
    # item 1: 2 words, nothing flagged, 0 errors; item 2: 4 words, 2 flagged, 1 flagged error, 2 errors
    items = [_item("u1", "sl", "en", "alpha beta"), _item("u2", "sl", "en", "a b c e")]

    class U(FakeSystem):
        def run(self, item):
            if item["id"] == "u1":
                ws = [{"text": "alpha", "unclear": False}, {"text": "beta", "unclear": False}]
            else:
                ws = [{"text": "a", "unclear": False}, {"text": "x", "unclear": True},
                      {"text": "c", "unclear": True}, {"text": "d", "unclear": False}]
            return {"text": " ".join(w["text"] for w in ws), "words": ws, "segments": [], "meta": {}}

    u = rs.run_suite([U()], items, tmp_path / "o.json", tmp_path / "c")["per_slice"]["sl"]["fake"]["unclear"]
    assert u["precision"] == pytest.approx(1 / 2) and u["recall"] == pytest.approx(1 / 2)  # not (1+.5)/2 = 0.75
    assert u["density"] == pytest.approx(2 / 6)
    assert (u["flagged"], u["flagged_errors"], u["errors"], u["words"]) == (2, 1, 2, 6)


def test_unclear_denominators_zero_give_none(tmp_path):
    class Clean(FakeSystem):
        def run(self, item):
            ws = [{"text": w, "unclear": False} for w in item["reference"].split()]
            return {"text": item["reference"], "words": ws, "segments": [], "meta": {}}

    u = rs.run_suite([Clean()], [_item("c", "sl", "en", "a b c")], tmp_path / "o.json",
                     tmp_path / "c")["per_slice"]["sl"]["fake"]["unclear"]
    assert u["precision"] is None and u["recall"] is None and u["density"] == 0.0


def test_der_missing_when_system_returns_no_speakers(tmp_path):
    speakers = [{"start": 0.0, "end": 5.0, "speaker": "A"}]
    item = _item("m", "mtg", "en", "a b c d", speakers=speakers)
    m = rs.run_suite([FakeSystem()], [item], tmp_path / "o.json", tmp_path / "c")["per_slice"]["mtg"]["fake"]
    assert m["der_missing"] == 1 and "der" not in m


# ----------------------------------------------------------------------------- publishable summaries


def test_summarize_keeps_only_aggregates(tmp_path):
    results = {
        "created_at": "2026-10-25T00:00:00+00:00", "machine": {"chip": "M1"},
        "config": {"systems": ["fake"], "n_items": 2, "manifest": "full", "split": "all", "slices": None,
                   "refresh": False, "cache_dir": "/Users/someone/.cache/suite"},
        "per_item": [{"id": "en_x-00", "reference": "SECRET REFERENCE TEXT", "text": "hyp", "audio": "/Users/a/b.wav",
                      "words": [{"text": "x"}]}],
        "per_slice": {"en_x": {"fake": {"n": 1, "wer": 0.1, "words": 10, "item_wer": {"en_x-00": 0.1},
                                        "splits": {"dev": {"n": 1, "wer": 0.1, "item_wer": {"en_x-00": 0.1}}}}}},
        "per_system": {"fake": {"n": 1, "wer": 0.1, "item_wer": {"en_x-00": 0.1}}},
    }
    summary = rs.summarize(results)
    blob = json.dumps(summary)
    assert "per_item" not in summary and "SECRET" not in blob and "/Users" not in blob and "item_wer" not in blob
    assert summary["per_slice"]["en_x"]["fake"]["wer"] == 0.1
    assert summary["per_slice"]["en_x"]["fake"]["splits"]["dev"]["n"] == 1
    assert summary["per_system"]["fake"]["wer"] == 0.1 and summary["config"]["n_items"] == 2
    assert "cache_dir" not in summary["config"]


def test_summarize_cli_writes_next_to_the_results_file(tmp_path):
    path = tmp_path / "auto-full.json"
    path.write_text(json.dumps({"per_item": [{"reference": "x"}], "per_slice": {}, "per_system": {"s": {"wer": 0.2}},
                                "config": {}}))
    rs.main(["--summarize", str(path)])
    out = tmp_path / "auto-full.summary.json"
    assert out.exists() and rs.summary_path(path) == out
    assert json.loads(out.read_text())["per_system"] == {"s": {"wer": 0.2}}
    assert "reference" not in out.read_text()

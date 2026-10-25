"""Task 14: auto systems in the suite runner, regression gate, profile generation (fakes only)."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import soundfile as sf

from benchmarks.suite import gate, make_profiles as mp, run_suite as rs
from benchmarks.suite.run_suite import AUTO_SYSTEMS, AutoSystem, build_system, score_item
from test_auto_pipeline import SR, Rig  # same fakes as the pipeline tests (tests/ is on sys.path)

# ----------------------------------------------------------------------------- synthetic results


def rec(system, item_id, slice_, wer, split="test", edits=None, **extra):
    r = {"system": system, "id": item_id, "slice": slice_, "split": split, "wer": wer, "empty_ref": wer is None,
         "duration": 60.0, "edits": edits or {}}
    r.update(extra)
    return r


def results(records):
    return {"per_item": records, "config": {}}


def wer_pattern(i):
    return 0.05 + 0.01 * (i % 4)


def two_slices(system="s", bump=0.0, bump_slice="en_clean", n=12, ids_from=0):
    out = []
    for i in range(ids_from, ids_from + n):
        out.append(rec(system, f"en_clean-{i}", "en_clean", wer_pattern(i) + (bump if bump_slice == "en_clean" else 0)))
        out.append(rec(system, f"es_mls-{i}", "es_mls", wer_pattern(i) + (bump if bump_slice == "es_mls" else 0)))
    return out


# ----------------------------------------------------------------------------- gate


def test_gate_flags_worse_slice():
    base = results(two_slices())
    cand = results(two_slices(bump=0.03, bump_slice="es_mls"))
    report = gate.compare(base, cand)
    assert report["slices"]["es_mls"]["worse"] is True
    assert report["slices"]["es_mls"]["delta"] == pytest.approx(0.03)
    lo, hi = report["slices"]["es_mls"]["ci"]
    assert lo > 0 and hi >= lo
    assert report["slices"]["en_clean"]["worse"] is False
    assert gate.passes(report) is False


def test_gate_passes_identical():
    base = results(two_slices())
    report = gate.compare(base, copy.deepcopy(base))
    assert all(not v["worse"] and v["delta"] == 0 for v in report["slices"].values())
    assert gate.passes(report) is True


def test_gate_improvement_is_not_worse():
    report = gate.compare(results(two_slices()), results(two_slices(bump=-0.02)))
    assert gate.passes(report) is True and report["slices"]["en_clean"]["delta"] < 0


def test_gate_intersects_item_ids_and_split():
    base = results(two_slices(n=10, ids_from=0))
    cand_recs = two_slices(n=10, ids_from=5)  # ids 5..14: only 5..9 overlap
    for r in cand_recs:
        if int(r["id"].rsplit("-", 1)[1]) >= 10:
            r["wer"] = 0.9  # items the base never saw must not count
    cand_recs.append(rec("s", "en_clean-x", "en_clean", 0.9, split="dev"))  # other split ignored
    report = gate.compare(base, results(cand_recs))
    assert report["slices"]["en_clean"]["n"] == 5
    assert report["slices"]["en_clean"]["delta"] == pytest.approx(0.0)
    assert report["slices"]["en_clean"]["n_base_only"] == 5 and report["slices"]["en_clean"]["n_cand_only"] == 5
    assert report["n_missing_in_cand"] == 10  # 5 per slice: the candidate lacks items the base completed
    assert gate.passes(report) is False


def test_gate_candidate_superset_still_passes():
    base = results(two_slices(n=5))
    cand = results(two_slices(n=8))  # extra candidate-only items are fine
    report = gate.compare(base, cand)
    assert report["slices"]["en_clean"]["n_cand_only"] == 3 and report["slices"]["en_clean"]["n_base_only"] == 0
    assert gate.passes(report) is True


def test_gate_fails_when_candidate_errored_on_shared_items():
    base = results(two_slices(n=5))
    cand_recs = two_slices(n=5)
    for r in cand_recs:
        if r["slice"] == "en_clean" and int(r["id"].rsplit("-", 1)[1]) >= 2:
            r.pop("wer")
            r["error"] = "RuntimeError: boom"  # 3/5 errored: only 2 paired items remain, and they look fine
    report = gate.compare(base, results(cand_recs))
    assert report["slices"]["en_clean"]["n"] == 2 and report["slices"]["en_clean"]["worse"] is False
    assert report["slices"]["en_clean"]["n_base_only"] == 3
    assert gate.passes(report) is False
    assert "lacks 3 item" in "\n".join(gate.failures(report))
    assert "base-only=3" in gate.format_report(report)


def test_gate_candidate_missing_a_whole_slice_fails():
    base = results(two_slices(n=4))
    cand = results([r for r in two_slices(n=4) if r["slice"] == "en_clean"])
    report = gate.compare(base, cand)
    assert report["slices"]["es_mls"]["n"] == 0 and report["slices"]["es_mls"]["delta"] is None
    assert gate.passes(report) is False
    gate.format_report(report)


def test_gate_harm_rate_rise_over_one_point_fails():
    def with_edits(harm):
        out = two_slices()
        for r in out:
            r["edits"] = {"refine": {"help": 20, "harm": harm, "neutral": 80 - harm}}
        return results(out)

    ok = gate.compare(with_edits(2), with_edits(3))  # +1 pp: allowed
    assert ok["harm"]["refine"]["delta"] == pytest.approx(0.01)
    assert gate.passes(ok) is True
    bad = gate.compare(with_edits(2), with_edits(4))  # +2 pp
    assert gate.passes(bad) is False


def test_gate_nospeech_insertions_rise_fails():
    def ns(ins):
        return [rec("s", f"ns-{i}", "nospeech", None, ins_per_min=ins) for i in range(3)]

    base, cand = results(two_slices() + ns(1.0)), results(two_slices() + ns(2.0))
    report = gate.compare(base, cand)
    assert report["nospeech"] == {"base": pytest.approx(1.0), "cand": pytest.approx(2.0), "delta": pytest.approx(1.0)}
    assert gate.passes(report) is False
    assert gate.passes(gate.compare(base, copy.deepcopy(base))) is True


def test_gate_requires_one_system_or_a_choice():
    two = results(two_slices("a") + two_slices("b", bump=0.03))
    with pytest.raises(ValueError):
        gate.compare(two, two)
    report = gate.compare(two, two, base_system="a", cand_system="b")
    assert report["slices"]["en_clean"]["delta"] == pytest.approx(0.03)


def test_gate_cli(tmp_path, capsys):
    base, cand = tmp_path / "b.json", tmp_path / "c.json"
    base.write_text(json.dumps(results(two_slices())))
    cand.write_text(json.dumps(results(two_slices(bump=0.03))))
    assert gate.main(["--base", str(base), "--cand", str(base)]) == 0
    assert gate.main(["--base", str(base), "--cand", str(cand)]) == 1
    assert "worse" in capsys.readouterr().out.lower()


# ----------------------------------------------------------------------------- make_profiles

BASE = {
    "version": 1, "generated_from": None, "trust": None,
    "profiles": {
        "en": {"engines": ["qwen3-asr-1.7b", "whisper-large-v3"], "pick": True, "verify": True, "refine": True,
               "margin": 0.05, "solid_min_prob": 0.5, "evidence": None},
        "es": {"engines": ["qwen3-asr-1.7b", "whisper-large-v3"], "pick": True, "verify": True, "refine": True,
               "margin": 0.05, "solid_min_prob": 0.5, "evidence": None},
        "other": {"engines": ["whisper-large-v3", "qwen3-asr-1.7b"], "pick": True, "verify": True, "refine": True,
                  "margin": 0.05, "solid_min_prob": 0.5, "evidence": None},
    },
    "default": {"engines": ["qwen3-asr-1.7b", "whisper-large-v3"], "pick": True, "verify": True, "refine": True,
                "margin": 0.05, "solid_min_prob": 0.5, "evidence": None},
}
SLICE_MAP = {"es_mls": "es", "es_fleurs": "es", "en_clean": "en", "nospeech": mp.ALL_KEYS}
CHAIN = ["auto:single-a", "auto:single-b", "auto:fused", "auto:+pick", "auto:+verify", "auto:+refine"]


def chain_results(wers: dict, harm: dict | None = None, slices=("es_mls", "es_fleurs"), n=8, nospeech=None,
                  only=None):
    """`wers` maps system -> base WER (a per-item pattern is added identically to all systems).

    `only` maps system -> predicate on the item index: the system completed just those items."""
    only = only or {}
    harm = harm or {}
    recs = []
    stage_of = {"auto:+pick": "pick", "auto:+verify": "verify", "auto:+refine": "refine"}
    for system, w in wers.items():
        for sl in slices:
            for i in range(n):
                if system in only and not only[system](i):
                    continue
                edits = {}
                if system in stage_of:
                    h = harm.get(stage_of[system], 0)
                    edits = {stage_of[system]: {"help": 50 - h, "harm": h, "neutral": 50}}
                recs.append(rec(system, f"{sl}-{i}", sl, w + wer_pattern(i) - 0.05, split="dev", edits=edits))
        for j, ins in enumerate((nospeech or {}).get(system, [])):
            recs.append(rec(system, f"ns-{j}", "nospeech", None, split="dev", ins_per_min=ins))
    return results(recs)


def test_make_profiles_turns_off_harmful_refine():
    res = chain_results({"auto:single-a": 0.20, "auto:single-b": 0.25, "auto:fused": 0.14, "auto:+pick": 0.12,
                         "auto:+verify": 0.10, "auto:+refine": 0.15})  # refine raises es WER
    out = mp.make_profiles(res, SLICE_MAP, base=BASE, evidence="results/x.json")
    es = out["profiles"]["es"]
    assert es["refine"] is False
    assert es["pick"] is True and es["verify"] is True
    assert es["engines"] == BASE["profiles"]["es"]["engines"]
    assert es["evidence"] == "results/x.json"
    assert out["generated_from"] == "results/x.json"
    # no data for en: the current entry is kept, uncalibrated
    assert out["profiles"]["en"] == BASE["profiles"]["en"]
    assert out["calibration"]["en"]["status"] == "unvalidated"
    assert out["calibration"]["es"]["status"] == "validated"


def test_make_profiles_turns_off_harmful_stage_by_harm_rate():
    res = chain_results({"auto:single-a": 0.20, "auto:single-b": 0.25, "auto:fused": 0.14, "auto:+pick": 0.12,
                         "auto:+verify": 0.11, "auto:+refine": 0.10}, harm={"refine": 5})  # 5 % > 2 %
    es = mp.make_profiles(res, SLICE_MAP, base=BASE)["profiles"]["es"]
    assert es["refine"] is False and es["verify"] is True


def test_make_profiles_keeps_full_chain_when_it_helps():
    res = chain_results({"auto:single-a": 0.20, "auto:single-b": 0.25, "auto:fused": 0.14, "auto:+pick": 0.12,
                         "auto:+verify": 0.10, "auto:+refine": 0.08})
    es = mp.make_profiles(res, SLICE_MAP, base=BASE)["profiles"]["es"]
    assert (es["pick"], es["verify"], es["refine"]) == (True, True, True)


def test_make_profiles_smallest_config_when_larger_ones_do_not_help():
    res = chain_results({"auto:single-a": 0.20, "auto:single-b": 0.25, "auto:fused": 0.10, "auto:+pick": 0.10,
                         "auto:+verify": 0.10, "auto:+refine": 0.10})
    es = mp.make_profiles(res, SLICE_MAP, base=BASE)["profiles"]["es"]
    assert (es["pick"], es["verify"], es["refine"]) == (False, False, False)
    assert len(es["engines"]) == 2


def test_make_profiles_single_engine_when_fusion_loses():
    res = chain_results({"auto:single-a": 0.10, "auto:single-b": 0.12, "auto:fused": 0.13, "auto:+pick": 0.13,
                         "auto:+verify": 0.12, "auto:+refine": 0.12})
    es = mp.make_profiles(res, SLICE_MAP, base=BASE)["profiles"]["es"]
    assert es["engines"] == [BASE["profiles"]["es"]["engines"][0]]  # engine A is the best single
    assert (es["pick"], es["verify"], es["refine"]) == (False, False, False)
    res_b = chain_results({"auto:single-a": 0.14, "auto:single-b": 0.10, "auto:fused": 0.13, "auto:+pick": 0.13,
                           "auto:+verify": 0.12, "auto:+refine": 0.12})
    assert mp.make_profiles(res_b, SLICE_MAP, base=BASE)["profiles"]["es"]["engines"] == [
        BASE["profiles"]["es"]["engines"][1]]


def test_make_profiles_compares_paired_items_when_a_system_is_missing_items():
    # +refine completed only the hardest items (i % 4 == 3). It is better than +verify on those items, but its
    # unpaired mean WER (hard items only) is above +verify's mean over all items: unpaired means would switch it off.
    wers = {"auto:single-a": 0.20, "auto:single-b": 0.25, "auto:fused": 0.14, "auto:+pick": 0.12,
            "auto:+verify": 0.10, "auto:+refine": 0.09}
    res = chain_results(wers, only={"auto:+refine": lambda i: i % 4 == 3})
    per_item = {(r["system"], r["id"]): r["wer"] for r in res["per_item"]}
    refine_mean = np.mean([v for (sy, _), v in per_item.items() if sy == "auto:+refine"])
    verify_mean = np.mean([v for (sy, _), v in per_item.items() if sy == "auto:+verify"])
    assert refine_mean > verify_mean  # the trap
    out = mp.make_profiles(res, SLICE_MAP, base=BASE)
    es = out["profiles"]["es"]
    assert (es["pick"], es["verify"], es["refine"]) == (True, True, True)
    assert out["calibration"]["es"]["chain_items"] == 4  # 2 slices x 2 shared items


def test_make_profiles_best_single_uses_items_both_singles_completed():
    # single-a only completed the easy items, so its raw mean is lower, but on shared items single-b is better
    wers = {"auto:single-a": 0.11, "auto:single-b": 0.10, "auto:fused": 0.05, "auto:+pick": 0.05,
            "auto:+verify": 0.05, "auto:+refine": 0.05}
    res = chain_results(wers, only={"auto:single-a": lambda i: i % 4 == 0})
    es = mp.make_profiles(res, SLICE_MAP, base=BASE)["profiles"]["es"]
    assert len(es["engines"]) == 2  # fused (.05) beats b (.10) on shared items; a is not the baseline
    res2 = chain_results({**wers, "auto:fused": 0.105, "auto:+pick": 0.105, "auto:+verify": 0.105,
                          "auto:+refine": 0.105},
                         only={"auto:single-a": lambda i: i % 4 == 0})
    assert mp.make_profiles(res2, SLICE_MAP, base=BASE)["profiles"]["es"]["engines"] == [
        BASE["profiles"]["es"]["engines"][1]]  # best single is b (paired), fusion is worse than it


def noisy_results(fused_diffs, n=12):
    """single-a at 0.10, single-b 0.12, fused = single-a + per-item diffs, later stages equal to fused."""
    recs = []
    for sl in ("es_mls",):
        for i in range(n):
            recs.append(rec("auto:single-a", f"{sl}-{i}", sl, 0.10, split="dev"))
            recs.append(rec("auto:single-b", f"{sl}-{i}", sl, 0.12, split="dev"))
            for system in ("auto:fused", "auto:+pick", "auto:+verify", "auto:+refine"):
                recs.append(rec(system, f"{sl}-{i}", sl, 0.10 + fused_diffs[i % len(fused_diffs)], split="dev"))
    return results(recs)


def test_fusion_needs_a_significant_win_over_the_best_single_engine():
    # mean diff is negative (-0.005) but the paired CI crosses 0: not significantly better -> single engine
    res = noisy_results([-0.04, 0.03])
    out = mp.make_profiles(res, SLICE_MAP, base=BASE)
    es = out["profiles"]["es"]
    assert es["engines"] == [BASE["profiles"]["es"]["engines"][0]]
    assert (es["pick"], es["verify"], es["refine"]) == (False, False, False)
    assert "not significantly better" in out["calibration"]["es"]["decision"]
    # a consistent small win (CI upper bound < 0) keeps both engines
    es2 = mp.make_profiles(noisy_results([-0.01, -0.02]), SLICE_MAP, base=BASE)["profiles"]["es"]
    assert es2["engines"] == BASE["profiles"]["es"]["engines"]


def test_make_profiles_uses_dev_split_only():
    res = chain_results({"auto:single-a": 0.20, "auto:single-b": 0.25, "auto:fused": 0.14, "auto:+pick": 0.12,
                         "auto:+verify": 0.10, "auto:+refine": 0.08})
    # a test split where refine is awful must not matter
    res["per_item"] += [rec("auto:+refine", f"es_mls-t{i}", "es_mls", 0.9, split="test") for i in range(8)]
    es = mp.make_profiles(res, SLICE_MAP, base=BASE)["profiles"]["es"]
    assert es["refine"] is True


def test_make_profiles_nospeech_insertions_switch_off_a_raising_stage():
    wers = {"auto:single-a": 0.20, "auto:single-b": 0.25, "auto:fused": 0.14, "auto:+pick": 0.12,
            "auto:+verify": 0.10, "auto:+refine": 0.09}
    ns = {"auto:fused": [1.0, 1.0], "auto:+pick": [1.0, 1.0], "auto:+verify": [1.0, 1.0], "auto:+refine": [3.0, 3.0]}
    es = mp.make_profiles(chain_results(wers, nospeech=ns), SLICE_MAP, base=BASE)["profiles"]["es"]
    assert es["refine"] is False and es["verify"] is True


def test_make_profiles_creates_new_key_from_fallback_entry_and_validates():
    res = chain_results({"auto:single-a": 0.2, "auto:single-b": 0.25, "auto:fused": 0.1, "auto:+pick": 0.1,
                         "auto:+verify": 0.1, "auto:+refine": 0.1}, slices=("es_phone",))
    out = mp.make_profiles(res, {"es_phone": "es-phone"}, base=BASE)
    assert "es-phone" in out["profiles"] and out["profiles"]["es-phone"]["margin"] == 0.05
    from transcribe.auto.profiles import validate_profiles
    validate_profiles(out)


def test_default_slice_to_profile_mapping():
    m = mp.default_slice_to_profile(
        ["en_clean", "en_talks", "en_phone", "en_noise", "es_mls", "es_fleurs", "es_phone", "es_noise", "nospeech"])
    assert m == {"en_clean": "en", "en_talks": "en", "en_phone": "en-phone", "en_noise": "en-noisy", "es_mls": "es",
                 "es_fleurs": "es", "es_phone": "es-phone", "es_noise": "es-noisy", "nospeech": mp.ALL_KEYS}


def test_trust_coef_fit_recovers_linear_relation():
    rng = np.random.default_rng(3)
    names = ["agreement_rate", "unclear_rate", "mean_confidence", "hallucination_flag_rate", "skipped_stages"]
    true_w = {"agreement_rate": -0.3, "unclear_rate": 0.8, "mean_confidence": -0.2, "hallucination_flag_rate": 0.5,
              "skipped_stages": 0.02}
    recs = []
    for i in range(40):
        feats = {n: float(rng.random()) for n in names}
        feats["skipped_stages"] = float(rng.integers(0, 3))
        wer = 0.4 + sum(true_w[n] * feats[n] for n in names)
        recs.append(rec("auto:+refine", f"es_mls-{i}", "es_mls", wer, split="dev", trust_features=feats))
    recs.append(rec("auto:+refine", "es_mls-t", "es_mls", 5.0, split="test",
                    trust_features={n: 0.0 for n in names}))  # test split never used
    coef = mp.fit_trust_coef(results(recs))
    assert coef["intercept"] == pytest.approx(0.4, abs=1e-6)
    for n in names:
        assert coef["weights"][n] == pytest.approx(true_w[n], abs=1e-6)
    out = mp.make_profiles(results(recs), SLICE_MAP, base=BASE)
    assert out["trust"]["coef"]["intercept"] == pytest.approx(0.4, abs=1e-6)


def _trust_rows(n, names, vary, true_w=None):
    rng = np.random.default_rng(7)
    true_w = true_w or {k: 0.1 * (j + 1) for j, k in enumerate(names)}
    recs = []
    for i in range(n):
        feats = {k: (float(rng.random()) if k in vary else 0.5) for k in names}
        recs.append(rec("auto:+refine", f"es_mls-{i}", "es_mls", 0.2 + sum(true_w[k] * feats[k] for k in vary),
                        split="dev", trust_features=feats))
    return results(recs)


NAMES = ["agreement_rate", "unclear_rate", "mean_confidence", "hallucination_flag_rate", "skipped_stages"]


def test_trust_fit_drops_zero_variance_features():
    res = _trust_rows(30, NAMES, vary={"agreement_rate", "unclear_rate"})  # the other 3 are constant
    coef = mp.fit_trust_coef(res)
    assert set(coef["weights"]) == {"agreement_rate", "unclear_rate"}
    assert coef["weights"]["agreement_rate"] == pytest.approx(0.1, abs=1e-6)
    assert coef["weights"]["unclear_rate"] == pytest.approx(0.2, abs=1e-6)
    out = mp.make_profiles(res, SLICE_MAP, base=BASE)
    assert out["trust_fit"]["status"] == "ok"
    assert out["trust_fit"]["dropped"] == ["mean_confidence", "hallucination_flag_rate", "skipped_stages"]
    assert out["trust"]["coef"]["weights"].keys() == coef["weights"].keys()


def test_trust_fit_skipped_with_too_few_rows():
    res = _trust_rows(10, NAMES, vary=set(NAMES))  # 10 rows < 3 x 5 features
    assert mp.fit_trust_coef(res) is None
    out = mp.make_profiles(res, SLICE_MAP, base=BASE)
    assert out["trust_fit"]["status"].startswith("skipped") and "10 rows" in out["trust_fit"]["status"]
    assert out["trust"] == BASE["trust"]  # the current coefficients are kept
    assert out["calibration"]["trust"]["status"] == "unvalidated"
    assert mp.fit_trust_coef(_trust_rows(15, NAMES, vary=set(NAMES))) is not None  # exactly 3 x 5 rows is enough
    assert mp.fit_trust_coef(_trust_rows(14, NAMES, vary=set(NAMES))) is None


def test_trust_fit_skipped_when_every_feature_is_constant():
    res = _trust_rows(30, NAMES, vary=set())
    out = mp.make_profiles(res, SLICE_MAP, base=BASE)
    assert out["trust_fit"]["status"].startswith("skipped") and "zero variance" in out["trust_fit"]["status"]


def test_trust_coef_none_without_data():
    assert mp.fit_trust_coef(results([rec("auto:+refine", "a", "es_mls", 0.1, split="dev")])) is None


def test_make_profiles_cli_dry_run_and_write(tmp_path, capsys):
    res = chain_results({"auto:single-a": 0.20, "auto:single-b": 0.25, "auto:fused": 0.14, "auto:+pick": 0.12,
                         "auto:+verify": 0.10, "auto:+refine": 0.08})
    src, dst = tmp_path / "res.json", tmp_path / "profiles.json"
    src.write_text(json.dumps(res))
    base = tmp_path / "base.json"
    base.write_text(json.dumps(BASE))
    assert mp.main(["--results", str(src), "--base", str(base)]) == 0
    assert "es" in capsys.readouterr().out and not dst.exists()
    assert mp.main(["--results", str(src), "--base", str(base), "--write", str(dst)]) == 0
    written = json.loads(dst.read_text())
    assert written["profiles"]["es"]["evidence"] == str(src)


# ----------------------------------------------------------------------------- auto systems


@pytest.fixture
def audio(tmp_path):
    t = np.arange(int(6 * SR)) / SR
    path = tmp_path / "talk.wav"
    sf.write(str(path), (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), SR)
    return path


def make_item(audio, **kw):
    item = {"id": "en_clean-00", "slice": "en_clean", "split": "dev", "lang": "en", "audio": str(audio),
            "duration": 6.0, "reference": "the hat sat on the mat",
            "speakers": [{"start": 0.0, "end": 3.0, "speaker": "A"}, {"start": 3.0, "end": 6.0, "speaker": "B"}],
            "condition": "clean", "meta": {}}
    item.update(kw)
    return item


EXPECTED_STAGE_TEXTS = {
    "auto:single-a": {"fused"}, "auto:single-b": {"fused"}, "auto:fused": {"fused"},
    "auto:+pick": {"fused", "pick"}, "auto:+verify": {"fused", "pick", "verify"},
    "auto:+refine": {"fused", "pick", "verify", "refine"},
}
EXPECTED_EDIT_STAGES = {
    "auto:single-a": set(), "auto:single-b": set(), "auto:fused": set(), "auto:+pick": {"pick"},
    "auto:+verify": {"pick", "verify"}, "auto:+refine": {"pick", "verify", "refine"},
}


def test_auto_system_names_and_registry():
    assert AUTO_SYSTEMS == ("auto:single-a", "auto:single-b", "auto:fused", "auto:+pick", "auto:+verify",
                            "auto:+refine")
    for name in AUTO_SYSTEMS:
        system = build_system(name)
        assert isinstance(system, AutoSystem) and system.name == name


def test_auto_systems_emit_stage_texts_and_share_engine_cache(tmp_path, audio):
    rig = Rig(tmp_path)
    item = make_item(audio)
    outs = {}
    for name in AUTO_SYSTEMS:
        outs[name] = AutoSystem(name, factories=rig.factories()).run(item)
    for name, out in outs.items():
        assert set(out["meta"]["stage_texts"]) == EXPECTED_STAGE_TEXTS[name], name
        assert set(out["meta"]["stages"]) == EXPECTED_EDIT_STAGES[name], name
        for st in out["meta"]["stages"].values():
            assert set(st) == {"before", "after"}
        assert out["words"] and all("unclear" in w for w in out["words"])
        assert out["text"] == " ".join(w["text"] for w in out["words"])
        assert out["meta"]["trust"]["features"]
    assert outs["auto:single-a"]["text"] == "the cat sat on the mat"
    assert outs["auto:single-b"]["text"] == "the hat sat on the mat"
    # fused: disputed span goes to engine A, unclear
    assert outs["auto:fused"]["text"] == "the cat sat on the mat"
    assert any(w["unclear"] for w in outs["auto:fused"]["words"])
    assert outs["auto:+pick"]["meta"]["stages"]["pick"] == {"before": "the cat sat on the mat",
                                                            "after": "the hat sat on the mat"}
    assert outs["auto:+verify"]["meta"]["stages"]["verify"]["before"] == "the hat sat on the mat"
    assert outs["auto:+refine"]["meta"]["stages"]["refine"]["before"] == outs["auto:+verify"]["text"]
    # one ASR run per engine for the whole ablation ladder
    assert sorted(rig.engine_calls) == ["qwen3-asr-1.7b", "whisper-large-v3"]
    # two reference speakers -> diarizer asked for 2
    assert set(rig.diarize_calls) == {2}
    assert rig.picker.__class__.__name__ == "FakePicker"
    assert all("speaker" in s for s in outs["auto:+refine"]["segments"])


def test_auto_system_scores_through_suite_runner(tmp_path, audio):
    rig = Rig(tmp_path)
    item = make_item(audio)
    result = rs.run_suite([AutoSystem("auto:+pick", factories=rig.factories())], [item], tmp_path / "o.json",
                          tmp_path / "c")
    r = result["per_item"][0]
    assert r["wer"] == 0.0
    assert r["edits"]["pick"]["help"] == 1
    assert r["unclear"] is not None
    assert r["trust_features"]["agreement_rate"] > 0
    assert "der" in r


def test_auto_system_title_from_item_meta_and_language_none(tmp_path, audio):
    rig = Rig(tmp_path)
    seen = {}
    import transcribe.auto.pipeline as pl

    original = pl.run_auto

    def spy(path, metadata, **kw):
        seen.update(metadata=metadata, language=kw["language"], num_speakers=kw["num_speakers"],
                    refine=kw["refine"])
        return original(path, metadata, **kw)

    system = AutoSystem("auto:fused", factories=rig.factories())
    import benchmarks.suite.run_suite as run_mod
    mp_ = pytest.MonkeyPatch()
    try:
        mp_.setattr(pl, "run_auto", spy)
        system.run(make_item(audio, lang="none", speakers=None, meta={"title": "A Talk About Whisper"}))
    finally:
        mp_.undo()
    assert seen == {"metadata": {"title": "A Talk About Whisper"}, "language": None, "num_speakers": None,
                    "refine": True}
    seen.clear()
    mp_ = pytest.MonkeyPatch()
    try:
        mp_.setattr(pl, "run_auto", spy)
        system.run(make_item(audio, speakers=[{"start": 0, "end": 6, "speaker": "A"}]))
    finally:
        mp_.undo()
    assert seen["metadata"] == {} and seen["language"] == "en" and seen["num_speakers"] == 1


def test_auto_system_profiles_override_every_key():
    system = AutoSystem("auto:+pick")
    table = system.profiles_for({"lang": "es"})
    base_es = rs_base()["profiles"]["es"]["engines"]
    for key, entry in list(table["profiles"].items()) + [("default", table["default"])]:
        assert entry["engines"] == base_es, key
        assert (entry["pick"], entry["verify"], entry["refine"]) == (True, False, False), key
    other = system.profiles_for({"lang": "none"})
    assert other["profiles"]["en"]["engines"] == rs_base()["profiles"]["other"]["engines"]
    single_b = AutoSystem("auto:single-b").profiles_for({"lang": "en"})
    assert single_b["profiles"]["en"]["engines"] == [rs_base()["profiles"]["en"]["engines"][1]]
    single_a = AutoSystem("auto:single-a").profiles_for({"lang": "en"})
    assert single_a["default"]["engines"] == [rs_base()["profiles"]["en"]["engines"][0]]
    full = AutoSystem("auto:+refine").profiles_for({"lang": "en"})["profiles"]["en"]
    assert (full["pick"], full["verify"], full["refine"]) == (True, True, True)
    fused = AutoSystem("auto:fused").profiles_for({"lang": "en"})["profiles"]["en"]
    assert (fused["pick"], fused["verify"], fused["refine"]) == (False, False, False) and len(fused["engines"]) == 2


def rs_base():
    from transcribe.auto.profiles import load_profiles
    return load_profiles()


def test_auto_cache_key_changes_with_override_and_config(monkeypatch):
    keys = {name: AutoSystem(name).cache_key() for name in AUTO_SYSTEMS}
    assert len(set(keys.values())) == len(AUTO_SYSTEMS)
    again = AutoSystem("auto:+pick").cache_key()
    assert again == keys["auto:+pick"]  # deterministic
    # a different profiles override (here: engine table) changes the key
    tweaked = rs_base()
    tweaked["profiles"]["en"]["engines"] = ["whisper-large-v3", "qwen3-asr-1.7b"]
    assert AutoSystem("auto:+pick", base_profiles=tweaked).cache_key() != keys["auto:+pick"]
    # pipeline cache version and model ids change it too
    import transcribe.auto.cache as cache_mod
    monkeypatch.setattr(cache_mod, "CACHE_VERSION", cache_mod.CACHE_VERSION + 1)
    assert AutoSystem("auto:+pick").cache_key() != keys["auto:+pick"]
    monkeypatch.undo()
    import transcribe.auto.picker as picker_mod
    monkeypatch.setattr(picker_mod, "PICKER_MODEL", "other/picker")
    assert AutoSystem("auto:+pick").cache_key() != keys["auto:+pick"]
    monkeypatch.undo()
    monkeypatch.setenv("TRANSCRIBE_REFINE_MODEL", "opus")
    assert AutoSystem("auto:+pick").cache_key() != keys["auto:+pick"]


def test_auto_cache_key_covers_margin_trust_and_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    key = AutoSystem("auto:+pick").cache_key()
    margin = rs_base()
    margin["profiles"]["es"]["margin"] = 0.2
    assert AutoSystem("auto:+pick", base_profiles=margin).cache_key() != key
    solid = rs_base()
    solid["profiles"]["en"]["solid_min_prob"] = 0.9
    assert AutoSystem("auto:+pick", base_profiles=solid).cache_key() != key
    other = rs_base()
    other["profiles"]["other"]["margin"] = 0.3
    assert AutoSystem("auto:+pick", base_profiles=other).cache_key() != key
    trust = rs_base()
    trust["trust"] = {"coef": {"intercept": 0.1, "weights": {"unclear_rate": 1.0}}}
    assert AutoSystem("auto:+pick", base_profiles=trust).cache_key() != key
    monkeypatch.setenv("HF_TOKEN", "x")
    with_token = AutoSystem("auto:+pick").cache_key()
    assert with_token != key
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "y")
    assert AutoSystem("auto:+pick").cache_key() == with_token  # only presence matters, either variable


def test_unknown_auto_system_raises():
    with pytest.raises(ValueError):
        build_system("auto:nope")


def test_score_item_carries_trust_features():
    item = {"id": "x", "slice": "s", "split": "dev", "lang": "en", "duration": 5.0, "reference": "a b c"}
    out = {"text": "a b c", "seconds": 1.0, "meta": {"trust": {"features": {"agreement_rate": 0.9}}}}
    assert score_item(item, "auto:x", out)["trust_features"] == {"agreement_rate": 0.9}
    assert "trust_features" not in score_item(item, "auto:x", {"text": "a b c", "seconds": 1.0, "meta": {}})


def test_cached_picker_memoizes_llm_answers(tmp_path, monkeypatch):
    from transcribe.auto.picker import MlxPicker

    calls = []

    def fake_run(self, prompt, max_tokens):
        calls.append((prompt, max_tokens))
        return '{"choices": []}'

    monkeypatch.setattr(MlxPicker, "_run_model", fake_run)
    factory = rs.cached_picker_factory(tmp_path / "picker")
    first, second = factory(), factory()
    assert first._run_model("p", 10) == '{"choices": []}'
    assert second._run_model("p", 10) == '{"choices": []}'  # served from disk, not the model
    assert second._run_model("p", 11) == '{"choices": []}'  # a different budget is a different prompt
    assert calls == [("p", 10), ("p", 11)]
    assert build_system("auto:+pick").factories["picker"].__name__ == "CachedPicker"


def test_cached_scorer_memoizes_scores(tmp_path, monkeypatch):
    from transcribe.auto.verify import WhisperScorer

    calls = []

    def fake_detail(self, samples, start, end, text, lang):
        calls.append(text)
        return -float(len(text)), len(text.split()) + 1

    monkeypatch.setattr(WhisperScorer, "score_detail", fake_detail)
    factory = rs.cached_scorer_factory(tmp_path / "scorer")
    samples = np.zeros(16000, dtype=np.float32)
    first, second = factory(), factory()
    assert first.score_detail(samples, 0.0, 1.0, "a b", "en") == (-3.0, 3)
    assert second.score_detail(samples.copy(), 0.0, 1.0, "a b", "en") == (-3.0, 3)  # same audio content: disk hit
    second.score_detail(samples, 0.0, 1.0, "a c", "en")  # different text
    second.score_detail(np.ones(16000, dtype=np.float32), 0.0, 1.0, "a b", "en")  # different audio
    second.score_detail(samples, 0.0, 1.5, "a b", "en")  # different window
    assert calls == ["a b", "a c", "a b", "a b"]
    assert build_system("auto:+verify").factories["scorer"].__name__ == "CachedScorer"


def test_auto_ladder_keeps_engine_pair_after_single_engine_calibration():
    calibrated = rs_base()
    calibrated["profiles"]["es"]["engines"] = ["whisper-large-v3"]  # what make_profiles writes for a single engine
    pair = calibrated["default"]["engines"]
    item = {"lang": "es"}
    assert AutoSystem("auto:single-a", base_profiles=calibrated).profiles_for(item)["profiles"]["es"]["engines"] == pair[:1]
    assert AutoSystem("auto:single-b", base_profiles=calibrated).profiles_for(item)["profiles"]["es"]["engines"] == pair[1:2]
    assert AutoSystem("auto:fused", base_profiles=calibrated).profiles_for(item)["profiles"]["es"]["engines"] == pair
    broken = copy.deepcopy(calibrated)
    broken["default"]["engines"] = ["whisper-large-v3"]  # neither the entry nor default is a pair: frozen pair
    assert AutoSystem("auto:fused", base_profiles=broken).profiles_for(item)["profiles"]["es"]["engines"] == list(
        rs.FROZEN_PAIRS["es"])


def _calibrated_base():
    """BASE after a calibration that made `es` single-engine Qwen3 (engine A)."""
    base = copy.deepcopy(BASE)
    base["profiles"]["es"].update(engines=["qwen3-asr-1.7b"], pick=False, verify=False, refine=False)
    return base


def test_make_profiles_rerun_on_calibrated_base_maps_winner_to_the_ladder_pair():
    # flipped winner: Whisper (single-b) is now the best single engine
    res = chain_results({"auto:single-a": 0.14, "auto:single-b": 0.10, "auto:fused": 0.13, "auto:+pick": 0.13,
                         "auto:+verify": 0.12, "auto:+refine": 0.12})
    es = mp.make_profiles(res, SLICE_MAP, base=_calibrated_base())["profiles"]["es"]
    assert es["engines"] == ["whisper-large-v3"]


def test_make_profiles_fusion_win_on_calibrated_base_restores_the_pair():
    res = chain_results({"auto:single-a": 0.20, "auto:single-b": 0.25, "auto:fused": 0.14, "auto:+pick": 0.12,
                         "auto:+verify": 0.10, "auto:+refine": 0.08})
    es = mp.make_profiles(res, SLICE_MAP, base=_calibrated_base())["profiles"]["es"]
    assert es["engines"] == ["qwen3-asr-1.7b", "whisper-large-v3"] and es["pick"] is True


def test_ladder_pair_falls_back_entry_default_frozen():
    base = _calibrated_base()
    assert rs.ladder_pair(base, "es-noisy") == base["default"]["engines"]
    assert rs.ladder_pair(BASE, "other") == BASE["profiles"]["other"]["engines"]
    base["default"]["engines"] = ["qwen3-asr-1.7b"]
    assert rs.ladder_pair(base, "es") == list(rs.FROZEN_PAIRS["es"])


def test_memos_miss_after_version_bump_and_bypass_on_refresh(tmp_path, monkeypatch):
    from transcribe.auto import cache as auto_cache
    from transcribe.auto.picker import MlxPicker
    from transcribe.auto.verify import WhisperScorer

    picks, scores = [], []
    monkeypatch.setattr(MlxPicker, "_run_model", lambda self, p, m: picks.append(p) or "{}")
    monkeypatch.setattr(WhisperScorer, "score_detail", lambda self, x, a, b, t, l: scores.append(t) or (-1.0, 2))
    samples = np.zeros(16000, dtype=np.float32)
    picker, scorer = rs.cached_picker_factory(tmp_path / "p"), rs.cached_scorer_factory(tmp_path / "s")
    picker()._run_model("p", 10), scorer().score_detail(samples, 0.0, 1.0, "t", "en")
    picker()._run_model("p", 10), scorer().score_detail(samples, 0.0, 1.0, "t", "en")
    assert (len(picks), len(scores)) == (1, 1)  # memo hits
    monkeypatch.setattr(rs, "PICKER_MEMO_VERSION", rs.PICKER_MEMO_VERSION + 1)
    monkeypatch.setattr(rs, "SCORER_MEMO_VERSION", rs.SCORER_MEMO_VERSION + 1)
    picker()._run_model("p", 10), scorer().score_detail(samples, 0.0, 1.0, "t", "en")
    assert (len(picks), len(scores)) == (2, 2)  # logic versions are part of the key
    monkeypatch.setattr(auto_cache, "CACHE_VERSION", auto_cache.CACHE_VERSION + 1)
    picker()._run_model("p", 10), scorer().score_detail(samples, 0.0, 1.0, "t", "en")
    assert (len(picks), len(scores)) == (3, 3)  # so is the pipeline cache version
    monkeypatch.setattr(rs, "PICKER_DECODING", {**rs.PICKER_DECODING, "temp": 0.7})
    picker()._run_model("p", 10)
    assert len(picks) == 4  # decoding settings too
    fresh_p = rs.cached_picker_factory(tmp_path / "p", refresh=True)
    fresh_s = rs.cached_scorer_factory(tmp_path / "s", refresh=True)
    fresh_p()._run_model("p", 10), fresh_s().score_detail(samples, 0.0, 1.0, "t", "en")
    assert (len(picks), len(scores)) == (5, 4)  # refresh never reads the memo


def test_build_system_threads_refresh_into_memos(monkeypatch):
    seen = {}
    monkeypatch.setattr(rs, "cached_picker_factory", lambda refresh=False: seen.setdefault("picker", refresh))
    monkeypatch.setattr(rs, "cached_scorer_factory", lambda refresh=False: seen.setdefault("scorer", refresh))
    rs.build_system("auto:+verify", refresh=True)
    assert seen == {"picker": True, "scorer": True}


# ----------------------------------------------------------------------------- per-profile trust validation


def _single_a_results(informative: bool, n=8):
    res = chain_results({"auto:single-a": 0.10, "auto:single-b": 0.20, "auto:fused": 0.12, "auto:+pick": 0.12,
                         "auto:+verify": 0.12, "auto:+refine": 0.12}, n=n)
    for r in res["per_item"]:
        if r["system"] == "auto:single-a" and r.get("wer") is not None:
            agreement = 1.0 - 2.0 * r["wer"] if informative else 0.9
            r["trust_features"] = {"agreement_rate": agreement, "unclear_rate": 0.0, "mean_confidence": 0.8,
                                   "hallucination_flag_rate": 0.0, "skipped_stages": 0}
    return res


def test_make_profiles_records_trust_spearman_for_the_rung_the_profile_runs():
    out = mp.make_profiles(_single_a_results(informative=True), SLICE_MAP, base=BASE)
    es = out["profiles"]["es"]
    assert es["engines"] == ["qwen3-asr-1.7b"]  # single-a won
    assert es["trust_rung"] == "auto:single-a"
    assert es["trust_spearman"] == pytest.approx(1.0)  # per-item WER pattern ties: ranks still monotone
    assert out["calibration"]["es"]["trust_rows"] == 16


def test_make_profiles_constant_features_give_spearman_zero():
    out = mp.make_profiles(_single_a_results(informative=False), SLICE_MAP, base=BASE)
    assert out["profiles"]["es"]["trust_spearman"] == 0.0


def test_make_profiles_leaves_trust_spearman_unset_with_too_few_rows():
    out = mp.make_profiles(_single_a_results(informative=True, n=3), SLICE_MAP, base=BASE)  # 6 rows < 8
    assert "trust_spearman" not in out["profiles"]["es"]
    assert out["calibration"]["es"]["trust_spearman"] is None


def test_make_profiles_drops_a_stale_trust_spearman_when_the_new_fit_has_too_few_rows():
    base = copy.deepcopy(BASE)
    base["profiles"]["es"]["trust_spearman"] = 0.9
    out = mp.make_profiles(_single_a_results(informative=True, n=3), SLICE_MAP, base=base)
    assert "trust_spearman" not in out["profiles"]["es"]


def test_validate_entry_rejects_non_numeric_trust_spearman():
    from transcribe.auto.profiles import validate_entry
    from transcribe.models import TranscriptionError

    entry = {**BASE["default"], "trust_spearman": 0.5}
    validate_entry("x", entry)
    with pytest.raises(TranscriptionError, match="trust_spearman"):
        validate_entry("x", {**entry, "trust_spearman": "high"})

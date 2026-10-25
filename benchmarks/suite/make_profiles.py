"""Generate `transcribe/profiles.json` from a suite results file (dev split only).

    .venv/bin/python -m benchmarks.suite.make_profiles --results X.json [--write transcribe/profiles.json]

Without `--write` it is a dry run that prints the decision table. Needs the `auto:*` systems in the results
(`auto:single-a`, `auto:single-b`, `auto:fused`, `auto:+pick`, `auto:+verify`, `auto:+refine`).

Decision per profile key (slices mapped by `slice_to_profile`; `nospeech` maps to `ALL_KEYS`):

1. Engines: the better single engine (lower mean per-item dev WER on the items both singles completed) is the
   baseline. Fusion (`auto:fused`) is kept only when it is significantly better: the paired-bootstrap 95 % CI of
   (fused - best single) per-item WER must have an upper bound < 0 (the same CI machinery as the stage decisions);
   otherwise the profile is single-engine with pick/verify/refine off (refine is unmeasured without a second engine, so
   it stays off).
2. Stages: configurations form a prefix chain fused < +pick < +verify < +refine, and every comparison inside the chain
   (means, CIs, harm rates, nospeech insertions) uses only the item ids completed by ALL systems in the chain, so a
   system that errored on some items cannot look better or worse through a different item set. A configuration is
   eligible when every stage it contains has dev harm rate <= 2 % (harm / changed regions, from `edits`). The smallest
   eligible configuration that no larger configuration beats significantly (paired bootstrap CI of the WER difference,
   lower bound > 0 means worse) wins; when none qualifies, the eligible one with the lowest mean WER. Finally a stage is
   switched off while it raises mean WER or nospeech insertions per minute over the previous configuration.
3. A key whose slices have no dev data keeps its current entry (and is reported "unvalidated").

The trust coefficients are a least-squares fit of per-file dev WER on `meta["trust"]["features"]` (see `fit_trust_coef`):
zero-variance features are dropped and the fit is skipped (status recorded in `trust_fit`) with fewer than 3 rows per
remaining feature.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.suite import metrics as M

ALL_KEYS = "*"  # slice_to_profile value for slices that apply to every profile key (nospeech)
HARM_LIMIT = 0.02
SINGLE_A, SINGLE_B, FUSED = "auto:single-a", "auto:single-b", "auto:fused"
CHAIN = (FUSED, "auto:+pick", "auto:+verify", "auto:+refine")
CHAIN_STAGE = {"auto:+pick": "pick", "auto:+verify": "verify", "auto:+refine": "refine"}
TRUST_FEATURES = ("agreement_rate", "unclear_rate", "mean_confidence", "hallucination_flag_rate", "skipped_stages")
TRUST_SYSTEM_PREFERENCE = ("auto:+refine", "auto:+verify", "auto:+pick", FUSED)
_EPS = 1e-9


def default_slice_to_profile(slices: list[str]) -> dict[str, str]:
    """en_* -> en, es_* -> es, *_phone -> {lang}-phone, *_noise -> {lang}-noisy, nospeech -> ALL_KEYS."""
    out: dict[str, str] = {}
    for sl in slices:
        if sl == "nospeech":
            out[sl] = ALL_KEYS
            continue
        lang = sl.split("_", 1)[0]
        if lang not in ("en", "es"):
            continue
        if sl.endswith("_phone"):
            out[sl] = f"{lang}-phone"
        elif sl.endswith("_noise"):
            out[sl] = f"{lang}-noisy"
        else:
            out[sl] = lang
    return out


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------


def _dev(results: dict) -> list[dict]:
    return [r for r in results.get("per_item", []) if r.get("split") == "dev" and not r.get("skipped")
            and not r.get("error")]


def _speech_wers(recs: list[dict], system: str, slices: set[str]) -> dict[str, float]:
    return {r["id"]: r["wer"] for r in recs
            if r["system"] == system and r["slice"] in slices and r.get("wer") is not None}


def _harm_rate(recs: list[dict], system: str, slices: set[str], stage: str, ids: set[str] | None = None) -> float:
    harm = total = 0
    for r in recs:
        if r["system"] == system and r["slice"] in slices and (ids is None or r["id"] in ids):
            eff = (r.get("edits") or {}).get(stage)
            if eff:
                harm += eff["harm"]
                total += eff["help"] + eff["harm"] + eff["neutral"]
    return harm / total if total else 0.0


def _ins_per_min(recs: list[dict], system: str, ids: set[str] | None = None) -> float | None:
    silent = [r for r in recs if r["system"] == system and r.get("empty_ref") and r.get("ins_per_min") is not None
              and (ids is None or r["id"] in ids)]
    minutes = sum(r["duration"] for r in silent) / 60.0
    if not silent or not minutes:
        return None
    return sum(r["ins_per_min"] * r["duration"] / 60.0 for r in silent) / minutes


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def _diff_ci(wers_a: dict[str, float], wers_b: dict[str, float]) -> tuple[float, float] | None:
    """Paired bootstrap 95 % CI of the mean per-item WER difference (A - B) on the shared ids; None without any."""
    shared = [i for i in wers_a if i in wers_b]
    return M.bootstrap_ci([wers_a[i] - wers_b[i] for i in shared]) if shared else None


def _worse(wers_a: dict[str, float], wers_b: dict[str, float]) -> bool:
    """True when system A is significantly worse than B on their shared items (CI lower bound > 0)."""
    ci = _diff_ci(wers_a, wers_b)
    return ci is not None and ci[0] > _EPS


def _significantly_better(wers_a: dict[str, float], wers_b: dict[str, float]) -> bool:
    """True when system A is significantly better than B on their shared items (CI upper bound < 0)."""
    ci = _diff_ci(wers_a, wers_b)
    return ci is not None and ci[1] < -_EPS


def _restrict(wers: dict[str, dict[str, float]], systems: list[str]) -> tuple[dict[str, dict[str, float]], set[str]]:
    """WERs of `systems` limited to the ids every one of them completed (and that id set)."""
    ids = set.intersection(*(set(wers[s]) for s in systems)) if systems else set()
    return {s: {i: v for i, v in wers[s].items() if i in ids} for s in systems}, ids


# --------------------------------------------------------------------------
# Trust coefficients
# --------------------------------------------------------------------------


MIN_ROWS_PER_FEATURE = 3


def _fit_trust(results: dict, systems: tuple[str, ...] = TRUST_SYSTEM_PREFERENCE) -> tuple[dict | None, dict]:
    """(coef or None, info) where info is `{"status": "ok" | "skipped: <reason>", ...}`.

    Fits on the dev rows of the first system in `systems` that has any (the most complete rung by default; pass a single
    rung to fit that rung alone).
    """
    recs = _dev(results)
    for system in systems:
        rows = [r for r in recs if r["system"] == system and r.get("wer") is not None and r.get("trust_features")]
        if rows:
            break
    else:
        return None, {"status": "skipped: no dev items with trust features"}
    raw = np.array([[float(r["trust_features"].get(n, 0.0)) for n in TRUST_FEATURES] for r in rows])
    keep = [j for j in range(len(TRUST_FEATURES)) if np.ptp(raw[:, j]) > 0]
    dropped = [TRUST_FEATURES[j] for j in range(len(TRUST_FEATURES)) if j not in keep]
    info = {"system": system, "rows": len(rows), "features": [TRUST_FEATURES[j] for j in keep], "dropped": dropped}
    if not keep:
        return None, {**info, "status": "skipped: every trust feature has zero variance"}
    if len(rows) < MIN_ROWS_PER_FEATURE * len(keep):
        return None, {**info, "status": f"skipped: {len(rows)} rows < {MIN_ROWS_PER_FEATURE} x {len(keep)} features"}
    x = np.hstack([np.ones((len(rows), 1)), raw[:, keep]])
    y = np.array([float(r["wer"]) for r in rows])
    sol, *_ = np.linalg.lstsq(x, y, rcond=None)
    coef = {"intercept": float(sol[0]), "weights": {TRUST_FEATURES[j]: float(w) for j, w in zip(keep, sol[1:])}}
    return coef, {**info, "status": "ok"}


MIN_SPEARMAN_ROWS = 8


def _trust_estimate(features: dict, coef: dict | None) -> float:
    """The trust score's estimate (`assemble.trust_score` semantics, without the stage penalty) for one item."""
    if coef:
        return float(coef.get("intercept", 0.0)) + sum(
            float(w) * float(features.get(n, 0.0)) for n, w in (coef.get("weights") or {}).items()
            if n != "skipped_stages")
    return 0.5 * (1.0 - float(features.get("agreement_rate", 0.0))) + float(features.get("unclear_rate", 0.0))


def _trust_spearman(recs: list[dict], system: str, slices: set[str], coef: dict | None) -> tuple[float | None, int]:
    """(Spearman of the trust estimate vs per-file WER on the dev rows of `system` in `slices`, rows).

    None when there are fewer than `MIN_SPEARMAN_ROWS` rows (the trust score then stays unvalidated for the profile).
    Constant features give 0.0 (`metrics.spearman` is 0 for constant input).
    """
    rows = [r for r in recs if r["system"] == system and r["slice"] in slices and r.get("wer") is not None
            and r.get("trust_features")]
    if len(rows) < MIN_SPEARMAN_ROWS:
        return None, len(rows)
    est = [_trust_estimate(r["trust_features"], coef) for r in rows]
    return float(M.spearman(est, [float(r["wer"]) for r in rows])), len(rows)


def fit_trust_coef(results: dict) -> dict | None:
    """Least squares of per-file dev WER on the trust features; `{"intercept", "weights"}` or None.

    Uses the most complete auto system present (`+refine`, else `+verify`, `+pick`, `fused`). Zero-variance features
    are dropped (absent from `weights`); fewer than 3 rows per remaining feature returns None (see `_fit_trust`).
    """
    return _fit_trust(results)[0]


# --------------------------------------------------------------------------
# Profile decisions
# --------------------------------------------------------------------------


def _select_entry(base: dict, key: str) -> dict:
    table = base["profiles"]
    entry = table.get(key) or table.get(key.split("-", 1)[0]) or base["default"]
    return copy.deepcopy(entry)


def _decide_key(key: str, slices: set[str], recs: list[dict], entry: dict,
                pair: list[str] | None = None) -> tuple[dict | None, dict]:
    """(new entry or None when there is no data, calibration notes).

    `pair` is the (A, B) engine pair the ladder's single-a/single-b/fused rungs ran with (`run_suite.ladder_pair` on the
    base table); it defaults to the entry's engines only when they are a pair. Taking A/B from a calibrated
    single-engine entry would map the single-b winner to the wrong engine.
    """
    wers = {s: _speech_wers(recs, s, slices) for s in (SINGLE_A, SINGLE_B, FUSED, *CHAIN)}
    wers = {s: w for s, w in wers.items() if w}
    if not wers:
        return None, {"status": "unvalidated", "reason": "no dev data for the key's slices"}

    notes: dict[str, Any] = {"status": "validated", "slices": sorted(slices),
                             "dev_wer": {s: _mean(w.values()) for s, w in wers.items()}}
    if pair is None:
        from benchmarks.suite.run_suite import FROZEN_PAIRS, lang_key_of

        engines = list(entry["engines"])
        pair = engines[:2] if len(engines) >= 2 else list(FROZEN_PAIRS[lang_key_of(key)])
    engines = list(pair[:2])
    engine_a, engine_b = engines

    singles = [s for s in (SINGLE_A, SINGLE_B) if s in wers]
    best_single = None
    if singles:
        paired, _ = _restrict(wers, singles)  # compare the singles on the items both completed
        best_single = min(singles, key=lambda s: _mean(paired[s].values()) if paired[s] else float("inf"))
    chain: list[str] = []  # contiguous prefix of the ablation ladder that is present
    for system in CHAIN:
        if system not in wers:
            break
        chain.append(system)
    if not chain and best_single is None:
        return None, {"status": "unvalidated", "reason": "results lack the auto:* systems for these slices"}
    # one significance rule everywhere: fusion must beat the best single engine by a paired bootstrap CI (upper < 0)
    fusion_wins = bool(chain) and (best_single is None or _significantly_better(wers[chain[0]], wers[best_single]))

    if not fusion_wins:
        entry.update(engines=[engine_a if best_single == SINGLE_A else engine_b], pick=False, verify=False,
                     refine=False)
        notes["decision"] = f"single engine ({best_single}): fusion is not significantly better"
        notes["rung"] = best_single
        return entry, notes

    entry["engines"] = engines[:2]
    shared_wers, ids = _restrict(wers, chain)  # every comparison below uses the ids all chain systems completed
    if not ids:
        return None, {"status": "unvalidated", "reason": "the chain's systems share no completed dev items"}
    notes["chain_items"] = len(ids)
    silent_ids = [{r["id"] for r in recs if r["system"] == s and r.get("empty_ref")} for s in chain]
    shared_silent = set.intersection(*silent_ids)  # nospeech items every chain system completed
    ins = {s: _ins_per_min(recs, s, shared_silent) for s in chain}
    mean_wer = {s: _mean(shared_wers[s].values()) for s in chain}
    harm = {s: _harm_rate(recs, s, slices, CHAIN_STAGE[s], ids) for s in chain if s in CHAIN_STAGE}
    notes["chain_dev_wer"] = mean_wer
    notes["harm_rate"] = {CHAIN_STAGE[s]: h for s, h in harm.items()}
    notes["nospeech_ins_per_min"] = {s: v for s, v in ins.items() if v is not None}

    def eligible(k: int) -> bool:
        return all(harm.get(chain[j], 0.0) <= HARM_LIMIT for j in range(1, k + 1))

    candidates = [k for k in range(len(chain)) if eligible(k)]
    chosen = next((k for k in candidates
                   if not any(_worse(shared_wers[chain[k]], shared_wers[chain[j]])
                              for j in range(k + 1, len(chain)))), None)
    if chosen is None:
        chosen = min(candidates, key=lambda k: mean_wer[chain[k]])
    while chosen > 0:  # a stage that raises WER (or nospeech insertions) is switched off
        prev, cur = chain[chosen - 1], chain[chosen]
        raises_ins = ins[cur] is not None and ins[prev] is not None and ins[cur] > ins[prev] + _EPS
        if mean_wer[cur] > mean_wer[prev] + _EPS or raises_ins:
            chosen -= 1
        else:
            break
    entry.update(pick=chosen >= 1 and "auto:+pick" in chain, verify=chosen >= 2 and "auto:+verify" in chain,
                 refine=chosen >= 3 and "auto:+refine" in chain)
    notes["decision"] = f"fused engines + {chain[chosen]}"
    notes["rung"] = chain[chosen]
    return entry, notes


def make_profiles(results: dict, slice_to_profile: dict[str, str], base: dict | None = None,
                  evidence: str | None = None) -> dict:
    """New profiles table (the `profiles.json` content) from the dev-split results.

    `base` is the current table (default: `transcribe/profiles.json`); `evidence` is recorded on every calibrated entry
    and as `generated_from`. Also adds a `calibration` block with the per-key decision and dev numbers.
    """
    if base is None:
        from transcribe.auto.profiles import load_profiles

        base = load_profiles()
    out = copy.deepcopy(base)
    recs = _dev(results)
    keys: dict[str, set[str]] = {}
    for sl, key in slice_to_profile.items():
        if key != ALL_KEYS:
            keys.setdefault(key, set()).add(sl)
    for key in out["profiles"]:  # existing keys without slices are reported, not touched
        keys.setdefault(key, set())
    calibration: dict[str, dict] = {}
    decided: dict[str, str] = {}  # key -> the ladder rung the profile actually runs
    for key, slices in sorted(keys.items()):
        current = _select_entry(base, key)
        from benchmarks.suite.run_suite import ladder_pair

        entry, notes = _decide_key(key, slices, recs, current, ladder_pair(base, key)) if slices else (
            None, {"status": "unvalidated", "reason": "no slices map to this key"})
        if entry is None:
            calibration[key] = notes
            continue  # keep the current entry (or no entry: the pipeline falls back, flagged unvalidated)
        entry["evidence"] = evidence
        out["profiles"][key] = entry
        calibration[key] = notes
        if notes.get("rung"):
            decided[key] = notes["rung"]
    out["calibration"] = calibration
    out["generated_from"] = evidence
    coef, info = _fit_trust(results)
    out["trust_fit"] = info
    if coef is not None:
        out["trust"] = {"coef": coef, "fit": "least squares of per-file dev WER on trust features", "evidence": evidence}
        out["calibration"]["trust"] = {"status": "validated", "rows": info["rows"]}
    else:
        out["calibration"]["trust"] = {"status": "unvalidated", "reason": info["status"]}

    # Per-rung coefficients and the per-profile trust validation (Spearman on the dev rows of the rung the profile runs).
    # A profile without `trust_spearman` (too few rows) or with one < 0.7 reports "uncalibrated" at run time.
    by_rung: dict[str, dict] = {}
    for rung in sorted(set(decided.values())):
        rung_coef, rung_info = _fit_trust(results, (rung,))
        by_rung[rung] = {"fit": rung_info, **({"coef": rung_coef} if rung_coef is not None else {})}
    for key, rung in decided.items():
        rung_coef = by_rung[rung].get("coef")
        spearman, rows = _trust_spearman(recs, rung, keys[key], rung_coef)
        entry = out["profiles"][key]
        entry["trust_rung"] = rung
        if spearman is None:
            entry.pop("trust_spearman", None)
        else:
            entry["trust_spearman"] = spearman
        calibration[key]["trust_spearman"] = spearman
        calibration[key]["trust_rows"] = rows
    if by_rung and isinstance(out.get("trust"), dict):
        out["trust"]["by_rung"] = by_rung
    from transcribe.auto.profiles import validate_profiles

    validate_profiles(out)
    return out


def format_table(profiles: dict) -> str:
    rows = [("key", "engines", "pick", "verify", "refine", "status", "decision")]
    for key, entry in profiles["profiles"].items():
        cal = profiles.get("calibration", {}).get(key, {})
        rows.append((key, "+".join(entry["engines"]), str(entry["pick"]), str(entry["verify"]), str(entry["refine"]),
                     cal.get("status", "unvalidated"), cal.get("decision", cal.get("reason", ""))))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows]
    trust = profiles.get("trust")
    status = profiles.get("trust_fit", {}).get("status", "n/a")
    lines.append(f"trust coef: {json.dumps(trust['coef']) if trust else 'none'} (fit: {status})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--write", type=Path, default=None, help="write the table here (default: dry run)")
    ap.add_argument("--base", type=Path, default=None, help="current table (default: transcribe/profiles.json)")
    args = ap.parse_args(argv)
    results = json.loads(args.results.read_text())
    base = None
    if args.base:
        base = json.loads(args.base.read_text())
    slices = list(dict.fromkeys(r["slice"] for r in results.get("per_item", [])))
    out = make_profiles(results, default_slice_to_profile(slices), base=base, evidence=str(args.results))
    print(format_table(out))
    if args.write:
        args.write.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {args.write}")
    else:
        print("dry run: pass --write PATH to write")
    return 0


if __name__ == "__main__":
    sys.exit(main())

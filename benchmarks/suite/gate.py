"""Regression gate: compare a candidate results file against a base one.

    .venv/bin/python -m benchmarks.suite.gate --base base.json --cand cand.json [--split test]

Exit code 0 when `passes(report)`, else 1. Both files are `run_suite` outputs; each must hold exactly one system
(or name it with `--base-system` / `--cand-system`). Items are paired by id: only ids present in both files (same
split) count, so the systems may cover different items.

Report (`compare`):

    {"split", "n_items", "n_missing_in_cand", "slices": {slice: {"delta", "ci", "worse", "n", "n_base_only",
                                                                   "n_cand_only"}},
     "harm": {stage: {"base", "cand", "delta"}},   # harm / (help + harm + neutral); a stage missing on one side is 0
     "nospeech": {"base", "cand", "delta"} | None} # pooled hypothesis words per audio minute on empty-reference items

`delta` is the mean per-item WER difference (cand - base) and `ci` its paired 95 % bootstrap interval (n=2000,
seed 0); a slice is `worse` when the interval's lower bound is above 0. `n_base_only` counts items the base completed
that the candidate lacks (errored, skipped or missing); `n_cand_only` the reverse. `passes` is false when any slice is
worse, the candidate lacks any item the base completed, any stage's harm rate rises by more than 1 pp, or nospeech
insertions rise (a candidate that errors on items cannot pass on the few that remain).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmarks.suite import metrics as M

HARM_RISE_LIMIT = 0.01
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 0
_EPS = 1e-9


def _records(results: dict, split: str, system: str | None, role: str) -> dict[str, dict]:
    """`{item_id: record}` for the split (skipped/errored items dropped); one system only."""
    recs = [r for r in results.get("per_item", []) if r.get("split") == split
            and not r.get("skipped") and not r.get("error")]
    systems = list(dict.fromkeys(r["system"] for r in results.get("per_item", [])))
    if system is None:
        if len(systems) > 1:
            raise ValueError(f"{role} results hold several systems ({', '.join(systems)}); name one")
    else:
        recs = [r for r in recs if r["system"] == system]
    return {r["id"]: r for r in recs}


def _harm_rates(recs: list[dict]) -> dict[str, float]:
    totals: dict[str, list[int]] = {}
    for r in recs:
        for stage, eff in (r.get("edits") or {}).items():
            t = totals.setdefault(stage, [0, 0])
            t[0] += eff["harm"]
            t[1] += eff["help"] + eff["harm"] + eff["neutral"]
    return {stage: (harm / n if n else 0.0) for stage, (harm, n) in totals.items()}


def _ins_per_min(recs: list[dict]) -> float | None:
    silent = [r for r in recs if r.get("empty_ref") and r.get("ins_per_min") is not None]
    minutes = sum(r["duration"] for r in silent) / 60.0
    if not silent or not minutes:
        return None
    return sum(r["ins_per_min"] * r["duration"] / 60.0 for r in silent) / minutes


def compare(base: dict, cand: dict, split: str = "test", base_system: str | None = None,
            cand_system: str | None = None) -> dict:
    b = _records(base, split, base_system, "base")
    c = _records(cand, split, cand_system, "cand")
    shared = [i for i in b if i in c]
    base_only = [i for i in b if i not in c]
    cand_only = [i for i in c if i not in b]
    report: dict = {"split": split, "n_items": len(shared), "n_missing_in_cand": len(base_only),
                    "slices": {}, "harm": {}, "nospeech": None}

    by_slice: dict[str, list[float]] = {}
    for i in shared:
        if b[i].get("wer") is None or c[i].get("wer") is None:
            continue
        by_slice.setdefault(b[i]["slice"], []).append(c[i]["wer"] - b[i]["wer"])
    slice_names = list(dict.fromkeys([*by_slice, *(b[i]["slice"] for i in base_only),
                                      *(c[i]["slice"] for i in cand_only)]))
    for sl in slice_names:
        diffs = by_slice.get(sl, [])
        entry = {"delta": None, "ci": None, "worse": False, "n": len(diffs),
                 "n_base_only": sum(1 for i in base_only if b[i]["slice"] == sl),
                 "n_cand_only": sum(1 for i in cand_only if c[i]["slice"] == sl)}
        if diffs:
            lo, hi = M.bootstrap_ci(diffs, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
            entry.update(delta=sum(diffs) / len(diffs), ci=(lo, hi), worse=lo > 0)
        report["slices"][sl] = entry

    hb, hc = _harm_rates([b[i] for i in shared]), _harm_rates([c[i] for i in shared])
    for stage in sorted(set(hb) | set(hc)):
        rb, rc = hb.get(stage, 0.0), hc.get(stage, 0.0)
        report["harm"][stage] = {"base": rb, "cand": rc, "delta": rc - rb}

    nb, nc = _ins_per_min([b[i] for i in shared]), _ins_per_min([c[i] for i in shared])
    if nb is not None and nc is not None:
        report["nospeech"] = {"base": nb, "cand": nc, "delta": nc - nb}
    return report


def failures(report: dict) -> list[str]:
    """Human-readable reasons the report does not pass (empty when it passes)."""
    out = [f"slice {sl}: WER delta {v['delta']:+.4f} CI [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] is worse"
           for sl, v in report["slices"].items() if v["worse"]]
    out += [f"slice {sl}: candidate lacks {v['n_base_only']} item(s) the base completed (errored or missing)"
            for sl, v in report["slices"].items() if v["n_base_only"]]
    out += [f"stage {st}: harm rate {100 * v['base']:.1f}% -> {100 * v['cand']:.1f}% (+{100 * v['delta']:.1f} pp)"
            for st, v in report["harm"].items() if v["delta"] > HARM_RISE_LIMIT + _EPS]
    ns = report.get("nospeech")
    if ns and ns["delta"] > _EPS:
        out.append(f"nospeech insertions {ns['base']:.2f} -> {ns['cand']:.2f} per minute")
    return out


def passes(report: dict) -> bool:
    return not failures(report)


def format_report(report: dict) -> str:
    lines = [f"split={report['split']} paired items={report['n_items']}",
             f"{'slice':16} {'n':>4} {'dWER':>9}  95% CI"]
    for sl, v in sorted(report["slices"].items()):
        gap = f"  base-only={v['n_base_only']}" if v["n_base_only"] else ""
        gap += f"  cand-only={v['n_cand_only']}" if v["n_cand_only"] else ""
        if v["delta"] is None:
            lines.append(f"{sl:16} {v['n']:>4} {'-':>9}{gap}")
            continue
        lines.append(f"{sl:16} {v['n']:>4} {100 * v['delta']:+8.2f}%  [{100 * v['ci'][0]:+.2f}, "
                     f"{100 * v['ci'][1]:+.2f}]{'  WORSE' if v['worse'] else ''}{gap}")
    for st, v in report["harm"].items():
        lines.append(f"harm {st:10} {100 * v['base']:5.1f}% -> {100 * v['cand']:5.1f}% ({100 * v['delta']:+.1f} pp)")
    if report.get("nospeech"):
        ns = report["nospeech"]
        lines.append(f"nospeech ins/min {ns['base']:.2f} -> {ns['cand']:.2f}")
    problems = failures(report)
    lines.append("PASS" if not problems else "FAIL")
    lines += [f"  - {p}" for p in problems]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--cand", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=("dev", "test"))
    ap.add_argument("--base-system", default=None)
    ap.add_argument("--cand-system", default=None)
    args = ap.parse_args(argv)
    report = compare(json.loads(args.base.read_text()), json.loads(args.cand.read_text()), args.split,
                     args.base_system, args.cand_system)
    print(format_report(report))
    return 0 if passes(report) else 1


if __name__ == "__main__":
    sys.exit(main())

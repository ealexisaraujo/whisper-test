"""Speed + accuracy benchmark: EG2 (EmbeddingGemma 2, Core ML) vs mlx-whisper on LibriSpeech.

Each LibriSpeech chapter is concatenated into one long-form file (minutes long), so the
engines exercise their own windowing/stitching. Audio is decoded once up front and passed as
16 kHz float32 arrays, so timings cover transcription only. Model load and one warm-up file
are excluded from timing.

    python benchmarks/bench_asr.py --librispeech /path/LibriSpeech/test-clean \
        --engines eg2,eg2-ane,mlx-whisper --out benchmarks/results.json

`--refine` also passes each engine's segments through the refine stage (transcribe/refine.py,
Claude via the Claude Code CLI) and records the refined WER plus refine cost/time per engine.
Refine time is reported separately and is not part of the ASR wall time.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SR = 16000


def load_chapters(root: Path, limit: int) -> List[Tuple[str, np.ndarray, str]]:
    chapters = []
    for trans in sorted(root.glob("*/*/*.trans.txt")):
        audio, refs = [], []
        for line in trans.read_text().splitlines():
            uid, ref = line.split(" ", 1)
            samples, sr = sf.read(trans.parent / f"{uid}.flac", dtype="float32")
            assert sr == SR
            audio.append(samples)
            refs.append(ref)
        chapters.append((trans.parent.parent.name + "-" + trans.parent.name, np.concatenate(audio), " ".join(refs)))
        if limit and len(chapters) >= limit:
            break
    return chapters


# Each engine returns {"text", "segments", "raw"} so the refine stage can work on segments.
Engine = Callable[[np.ndarray], Dict[str, Any]]


def make_eg2(workers: int, gpu_encoder: bool = False) -> Engine:
    from transcribe.eg2 import EG2Engine, words_to_segments

    engine = EG2Engine(workers=workers, gpu_encoder=gpu_encoder, punctuate=True)

    def run(samples: np.ndarray) -> Dict[str, Any]:
        words = engine.transcribe(samples)
        tokens = engine.render(words)
        raw_words = [{"word": tok, "start": w.start, "end": w.end, "score": w.score} for w, tok in zip(words, tokens)]
        return {"text": " ".join(tokens), "segments": words_to_segments(words, tokens), "raw": {"words": raw_words}}

    return run


def make_mlx_whisper(model: str) -> Engine:
    import mlx_whisper

    from transcribe.backends import normalize_whisper_segments

    def run(samples: np.ndarray) -> Dict[str, Any]:
        out = mlx_whisper.transcribe(samples, path_or_hf_repo=model, language="en", verbose=None)
        return {"text": out["text"], "segments": normalize_whisper_segments(out.get("segments") or []), "raw": {}}

    return run


def refine_output(refiner: Any, out: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    from transcribe.models import BackendResult

    result = BackendResult(text=out["text"], segments=out["segments"], language="en", duration_sec=None, raw=out["raw"])
    refined = refiner.refine(result, {"language": "en", "notes": "Read-aloud audiobook chapter."})
    return refined.text, refined.raw["refine"]


ENGINES: Dict[str, Callable[[], Engine]] = {
    "eg2": lambda: make_eg2(workers=2),  # two pipelined ANE workers, scorer on GPU
    "eg2-1w": lambda: make_eg2(workers=1),
    "eg2-3w": lambda: make_eg2(workers=3),
    "eg2-gpuenc": lambda: make_eg2(workers=2, gpu_encoder=True),  # app layout: 2nd encoder on GPU
    "mlx-whisper": lambda: make_mlx_whisper("mlx-community/whisper-turbo"),
}


def machine() -> Dict[str, str]:
    def sysctl(key: str) -> str:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()

    return {
        "cpu": sysctl("machdep.cpu.brand_string"),
        "memory_gb": str(round(int(sysctl("hw.memsize") or 0) / 2**30)),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech", type=Path, required=True)
    parser.add_argument("--engines", default="eg2,mlx-whisper")
    parser.add_argument("--limit", type=int, default=0, help="number of chapters (0 = all)")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results.json"))
    parser.add_argument("--refine", action="store_true", help="also score each engine after the Claude refine stage")
    parser.add_argument("--refine-model", default=None, help="Claude model alias (default: TRANSCRIBE_REFINE_MODEL or sonnet)")
    args = parser.parse_args()

    import jiwer
    from whisper_normalizer.english import EnglishTextNormalizer

    normalize = EnglishTextNormalizer()
    refiner = None
    if args.refine:
        from transcribe.refine import ClaudeRefiner

        refiner = ClaudeRefiner(model=args.refine_model)
    chapters = load_chapters(args.librispeech, args.limit)
    total_audio = sum(len(a) for _, a, _ in chapters) / SR
    print(f"{len(chapters)} chapters, {total_audio / 3600:.2f} h audio", flush=True)

    results = {"machine": machine(), "dataset": str(args.librispeech), "chapters": len(chapters),
               "audio_seconds": total_audio, "engines": {}}
    for name in args.engines.split(","):
        t = time.perf_counter()
        run = ENGINES[name]()
        run(chapters[0][1])  # warm-up: compiles / loads weights
        load_s = time.perf_counter() - t

        refs, hyps, refined_hyps, per = [], [], [], []
        wall = 0.0
        refine_cost = refine_wall = 0.0
        refine_changed = refine_guarded = 0
        for cid, samples, ref in chapters:
            t = time.perf_counter()
            out = run(samples)
            dt = time.perf_counter() - t
            wall += dt
            r, h = normalize(ref), normalize(out["text"])
            refs.append(r)
            hyps.append(h)
            row = {"chapter": cid, "seconds": len(samples) / SR, "wall": dt, "wer": jiwer.wer(r, h)}
            if refiner is not None:
                refined_text, info = refine_output(refiner, out)
                rh = normalize(refined_text)
                refined_hyps.append(rh)
                refine_cost += info["cost_usd"]
                refine_wall += info["duration_s"]
                refine_changed += info["changed_segments"]
                refine_guarded += info["guarded_segments"]
                row.update({"refined_wer": jiwer.wer(r, rh), "refine_cost_usd": info["cost_usd"],
                            "refine_s": info["duration_s"]})
            per.append(row)
        m = jiwer.process_words(refs, hyps)
        summary = {
            "load_and_warmup_s": load_s,
            "wall_s": wall,
            "x_realtime": total_audio / wall,
            "seconds_per_audio_hour": wall / total_audio * 3600,
            "wer": m.wer,
            "substitutions": m.substitutions,
            "deletions": m.deletions,
            "insertions": m.insertions,
            "per_chapter": per,
            "sample": {"ref": refs[0][:400], "hyp": hyps[0][:400]},
        }
        if refiner is not None:
            rm = jiwer.process_words(refs, refined_hyps)
            summary["refine"] = {
                "model": refiner.model,
                "wer": rm.wer,
                "substitutions": rm.substitutions,
                "deletions": rm.deletions,
                "insertions": rm.insertions,
                "cost_usd": refine_cost,
                "cost_usd_per_audio_hour": refine_cost / total_audio * 3600,
                "wall_s": refine_wall,
                "changed_segments": refine_changed,
                "guarded_segments": refine_guarded,
                "sample": refined_hyps[0][:400],
            }
        results["engines"][name] = summary
        print(f"{name:12s} {summary['x_realtime']:7.1f}x RT  {summary['seconds_per_audio_hour']:6.1f} s/h  "
              f"WER {100 * m.wer:5.2f}%  (load+warmup {load_s:.0f}s)", flush=True)
        if refiner is not None:
            info = summary["refine"]
            print(f"{'  +refine':12s} WER {100 * info['wer']:5.2f}%  cost ${info['cost_usd']:.3f} "
                  f"(${info['cost_usd_per_audio_hour']:.3f}/h)  {info['wall_s']:.0f}s  "
                  f"changed {refine_changed} guarded {refine_guarded}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""Speed + accuracy benchmark: EG2 (EmbeddingGemma 2, Core ML) vs mlx-whisper on LibriSpeech.

Each LibriSpeech chapter is concatenated into one long-form file (minutes long), so the
engines exercise their own windowing/stitching. Audio is decoded once up front and passed as
16 kHz float32 arrays, so timings cover transcription only. Model load and one warm-up file
are excluded from timing.

    python benchmarks/bench_asr.py --librispeech /path/LibriSpeech/test-clean \
        --engines eg2,eg2-ane,mlx-whisper --out benchmarks/results.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

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


def make_eg2(workers: int, gpu_encoder: bool = False) -> Callable[[np.ndarray], str]:
    from transcribe.eg2 import EG2Engine

    engine = EG2Engine(workers=workers, gpu_encoder=gpu_encoder, punctuate=True)

    def run(samples: np.ndarray) -> str:
        return " ".join(engine.render(engine.transcribe(samples)))

    return run


def make_mlx_whisper(model: str) -> Callable[[np.ndarray], str]:
    import mlx_whisper

    def run(samples: np.ndarray) -> str:
        return mlx_whisper.transcribe(samples, path_or_hf_repo=model, language="en", verbose=None)["text"]

    return run


ENGINES: Dict[str, Callable[[], Callable[[np.ndarray], str]]] = {
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
    args = parser.parse_args()

    import jiwer
    from whisper_normalizer.english import EnglishTextNormalizer

    normalize = EnglishTextNormalizer()
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

        refs, hyps, per = [], [], []
        wall = 0.0
        for cid, samples, ref in chapters:
            t = time.perf_counter()
            hyp = run(samples)
            dt = time.perf_counter() - t
            wall += dt
            r, h = normalize(ref), normalize(hyp)
            refs.append(r)
            hyps.append(h)
            per.append({"chapter": cid, "seconds": len(samples) / SR, "wall": dt, "wer": jiwer.wer(r, h)})
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
        results["engines"][name] = summary
        print(f"{name:12s} {summary['x_realtime']:7.1f}x RT  {summary['seconds_per_audio_hour']:6.1f} s/h  "
              f"WER {100 * m.wer:5.2f}%  (load+warmup {load_s:.0f}s)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

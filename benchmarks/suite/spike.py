"""Environment spike for the `auto` pipeline: prove every model runs on this Mac and measure it.

Run from the repo root (each step is a subcommand; `asr` runs one engine on one clip in-process so
its peak RSS is clean; `all-asr` spawns one subprocess per engine/clip and writes the results JSON):

    .venv/bin/python -m benchmarks.suite.spike prep
    .venv/bin/python -m benchmarks.suite.spike asr --engine parakeet-v3 --clip en
    .venv/bin/python -m benchmarks.suite.spike all-asr
    .venv/bin/python -m benchmarks.suite.spike picker --model mlx-community/Qwen3-30B-A3B-4bit
    .venv/bin/python -m benchmarks.suite.spike score
    .venv/bin/python -m benchmarks.suite.spike vad

Results land in benchmarks/suite/spike-results.json; the decisions are written up in
docs/agents/auto-pipeline-spike.md. Audio lives under ~/.cache/whisper_translate/suite/spike/.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SR = 16000
SPIKE_DIR = Path.home() / ".cache" / "whisper_translate" / "suite" / "spike"
RESULTS = Path(__file__).with_name("spike-results.json")
# en_30s / es_30s are the paths later slow tests rely on; the two *_hard clips make the engine comparison less trivial.
CLIPS = {"en": "en_30s", "es": "es_30s", "en_hard": "en_hard_30s", "es_hard": "es_hard_30s"}
WHISPER_LARGE = "mlx-community/whisper-large-v3-mlx"
QWEN3_ASR = "mlx-community/Qwen3-ASR-1.7B-bf16"
PARAKEET = "mlx-community/parakeet-tdt-0.6b-v3"


def peak_rss_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)  # bytes on macOS


def load_clip(lang: str) -> Tuple[np.ndarray, str]:
    wav = SPIKE_DIR / f"{CLIPS[lang]}.wav"  # `lang` is the clip key
    samples, sr = sf.read(wav, dtype="float32")
    assert sr == SR and samples.ndim == 1, (sr, samples.shape)
    return samples, wav.with_suffix(".txt").read_text().strip()


def wer(ref: str, hyp: str, lang: str) -> float:
    import jiwer
    from whisper_normalizer.basic import BasicTextNormalizer
    from whisper_normalizer.english import EnglishTextNormalizer

    norm = EnglishTextNormalizer() if lang == "en" else BasicTextNormalizer()
    return round(jiwer.wer(norm(ref), norm(hyp)), 4)


# ----------------------------------------------------------------------------- prep (clips)

def _decode(audio: Dict[str, Any]) -> np.ndarray:
    """Decode a HF audio blob with soundfile (torchcodec is not usable with this torch build)."""
    samples, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    if sr != SR:
        import librosa

        samples = librosa.resample(samples, orig_sr=sr, target_sr=SR)
    return samples


def _build_clip(rows, ref_key: str, name: str, target_s: float = 30.0) -> None:
    """Concatenate utterances (0.3 s apart) until the clip is 27-33 s long; utterances that overshoot are skipped."""
    gap = np.zeros(int(0.3 * SR), dtype=np.float32)
    parts: List[np.ndarray] = []
    refs: List[str] = []
    total = 0.0
    for row in rows:
        audio = _decode(row["audio"])
        if total + len(audio) / SR > target_s + 3:
            continue
        parts += [audio, gap]
        refs.append(row[ref_key].strip())
        total = sum(len(p) for p in parts) / SR
        if total >= target_s - 3:
            break
    audio = np.concatenate(parts[:-1])
    SPIKE_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(SPIKE_DIR / f"{name}.wav", audio, SR, subtype="PCM_16")
    (SPIKE_DIR / f"{name}.txt").write_text(" ".join(refs) + "\n")
    print(f"{name}: {len(audio) / SR:.1f}s, {len(' '.join(refs).split())} ref words")


def prep() -> None:
    from datasets import Audio, load_dataset

    def rows(name: str, cfg: str):
        ds = load_dataset(name, cfg, split="test", streaming=True)
        return iter(ds.cast_column("audio", Audio(decode=False)))

    # LibriSpeech references are upper case with no punctuation; keep them (WER normalises anyway).
    _build_clip(rows("openslr/librispeech_asr", "clean"), "text", "en_30s")
    _build_clip(rows("google/fleurs", "es_419"), "raw_transcription", "es_30s")
    # Harder clips: LibriSpeech test-other (noisier, accented) and Multilingual LibriSpeech Spanish (old audiobooks).
    _build_clip(rows("openslr/librispeech_asr", "other"), "text", "en_hard_30s")
    _build_clip(rows("facebook/multilingual_librispeech", "spanish"), "transcript", "es_hard_30s")


# ----------------------------------------------------------------------------- ASR engines

def make_whisper(lang: str) -> Callable[[np.ndarray], Dict[str, Any]]:
    import mlx.core as mx
    import mlx_whisper
    from mlx_whisper.transcribe import ModelHolder

    ModelHolder.get_model(WHISPER_LARGE, dtype=mx.float16)  # load once so load time is measured on its own

    def run(samples: np.ndarray) -> Dict[str, Any]:
        out = mlx_whisper.transcribe(samples, path_or_hf_repo=WHISPER_LARGE, language=lang, word_timestamps=True,
                                     verbose=None, condition_on_previous_text=False)
        words = [w for s in out["segments"] for w in s.get("words", [])]
        return {"text": out["text"], "word_ts": bool(words), "word_probs": bool(words) and "probability" in words[0],
                "n_words": len(words)}

    return run


def make_qwen3(lang: str) -> Callable[[np.ndarray], Dict[str, Any]]:
    from mlx_audio.stt.utils import load_model

    model = load_model(QWEN3_ASR)

    def run(samples: np.ndarray) -> Dict[str, Any]:
        out = model.generate(samples, language={"en": "English", "es": "Spanish"}[lang])
        segs = getattr(out, "segments", None) or []
        return {"text": out.text, "word_ts": False, "word_probs": False, "n_words": len(segs),
                "extra": f"{len(segs)} segment(s); word timestamps need the separate Qwen3 forced aligner"}

    return run


def make_parakeet(lang: str) -> Callable[[np.ndarray], Dict[str, Any]]:
    from parakeet_mlx import from_pretrained

    model = from_pretrained(PARAKEET)

    def run(samples: np.ndarray) -> Dict[str, Any]:
        tmp = SPIKE_DIR / "_parakeet_in.wav"  # parakeet-mlx's public API takes a path
        sf.write(tmp, samples, SR, subtype="PCM_16")
        result = model.transcribe(tmp)
        tmp.unlink()
        toks = result.tokens
        return {"text": result.text, "word_ts": bool(toks), "word_probs": bool(toks) and hasattr(toks[0], "confidence"),
                "n_words": len(toks), "extra": "per-token `confidence` field"}

    return run


def make_eg2(lang: str) -> Callable[[np.ndarray], Dict[str, Any]]:
    from transcribe.eg2 import EG2Engine

    engine = EG2Engine(workers=2)  # first ANE load compiles for ~110 s: part of load time

    def run(samples: np.ndarray) -> Dict[str, Any]:
        words = engine.transcribe(samples)
        tokens = engine.render(words)
        return {"text": " ".join(tokens), "word_ts": True, "word_probs": hasattr(words[0], "score"),
                "n_words": len(words), "extra": "per-word CTC margin `score`"}

    return run


ENGINES: Dict[str, Callable[[str], Callable[[np.ndarray], Dict[str, Any]]]] = {
    "whisper-large-v3": make_whisper, "qwen3-asr-1.7b": make_qwen3, "parakeet-v3": make_parakeet, "eg2": make_eg2,
}


def mlx_peak_gb() -> float:
    import mlx.core as mx

    return round(mx.get_peak_memory() / 2**30, 2)


def asr(engine: str, clip: str) -> Dict[str, Any]:
    """Load once (timed), transcribe twice: run 1 pays kernel compilation, run 2 gives the steady-state x RT."""
    lang = clip[:2]
    samples, ref = load_clip(clip)
    duration = len(samples) / SR
    t0 = time.perf_counter()
    run = ENGINES[engine](lang)
    load_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    run(samples)
    first_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    out = run(samples)
    warm_s = time.perf_counter() - t2
    return {"engine": engine, "clip": clip, "wer": wer(ref, out["text"], lang), "load_s": round(load_s, 1),
            "x_rt_first": round(duration / first_s, 1), "x_rt": round(duration / warm_s, 1), "rss_gb": peak_rss_gb(),
            "mlx_peak_gb": mlx_peak_gb(), "word_ts": out["word_ts"], "word_probs": out["word_probs"],
            "n_words": out["n_words"], "extra": out.get("extra", ""), "hyp": out["text"].strip()}


def spawn(*args: str, timeout: int = 3600) -> Dict[str, Any]:
    cmd = [sys.executable, "-m", "benchmarks.suite.spike", *args]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    line = next((l for l in reversed(proc.stdout.splitlines()) if l.startswith("{")), None)
    if line is None:
        return {"error": (proc.stderr or proc.stdout)[-600:], "args": list(args)}
    return json.loads(line)


def save(section: str, rows: Any) -> None:
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data[section] = rows
    RESULTS.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def all_asr() -> None:
    rows = []
    for engine in ENGINES:
        for clip in CLIPS:
            if engine == "eg2" and clip.startswith("es"):
                continue  # English only
            row = spawn("asr", "--engine", engine, "--clip", clip)
            print(json.dumps({k: v for k, v in row.items() if k != "hyp"}))
            rows.append(row)
    save("asr", rows)


# ----------------------------------------------------------------------------- picker LLMs

PICKER_TEMPLATE = (
    "Two automatic transcripts of the same audio disagree on one word or phrase. Choose the one that is correct "
    "{lang}. Answer \"unclear\" only if neither is plausible.\nContext: \"{ctx}\"\nA: \"{a}\"\nB: \"{b}\"\n"
    'Answer with JSON only: {{"choice": "A"|"B"|"unclear"}}'
)
# (language, context with ____, A, B, expected). The first item is the one-shot check from the brief.
PICKER_ITEMS = [
    ("English", "and then we shipped the new ____ to every customer", "pipeline", "pipe line", "A"),
    ("English", "the model was trained with ____ on a cluster of eight GPUs", "pie torch", "PyTorch", "B"),
    ("English", "she walked ____ the river to reach the village", "alongside", "a long side", "A"),
    ("Spanish", "el equipo decidió ____ el proyecto la semana pasada", "a plazar", "aplazar", "B"),
    ("Spanish", "no sé ____ vamos a llegar a tiempo", "si", "sí", "A"),
    ("English", "he said the results were ____ expected", "better then", "better than", "B"),
    ("English", "we deployed the service on ____ behind an nginx proxy", "Cooper Nettie's", "Kubernetes", "B"),
    ("English", "the council will ____ the proposal on Tuesday", "vote on", "boat on", "A"),
    ("English", "I think ____ going to be late for the meeting", "their", "they're", "B"),
    ("English", "the paper by ____ introduced the transformer architecture", "Vaswani et al.", "Vase Wani et al.", "A"),
    ("English", "it was a ____ day, so we stayed inside", "rainy", "rain knee", "A"),
    ("Spanish", "ayer ____ al mercado con mi hermana", "fui", "fue", "A"),
    ("Spanish", "ellos ____ que la reunión empezará a las nueve", "dijeron", "dijo", "A"),
    ("Spanish", "quiero ____ a la playa este verano", "ir", "hir", "A"),
    ("Spanish", "el informe ____ publicado ayer por el ministerio", "fue", "fuel", "A"),
    ("Spanish", "no me gusta el ____ de esa película", "final", "fin al", "A"),
]


def _parse_choice(text: str) -> Any:
    text = text.strip()
    text = text[text.find("{"): text.rfind("}") + 1] if "{" in text else text
    try:
        choice = json.loads(text)["choice"]
    except Exception:
        return None
    return choice if choice in ("A", "B", "unclear") else None


def picker(model_id: str) -> Dict[str, Any]:
    from mlx_lm import load, stream_generate

    t0 = time.perf_counter()
    model, tok = load(model_id)
    load_s = time.perf_counter() - t0
    rows, tokens, seconds, prompt_tps = [], 0, 0.0, []
    for lang, ctx, a, b, expected in PICKER_ITEMS:
        user = PICKER_TEMPLATE.format(lang=f"in {lang}", ctx=ctx, a=a, b=b)
        messages = [{"role": "user", "content": user}]
        try:
            prompt = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(messages, add_generation_prompt=True)
        text, last = "", None
        for last in stream_generate(model, tok, prompt=prompt, max_tokens=48):
            text += last.text
        if rows:  # the first item pays kernel compilation; steady-state speed comes from the rest
            tokens += last.generation_tokens
            seconds += last.generation_tokens / last.generation_tps
            prompt_tps.append(last.prompt_tps)
        rows.append({"choice": _parse_choice(text), "expected": expected, "raw": text.strip()[:120]})
    return {"model": model_id, "load_s": round(load_s, 1), "tok_s": round(tokens / seconds, 1),
            "prompt_tok_s": round(sum(prompt_tps) / len(prompt_tps)), "rss_gb": peak_rss_gb(),
            "mlx_peak_gb": mlx_peak_gb(), "json_valid": sum(r["choice"] is not None for r in rows),
            "correct": sum(r["choice"] == r["expected"] for r in rows), "n": len(rows),
            "first_item_valid": rows[0]["choice"] is not None, "items": rows}


# ----------------------------------------------------------------------------- Whisper teacher forcing

def score_text(model: Any, mel: Any, tokenizer: Any, text: str) -> float:
    """Mean token log-prob of `text` given the audio, teacher-forced through the Whisper decoder."""
    import mlx.core as mx

    prefix = list(tokenizer.sot_sequence_including_notimestamps)
    ids = tokenizer.encode(" " + text.strip())
    tokens = mx.array([prefix + ids])
    features = model.embed_audio(mel[None])
    logits = model.logits(tokens[:, :-1], features)[0]
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    target = tokens[0, 1:]
    picked = logp[mx.arange(target.shape[0]), target]
    return float(picked[len(prefix) - 1:].mean())


def corrupt(text: str, n: int = 3) -> str:
    words = text.split()
    idx = np.linspace(2, len(words) - 3, n * 2).astype(int)
    for a, b in zip(idx[::2], idx[1::2]):
        words[a], words[b] = words[b], words[a]
    return " ".join(words)


def score() -> Dict[str, Any]:
    import mlx.core as mx
    from mlx_whisper.audio import log_mel_spectrogram, pad_or_trim
    from mlx_whisper.load_models import load_model
    from mlx_whisper.tokenizer import get_tokenizer
    from huggingface_hub import snapshot_download

    samples, ref = load_clip("en")
    model = load_model(snapshot_download(WHISPER_LARGE), dtype=mx.float16)
    tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages, language="en", task="transcribe")
    mel = pad_or_trim(log_mel_spectrogram(samples, n_mels=model.dims.n_mels), 3000, axis=-2).astype(mx.float16)
    good, bad = score_text(model, mel, tokenizer, ref.lower()), score_text(model, mel, tokenizer, corrupt(ref.lower()))
    return {"ref_logprob": round(good, 3), "corrupted_logprob": round(bad, 3), "ref_higher": good > bad}


# ----------------------------------------------------------------------------- Qwen3 forced aligner

def _norm_word(w: str) -> str:
    import re

    return re.sub(r"[^\w]", "", w.lower())


def _matched_start_diffs(a: List[Tuple[str, float]], b: List[Tuple[str, float]]) -> List[float]:
    """|start_a - start_b| for words that match in a difflib alignment of the two normalized word sequences."""
    import difflib

    wa, wb = [_norm_word(w) for w, _ in a], [_norm_word(w) for w, _ in b]
    diffs = []
    for blk in difflib.SequenceMatcher(None, wa, wb, autojunk=False).get_matching_blocks():
        diffs += [abs(a[blk.a + k][1] - b[blk.b + k][1]) for k in range(blk.size)]
    return diffs


def align() -> Dict[str, Any]:
    """Qwen3-ASR has no word timestamps. Align the Qwen3 *hypothesis* (not the reference) with the forced aligner
    and compare its word starts with Whisper's own word times on the same clip."""
    import gc

    import mlx.core as mx
    import mlx_whisper
    from mlx_audio.stt.utils import load_model

    out: Dict[str, Any] = {}
    for clip, language in (("en", "English"), ("es", "Spanish")):
        samples, ref = load_clip(clip)
        qwen = load_model(QWEN3_ASR)
        hyp = qwen.generate(samples, language=language).text.strip()
        del qwen
        gc.collect()
        mx.clear_cache()
        aligner = load_model("mlx-community/Qwen3-ForcedAligner-0.6B-bf16")
        t1 = time.perf_counter()
        items = aligner.generate(samples, hyp, language=language).items
        secs = time.perf_counter() - t1
        del aligner
        gc.collect()
        mx.clear_cache()
        whisper = mlx_whisper.transcribe(samples, path_or_hf_repo=WHISPER_LARGE, language=clip, word_timestamps=True,
                                         verbose=None, condition_on_previous_text=False)
        ww = [(w["word"], w["start"]) for seg in whisper["segments"] for w in seg["words"]]
        diffs = np.array(_matched_start_diffs([(i.text, i.start_time) for i in items], ww))
        out[clip] = {"hyp_words": len(hyp.split()), "aligned_words": len(items), "aligner_s": round(secs, 2),
                     "matched_words": len(diffs), "median_abs_start_diff_s": round(float(np.median(diffs)), 3),
                     "p90_abs_start_diff_s": round(float(np.percentile(diffs, 90)), 3),
                     "aligner_last_start": round(items[-1].start_time, 2), "aligner_last_end": round(items[-1].end_time, 2),
                     "whisper_last_word_end": round(whisper["segments"][-1]["words"][-1]["end"], 2),
                     "clip_s": round(len(samples) / SR, 1)}
    out["rss_gb"], out["mlx_peak_gb"] = peak_rss_gb(), mlx_peak_gb()
    return out


# ----------------------------------------------------------------------------- VAD / diarization

def vad() -> Dict[str, Any]:
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    samples, _ = load_clip("en")
    audio = np.concatenate([samples, np.zeros(10 * SR, dtype=np.float32)])
    spans = get_speech_timestamps(torch.from_numpy(audio), load_silero_vad(), sampling_rate=SR, return_seconds=True)
    speech_end = len(samples) / SR
    out = {"clip_s": round(speech_end, 1), "n_spans": len(spans), "first_start": spans[0]["start"],
           "last_end": spans[-1]["end"], "excludes_silence": spans[-1]["end"] <= speech_end + 0.5}
    try:
        import pyannote.audio

        out["pyannote_import"] = pyannote.audio.__version__
    except Exception as exc:  # noqa: BLE001 - report whatever breaks the import
        out["pyannote_import"] = f"FAILED: {type(exc).__name__}: {str(exc)[:160]}"
    out["hf_token_set"] = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    return out


# ----------------------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    a = sub.add_parser("asr")
    a.add_argument("--engine", choices=list(ENGINES), required=True)
    a.add_argument("--clip", choices=list(CLIPS), required=True)
    sub.add_parser("all-asr")
    p = sub.add_parser("picker")
    p.add_argument("--model", required=True)
    sub.add_parser("score")
    sub.add_parser("vad")
    sub.add_parser("align")
    args = ap.parse_args()

    if args.cmd == "prep":
        prep()
    elif args.cmd == "all-asr":
        all_asr()
    else:
        result = {"asr": lambda: asr(args.engine, args.clip), "picker": lambda: picker(args.model),
                  "score": score, "vad": vad, "align": align}[args.cmd]()
        if args.cmd != "asr":  # `all-asr` and the per-step commands persist to the results JSON
            save(f"picker:{args.model}" if args.cmd == "picker" else args.cmd, result)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

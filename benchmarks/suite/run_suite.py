"""Suite runner: score transcription systems on the eval manifests.

    .venv/bin/python -m benchmarks.suite.run_suite --systems mlx-whisper-turbo,eg2 --manifest smoke \
        --out benchmarks/suite/results/baseline-smoke.json

A *system* is anything with a `name` and `run(item) -> dict`. The returned dict is
`{"text": str, "words": list[dict] | None, "segments": list[dict], "meta": dict}` where

- `words` (optional): `{"text", "start", "end", "unclear": bool, "speaker"}`; scored with `unclear_quality`.
- `segments`: `{"start", "end", "speaker", "text"}`; when the item has reference speakers and any segment carries
  a speaker label, DER is computed.
- `meta["stages"]` (optional): `{stage_name: {"before": str, "after": str}}`; each stage is scored with
  `edit_effects` (help/harm/neutral regions) and aggregated per stage name.

Optional system attributes: `cache_key()` (string folded into each cache entry's fingerprint with the item language
and audio size/mtime; "" for the built-in backends, a config/version hash for pipeline systems; legacy entries with no
fingerprint are accepted only when it is ""), `languages` (set of item languages the system supports, None = all; items outside it
are recorded as `"skipped": "language"`), `warmup()` (called once, untimed, before the first uncached item so model
loading does not pollute xrt) and `close()` (called after the system's last item to release models).

Systems run one after another (all items of one system, then the next) so only one model set is resident at a time.
Outputs are cached per system and item under `cache_dir/<system>/<item_id>.json` (text, words, segments, meta, wall
seconds, peak RSS), so reruns and interrupted runs resume; scores are recomputed from the cache every time.

Aggregation is corpus-level: WER/CER pool errors over words/characters, xrt pools audio seconds over wall seconds,
`ins_per_min` pools hypothesis words over audio minutes for reference-less (no-speech) items, DER is
duration-weighted, `unclear` precision/recall/density pool the flagged/error/word counts over items (None when a denominator is 0). `item_wer` maps item id to its
WER (for paired bootstrap across systems). `peak_rss_mb` is the process high-water mark (`ru_maxrss`) after the
item, so it never decreases within one process and later systems inherit earlier ones' peak.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import platform
import re
import resource
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from benchmarks.suite import metrics as M

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "whisper_translate" / "suite" / "cache"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BUILTIN_SYSTEMS = ("mlx-whisper-turbo", "eg2")
# Ablation ladder of `--backend auto`: stage flags per system (engines are resolved per item, see AutoSystem).
AUTO_SYSTEMS = ("auto:single-a", "auto:single-b", "auto:fused", "auto:+pick", "auto:+verify", "auto:+refine")
_AUTO_SPECS: dict[str, dict] = {
    "auto:single-a": {"engines": "a", "pick": False, "verify": False, "refine": False},
    "auto:single-b": {"engines": "b", "pick": False, "verify": False, "refine": False},
    "auto:fused": {"engines": "ab", "pick": False, "verify": False, "refine": False},
    "auto:+pick": {"engines": "ab", "pick": True, "verify": False, "refine": False},
    "auto:+verify": {"engines": "ab", "pick": True, "verify": True, "refine": False},
    "auto:+refine": {"engines": "ab", "pick": True, "verify": True, "refine": True},
}
# Engine pair of the ablation ladder when neither the language entry nor `default` holds two engines (Task 0 decisions).
FROZEN_PAIRS = {"en": ("qwen3-asr-1.7b", "whisper-large-v3"), "es": ("qwen3-asr-1.7b", "whisper-large-v3"),
                "other": ("whisper-large-v3", "qwen3-asr-1.7b")}


def lang_key_of(key_or_lang: str) -> str:
    """`en` / `es` / `other` for an item language or a profile key such as `en-noisy`."""
    lang = key_or_lang.split("-", 1)[0]
    return lang if lang in ("en", "es") else "other"


def ladder_pair(profiles: dict, key_or_lang: str) -> list[str]:
    """Engine pair (A, B) the ablation ladder compares for a language or profile key.

    The language entry if it holds two engines, else `default` (make_profiles never rewrites it), else
    `FROZEN_PAIRS`. A calibrated single-engine entry therefore never collapses the ladder; make_profiles uses the same
    rule to map its single-a/single-b decision back to an engine id.
    """
    lang = lang_key_of(key_or_lang)
    for entry in (profiles["profiles"].get(lang), profiles.get("default")):
        engines = list((entry or {}).get("engines") or [])
        if len(engines) >= 2:
            return engines[:2]
    return list(FROZEN_PAIRS[lang])


# fusion stages in pipeline order; a stage's edit effect compares its text with the previous stage's that ran
_STAGE_CHAIN = ("fused", "pick", "verify", "refine")


class SystemUnderTest(Protocol):
    name: str

    def run(self, item: dict) -> dict: ...


# --------------------------------------------------------------------------
# Built-in systems
# --------------------------------------------------------------------------


class BackendSystem:
    """Adapter over a `TranscriptionBackend`: direct `backend.transcribe`, no chunking, no refine."""

    def __init__(self, name: str, factory, languages: set[str] | None = None):
        self.name = name
        self.languages = languages
        self._factory = factory
        self._backend = None

    def _get(self):
        if self._backend is None:
            self._backend = self._factory()
        return self._backend

    def _call(self, audio: Path, lang: str | None):
        return self._get().transcribe(audio, lang, None, 0, {"progress_log_seconds": 20})

    def warmup(self) -> None:
        """Load models on two seconds of silence so the first real item is timed fairly."""
        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "warmup.wav"
            sf.write(str(path), np.zeros(32000, dtype="float32"), 16000)
            try:
                self._call(path, None)
            except Exception as exc:  # noqa: BLE001 - warmup is best effort
                _log(f"[{self.name}] warmup failed (ignored): {exc}")

    def run(self, item: dict) -> dict:
        lang = None if item["lang"] == "none" else item["lang"]
        result = self._call(Path(item["audio"]), lang)
        return {"text": result.text, "words": None, "segments": result.segments,
                "meta": {"language": result.language}}

    def close(self) -> None:
        self._backend = None
        gc.collect()
        try:  # drop mlx_whisper's cached model and Metal buffers
            from mlx_whisper.transcribe import ModelHolder

            ModelHolder.model = None
            ModelHolder.model_path = None
            import mlx.core as mx

            mx.clear_cache()
        except Exception:  # noqa: BLE001 - mlx may be absent or API may differ
            pass


class AutoSystem:
    """`--backend auto` with one rung of the ablation ladder (see `AUTO_SYSTEMS`).

    Each item runs `run_auto` with a profiles table whose every entry is overridden: engines A/B come from
    `transcribe/profiles.json` for the item's language key (en/es; `other` for none/others) and the pick/verify/refine
    flags come from the system. Engine outputs, VAD and probe are cached by the pipeline's StageCache, so the six
    systems share one ASR run per engine per item (their `seconds`/xrt then reflect only the later stages).
    """

    languages = None

    def __init__(self, name: str, factories: dict | None = None, base_profiles: dict | None = None):
        if name not in _AUTO_SPECS:
            raise ValueError(f"unknown auto system {name!r}; known: {', '.join(AUTO_SYSTEMS)}")
        self.name = name
        self.spec = _AUTO_SPECS[name]
        self.factories = factories
        self._base_profiles = base_profiles

    # -- profiles override

    def base_profiles(self) -> dict:
        if self._base_profiles is not None:
            return self._base_profiles
        from transcribe.auto.profiles import load_profiles

        return load_profiles()

    @staticmethod
    def _lang_key(lang: str) -> str:
        return lang_key_of(lang)

    def _engines(self, base: dict, lang_key: str) -> list[str]:
        engine_a, engine_b = ladder_pair(base, lang_key)  # never collapses after a single-engine calibration
        return {"a": [engine_a], "b": [engine_b], "ab": [engine_a, engine_b]}[self.spec["engines"]]

    def profiles_for(self, item: dict) -> dict:
        """Profiles table for the item: every entry (and `default`) overridden by this system's configuration."""
        base = copy.deepcopy(self.base_profiles())
        engines = self._engines(base, self._lang_key(item["lang"]))
        for entry in [*base["profiles"].values(), base["default"]]:
            entry.update(engines=list(engines), pick=self.spec["pick"], verify=self.spec["verify"],
                         refine=self.spec["refine"])
        return base

    def cache_key(self) -> str:
        """Hash of the system name, the full overridden profiles table, HF token presence (diarization on/off), the
        pipeline cache version and the picker/scorer/refine model ids."""
        from transcribe.auto import cache as auto_cache, picker, verify
        from transcribe.constants import DEFAULT_REFINE_MODEL

        # the full overridden table per language key (engines, flags, margin, solid_min_prob, trust coef, ...)
        payload = {
            "system": self.name, "spec": self.spec,
            "profiles": {k: self.profiles_for({"lang": k}) for k in ("en", "es", "other")},
            "diarization": bool(os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")),
            "cache_version": auto_cache.CACHE_VERSION,
            "picker": picker.PICKER_MODEL, "scorer": verify.DEFAULT_REPO,
            "refine": os.getenv("TRANSCRIBE_REFINE_MODEL") or DEFAULT_REFINE_MODEL,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    # -- run

    def run(self, item: dict) -> dict:
        from transcribe.auto import pipeline

        title = (item.get("meta") or {}).get("title")
        metadata = {"title": title} if title else {}
        speakers = {t.get("speaker") for t in item.get("speakers") or [] if t.get("speaker") is not None}
        result, meta = pipeline.run_auto(
            Path(item["audio"]), metadata,
            language=None if item["lang"] == "none" else item["lang"],
            num_speakers=len(speakers) or None,
            refine=True, profiles=self.profiles_for(item), factories=self.factories)
        words = meta.get("words") or []
        stage_texts = meta.get("stage_texts") or {}
        stages: dict[str, dict] = {}
        previous = stage_texts.get("fused")
        for stage in _STAGE_CHAIN[1:]:
            if stage not in stage_texts:
                continue
            if previous is not None:
                stages[stage] = {"before": previous, "after": stage_texts[stage]}
            previous = stage_texts[stage]
        return {
            "text": " ".join(w["text"] for w in words),  # best guess; the rendered text carries [unclear] markers
            "words": words, "segments": result.segments,
            "meta": {"stages": stages, "stage_texts": stage_texts, "trust": meta.get("trust"),
                     "pipeline_stages": meta.get("stages"), "engines": meta.get("engines"),
                     "profile_key": meta.get("profile_key"), "language": meta.get("language"),
                     "speakers": meta.get("speakers"), "refine": meta.get("refine"), "notes": meta.get("notes"),
                     "no_speech": meta.get("no_speech")},
        }

    def close(self) -> None:
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:  # noqa: BLE001 - mlx may be absent or API may differ
            pass


PICK_CACHE_DIR = DEFAULT_CACHE_DIR.parent / "picker"
# Bump when the picker's decoding or prompt plumbing changes, or the scorer's scoring logic changes: memo entries
# are keyed by these versions (plus the pipeline CACHE_VERSION), so old answers/scores stop matching.
PICKER_MEMO_VERSION = 1
SCORER_MEMO_VERSION = 1
PICKER_DECODING = {"temp": 0.0, "chat_template": True, "enable_thinking": False}


def _picker_memo_key(repo: str, prompt: str, max_tokens: int) -> str:
    from transcribe.auto import cache as auto_cache

    return hashlib.sha256(json.dumps([PICKER_MEMO_VERSION, auto_cache.CACHE_VERSION, PICKER_DECODING, repo, prompt,
                                      max_tokens], sort_keys=True).encode()).hexdigest()


def _scorer_memo_key(repo: str, digest: str, start: float, end: float, text: str, lang) -> str:
    from transcribe.auto import cache as auto_cache, verify

    logic = [SCORER_MEMO_VERSION, auto_cache.CACHE_VERSION, verify.WINDOW_PAD_S, verify.MAX_WINDOW_S,
             verify.DEFAULT_LANG, verify._MAX_TEXT_TOKENS]
    return hashlib.sha256(json.dumps([logic, repo, digest, round(float(start), 4), round(float(end), 4), text,
                                      lang]).encode()).hexdigest()


def cached_picker_factory(cache_dir: Path = PICK_CACHE_DIR, refresh: bool = False):
    """Picker factory whose LLM answers are memoized on disk by (memo versions, decoding, repo, prompt, max_tokens).

    `refresh=True` never reads the memo (answers are recomputed and rewritten), like run_suite's `--refresh`.

    The picker decodes at temperature 0, so `+pick`, `+verify` and `+refine` ask it identical prompts for the same
    item; the memo makes the later rungs reuse the first rung's answers instead of reloading the 13 GB model. The model
    still loads lazily (only on a miss) and `release()` still frees it, so memory behaves as in production.
    """
    from transcribe.auto.picker import MlxPicker

    class CachedPicker(MlxPicker):
        def _run_model(self, prompt: str, max_tokens: int) -> str:
            key = _picker_memo_key(self.repo, prompt, max_tokens)
            path = Path(cache_dir) / f"{key}.json"
            if not refresh:
                try:
                    return json.loads(path.read_text())["text"]
                except (OSError, ValueError, KeyError):
                    pass
            text = super()._run_model(prompt, max_tokens)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
                with os.fdopen(fd, "w") as handle:
                    json.dump({"text": text}, handle, ensure_ascii=False)
                os.replace(tmp, path)
            except OSError:  # a memo that cannot write must never fail the run
                pass
            return text

    return CachedPicker


SCORE_CACHE_DIR = DEFAULT_CACHE_DIR.parent / "scorer"


def cached_scorer_factory(cache_dir: Path = SCORE_CACHE_DIR, refresh: bool = False):
    """Verify-scorer factory whose `score_detail` results are memoized on disk.

    Key: (memo versions + scoring constants, repo, sha256 of the samples, window start/end, text, lang); `refresh=True`
    never reads the memo. `+verify` and `+refine` (and margin sweeps)
    score identical windows, so the later systems reuse the scores instead of re-running Whisper. The model loads
    only on a miss; `release()` still frees it.
    """
    from transcribe.auto.verify import WhisperScorer

    class CachedScorer(WhisperScorer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._digests: dict[int, tuple] = {}

        def _digest(self, samples) -> str:
            hit = self._digests.get(id(samples))
            if hit is None or hit[0] is not samples:  # keep a reference so the id can't be reused
                hit = (samples, hashlib.sha256(memoryview(samples).cast("B")).hexdigest())
                self._digests[id(samples)] = hit
            return hit[1]

        def score_detail(self, samples, start, end, text, lang):
            import numpy as np

            samples_c = np.ascontiguousarray(samples)
            key = _scorer_memo_key(self.repo, self._digest(samples_c), start, end, text, lang)
            path = Path(cache_dir) / key[:2] / f"{key}.json"
            if not refresh:
                try:
                    data = json.loads(path.read_text())
                    return float(data["total"]), int(data["count"])
                except (OSError, ValueError, KeyError):
                    pass
            total, count = super().score_detail(samples, start, end, text, lang)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
                with os.fdopen(fd, "w") as handle:
                    json.dump({"total": total, "count": count}, handle)
                os.replace(tmp, path)
            except OSError:
                pass
            return total, count

        def release(self) -> None:
            self._digests.clear()
            super().release()

    return CachedScorer


def build_system(name: str, refresh: bool = False) -> SystemUnderTest:
    """`refresh=True` also bypasses the auto systems' picker/scorer memos (run_suite `--refresh`)."""
    if name.startswith("auto:"):
        return AutoSystem(name, factories={"picker": cached_picker_factory(refresh=refresh),
                                           "scorer": cached_scorer_factory(refresh=refresh)})
    if name == "mlx-whisper-turbo":
        def factory():
            from transcribe.backends import MLXWhisperBackend
            return MLXWhisperBackend("mlx-community/whisper-turbo")
        return BackendSystem(name, factory)
    if name == "eg2":
        def factory():
            from transcribe.backends import EG2Backend
            return EG2Backend("eg2demo-v5")
        # English only; language-less items (no-speech clips) still run so hallucinations are measured
        return BackendSystem(name, factory, languages={"en", "none"})
    raise ValueError(f"unknown system {name!r}; built-ins: {', '.join(BUILTIN_SYSTEMS + AUTO_SYSTEMS)}")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024, 1)


def _sysctl(key: str) -> str | None:
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def machine_info() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()}
    chip = _sysctl("machdep.cpu.brand_string")
    mem = _sysctl("hw.memsize")
    if chip:
        info["chip"] = chip
    if mem and mem.isdigit():
        info["memory_gb"] = round(int(mem) / 1024 ** 3, 1)
    return info


def _safe(name: str) -> str:
    """Reversible, collision-free file name: safe characters pass through, others become %XX (so `:` is `%3A`)."""
    return "".join(c if (c.isascii() and (c.isalnum() or c in "._-")) else "".join(f"%{b:02X}" for b in c.encode())
                   for c in name)


def _cache_path(cache_dir: Path, system: str, item_id: str) -> Path:
    return cache_dir / _safe(system) / f"{_safe(item_id)}.json"


def _cache_key(system: Any) -> str:
    fn = getattr(system, "cache_key", None)
    return str(fn()) if callable(fn) else ""


def fingerprint(item: dict, system: Any) -> dict:
    """What a cached output depends on besides the item id: language, the audio file, the system config."""
    try:
        st = Path(item["audio"]).stat()
        size, mtime = st.st_size, st.st_mtime_ns
    except OSError:
        size = mtime = None
    return {"lang": item["lang"], "audio_size": size, "audio_mtime_ns": mtime, "cache_key": _cache_key(system)}


def _load_cache(path: Path, item: dict, system: Any) -> dict | None:
    """Cached run for the item, or None when missing/stale.

    Entries carry a fingerprint that must match. Legacy entries (no fingerprint) are accepted only when the
    system's cache_key() is "" (built-in backends) and the audio duration still matches.
    """
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if abs(float(data.get("duration", -1)) - float(item["duration"])) > 0.01:
        return None  # audio changed since the cached run
    stored = data.get("fingerprint")
    if stored is None:
        return data if _cache_key(system) == "" else None
    return data if stored == fingerprint(item, system) else None


def _json_default(obj: Any) -> Any:
    """Serialize numpy scalars/arrays that leak into words/segments/meta."""
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def select_items(manifest: list[dict], split: str = "all", slices: list[str] | None = None) -> list[dict]:
    items = [i for i in manifest if split == "all" or i["split"] == split]
    if slices:
        items = [i for i in items if i["slice"] in slices]
    return items


def load_manifest(name: str) -> list[dict]:
    """`smoke` or `full` (every slice manifest, in SLICE_ORDER)."""
    from benchmarks.suite import datasets

    if name == "smoke":
        items = datasets.read_manifest("smoke")
        if not items:
            raise SystemExit("smoke manifest missing or its audio is not on disk; run benchmarks.suite.datasets prepare")
        return items
    if name != "full":
        raise SystemExit(f"unknown manifest {name!r} (smoke|full)")
    items: list[dict] = []
    for slice_name in datasets.SLICE_ORDER:
        part = datasets.read_manifest(slice_name)
        if not part:
            _log(f"warning: slice {slice_name} manifest missing or incomplete audio; skipped")
        items.extend(part)
    if not items:
        raise SystemExit("no slice manifests available")
    return items


# --------------------------------------------------------------------------
# Per-item scoring
# --------------------------------------------------------------------------


def score_item(item: dict, system: str, out: dict) -> dict:
    """Per-item record from a system output (`out` is the cached/raw run dict)."""
    lang, ref, text = item["lang"], item["reference"], out.get("text") or ""
    seconds = float(out.get("seconds") or 0.0)
    rec: dict[str, Any] = {
        "system": system, "id": item["id"], "slice": item["slice"], "split": item["split"], "lang": lang,
        "condition": item.get("condition"), "duration": item["duration"], "seconds": seconds,
        "xrt": item["duration"] / seconds if seconds > 0 else None,
        "peak_rss_mb": out.get("peak_rss_mb"), "cached": bool(out.get("cached")),
        "reference": ref, "text": text, "has_speakers": bool(item.get("speakers")),
    }
    empty_ref = not M.normalize(ref, lang)
    rec["empty_ref"] = empty_ref
    if empty_ref:
        rec["ins_per_min"] = M.insertions_per_minute(M.normalize(text, lang), item["duration"])
        rec["wer"] = None
    else:
        r = M.wer_cer([ref], [text], lang)
        rec.update({"wer": r["wer"], "cer": r["cer"], "sub": r["sub"], "del": r["del"], "ins": r["ins"],
                    "words": r["words"], "chars": r["chars"]})
        rec["entity"] = M.entity_number_error(ref, text, lang)
        trust = ((out.get("meta") or {}).get("trust") or {}).get("features")
        if trust:
            rec["trust_features"] = trust  # make_profiles fits the trust coefficients on these
        stages = (out.get("meta") or {}).get("stages") or {}
        if stages:
            rec["edits"] = {name: M.edit_effects(ref, st["before"], st["after"], lang) for name, st in stages.items()}
        words = out.get("words")
        if words:
            q = M.unclear_quality(ref, words, lang)
            rec["unclear"] = q
    turns = [{"start": s["start"], "end": s["end"], "speaker": s["speaker"]}
             for s in out.get("segments") or [] if s.get("speaker") is not None]
    if item.get("speakers") and turns:
        rec["der"] = M.der(item["speakers"], turns)
    return rec


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def _aggregate(recs: list[dict]) -> dict:
    """Metrics for one group of per-item records (one system, one slice/split/overall)."""
    ran = [r for r in recs if not r.get("skipped") and not r.get("error")]
    speech = [r for r in ran if not r["empty_ref"]]
    out: dict[str, Any] = {
        "n": len(ran), "n_skipped": sum(1 for r in recs if r.get("skipped")),
        "n_errors": sum(1 for r in recs if r.get("error")),
    }
    sub = sum(r["sub"] for r in speech)
    dele = sum(r["del"] for r in speech)
    ins = sum(r["ins"] for r in speech)
    words = sum(r["words"] for r in speech)
    chars = sum(r["chars"] for r in speech)
    cer_err = sum(r["cer"] * r["chars"] for r in speech)
    out.update({"wer": (sub + dele + ins) / words if words else None, "cer": cer_err / chars if chars else None,
                "sub": sub, "del": dele, "ins": ins, "words": words})
    ent_total = sum(r["entity"]["total"] for r in speech)
    ent_missed = sum(r["entity"]["missed"] for r in speech)
    out["entity_err"] = ent_missed / ent_total if ent_total else None
    out["entity_total"] = ent_total
    out["item_wer"] = {r["id"]: r["wer"] for r in speech}

    silent = [r for r in ran if r["empty_ref"]]
    if silent:
        minutes = sum(r["duration"] for r in silent) / 60.0
        total_words = sum(r["ins_per_min"] * r["duration"] / 60.0 for r in silent)
        out["ins_per_min"] = total_words / minutes if minutes else 0.0

    audio = sum(r["duration"] for r in ran)
    wall = sum(r["seconds"] for r in ran)
    out["audio_seconds"] = audio
    out["seconds"] = wall
    out["xrt"] = audio / wall if wall > 0 else None
    rss = [r["peak_rss_mb"] for r in ran if r.get("peak_rss_mb")]
    out["peak_rss_mb"] = max(rss) if rss else None

    edits: dict[str, dict[str, int]] = {}
    for r in speech:
        for stage, eff in (r.get("edits") or {}).items():
            tot = edits.setdefault(stage, {"help": 0, "harm": 0, "neutral": 0})
            for k in tot:
                tot[k] += eff[k]
    if edits:
        out["edits"] = edits

    unclear = [r["unclear"] for r in speech if r.get("unclear")]
    if unclear:
        flagged = sum(u["flagged"] for u in unclear)
        hits = sum(u["flagged_errors"] for u in unclear)
        errors = sum(u["errors"] for u in unclear)
        n_words = sum(u["words"] for u in unclear)
        out["unclear"] = {
            "precision": hits / flagged if flagged else None,
            "recall": hits / errors if errors else None,
            "density": flagged / n_words if n_words else None,
            "flagged": flagged, "flagged_errors": hits, "errors": errors, "words": n_words,
        }

    ders = [r for r in ran if r.get("der") is not None]
    if ders:
        total = sum(r["duration"] for r in ders)
        out["der"] = sum(r["der"] * r["duration"] for r in ders) / total if total else None
    der_missing = sum(1 for r in ran if r.get("has_speakers") and r.get("der") is None)
    if der_missing:
        out["der_missing"] = der_missing  # items with reference speakers whose system returned no speaker labels
    return out


def aggregate(records: list[dict]) -> tuple[dict, dict]:
    """(per_slice {slice: {system: metrics}}, per_system {system: metrics}); each metrics has `splits`."""

    def with_splits(recs: list[dict]) -> dict:
        m = _aggregate(recs)
        m["splits"] = {sp: _aggregate([r for r in recs if r["split"] == sp])
                       for sp in sorted({r["split"] for r in recs})}
        return m

    per_slice: dict[str, dict] = {}
    per_system: dict[str, dict] = {}
    slices = list(dict.fromkeys(r["slice"] for r in records))
    systems = list(dict.fromkeys(r["system"] for r in records))
    for sl in slices:
        per_slice[sl] = {s: with_splits([r for r in records if r["slice"] == sl and r["system"] == s])
                         for s in systems if any(r["slice"] == sl and r["system"] == s for r in records)}
    for s in systems:
        per_system[s] = with_splits([r for r in records if r["system"] == s])
    return per_slice, per_system


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def _log_item(system: Any, idx: int, todo: int, rec: dict) -> None:
    wer = rec["wer"]
    _log(f"[{system.name}] {idx}/{todo} {rec['id']} "
         f"wer={'-' if wer is None else f'{wer:.3f}'} xrt={rec['xrt'] or 0:.1f}"
         f"{' (cached)' if rec['cached'] else ''}")


def run_suite(systems: list[SystemUnderTest], manifest: list[dict], out_path: Path, cache_dir: Path,
              config: dict | None = None, refresh: bool = False) -> dict:
    """Run every system over the manifest. `refresh=True` ignores cached outputs (and overwrites them)."""
    out_path, cache_dir = Path(out_path), Path(cache_dir)
    records: list[dict] = []
    for system in systems:
        langs = getattr(system, "languages", None)
        warmed = False
        warm_error: str | None = None
        todo = len(manifest)
        _log(f"[{system.name}] {todo} items")
        for idx, item in enumerate(manifest, 1):
            base = {"system": system.name, "id": item["id"], "slice": item["slice"], "split": item["split"],
                    "lang": item["lang"], "duration": item["duration"]}
            if langs is not None and item["lang"] not in langs:
                records.append({**base, "skipped": "language"})
                continue
            try:
                path = _cache_path(cache_dir, system.name, item["id"])
                out = None if refresh else _load_cache(path, item, system)
                if out is not None:
                    out["cached"] = True
                else:
                    if not warmed and hasattr(system, "warmup"):
                        warmed = True
                        try:
                            system.warmup()
                        except Exception as exc:  # noqa: BLE001
                            warm_error = f"warmup failed: {type(exc).__name__}: {exc}"
                            _log(f"[{system.name}] {warm_error}")
                    warmed = True
                    if warm_error:
                        raise RuntimeError(warm_error)
                    t0 = time.perf_counter()
                    raw = system.run(item)
                    seconds = time.perf_counter() - t0
                    entry = {"text": raw.get("text") or "", "words": raw.get("words"),
                             "segments": raw.get("segments") or [], "meta": raw.get("meta") or {},
                             "seconds": seconds, "peak_rss_mb": _peak_rss_mb(), "duration": item["duration"],
                             "fingerprint": fingerprint(item, system)}
                    payload = json.dumps(entry, ensure_ascii=False, default=_json_default)
                    out = json.loads(payload)  # score exactly what the cache would hold
                    out["cached"] = False
                    rec = score_item(item, system.name, out)  # score before caching: a scoring failure is retried
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(payload)
                    records.append(rec)
                    _log_item(system, idx, todo, rec)
                    continue
                rec = score_item(item, system.name, out)
            except Exception as exc:  # noqa: BLE001 - one bad item must not abort a multi-hour run
                _log(f"[{system.name}] {idx}/{todo} {item['id']} ERROR {type(exc).__name__}: {exc}")
                records.append({**base, "error": f"{type(exc).__name__}: {exc}"})
                continue
            records.append(rec)
            _log_item(system, idx, todo, rec)
        if hasattr(system, "close"):
            try:
                system.close()
            except Exception as exc:  # noqa: BLE001
                _log(f"[{system.name}] close failed (ignored): {type(exc).__name__}: {exc}")

    per_slice, per_system = aggregate(records)
    cfg = {"systems": [s.name for s in systems], "n_items": len(manifest), "cache_dir": str(cache_dir)}
    cfg.update(config or {})
    result = {"config": cfg, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "machine": machine_info(), "per_item": records, "per_slice": per_slice, "per_system": per_system}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return result


# --------------------------------------------------------------------------
# Reporting and CLI
# --------------------------------------------------------------------------


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{100 * v:.1f}"


def format_table(result: dict) -> str:
    rows = [("slice", "system", "n", "WER%", "CER%", "Ent%", "xrt", "RSS MB", "extra")]

    def row(label: str, system: str, m: dict) -> tuple:
        extra = []
        if "ins_per_min" in m:
            extra.append(f"ins/min={m['ins_per_min']:.1f}")
        if "der" in m:
            extra.append(f"DER={_pct(m['der'])}%")
        if "der_missing" in m:
            extra.append(f"der_missing={m['der_missing']}")
        if m["n_skipped"]:
            extra.append(f"skipped={m['n_skipped']}")
        if m["n_errors"]:
            extra.append(f"errors={m['n_errors']}")
        return (label, system, str(m["n"]), _pct(m["wer"]), _pct(m["cer"]), _pct(m["entity_err"]),
                "-" if m["xrt"] is None else f"{m['xrt']:.1f}",
                "-" if m["peak_rss_mb"] is None else f"{m['peak_rss_mb']:.0f}", " ".join(extra))

    for sl, systems in result["per_slice"].items():
        for system, m in systems.items():
            rows.append(row(sl, system, m))
    for system, m in result.get("per_system", {}).items():
        rows.append(row("ALL", system, m))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows)


_PER_ITEM_KEYS = ("item_wer",)  # aggregate dicts keyed by item id: dropped from published summaries
_SUMMARY_CONFIG_KEYS = ("systems", "n_items", "manifest", "split", "slices", "refresh")


def _strip_items(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _strip_items(v) for k, v in node.items() if k not in _PER_ITEM_KEYS}
    return node


def summarize(results: dict) -> dict:
    """Publishable aggregate of a results file: per-slice/per-system metrics only.

    The full results file embeds per-item references and hypotheses (TED-LIUM is CC BY-NC-ND) and local paths, so it is
    git-ignored; this keeps only the pooled metrics (no `per_item`, no text, no per-item ids, no paths).
    """
    config = results.get("config") or {}
    return {
        "summary": True,
        "created_at": results.get("created_at"),
        "machine": results.get("machine"),
        "config": {k: config[k] for k in _SUMMARY_CONFIG_KEYS if k in config},
        "per_slice": _strip_items(results.get("per_slice") or {}),
        "per_system": _strip_items(results.get("per_system") or {}),
    }


def summary_path(results_path: Path) -> Path:
    results_path = Path(results_path)
    return results_path.with_name(f"{results_path.stem}.summary.json")


def write_summary(results_path: Path) -> Path:
    """Write `<name>.summary.json` next to the results file `<name>.json`; returns its path."""
    results_path = Path(results_path)
    out = summary_path(results_path)
    out.write_text(json.dumps(summarize(json.loads(results_path.read_text())), indent=2, ensure_ascii=False) + "\n")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--summarize", type=Path, default=None, metavar="RESULTS_JSON",
                    help="write <name>.summary.json (aggregate metrics only, committable) for a results file and exit")
    ap.add_argument("--systems", default=",".join(BUILTIN_SYSTEMS), help="comma-separated built-in systems")
    ap.add_argument("--manifest", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--split", choices=("dev", "test", "all"), default="all")
    ap.add_argument("--slices", default=None, help="comma-separated slice filter")
    ap.add_argument("--out", type=Path, default=None, help="default: results/<systems>-<manifest>.json")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--refresh", action="store_true", help="ignore cached outputs and rerun every item")
    args = ap.parse_args(argv)
    if args.summarize:
        _log(f"wrote {write_summary(args.summarize)}")
        return

    names = [s.strip() for s in args.systems.split(",") if s.strip()]
    systems = [build_system(n, refresh=args.refresh) for n in names]
    manifest = select_items(load_manifest(args.manifest), args.split,
                            [s.strip() for s in args.slices.split(",")] if args.slices else None)
    out = args.out or RESULTS_DIR / f"{'+'.join(names)}-{args.manifest}.json"
    result = run_suite(systems, manifest, out, args.cache_dir,
                       config={"manifest": args.manifest, "split": args.split, "slices": args.slices,
                               "refresh": args.refresh},
                       refresh=args.refresh)
    print(format_table(result))
    _log(f"wrote {out}")


if __name__ == "__main__":
    main()

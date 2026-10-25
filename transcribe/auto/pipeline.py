"""Orchestration of `--backend auto`.

Order: load audio -> VAD/chunks -> probe -> (diarize) -> engine A -> engine B -> guards -> align ->
pick -> verify/resolve -> refine -> assemble (speakers + segments) -> trust.

Failure policy (spec section 8a):

* Core stages (`load`, `vad`, `probe`, `engine_a`, `engine_b`) raise `TranscriptionError`
  (`CapabilityError` is kept as is). Engines never fall back to each other.
* Enhancement stages (`diarize`, `guards`, `align`, `pick`, `verify`, `refine`, `trust`) record
  `"skipped: <ExcName>: <msg>"` and the pipeline continues with the best data it has. `assemble`
  has no meaningful fallback, so its failure is a `TranscriptionError` as well.
* A stage that is switched off on purpose (single-engine profile, `--no-refine`, `--num-speakers 1`,
  unsupported language) records status `"off"`; `refine` may also report `"partial: ..."`.

Memory: every engine, the picker and the verify scorer are released as soon as their stage ends
(also on failure). `factories` injects each collaborator for tests; models load lazily inside them.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..models import BackendResult, CapabilityError, TranscriptionError
from .align import align_chunk, build_spans, single_engine_spans
from .assemble import assign_speakers, build_segments, trust_score
from .cache import StageCache, engine_output_from_dict, file_sha256, profile_from_dict
from . import chunks as chunk_cfg
from .chunks import make_chunks, speech_spans, vad_config
from .diarize import diarize
from .engines import ENGINES_WITH_APPROX_TIMES, WHISPER_REPO, engine_cache_config, make_engine
from .guards import apply_guards
from .picker import MlxPicker
from .probe import build_profile, probe_config, whisper_lid
from .profiles import load_profiles, select
from .refine_gated import GatedRefiner, refine_targets
from .resolve import FusedWord, resolve
from .terms import build_terms
from .types import AudioProfile, Chunk, EngineOutput, StageStatus
from .verify import WhisperScorer

LOGGER = logging.getLogger("transcribe_youtube")

SAMPLE_RATE = 16000
SUPPORTED_LANGUAGES = ("en", "es")
STATUS_OK = "ok"
STATUS_OFF = "off"
NO_SPEECH = "no speech detected"
NO_WORDS = "no words recognized"


def default_factories() -> Dict[str, Any]:
    """Real collaborators. Nothing here loads a model; each one loads lazily on first use."""
    return {
        "engine": make_engine,
        "scorer": WhisperScorer,
        "picker": MlxPicker,
        "refiner": lambda model: GatedRefiner(model=model),
        "diarizer": diarize,
        "lid": lambda: whisper_lid(WHISPER_REPO),
        "vad": speech_spans,
        "cache": StageCache(),
    }


# ----------------------------------------------------------------------------- stage recorder


class _Stage:
    """Mutable handle a stage function uses to override its recorded status/detail."""

    def __init__(self) -> None:
        self.status = STATUS_OK
        self.detail = ""


_SKIPPED = object()


class _Recorder:
    def __init__(self) -> None:
        self.stages: List[StageStatus] = []

    def add(self, name: str, status: str, detail: str = "", seconds: float = 0.0) -> None:
        self.stages.append(StageStatus(name, status, detail, round(seconds, 3)))
        # INFO only: the CLI (`_log_auto_warnings`) emits the single WARNING per degraded stage.
        LOGGER.info("auto stage %s: %s%s (%.1fs)", name, status, f" [{detail}]" if detail else "", seconds)

    def run(self, name: str, fn: Callable[[_Stage], Any], *, core: bool = False) -> Any:
        """Time `fn`; returns its value, or `_SKIPPED` for an enhancement stage that raised."""
        handle = _Stage()
        started = time.perf_counter()
        try:
            value = fn(handle)
        except (TranscriptionError, CapabilityError) as exc:
            seconds = time.perf_counter() - started
            if core:
                self.add(name, f"failed: {type(exc).__name__}: {exc}", seconds=seconds)
                raise
            self.add(name, f"skipped: {type(exc).__name__}: {exc}", seconds=seconds)
            return _SKIPPED
        except Exception as exc:
            seconds = time.perf_counter() - started
            if core:
                self.add(name, f"failed: {type(exc).__name__}: {exc}", seconds=seconds)
                raise TranscriptionError(
                    f"auto pipeline stage '{name}' failed: {type(exc).__name__}: {exc}"
                ) from exc
            self.add(name, f"skipped: {type(exc).__name__}: {exc}", seconds=seconds)
            LOGGER.debug("auto stage %s failed", name, exc_info=True)
            return _SKIPPED
        self.add(name, handle.status, handle.detail, time.perf_counter() - started)
        return value


# ----------------------------------------------------------------------------- helpers


def _load_audio(path: Path) -> Tuple[np.ndarray, str]:
    import soundfile as sf

    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if int(rate) != SAMPLE_RATE:
        raise TranscriptionError(f"auto pipeline expects {SAMPLE_RATE} Hz normalized audio, got {rate} Hz")
    return np.ascontiguousarray(data.mean(axis=1), dtype=np.float32), file_sha256(path)


def _release(obj: Any) -> None:
    try:
        obj.release()
    except Exception as exc:  # a failing release must not mask the stage result
        LOGGER.warning("auto pipeline: release failed: %s: %s", type(exc).__name__, exc)


def _release_whisper_holder() -> None:
    """mlx-whisper keeps the LID model in a process-global holder: drop it after the probe."""
    try:
        from mlx_whisper.transcribe import ModelHolder

        ModelHolder.model = None
        ModelHolder.model_path = None
    except ImportError:
        return
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


def _chunk_rms(samples: np.ndarray, chunks: List[Chunk]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for chunk in chunks:
        lo = max(0, int(round(chunk.start * SAMPLE_RATE)))
        hi = min(len(samples), int(round(chunk.end * SAMPLE_RATE)))
        part = samples[lo:hi].astype(np.float64)
        out[chunk.index] = float(np.sqrt(np.mean(part * part))) if len(part) else 0.0
    return out


def _fallback_words(out: EngineOutput) -> List[FusedWord]:
    """Engine A's words as-is, used only when the align stage itself failed."""
    words: List[FusedWord] = []
    for index in sorted(out.chunks):
        for w in out.chunks[index]:
            words.append(FusedWord(w.text, w.start, w.end, "solid", 0.5 if w.prob is None else w.prob, "a"))
    return words


def _joined(words: List[FusedWord]) -> str:
    """Best-guess text of a word list (no `[unclear: ...]` markers)."""
    return " ".join(w.text for w in words)


def _trust_coef(profiles: Dict[str, Any], entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Trust coefficients of the rung the profile runs (`trust.by_rung`), else the global fit."""
    trust = profiles.get("trust") or {}
    rung = entry.get("trust_rung")
    per_rung = ((trust.get("by_rung") or {}).get(rung) or {}).get("coef") if rung else None
    return per_rung or trust.get("coef")


def _profile_key(profile: AudioProfile, language: Optional[str]) -> str:
    """Routing key; an explicit `--language` replaces the detected language part."""
    suffix = profile.key.partition("-")[2]
    if language:
        lang_key = language if language in SUPPORTED_LANGUAGES else "other"
    else:
        lang_key = profile.key.partition("-")[0]
    return lang_key + (f"-{suffix}" if suffix else "")


# ----------------------------------------------------------------------------- pipeline


def run_auto(
    audio_path: Path,
    metadata: Dict[str, Any],
    *,
    language: Optional[str] = None,
    terms: Optional[List[str]] = None,
    num_speakers: Optional[int] = None,
    refine: bool = True,
    profiles: Optional[Dict[str, Any]] = None,
    factories: Optional[Dict[str, Any]] = None,
    refine_model: Optional[str] = None,
    original_sample_rate: Optional[int] = None,
) -> Tuple[BackendResult, Dict[str, Any]]:
    """Run the auto pipeline on a 16 kHz mono WAV; returns the result and the `metadata["auto"]` dict."""
    started = time.perf_counter()
    audio_path = Path(audio_path)
    metadata = metadata or {}
    default_lid = not (factories and "lid" in factories)
    fac = {**default_factories(), **(factories or {})}
    cache: Optional[StageCache] = fac.get("cache")
    profiles = profiles if profiles is not None else load_profiles()
    override = language.strip().lower() if language and language.strip() else None
    rec = _Recorder()
    notes: List[str] = []

    # ---- load (core)
    samples, audio_sha = rec.run("load", lambda st: _load_audio(audio_path), core=True)
    duration = len(samples) / SAMPLE_RATE

    # ---- vad + chunks (core)
    vad_key = {"audio": audio_sha, "config": vad_config()}

    def do_vad(st: _Stage):
        hit = cache.get("vad", vad_key) if cache else None
        if hit is not None:
            spans = [(float(a), float(b)) for a, b in hit]
            st.detail = "cache hit"
        else:
            spans = [(float(a), float(b)) for a, b in fac["vad"](samples, SAMPLE_RATE)]
            if cache:
                cache.put("vad", vad_key, [[a, b] for a, b in spans])
        made = make_chunks(
            spans, samples, SAMPLE_RATE, chunk_cfg.MAX_CHUNK_SECONDS, chunk_cfg.MERGE_GAP_SECONDS
        )
        return spans, made

    spans, chunks = rec.run("vad", do_vad, core=True)

    # ---- probe (core)
    def do_probe(st: _Stage) -> AudioProfile:
        key = {
            "audio": audio_sha,
            "original_sample_rate": original_sample_rate,
            "config": probe_config(fac.get("lid_repo", WHISPER_REPO)),
            "vad": vad_key["config"],  # the probe consumes the VAD spans
        }
        hit = cache.get("probe", key) if cache else None
        if hit is not None:
            st.detail = "cache hit"
            return profile_from_dict(hit)
        try:
            built = build_profile(samples, SAMPLE_RATE, spans, fac["lid"](), original_sample_rate)
        finally:
            if default_lid:
                _release_whisper_holder()
        if cache:
            cache.put("probe", key, built)
        return built

    profile: AudioProfile = rec.run("probe", do_probe, core=True)

    # ---- routing
    if not chunks:
        notes.append(NO_SPEECH)
        return _finish_empty(rec, profile, _profile_key(profile, override), override, duration, notes, started)

    lang = override or profile.language
    key = _profile_key(profile, override)
    entry = select(profiles, key)
    lang_key = key.partition("-")[0]  # probe already maps code-switching audio to "other"
    engine_lang = override or (None if lang_key == "other" else lang)
    if entry.get("unvalidated"):
        notes.append(f"profile '{key}' is not calibrated; using the closest entry")
    if lang_key == "other":
        notes.append(f"language '{lang}' is best effort: no refine and no phonetic gates")
    terms_list = _safe_terms(metadata, terms)

    # ---- diarize (enhancement)
    turns: List[dict] = []
    if num_speakers == 1:
        rec.add("diarize", STATUS_OFF, "num_speakers=1")
    else:
        def do_diarize(st: _Stage) -> List[dict]:
            found, status = fac["diarizer"](audio_path, num_speakers)
            st.status = status
            return found

        diarized = rec.run("diarize", do_diarize)
        turns = [] if diarized is _SKIPPED else diarized

    # ---- engines (core, one at a time)
    engine_ids = list(entry["engines"])[:2]
    outputs: List[EngineOutput] = []
    chunk_key = [[c.index, c.start, c.end] for c in chunks]
    for position, engine_id in enumerate(engine_ids):
        name = "engine_a" if position == 0 else "engine_b"

        def do_engine(st: _Stage, engine_id: str = engine_id) -> EngineOutput:
            ckey = {
                "audio": audio_sha,
                "engine": fac.get("engine_config", engine_cache_config)(engine_id),
                "language": engine_lang,
                "chunks": chunk_key,
            }
            hit = cache.get("engine", ckey) if cache else None
            if hit is not None:
                st.detail = f"{engine_id} (cache hit)"
                return engine_output_from_dict(hit)
            engine = fac["engine"](engine_id)
            try:
                out = engine.transcribe_chunks(samples, chunks, engine_lang)
            finally:
                _release(engine)
            if cache:
                cache.put("engine", ckey, out)
            st.detail = engine_id
            return out

        outputs.append(rec.run(name, do_engine, core=True))

    # ---- guards (enhancement: raw outputs if it fails)
    flag_rate = 0.0  # fraction of (engine, chunk) pairs where a guard removed or collapsed text

    def do_guards(st: _Stage) -> List[EngineOutput]:
        rms = _chunk_rms(samples, chunks)
        return [apply_guards(o, rms, spans, chunks) for o in outputs]

    guarded = rec.run("guards", do_guards)
    if guarded is _SKIPPED:
        guarded = outputs
    else:
        flagged = sum(1 for o in guarded for reasons in o.flags.values() if reasons)
        flag_rate = flagged / max(1, len(guarded) * len(chunks))

    # ---- align (enhancement: engine A's words if it fails)
    def do_align(st: _Stage):
        if len(guarded) >= 2:
            slots = {}
            for chunk in chunks:
                a = guarded[0].chunks.get(chunk.index, [])
                b = guarded[1].chunks.get(chunk.index, [])
                if a or b:
                    slots[chunk.index] = align_chunk(a, b, lang)
            return build_spans(slots, entry["solid_min_prob"])
        return single_engine_spans(guarded[0], lang, entry["solid_min_prob"])

    aligned = rec.run("align", do_align)
    align_failed = aligned is _SKIPPED
    span_list = [] if align_failed else aligned
    nonsolid = [s for s in span_list if s.tier != "solid"]

    # ---- pick (enhancement; the picker is released right after)
    picks: Optional[Dict[int, str]] = None
    if align_failed:
        rec.add("pick", STATUS_OFF, "align failed")
    elif len(guarded) < 2:
        rec.add("pick", STATUS_OFF, "single-engine profile")
    elif not entry.get("pick"):
        rec.add("pick", STATUS_OFF, "disabled by profile")
    elif not nonsolid:
        rec.add("pick", STATUS_OK, "no disputed spans")
    else:
        def do_pick(st: _Stage) -> Dict[int, str]:
            picker = fac["picker"]()
            try:
                return picker.pick(nonsolid, lang, terms_list)
            finally:
                _release(picker)

        result_picks = rec.run("pick", do_pick)
        picks = None if result_picks is _SKIPPED else result_picks

    # Best-guess text after each fusion stage, for the benchmark suite's per-stage edit scoring. The fused/pick texts
    # come from pure-Python resolves without an audio scorer, so this costs nothing.
    stage_texts: Dict[str, str] = {}
    if not align_failed:
        try:
            stage_texts["fused"] = _joined(resolve(span_list, None, None, samples, lang, entry["margin"], "a"))
            if picks is not None:
                stage_texts["pick"] = _joined(resolve(span_list, picks, None, samples, lang, entry["margin"], "a"))
        except Exception as exc:  # diagnostics only: never fail a run over them
            LOGGER.debug("auto stage_texts failed: %s: %s", type(exc).__name__, exc)
            stage_texts.clear()

    # ---- verify + resolve (enhancement; the scorer is released right after)
    words: Optional[List[FusedWord]] = None
    verify_ok = False
    if align_failed:
        words = _fallback_words(guarded[0])
        rec.add("verify", STATUS_OFF, "align failed")
    elif not entry.get("verify"):
        rec.add("verify", STATUS_OFF, "disabled by profile")
    elif not nonsolid:
        rec.add("verify", STATUS_OK, "no disputed spans")
        verify_ok = True
    else:
        def do_verify(st: _Stage) -> List[FusedWord]:
            scorer = fac["scorer"]()
            try:
                return resolve(span_list, picks, scorer, samples, lang, entry["margin"], "a")
            finally:
                _release(scorer)

        verified = rec.run("verify", do_verify)
        if verified is not _SKIPPED:
            words, verify_ok = verified, True
    if words is None:  # verify off or skipped: resolve without audio scores, as a recorded core stage
        words = rec.run(
            "resolve",
            lambda st: resolve(span_list, picks, None, samples, lang, entry["margin"], "a"),
            core=True,
        )

    if verify_ok and not align_failed:
        stage_texts["verify"] = _joined(words)

    # ---- refine (enhancement; gated Claude)
    refine_stats: Optional[Dict[str, Any]] = None
    refine_on = refine and entry.get("refine") and lang_key != "other" and lang in SUPPORTED_LANGUAGES
    if not refine:
        rec.add("refine", STATUS_OFF, "disabled")
    elif not refine_on:
        rec.add("refine", STATUS_OFF, "disabled by profile or unsupported language")
    else:
        stats_box: Dict[str, Any] = {}

        def do_refine(st: _Stage) -> List[FusedWord]:
            targets = refine_targets(words, terms_list, lang)
            if not targets:
                st.detail = "no targets"
                return words
            scorer = None
            if entry.get("verify") and verify_ok:
                try:
                    scorer = fac["scorer"]()
                except Exception as exc:
                    LOGGER.warning("auto refine: audio check unavailable (%s: %s)", type(exc).__name__, exc)
            try:
                refiner = fac["refiner"](refine_model)
                new_words, stats = refiner.refine(words, targets, terms_list, lang, scorer, samples)
            finally:
                if scorer is not None:
                    _release(scorer)
            stats_box.update(stats)
            status = str(stats.get("status", STATUS_OK))
            st.status = status
            st.detail = f"targets={stats.get('targets')} accepted={stats.get('accepted')}"
            return new_words

        refined = rec.run("refine", do_refine)
        if refined is not _SKIPPED:
            words = refined
            stage_texts["refine"] = _joined(words)
        refine_stats = dict(stats_box) or None

    # ---- assemble (speakers + segments)
    def do_assemble(st: _Stage) -> List[dict]:
        assign_speakers(words, turns)
        return build_segments(words)

    segments = rec.run("assemble", do_assemble, core=True)

    # ---- trust (enhancement)
    coef = _trust_coef(profiles, entry)
    trust = rec.run("trust", lambda st: trust_score(words, list(rec.stages), coef, flag_rate, entry))
    if trust is _SKIPPED:
        trust = {"estimated_wer": None, "features": {}, "level": "unknown"}

    meta: Dict[str, Any] = {
        "profile": asdict(profile),
        "profile_key": key,
        "profile_entry": entry,
        "language": lang,
        "engines": engine_ids,
        # "approximate": the engine whose words are the best guess (engine A) has no native word times (words are
        # spread over each chunk by character length), so SRT cue and word times are approximate.
        "timestamps": "approximate" if engine_ids[0] in ENGINES_WITH_APPROX_TIMES else "engine",
        "terms": terms_list,
        "stages": [asdict(s) for s in rec.stages],
        "trust": trust,
        "refine": refine_stats,
        "notes": notes,
        "no_speech": False,
        "no_speech_reason": None,
        "speakers": len({w.speaker for w in words if w.speaker}),
        "seconds": round(time.perf_counter() - started, 3),
        "words": [asdict(w) for w in words],
        "stage_texts": stage_texts,
    }
    if not segments:  # chunks existed but no engine/guard left any word: still a valid empty result
        notes.append(NO_WORDS)
        meta["no_speech"] = True
        meta["no_speech_reason"] = NO_WORDS
        meta["trust"] = {"estimated_wer": None, "features": {}, "level": "n/a"}
    text = " ".join(seg["text"] for seg in segments)
    result = BackendResult(text=text, segments=segments, language=lang, duration_sec=duration, raw=meta)
    return result, meta


def _safe_terms(metadata: Dict[str, Any], extra: Optional[List[str]]) -> List[str]:
    try:
        return build_terms(metadata, extra)
    except Exception as exc:  # terms are only hints
        LOGGER.warning("auto pipeline: term extraction failed (%s: %s)", type(exc).__name__, exc)
        return [t for t in (extra or []) if t]


def _finish_empty(
    rec: _Recorder,
    profile: AudioProfile,
    key: str,
    override: Optional[str],
    duration: float,
    notes: List[str],
    started: float,
) -> Tuple[BackendResult, Dict[str, Any]]:
    """No speech: empty segments, no engine is ever created."""
    meta: Dict[str, Any] = {
        "profile": asdict(profile),
        "profile_key": key,
        "profile_entry": None,
        "language": override,
        "engines": [],
        "timestamps": "n/a",
        "terms": [],
        "stages": [asdict(s) for s in rec.stages],
        "trust": {"estimated_wer": None, "features": {}, "level": "n/a"},
        "refine": None,
        "notes": notes,
        "no_speech": True,
        "no_speech_reason": NO_SPEECH,
        "speakers": 0,
        "seconds": round(time.perf_counter() - started, 3),
        "words": [],
        "stage_texts": {},
    }
    result = BackendResult(text="", segments=[], language=override, duration_sec=duration, raw=meta)
    return result, meta

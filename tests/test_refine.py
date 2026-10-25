import json
import subprocess
from typing import Any, Dict, List

import pytest

import transcribe.refine as refine_mod
from transcribe_youtube import (
    BackendResult,
    CapabilityError,
    ClaudeRefiner,
    OutputWriter,
    TranscriptionError,
    build_refine_context,
    map_uncertain_words,
    parse_args,
    refine_diverges,
)


def _segments(texts: List[str]) -> List[Dict[str, Any]]:
    return [{"start": float(i), "end": float(i) + 1.0, "speaker": None, "text": t} for i, t in enumerate(texts)]


def _result(texts: List[str], raw: Dict[str, Any] = None) -> BackendResult:
    return BackendResult(text=" ".join(texts), segments=_segments(texts), language="en", duration_sec=10.0, raw=raw or {})


def _envelope(items: List[Dict[str, Any]], cost: float = 0.01) -> str:
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "",
            "duration_ms": 1200,
            "total_cost_usd": cost,
            "structured_output": {"segments": items},
        }
    )


class FakeRun:
    """Records subprocess.run calls; `responder(prompt_items) -> (returncode, stdout)`."""

    def __init__(self, responder):
        self.responder = responder
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, cmd, input=None, capture_output=None, text=None, timeout=None):
        self.calls.append({"cmd": cmd, "input": input})
        code, stdout = self.responder(_prompt_items(input), len(self.calls))
        return subprocess.CompletedProcess(cmd, code, stdout=stdout, stderr="boom" if code else "")


def _prompt_items(prompt: str) -> List[Dict[str, Any]]:
    tail = prompt.split("return each id exactly once.\n", 1)[1]
    return json.loads(tail)


def _echo(fix=lambda t: t, cost=0.01):
    def responder(items, call_no):
        return 0, _envelope([{"id": it["id"], "text": fix(it["text"])} for it in items], cost)

    return responder


@pytest.fixture
def refiner(monkeypatch: pytest.MonkeyPatch) -> ClaudeRefiner:
    monkeypatch.setattr(refine_mod.shutil, "which", lambda name: f"/usr/local/bin/{name}")
    return ClaudeRefiner(model="sonnet", batch_size=3, max_workers=2, context_segments=2)


def test_command_uses_safe_mode_flags_and_model(monkeypatch: pytest.MonkeyPatch, refiner: ClaudeRefiner) -> None:
    fake = FakeRun(_echo())
    monkeypatch.setattr(refine_mod.subprocess, "run", fake)
    refiner.refine(_result(["hello there"]), {"title": "Demo"})

    cmd = fake.calls[0]["cmd"]
    assert cmd[0] == "/usr/local/bin/claude"
    for flag in ("-p", "--safe-mode", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
        assert flag in cmd
    assert "--bare" not in cmd
    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--model") + 1] == "sonnet"
    assert cmd[cmd.index("--output-format") + 1] == "json"
    schema = json.loads(cmd[cmd.index("--json-schema") + 1])
    assert schema["required"] == ["segments"]
    assert schema["properties"]["segments"]["items"]["required"] == ["id", "text"]
    assert "--system-prompt" in cmd
    assert '"title": "Demo"' in fake.calls[0]["input"]


def test_model_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(refine_mod.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setenv("TRANSCRIBE_REFINE_MODEL", "opus")
    assert ClaudeRefiner().model == "opus"
    assert ClaudeRefiner(model="haiku").model == "haiku"


def test_batching_with_read_only_context(monkeypatch: pytest.MonkeyPatch, refiner: ClaudeRefiner) -> None:
    fake = FakeRun(_echo(lambda t: t.replace("in graphic", "infographic")))
    monkeypatch.setattr(refine_mod.subprocess, "run", fake)
    texts = [f"segment number {i} about an in graphic tool" for i in range(7)]

    out = refiner.refine(_result(texts), {})

    assert len(fake.calls) == 3  # 3 + 3 + 1
    sent = sorted((it["id"] for call in fake.calls for it in _prompt_items(call["input"])))
    assert sent == list(range(7))
    second = next(c["input"] for c in fake.calls if _prompt_items(c["input"])[0]["id"] == 3)
    assert "Read-only context" in second
    readonly = second.split("Do NOT return them.\n", 1)[1].split("\n\nSegments to correct", 1)[0]
    assert json.loads(readonly) == texts[1:3]  # last 2 of the previous batch
    first = next(c["input"] for c in fake.calls if _prompt_items(c["input"])[0]["id"] == 0)
    assert "Read-only context" not in first

    assert [s["text"] for s in out.segments] == [t.replace("in graphic", "infographic") for t in texts]
    assert [(s["start"], s["end"]) for s in out.segments] == [(s["start"], s["end"]) for s in _segments(texts)]
    assert out.raw["refine"]["changed_segments"] == 7
    assert out.raw["refine"]["batches"] == 3
    assert out.raw["refine"]["original_texts"] == texts
    assert out.text == " ".join(s["text"] for s in out.segments)


def test_id_mismatch_retries_then_raises(monkeypatch: pytest.MonkeyPatch, refiner: ClaudeRefiner) -> None:
    def responder(items, call_no):
        return 0, _envelope([{"id": it["id"] + 100, "text": it["text"]} for it in items])

    fake = FakeRun(responder)
    monkeypatch.setattr(refine_mod.subprocess, "run", fake)
    with pytest.raises(TranscriptionError, match=r"Refine batch 1/1 \(segments 0-1\) failed after 2 attempts"):
        refiner.refine(_result(["one two", "three four"]), {})
    assert len(fake.calls) == 2


def test_retry_recovers_after_one_failure(monkeypatch: pytest.MonkeyPatch, refiner: ClaudeRefiner) -> None:
    def responder(items, call_no):
        if call_no == 1:
            return 1, ""
        return 0, _envelope([{"id": it["id"], "text": it["text"]} for it in items], cost=0.02)

    monkeypatch.setattr(refine_mod.subprocess, "run", FakeRun(responder))
    out = refiner.refine(_result(["one two"]), {})
    assert out.segments[0]["text"] == "one two"
    assert out.raw["refine"]["cost_usd"] == pytest.approx(0.02)


def test_is_error_envelope_counts_as_failure(monkeypatch: pytest.MonkeyPatch, refiner: ClaudeRefiner) -> None:
    error = json.dumps({"is_error": True, "subtype": "error_max_turns", "result": "nope", "total_cost_usd": 0.001})
    monkeypatch.setattr(refine_mod.subprocess, "run", FakeRun(lambda items, n: (0, error)))
    with pytest.raises(TranscriptionError, match="error_max_turns"):
        refiner.refine(_result(["one two"]), {})


def test_divergence_and_empty_guards_keep_original(monkeypatch: pytest.MonkeyPatch, refiner: ClaudeRefiner) -> None:
    texts = [
        "so your manage says the boring park is next",
        "and then we talked about the weather for a while",
        "keep this one",
    ]
    rewrites = {
        texts[0]: "So your manager says the boring part is next.",
        texts[1]: "Overall, the conversation covered several unrelated topics.",
        texts[2]: "",
    }
    monkeypatch.setattr(refine_mod.subprocess, "run", FakeRun(_echo(lambda t: rewrites[t])))

    out = refiner.refine(_result(texts), {})

    assert out.segments[0]["text"] == rewrites[texts[0]]
    assert out.segments[1]["text"] == texts[1]
    assert out.segments[2]["text"] == texts[2]
    assert out.raw["refine"]["changed_segments"] == 1
    assert out.raw["refine"]["guarded_segments"] == 2


def test_refine_diverges_thresholds() -> None:
    assert not refine_diverges("turn any idea into an in graphic", "Turn any idea into an infographic.")
    assert refine_diverges("we went to the store yesterday", "Prices rose sharply in the third quarter.")
    assert not refine_diverges("open eye", "OpenAI")
    assert refine_diverges("yeah", "completely different")


def test_missing_binary_raises_capability_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(refine_mod.shutil, "which", lambda name: None)
    monkeypatch.delenv("CLAUDE_BIN", raising=False)
    with pytest.raises(CapabilityError, match="Claude Code CLI"):
        ClaudeRefiner()


def test_claude_bin_env_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    monkeypatch.setattr(refine_mod.shutil, "which", lambda name: seen.append(name) or name)
    monkeypatch.setenv("CLAUDE_BIN", "/opt/claude/bin/claude")
    assert ClaudeRefiner().claude_bin == "/opt/claude/bin/claude"
    assert seen == ["/opt/claude/bin/claude"]


def test_text_only_result_raises_capability_error(refiner: ClaudeRefiner) -> None:
    result = BackendResult(text="plain text", segments=[], language="en", duration_sec=1.0, raw={})
    with pytest.raises(CapabilityError):
        refiner.refine(result, {})


def test_uncertain_word_mapping_from_raw_words() -> None:
    segments = [
        {"start": 0.0, "end": 2.0, "text": "a b c"},
        {"start": 2.0, "end": 4.0, "text": "d e f"},
    ]
    words = [{"word": w, "start": s, "end": s + 0.04, "score": sc} for w, s, sc in [
        ("a", 0.1, 5.0), ("b,", 0.5, 0.2), ("c", 1.0, 0.1),
        ("d", 2.1, 4.0), ("e", 2.5, 6.0), ("f.", 3.0, 0.3),
    ] + [(f"x{i}", 3.5, 9.0) for i in range(24)]]

    hints = map_uncertain_words(segments, words, quantile=0.1, per_segment=2)

    assert hints == [["b", "c"], ["f"]]  # time order, punctuation stripped
    assert map_uncertain_words(segments, []) == [[], []]
    assert map_uncertain_words([{"start": None, "end": None, "text": "x"}], words) == [[]]


def test_uncertain_hints_reach_the_prompt(monkeypatch: pytest.MonkeyPatch, refiner: ClaudeRefiner) -> None:
    fake = FakeRun(_echo())
    monkeypatch.setattr(refine_mod.subprocess, "run", fake)
    raw = {"words": [{"word": "boring", "start": 0.1, "score": 3.0}, {"word": "park", "start": 0.5, "score": -1.0}]}
    refiner.refine(_result(["boring park"], raw), {})
    assert _prompt_items(fake.calls[0]["input"]) == [{"id": 0, "text": "boring park", "uncertain": ["park"]}]


def test_cost_aggregates_across_batches(monkeypatch: pytest.MonkeyPatch, refiner: ClaudeRefiner) -> None:
    monkeypatch.setattr(refine_mod.subprocess, "run", FakeRun(_echo(cost=0.0125)))
    out = refiner.refine(_result([f"text {i}" for i in range(8)]), {})
    assert out.raw["refine"]["batches"] == 3
    assert out.raw["refine"]["cost_usd"] == pytest.approx(0.0375)


def test_refined_result_serializes_to_all_outputs(monkeypatch: pytest.MonkeyPatch, refiner: ClaudeRefiner, tmp_path) -> None:
    monkeypatch.setattr(refine_mod.subprocess, "run", FakeRun(_echo(lambda t: t.replace("park", "part"))))
    out = refiner.refine(_result(["the boring park"], {"words": []}), {})
    payload = {"segments": out.segments, "text": out.text, "metadata": {"refine": out.raw["refine"]}}
    written = OutputWriter.write(payload, ["txt", "srt", "json"], tmp_path, "refined")
    assert "the boring part" in written["txt"].read_text()
    assert "the boring part" in written["srt"].read_text()
    data = json.loads(written["json"].read_text())
    assert data["metadata"]["refine"]["original_texts"] == ["the boring park"]


def test_build_refine_context_uses_metadata_and_truncates() -> None:
    meta = {
        "platform": "youtube",
        "youtube_title": "Turn Any Idea into an AI Infographic",
        "youtube_channel": "Tool Finder",
        "youtube_description": "x" * 5000,
    }
    ctx = build_refine_context(meta, language="en", notes="Hotwords: Canva")
    assert ctx["title"] == "Turn Any Idea into an AI Infographic"
    assert ctx["channel"] == "Tool Finder"
    assert len(ctx["description"]) <= 1505
    assert ctx["notes"] == "Hotwords: Canva"
    assert build_refine_context({}, title="meeting notes") == {"title": "meeting notes"}


def test_cli_refine_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRANSCRIBE_REFINE", raising=False)
    monkeypatch.delenv("TRANSCRIBE_REFINE_MODEL", raising=False)
    args = parse_args(["video.mp4"])
    assert args.refine is None  # unset: off for existing backends, on for auto
    assert args.refine_model == "sonnet"

    args = parse_args(["video.mp4", "--refine", "--refine-model", "opus"])
    assert args.refine is True
    assert args.refine_model == "opus"

    monkeypatch.setenv("TRANSCRIBE_REFINE", "1")
    monkeypatch.setenv("TRANSCRIBE_REFINE_MODEL", "haiku")
    args = parse_args(["video.mp4"])
    assert args.refine is True
    assert args.refine_model == "haiku"
    assert parse_args(["video.mp4", "--no-refine"]).refine is False


def test_cli_main_runs_refine_before_writing(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import transcribe.cli as cli
    from transcribe_youtube import NormalizedMedia

    media = tmp_path / "boring_talk.wav"
    media.write_bytes(b"audio")
    monkeypatch.setattr(refine_mod.shutil, "which", lambda name: f"/bin/{name}")
    fake = FakeRun(_echo(lambda t: t.replace("park", "part")))
    monkeypatch.setattr(refine_mod.subprocess, "run", fake)
    monkeypatch.setattr(cli.AudioNormalizer, "normalize", lambda self, path, backend, temp: NormalizedMedia(path, 2.0))
    monkeypatch.setattr(cli.BackendFactory, "create", staticmethod(lambda args: object()))
    monkeypatch.setattr(cli, "transcribe_with_chunking", lambda **kw: _result(["the boring park"]))

    code = cli.main([str(media), "--backend", "eg2", "--refine", "--output-dir", str(tmp_path),
                     "--output-basename", "out", "--output-formats", "txt,json"])

    assert code == 0
    assert '"title": "boring_talk"' in fake.calls[0]["input"]
    assert (tmp_path / "out.txt").read_text().strip().endswith("the boring part")
    data = json.loads((tmp_path / "out.json").read_text())
    assert data["segments"][0]["text"] == "the boring part"
    assert data["metadata"]["refine"]["original_texts"] == ["the boring park"]


def test_cli_refine_rejects_text_only_openai_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from transcribe_youtube import validate_refine_constraints

    with pytest.raises(CapabilityError):
        validate_refine_constraints("openai", "gpt-4o-transcribe")
    validate_refine_constraints("openai", "whisper-1")
    validate_refine_constraints("eg2", "eg2demo-v5")

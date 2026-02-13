import logging
import time
from concurrent.futures import TimeoutError as FutureTimeoutError

import pytest

from logger_utils import format_bytes, log_stage
from transcribe_youtube import _run_with_progress


def test_format_bytes_human_readable() -> None:
    assert format_bytes(512) == "512.00B"
    assert format_bytes(1024) == "1.00KB"
    assert format_bytes(1024 * 1024) == "1.00MB"


def test_log_stage_emits_start_and_end(caplog) -> None:
    logger = logging.getLogger("test_log_stage")
    with caplog.at_level(logging.INFO):
        with log_stage(logger, "unit_stage", backend="whisper"):
            pass

    text = caplog.text
    assert "Stage started: unit_stage (backend=whisper)" in text
    assert "Stage completed: unit_stage" in text


def test_run_with_progress_timeout_returns_quickly() -> None:
    started = time.perf_counter()
    with pytest.raises(FutureTimeoutError):
        _run_with_progress(
            lambda: time.sleep(5),
            activity="timeout_probe",
            timeout_seconds=1,
            progress_log_seconds=1,
        )
    elapsed = time.perf_counter() - started
    assert elapsed < 3.0

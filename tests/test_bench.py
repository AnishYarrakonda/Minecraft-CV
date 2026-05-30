"""Tests for the mcv-bench latency summarizer (Task 4 JSON profiling report).

The summarizer is pure: it turns a list of per-frame latencies (ms) into the stats the JSON
report carries. The live tracking benchmark itself needs MediaPipe + a camera/clip and is not
unit-tested here.
"""

from __future__ import annotations

import pytest

from minecraft_cv.cli import _summarize


def test_summarize_basic_stats() -> None:
    s = _summarize([10.0, 10.0, 10.0, 10.0])
    assert s["count"] == 4
    assert s["mean"] == pytest.approx(10.0)
    assert s["min"] == pytest.approx(10.0)
    assert s["max"] == pytest.approx(10.0)
    assert s["jitter_std"] == pytest.approx(0.0)
    assert s["fps_mean"] == pytest.approx(100.0)


def test_summarize_jitter_is_nonzero_for_spread() -> None:
    steady = _summarize([16.0] * 10)
    spiky = _summarize([16.0] * 9 + [60.0])
    assert spiky["jitter_std"] > steady["jitter_std"]
    # A single 60 ms spike pushes p99/max up while the median stays low.
    assert spiky["p99"] >= spiky["p50"]
    assert spiky["max"] == pytest.approx(60.0)


def test_summarize_fps_from_mean() -> None:
    s = _summarize([20.0, 20.0])  # 20 ms mean -> 50 FPS
    assert s["fps_mean"] == pytest.approx(50.0)


def test_summarize_percentiles_ordered() -> None:
    s = _summarize([float(x) for x in range(1, 101)])
    assert s["p50"] <= s["p95"] <= s["p99"] <= s["max"]
    assert s["min"] <= s["p50"]

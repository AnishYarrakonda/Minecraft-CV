"""Deterministic tests for the One-Euro mouse-look filter.

Pure and clock-free: every sample is fed with an explicit timestamp, so the filter's
behavior is fully reproducible with no camera and no wall-clock dependence.
"""

from __future__ import annotations

import numpy as np
import pytest

from minecraft_cv.joystick.one_euro import OneEuroFilter


def test_first_sample_passes_through() -> None:
    f = OneEuroFilter()
    out = f.filter(np.array([0.3, -0.2]), timestamp=0.0)
    assert np.allclose(out, [0.3, -0.2])


def test_smooths_stationary_jitter() -> None:
    """A noisy-but-stationary signal is pulled toward its mean (jitter is attenuated)."""
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=0.5, beta=0.0)  # fixed low cutoff -> heavy smoothing
    t = 0.0
    last = np.zeros(2)
    raw_spread = []
    filt_spread = []
    for _ in range(200):
        t += 1.0 / 60.0
        raw = np.array([1.0, 1.0]) + rng.normal(0.0, 0.1, size=2)
        last = f.filter(raw, timestamp=t)
        raw_spread.append(np.linalg.norm(raw - np.array([1.0, 1.0])))
        filt_spread.append(np.linalg.norm(last - np.array([1.0, 1.0])))
    # After warm-up, the filtered signal sits far closer to the true mean than the raw.
    assert np.mean(filt_spread[50:]) < 0.5 * np.mean(raw_spread[50:])


def test_tracks_fast_motion_with_low_lag() -> None:
    """With a high beta, a fast ramp is followed with small lag (responsiveness)."""
    f = OneEuroFilter(min_cutoff=1.0, beta=1.0)
    t = 0.0
    out = np.zeros(2)
    for i in range(60):
        t += 1.0 / 60.0
        out = f.filter(np.array([0.1 * i, 0.0]), timestamp=t)
    # The filtered value should be close to the latest input (0.1 * 59 = 5.9).
    assert out[0] > 5.0


def test_reset_clears_history() -> None:
    f = OneEuroFilter()
    f.filter(np.array([5.0, 5.0]), timestamp=0.0)
    f.filter(np.array([5.0, 5.0]), timestamp=0.1)
    f.reset()
    # After reset, the next sample passes through unchanged (re-seeds).
    out = f.filter(np.array([-1.0, 2.0]), timestamp=0.2)
    assert np.allclose(out, [-1.0, 2.0])


def test_non_advancing_timestamp_returns_raw() -> None:
    f = OneEuroFilter()
    f.filter(np.array([1.0, 1.0]), timestamp=1.0)
    out = f.filter(np.array([2.0, 2.0]), timestamp=1.0)  # dt == 0
    assert np.allclose(out, [2.0, 2.0])


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_invalid_min_cutoff_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        OneEuroFilter(min_cutoff=bad)


def test_negative_beta_rejected() -> None:
    with pytest.raises(ValueError):
        OneEuroFilter(beta=-0.1)

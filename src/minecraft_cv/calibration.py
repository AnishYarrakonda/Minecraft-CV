"""Auto-calibration math + safe config persistence for the spatial joysticks (Task 3).

Hand-tuning ``joystick.deadzone_radius`` and ``joystick.sensitivity`` in ``config.yaml`` is
tedious and error-prone. The ``mcv-calibrate`` wizard (see ``cli.py``) instead samples the
user's anchor position across a few guided poses — *neutral*, then a full push in each cardinal
direction — and this module turns those samples into concrete settings:

  * **deadzone_radius** from the resting-hand jitter at neutral, so tremor never leaks into
    movement (the same idea as the dynamic deadzone, but baked into the static config).
  * **sensitivity** from the user's full physical reach, so output saturates exactly when the
    hand reaches its comfortable extent — mitigating Gorilla Arm without manual guessing.

All positions are **normalized frame coordinates** (``[0, 1]`` in x/y), matching MediaPipe
landmark / anchor space. The sample-collection loop lives in the CLI; the math here is pure and
deterministic so it is unit-testable without a camera (see ``tests/test_calibration.py``).

Persistence is comment-lossy but value-safe: we load the existing YAML into a dict, deep-merge
the computed ``joystick`` keys (leaving every other setting untouched), validate the result
through :class:`~minecraft_cv.config.Settings`, then write atomically (temp file + ``os.replace``)
so a crash mid-write can never truncate the user's config.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# The cardinal pose names the wizard collects, in prompt order.
REACH_POSES: tuple[str, ...] = ("forward", "back", "left", "right")


@dataclass(frozen=True)
class CalibrationResult:
    """Computed joystick parameters plus the raw measurements behind them.

    Attributes:
        neutral: Resting anchor position ``(x, y)`` in normalized frame coords.
        deadzone_radius: Recommended spherical deadzone radius (normalized units).
        sensitivity: Recommended travel-to-saturation gain (output saturates once the
            displacement beyond the deadzone reaches ``1 / sensitivity``).
        resting_jitter: 95th-percentile resting displacement from ``neutral`` (normalized).
        mean_reach: Mean full-push distance from ``neutral`` across the cardinal poses.
    """

    neutral: tuple[float, float]
    deadzone_radius: float
    sensitivity: float
    resting_jitter: float
    mean_reach: float

    def joystick_overrides(self) -> dict[str, float]:
        """The subset of ``joystick`` settings this calibration writes back to config."""
        return {
            "deadzone_radius": round(self.deadzone_radius, 4),
            "sensitivity": round(self.sensitivity, 4),
        }


def _as_xy(samples: Sequence[Any] | np.ndarray) -> np.ndarray:
    """Coerce a sequence of ``(x, y)`` samples to ``(N, 2)`` float (empty -> ``(0, 2)``)."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return arr.reshape(-1, 2)


def compute_calibration(
    neutral_samples: Sequence[Any] | np.ndarray,
    reach_samples: Mapping[str, Sequence[Any] | np.ndarray],
    *,
    deadzone_margin: float = 1.5,
    deadzone_floor: float = 0.01,
    max_sensitivity: float = 50.0,
) -> CalibrationResult:
    """Derive joystick settings from collected pose samples.

    Args:
        neutral_samples: ``(N, 2)`` resting anchor positions (hand held still at center).
        reach_samples: Map of pose name -> ``(M, 2)`` positions captured while the hand is
            pushed fully in that direction. Only the magnitudes matter; direction labels are
            informational. Empty poses are ignored.
        deadzone_margin: Multiplier on the resting jitter that sets the deadzone radius.
        deadzone_floor: Minimum deadzone radius, so a perfectly steady hand still gets a small
            buffer against single-frame spikes.
        max_sensitivity: Upper clamp on the computed sensitivity (guards against a degenerate
            reach ≈ deadzone yielding an enormous gain).

    Returns:
        A :class:`CalibrationResult`.

    Raises:
        ValueError: If no neutral samples were provided (cannot establish a neutral).
    """
    neutral_arr = _as_xy(neutral_samples)
    if neutral_arr.shape[0] == 0:
        raise ValueError("Calibration needs at least one 'neutral' sample to set the origin.")

    neutral = neutral_arr.mean(axis=0)
    resting_radii = np.linalg.norm(neutral_arr - neutral, axis=1)
    jitter = float(np.percentile(resting_radii, 95)) if resting_radii.size else 0.0
    deadzone = max(deadzone_floor, jitter * deadzone_margin)

    reaches: list[float] = []
    for samples in reach_samples.values():
        arr = _as_xy(samples)
        if arr.shape[0] == 0:
            continue
        reaches.append(float(np.linalg.norm(arr - neutral, axis=1).max()))
    mean_reach = float(np.mean(reaches)) if reaches else deadzone * 2.0

    travel = mean_reach - deadzone
    sensitivity = max_sensitivity if travel <= 1e-6 else min(max_sensitivity, 1.0 / travel)

    return CalibrationResult(
        neutral=(float(neutral[0]), float(neutral[1])),
        deadzone_radius=float(deadzone),
        sensitivity=float(sensitivity),
        resting_jitter=jitter,
        mean_reach=mean_reach,
    )


def merge_calibration(existing: dict[str, Any], result: CalibrationResult) -> dict[str, Any]:
    """Return a deep copy of ``existing`` config data with the calibrated joystick keys merged.

    Every other section and key is preserved exactly. Only ``joystick.deadzone_radius`` and
    ``joystick.sensitivity`` are overwritten.
    """
    merged = copy.deepcopy(existing)
    joystick = dict(merged.get("joystick") or {})
    joystick.update(result.joystick_overrides())
    merged["joystick"] = joystick
    return merged


def load_config_data(path: str | Path) -> dict[str, Any]:
    """Load a ``config.yaml`` into a plain dict (``{}`` if the file is missing or empty)."""
    p = Path(path)
    if not p.is_file():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def save_config_data(path: str | Path, data: dict[str, Any]) -> None:
    """Atomically write ``data`` back to ``path`` as YAML (temp file + ``os.replace``).

    Comments in the original file are not preserved (PyYAML round-trips values only); all
    setting *values* are. The atomic replace guarantees the config is never left truncated.
    """
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))
    os.replace(tmp, p)


__all__ = [
    "REACH_POSES",
    "CalibrationResult",
    "compute_calibration",
    "load_config_data",
    "merge_calibration",
    "save_config_data",
]

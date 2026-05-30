"""One-Euro filter for smoothing the right-hand mouse-look signal.

Raw spatial-translation output jitters frame-to-frame because the underlying MediaPipe
landmarks jitter. A plain EMA (the joystick's ``smoothing``) trades jitter for lag uniformly:
turn it up to kill chatter when the hand is still and you also add lag when the hand moves
fast — exactly when low latency matters most for camera look.

The **One-Euro filter** (Casiez, Roussel, Vogel 2012) is velocity-adaptive: it cuts jitter
hard when the signal is slow (hand near rest) and relaxes the cutoff as the signal speeds up,
so fast looks stay responsive. This is the right tool for mouse-look where both steadiness at
rest and snappiness in motion are required.

All math is on small 2D ``float`` vectors and allocation-light, suitable for the per-frame
hot path. The filter is deterministic given an explicit timestamp, so it is unit-testable
without a clock (see ``tests/test_one_euro.py``).

Units: the filtered quantity is the joystick **output vector** (abstract joystick units,
``(dx, dy)``), filtered just before it is scaled to relative mouse-look pixels. Timestamps are
in seconds.
"""

from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    """Smoothing factor for a first-order low-pass with the given cutoff and step.

    Args:
        cutoff: Low-pass cutoff frequency in Hz (>0).
        dt: Time since the previous sample in seconds (>0).

    Returns:
        The EMA weight ``alpha`` in ``(0, 1]`` for ``y = alpha*x + (1-alpha)*y_prev``.
    """
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Velocity-adaptive low-pass filter over a 2D vector signal.

    The cutoff frequency rises with the estimated signal speed:
    ``cutoff = min_cutoff + beta * |dx/dt|``. Small ``min_cutoff`` removes jitter at rest;
    larger ``beta`` makes the filter follow fast motion with less lag.

    Attributes:
        min_cutoff: Baseline cutoff (Hz) applied when the signal is stationary. Lower =
            smoother at rest (and laggier).
        beta: Speed coefficient. Higher = less lag during fast motion (and more jitter).
        d_cutoff: Cutoff (Hz) for the internal derivative (speed) estimate.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        """Construct a One-Euro filter.

        Args:
            min_cutoff: Baseline cutoff frequency in Hz (>0).
            beta: Speed coefficient (>=0); 0 reduces this to a fixed-cutoff low-pass.
            d_cutoff: Derivative cutoff frequency in Hz (>0).
        """
        if min_cutoff <= 0.0:
            raise ValueError(f"min_cutoff must be > 0 (got {min_cutoff}).")
        if d_cutoff <= 0.0:
            raise ValueError(f"d_cutoff must be > 0 (got {d_cutoff}).")
        if beta < 0.0:
            raise ValueError(f"beta must be >= 0 (got {beta}).")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    def reset(self) -> None:
        """Forget all history so the next sample passes through unfiltered.

        Called on tracking dropout / recenter so a stale pre-dropout velocity estimate does
        not jerk the cursor when the hand re-enters the frame.
        """
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def filter(self, x: np.ndarray, timestamp: float) -> np.ndarray:
        """Filter one 2D sample.

        Args:
            x: ``(2,)`` raw sample (joystick output ``(dx, dy)``).
            timestamp: Monotonic time of this sample in seconds. Must be non-decreasing.

        Returns:
            The filtered ``(2,)`` vector. The first sample after construction/``reset`` is
            returned unchanged (the filter seeds itself, so there is no start-up lag).
        """
        x = np.asarray(x, dtype=np.float64)[:2]
        if self._x_prev is None or self._t_prev is None or self._dx_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros(2, dtype=np.float64)
            self._t_prev = float(timestamp)
            return x.copy()

        dt = float(timestamp) - self._t_prev
        if dt <= 0.0:
            # Non-advancing clock: skip filtering, return the raw sample to avoid div-by-zero.
            return x.copy()

        # Low-pass the derivative (speed) estimate.
        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Adaptive cutoff: faster signal -> higher cutoff -> less smoothing -> less lag.
        speed = float(np.linalg.norm(dx_hat))
        cutoff = self.min_cutoff + self.beta * speed
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = float(timestamp)
        return x_hat


__all__ = ["OneEuroFilter"]

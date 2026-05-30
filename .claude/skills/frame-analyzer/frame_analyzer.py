#!/usr/bin/env python3
"""Offline frame/clip analyzer for the minecraft_cv hand-tracking pipeline.

Runs the tracking + Schmitt-trigger gesture stack over a recorded clip or still
image so gesture behavior is *deterministic and reproducible* — no live camera and
no OS input emission. See ``SKILL.md`` in this directory for usage and intent.

If the real ``minecraft_cv`` package is importable, its tracker + gesture state
machine are used. Otherwise the script falls back to a self-contained reference
implementation so threshold sweeps work even on a fresh scaffold; that fallback
doubles as a spec for ``src/minecraft_cv/gestures/``.

This module never emits keyboard/mouse events. It is safe to run while debugging.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

try:  # OpenCV is optional for the --self-test path
    import cv2
except ImportError:  # pragma: no cover - reported clearly at runtime
    cv2 = None  # type: ignore[assignment]


# --- MediaPipe Hands landmark indices (single source of truth) ----------------
WRIST = 0
THUMB_TIP = 4
MIDDLE_MCP = 9
FINGERTIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}

# gesture -> the fingertip whose thumb-pinch triggers it (per the design doc)
GESTURE_FINGER = {
    "jump": "index",     # left hand
    "sneak": "middle",   # left hand
    "attack": "index",   # right hand
    "use": "middle",     # right hand
    "hotbar_next": "ring",
    "hotbar_prev": "pinky",
}


class PinchState(Enum):
    RELEASED = "RELEASED"
    HOLDING = "HOLDING"


@dataclass
class SchmittTrigger:
    """Hysteresis gate for one discrete pinch gesture.

    Invariant: ``t_release > t_engage``. Distances are unitless ratios already
    normalized by hand scale, so thresholds are scale-invariant.
    """

    t_engage: float
    t_release: float
    state: PinchState = PinchState.RELEASED

    def __post_init__(self) -> None:
        if not self.t_release > self.t_engage:
            raise ValueError(
                f"T_release ({self.t_release}) must be strictly greater than "
                f"T_engage ({self.t_engage}); equal/inverted thresholds reintroduce "
                "the jitter chatter the Schmitt trigger exists to prevent."
            )

    def update(self, distance: float) -> str | None:
        """Feed one normalized distance; return 'KEY_DOWN'/'KEY_UP' on transition."""
        if self.state is PinchState.RELEASED and distance < self.t_engage:
            self.state = PinchState.HOLDING
            return "KEY_DOWN"
        if self.state is PinchState.HOLDING and distance > self.t_release:
            self.state = PinchState.RELEASED
            return "KEY_UP"
        return None


@dataclass
class FrameRecord:
    index: int
    distances: dict[str, float]
    state: PinchState
    transition: str | None = None


@dataclass
class AnalysisResult:
    gesture: str
    records: list[FrameRecord] = field(default_factory=list)
    stage_times_ms: dict[str, list[float]] = field(default_factory=dict)

    @property
    def transitions(self) -> list[FrameRecord]:
        return [r for r in self.records if r.transition is not None]

    def chatter_count(self, window: int = 6) -> int:
        """Count RELEASED->HOLDING->RELEASED flips inside `window` frames."""
        ts = self.transitions
        flips = 0
        for i in range(len(ts) - 2):
            a, b, c = ts[i], ts[i + 1], ts[i + 2]
            if (
                a.transition == "KEY_DOWN"
                and b.transition == "KEY_UP"
                and c.transition == "KEY_DOWN"
                and (c.index - a.index) <= window
            ):
                flips += 1
        return flips


# --- landmark math (vectorized; the reference impl for src/) -------------------
def normalized_distances(landmarks: np.ndarray) -> dict[str, float]:
    """Thumb-to-fingertip distances normalized by hand scale.

    Args:
        landmarks: (21, 3) float32 array of (x, y, z) hand keypoints.

    Returns:
        Mapping finger-name -> distance ratio (invariant to camera distance).
    """
    scale = float(np.linalg.norm(landmarks[MIDDLE_MCP] - landmarks[WRIST])) or 1e-6
    thumb = landmarks[THUMB_TIP]
    tips = np.array([landmarks[i] for i in FINGERTIPS.values()], dtype=np.float32)
    raw = np.linalg.norm(tips - thumb, axis=1) / scale  # one vectorized call
    return dict(zip(FINGERTIPS.keys(), (float(d) for d in raw)))


# --- frame sources ------------------------------------------------------------
def iter_frames(path: Path):
    """Yield BGR frames from an image or video file."""
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required to read clips/images.")
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        yield img
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open clip: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def _load_tracker(backend: str, device: str):
    """Return a callable frame(BGR)->(21,3) landmarks, real or stubbed."""
    try:
        from minecraft_cv.tracking import HandTracker  # type: ignore

        tracker = HandTracker.create(backend=backend, device=device)
        return lambda frame: tracker.landmarks(frame)
    except Exception:  # pragma: no cover - scaffold fallback
        rng = np.random.default_rng(0)

        def _stub(frame: np.ndarray) -> np.ndarray:
            # Deterministic-ish synthetic landmarks so threshold sweeps still run.
            base = rng.normal(0.5, 0.02, size=(21, 3)).astype(np.float32)
            return base

        print(
            "[frame-analyzer] minecraft_cv not importable — using synthetic landmark "
            "stub. Threshold/state-machine logic is still exercised.",
            file=sys.stderr,
        )
        return _stub


# --- main analysis ------------------------------------------------------------
def analyze(args: argparse.Namespace) -> AnalysisResult:
    finger = GESTURE_FINGER[args.gesture]
    trigger = SchmittTrigger(t_engage=args.engage, t_release=args.release)
    tracker = _load_tracker(args.backend, args.device)
    result = AnalysisResult(gesture=args.gesture)
    result.stage_times_ms = {"track": [], "gesture": []}

    for i, frame in enumerate(iter_frames(Path(args.path))):
        t0 = time.perf_counter()
        landmarks = tracker(frame)
        t1 = time.perf_counter()
        dists = normalized_distances(landmarks)
        transition = trigger.update(dists[finger])
        t2 = time.perf_counter()

        result.stage_times_ms["track"].append((t1 - t0) * 1e3)
        result.stage_times_ms["gesture"].append((t2 - t1) * 1e3)
        result.records.append(
            FrameRecord(index=i, distances=dists, state=trigger.state, transition=transition)
        )
    return result


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = min(len(s) - 1, int(round((p / 100) * (len(s) - 1))))
    return s[k]


def report(result: AnalysisResult, args: argparse.Namespace) -> None:
    rows = result.transitions if args.transitions else result.records
    finger = GESTURE_FINGER[args.gesture]
    print(f"\n=== frame-analyzer: gesture={args.gesture} finger={finger} "
          f"T_engage={args.engage} T_release={args.release} ===")
    print(f"frames={len(result.records)}  transitions={len(result.transitions)}  "
          f"chatter_events={result.chatter_count()}")
    if result.chatter_count() > 0:
        print("  ! CHATTER DETECTED — widen the hysteresis band or add One-Euro "
              "smoothing upstream (see rules §3).")

    print(f"\n{'frame':>6} {'dist':>8}  state        transition")
    for r in rows:
        t = r.transition or ""
        print(f"{r.index:>6} {r.distances[finger]:>8.3f}  {r.state.value:<11}  {t}")

    if args.timing:
        print("\n--- timing (ms) ---")
        for stage, xs in result.stage_times_ms.items():
            if not xs:
                continue
            print(f"{stage:>8}: p50={_pct(xs,50):6.2f}  p95={_pct(xs,95):6.2f}  "
                  f"p99={_pct(xs,99):6.2f}  mean={statistics.fmean(xs):6.2f}")
        total = [a + b for a, b in zip(*result.stage_times_ms.values())] \
            if all(result.stage_times_ms.values()) else []
        if total:
            fps = 1000.0 / statistics.fmean(total)
            print(f"   total: mean={statistics.fmean(total):6.2f} ms  (~{fps:5.1f} FPS)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline frame/clip analyzer for minecraft_cv.")
    p.add_argument("path", help="clip (.mp4/.mov) or image (.png/.jpg) to analyze")
    p.add_argument("--hand", choices=["left", "right"], default="right")
    p.add_argument("--gesture", choices=sorted(GESTURE_FINGER), default="attack")
    p.add_argument("--engage", type=float, default=0.30, help="T_engage (normalized)")
    p.add_argument("--release", type=float, default=0.45, help="T_release (normalized)")
    p.add_argument("--backend", default="mediapipe")
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--transitions", action="store_true", help="only print state changes")
    p.add_argument("--timing", action="store_true", help="print per-stage latency stats")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = analyze(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[frame-analyzer] error: {exc}", file=sys.stderr)
        return 1
    report(result, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line entrypoints: mcv-run, mcv-calibrate, mcv-analyze, mcv-bench.

Heavy libraries (OpenCV, MediaPipe) are imported lazily inside the command bodies so that
``import minecraft_cv.cli`` and ``--help`` stay fast and dependency-light.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from minecraft_cv.config import Settings

DEFAULT_CONFIG = "config.yaml"


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG, help="path to config.yaml (default: ./config.yaml)"
    )


def _load_settings(path: str, overrides: dict[str, Any] | None = None) -> Settings:
    config_path = path if Path(path).is_file() else None
    if config_path is None and path != DEFAULT_CONFIG:
        print(f"[mcv] warning: config '{path}' not found; using defaults.", file=sys.stderr)
    return Settings.load(config_path, overrides=overrides)


# --- mcv-run ------------------------------------------------------------------
def main_run(argv: list[str] | None = None) -> int:
    """Run the live controller (camera or clip). Default emits NO input."""
    p = argparse.ArgumentParser(prog="mcv-run", description="Run the gesture controller.")
    _add_config_arg(p)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--input", action="store_true", help="emit real OS input (live)")
    group.add_argument(
        "--no-input", action="store_true", help="force dry-run (NullEmitter); the default"
    )
    p.add_argument("--debug-overlay", action="store_true", help="show the OpenCV debug window")
    p.add_argument("--clip", help="run on a recorded clip instead of the live camera")
    args = p.parse_args(argv)

    overrides: dict[str, Any] = {"input": {}, "debug": {}}
    if args.input:
        overrides["input"]["enabled"] = True
    if args.no_input:
        overrides["input"]["enabled"] = False
    if args.debug_overlay:
        overrides["debug"]["overlay"] = True
    settings = _load_settings(args.config, overrides)

    from minecraft_cv.pipeline import run_pipeline

    source = None
    if args.clip:
        from minecraft_cv.capture.source import ClipSource

        source = ClipSource(args.clip)

    mode = "LIVE (emitting input)" if settings.input.enabled else "DRY-RUN (no input)"
    print(f"[mcv-run] {mode}; overlay={settings.debug.overlay}; "
          f"source={'clip:' + args.clip if args.clip else 'camera:' + str(settings.camera.index)}")
    try:
        run_pipeline(settings, source=source)
    except KeyboardInterrupt:
        print("\n[mcv-run] interrupted; releasing all input.", file=sys.stderr)
    except (PermissionError, RuntimeError, FileNotFoundError) as exc:
        print(f"[mcv-run] error: {exc}", file=sys.stderr)
        return 1
    return 0


# --- mcv-calibrate ------------------------------------------------------------
def main_calibrate(argv: list[str] | None = None) -> int:
    """Print live normalized thumb-to-fingertip distances to help tune thresholds."""
    p = argparse.ArgumentParser(prog="mcv-calibrate", description="Tune Schmitt thresholds.")
    _add_config_arg(p)
    p.add_argument("--clip", help="calibrate from a clip instead of the live camera")
    p.add_argument("--hand", choices=["Left", "Right"], default="Right")
    args = p.parse_args(argv)
    settings = _load_settings(args.config)

    import cv2

    from minecraft_cv.capture.source import AVFoundationSource, ClipSource
    from minecraft_cv.gestures.pinch import normalized_distances
    from minecraft_cv.tracking.tracker import HandTracker

    source = (
        ClipSource(args.clip)
        if args.clip
        else AVFoundationSource(
            settings.camera.index, settings.camera.width, settings.camera.height, settings.camera.fps
        )
    )
    tracker = HandTracker.create(settings.tracking.backend, settings.tracking.device)
    print("[mcv-calibrate] move your fingers; Ctrl-C to stop. Pick T_engage below the "
          "pinched distance and T_release above the open distance.")
    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for hand in tracker.detect(rgb):
                if hand.handedness != args.hand:
                    continue
                d = normalized_distances(hand.landmarks)
                print(
                    f"  index={d['index']:.3f}  middle={d['middle']:.3f}  "
                    f"ring={d['ring']:.3f}  pinky={d['pinky']:.3f}",
                    end="\r",
                )
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[mcv-calibrate] done.")
    finally:
        source.release()
        tracker.close()
    return 0


# --- mcv-analyze --------------------------------------------------------------
def main_analyze(argv: list[str] | None = None) -> int:
    """Offline: run tracking + a Schmitt trigger over a clip and print transitions/timing."""
    p = argparse.ArgumentParser(prog="mcv-analyze", description="Offline clip analysis.")
    p.add_argument("path", help="clip (.mp4/.mov) or image (.png/.jpg) to analyze")
    p.add_argument("--hand", choices=["Left", "Right"], default="Right")
    p.add_argument("--gesture", default="attack", help="gesture name (finger inferred)")
    p.add_argument("--finger", default="index", choices=["index", "middle", "ring", "pinky"])
    p.add_argument("--engage", type=float, default=0.30)
    p.add_argument("--release", type=float, default=0.45)
    p.add_argument("--backend", default="mediapipe")
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--transitions", action="store_true", help="print only state changes")
    p.add_argument("--timing", action="store_true", help="print per-stage latency stats")
    args = p.parse_args(argv)

    import cv2

    from minecraft_cv.capture.source import ClipSource
    from minecraft_cv.gestures.pinch import normalized_distances
    from minecraft_cv.gestures.schmitt import SchmittTrigger
    from minecraft_cv.tracking.tracker import HandTracker

    try:
        source = ClipSource(args.path)
        tracker = HandTracker.create(args.backend, args.device)
        trigger = SchmittTrigger(t_engage=args.engage, t_release=args.release)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[mcv-analyze] error: {exc}", file=sys.stderr)
        return 1

    track_ms: list[float] = []
    gesture_ms: list[float] = []
    n_frames = n_transitions = 0
    print(f"=== mcv-analyze gesture={args.gesture} finger={args.finger} "
          f"T_engage={args.engage} T_release={args.release} ===")
    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            n_frames += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t0 = time.perf_counter()
            hands = tracker.detect(rgb)
            t1 = time.perf_counter()
            lm = next((h.landmarks for h in hands if h.handedness == args.hand), None)
            if lm is None:
                trans = trigger.reset()
            else:
                trans = trigger.update(normalized_distances(lm)[args.finger])
            t2 = time.perf_counter()
            track_ms.append((t1 - t0) * 1e3)
            gesture_ms.append((t2 - t1) * 1e3)
            if trans is not None:
                n_transitions += 1
                print(f"  frame {n_frames:>5}: {trans}")
            elif not args.transitions and lm is not None:
                pass
    finally:
        source.release()
        tracker.close()

    print(f"frames={n_frames}  transitions={n_transitions}")
    if args.timing and track_ms:
        for name, xs in (("track", track_ms), ("gesture", gesture_ms)):
            print(f"  {name:>8}: p50={_pct(xs, 50):6.2f}  p95={_pct(xs, 95):6.2f}  "
                  f"p99={_pct(xs, 99):6.2f}  mean={statistics.fmean(xs):6.2f} ms")
    return 0


# --- mcv-bench ----------------------------------------------------------------
def main_bench(argv: list[str] | None = None) -> int:
    """Benchmark the tracker over N frames with warmup + p50/p95/p99 reporting."""
    p = argparse.ArgumentParser(prog="mcv-bench", description="Benchmark the tracking backend.")
    p.add_argument("--backend", default="mediapipe")
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--frames", type=int, default=500)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--clip", help="benchmark on a clip; otherwise synthetic noise frames")
    args = p.parse_args(argv)

    from minecraft_cv.tracking.tracker import HandTracker

    try:
        tracker = HandTracker.create(args.backend, args.device)
    except (RuntimeError, ValueError, NotImplementedError) as exc:
        print(f"[mcv-bench] error: {exc}", file=sys.stderr)
        return 1

    frames = _bench_frames(args)
    times: list[float] = []
    try:
        for i, frame in enumerate(frames):
            _maybe_mps_sync(args.device)
            t0 = time.perf_counter()
            tracker.detect(frame)
            _maybe_mps_sync(args.device)
            dt = (time.perf_counter() - t0) * 1e3
            if i >= args.warmup:  # discard warmup (shader compile / allocation)
                times.append(dt)
    finally:
        tracker.close()

    if not times:
        print("[mcv-bench] no timed frames.", file=sys.stderr)
        return 1
    fps = 1000.0 / statistics.fmean(times)
    print(f"=== mcv-bench backend={args.backend} device={args.device} "
          f"frames={len(times)} (warmup={args.warmup}) ===")
    print(f"  detect: p50={_pct(times, 50):6.2f}  p95={_pct(times, 95):6.2f}  "
          f"p99={_pct(times, 99):6.2f}  mean={statistics.fmean(times):6.2f} ms  (~{fps:5.1f} FPS)")
    return 0


def _bench_frames(args: argparse.Namespace) -> list[np.ndarray]:
    if args.clip:
        import cv2

        from minecraft_cv.capture.source import ClipSource

        source = ClipSource(args.clip)
        out: list[np.ndarray] = []
        try:
            while len(out) < args.frames:
                frame = source.read()
                if frame is None:
                    break
                out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            source.release()
        return out or _synthetic_frames(args.frames)
    return _synthetic_frames(args.frames)


def _synthetic_frames(n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8) for _ in range(n)]


def _maybe_mps_sync(device: str) -> None:
    if device != "mps":
        return
    try:  # pragma: no cover - torch is optional
        import torch

        if torch.backends.mps.is_available():
            torch.mps.synchronize()
    except ImportError:
        pass


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = min(len(s) - 1, int(round((p / 100) * (len(s) - 1))))
    return s[k]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main_run())

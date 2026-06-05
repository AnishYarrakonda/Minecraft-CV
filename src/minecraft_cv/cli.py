"""Command-line entrypoints: mcv-run, mcv-calibrate, mcv-analyze, mcv-bench.

Heavy libraries (OpenCV, MediaPipe) are imported lazily inside the command bodies so that
``import minecraft_cv.cli`` and ``--help`` stay fast and dependency-light.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# Prevent mediapipe from taking the GPU landmark-projection path when Qt has already
# initialised OpenGL. Must be set before PySide6 / QApplication is created.
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

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


# --- Unified Entrypoint -------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Unified entrypoint: mcv <command> [args...]."""
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print("Usage: mcv <command> [args...]")
        print("\nCommands:")
        print("  ui         Launch the polished desktop app")
        print("  overlay    Compact always-on-top overlay for use during gameplay")
        print("  run        Run the live gesture controller (headless / cv2 overlay)")
        print("  analyze    Offline clip analysis")
        print("  bench      Benchmark tracking backend latency")
        print("  doctor     Check system permissions and config")
        print("  gestures   Print the gesture reference card")
        return 0

    cmd = argv[0]
    sub_argv = argv[1:]

    if cmd == "ui":
        return main_ui(sub_argv)
    elif cmd == "overlay":
        return main_overlay(sub_argv)
    elif cmd == "run":
        return main_run(sub_argv)
    elif cmd == "analyze":
        return main_analyze(sub_argv)
    elif cmd == "bench":
        return main_bench(sub_argv)
    elif cmd == "doctor":
        return main_doctor(sub_argv)
    elif cmd == "gestures":
        return main_gestures(sub_argv)
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        return 1


# --- mcv doctor ---------------------------------------------------------------
def main_doctor(argv: list[str] | None = None) -> int:
    """Check AVFoundation permission and print config."""
    p = argparse.ArgumentParser(
        prog="mcv doctor", description="Check system permissions and config."
    )
    _add_config_arg(p)
    args = p.parse_args(argv)

    import cv2

    print("=== mcv doctor ===")
    print("\n--- Camera Access (AVFoundation) ---")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FAIL] Cannot open default camera. Check Terminal Camera permissions.")
    else:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[FAIL] Camera opened but cannot read frames. Check permissions.")
        else:
            print("[PASS] Camera access OK.")
        cap.release()

    print("\n--- Configuration ---")
    try:
        settings = _load_settings(args.config)
        print(settings.model_dump_json(indent=2))
    except Exception as e:
        print(f"[FAIL] Error loading config: {e}")

    return 0


# --- mcv gestures -------------------------------------------------------------
def main_gestures(argv: list[str] | None = None) -> int:
    """Print the gesture reference card."""
    print("=== minecraft_cv Gesture Reference ===")
    print("\n[Screen Joysticks]")
    print("  Left hand  -> WASD movement (8 slices)")
    print("  Right thumb -> Direct mouse/cursor movement, no deadzone")
    print("\n[Left Hand - Actions]")
    print("  Index Pinch  -> Jump (Space)")
    print("  Middle Pinch -> Inventory (E)")
    print("  Ring Pinch   -> Throw Item (Q)")
    print("  Pinky Pinch  -> Sneak (Shift)")
    print("  Peace Sign   -> Relocalize movement")
    print("\n[Right Hand - Holds]")
    print("  Index Pinch  -> Attack/Mine (Left Click)")
    print("  Middle Pinch -> Use/Place (Right Click)")
    print("  Ring Pinch   -> Hotbar Scroll Up")
    print("  Pinky Pinch  -> Hotbar Scroll Down")
    print("  Peace Sign   -> Reset thumb cursor point; held peace clutches mouse output")
    print("\nOpen hands are neutral; there is no modal inventory control mode.")
    return 0


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
    overlay_group = p.add_mutually_exclusive_group()
    overlay_group.add_argument(
        "--debug-overlay", action="store_true", help="show the OpenCV debug window"
    )
    overlay_group.add_argument(
        "--no-debug-overlay",
        action="store_true",
        help="hide the OpenCV debug window even if config enables it",
    )
    p.add_argument("--clip", help="run on a recorded clip instead of the live camera")
    args = p.parse_args(argv)

    overrides: dict[str, Any] = {"input": {}, "debug": {}}
    if args.input:
        overrides["input"]["enabled"] = True
    if args.no_input:
        overrides["input"]["enabled"] = False
    if args.debug_overlay:
        overrides["debug"]["overlay"] = True
    if args.no_debug_overlay:
        overrides["debug"]["overlay"] = False
    settings = _load_settings(args.config, overrides)

    from minecraft_cv.pipeline import run_pipeline

    source = None
    if args.clip:
        from minecraft_cv.capture.source import ClipSource

        source = ClipSource(args.clip)

    mode = "LIVE (emitting input)" if settings.input.enabled else "DRY-RUN (no input)"
    print(
        f"[mcv-run] {mode}; overlay={settings.debug.overlay}; "
        f"source={'clip:' + args.clip if args.clip else 'camera:' + str(settings.camera.index)}",
        flush=True,
    )
    try:
        run_pipeline(settings, source=source)
    except KeyboardInterrupt:
        print("\n[mcv-run] interrupted; releasing all input.", file=sys.stderr)
    except (PermissionError, RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"[mcv-run] error: {exc}", file=sys.stderr)
        return 1
    return 0


# --- mcv ui -------------------------------------------------------------------
_UI_INSTALL_HINT = (
    "[mcv ui] The desktop app needs PySide6, which is not installed.\n"
    "         Install the project (pulls in PySide6):  pip install -e .\n"
    "         or just PySide6:                         pip install PySide6"
)


def main_ui(argv: list[str] | None = None) -> int:
    """Launch the polished PySide6 desktop app (camera + live HUD). Defaults to Dry-Run."""
    p = argparse.ArgumentParser(prog="mcv ui", description="Launch the desktop app.")
    _add_config_arg(p)
    p.add_argument("--clip", help="drive the app from a recorded clip instead of the camera")
    args = p.parse_args(argv)
    settings = _load_settings(args.config)

    try:
        from minecraft_cv.ui.app import run_app
    except ImportError:
        print(_UI_INSTALL_HINT, file=sys.stderr)
        return 1

    source = None
    if args.clip:
        from minecraft_cv.capture.source import ClipSource

        source = ClipSource(args.clip)
    return run_app(settings, source=source)


# --- mcv overlay --------------------------------------------------------------
_OVERLAY_INSTALL_HINT = (
    "[mcv overlay] The overlay needs PySide6, which is not installed.\n"
    "              Install the project (pulls in PySide6):  pip install -e .\n"
    "              or just PySide6:                         pip install PySide6"
)


def main_overlay(argv: list[str] | None = None) -> int:
    """Launch the compact always-on-top overlay window. Defaults to the mode in config."""
    p = argparse.ArgumentParser(
        prog="mcv overlay",
        description="Compact always-on-top overlay for use alongside Minecraft.",
    )
    _add_config_arg(p)
    p.add_argument(
        "--live", action="store_true", help="enable real OS input on launch (default: dry-run)"
    )
    args = p.parse_args(argv)

    overrides: dict[str, Any] = {}
    if args.live:
        overrides["input"] = {"enabled": True}
    settings = _load_settings(args.config, overrides if overrides else None)

    try:
        from minecraft_cv.ui.overlay import run_overlay
    except ImportError:
        print(_OVERLAY_INSTALL_HINT, file=sys.stderr)
        return 1

    return run_overlay(settings)


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
    """Benchmark the tracker over N frames; emit a detailed JSON profiling report (Task 4)."""
    p = argparse.ArgumentParser(prog="mcv-bench", description="Benchmark the tracking backend.")
    p.add_argument("--backend", default="mediapipe")
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--frames", type=int, default=500)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--clip", help="benchmark on a clip; otherwise synthetic noise frames")
    p.add_argument("--target-fps", type=float, default=60.0,
                   help="FPS the mean frame time is checked against (default: 60)")
    p.add_argument("--json", dest="json_path", nargs="?", const="-", default=None,
                   help="write the full JSON report to PATH (or stdout if given with no value)")
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

    summary = _summarize(times)
    report = {
        "backend": args.backend,
        "device": args.device,
        "frames": len(times),
        "warmup": args.warmup,
        "source": f"clip:{args.clip}" if args.clip else "synthetic",
        "target_fps": args.target_fps,
        "meets_target": summary["fps_mean"] >= args.target_fps,
        "detect_ms": summary,
    }
    print(f"=== mcv-bench backend={args.backend} device={args.device} "
          f"frames={len(times)} (warmup={args.warmup}) ===")
    print(f"  detect: p50={summary['p50']:6.2f}  p95={summary['p95']:6.2f}  "
          f"p99={summary['p99']:6.2f}  mean={summary['mean']:6.2f} ms  "
          f"jitter(std)={summary['jitter_std']:5.2f}  (~{summary['fps_mean']:5.1f} FPS)")
    status = "PASS" if report["meets_target"] else "FAIL"
    print(f"  target {args.target_fps:.0f} FPS: {status}")

    if args.json_path is not None:
        blob = json.dumps(report, indent=2)
        if args.json_path == "-":
            print(blob)
        else:
            Path(args.json_path).write_text(blob + "\n")
            print(f"  wrote JSON report to {args.json_path}")
    return 0


def _summarize(times: list[float]) -> dict[str, float]:
    """Summary statistics for a list of per-frame latencies in milliseconds.

    Args:
        times: Per-frame detect latencies (ms). Must be non-empty.

    Returns:
        A dict with count, p50/p95/p99, mean, min, max, ``jitter_std`` (latency standard
        deviation — the tail-instability signal that matters for a real-time loop), and
        ``fps_mean`` (``1000 / mean``).
    """
    mean = statistics.fmean(times)
    return {
        "count": float(len(times)),
        "p50": _pct(times, 50),
        "p95": _pct(times, 95),
        "p99": _pct(times, 99),
        "mean": mean,
        "min": min(times),
        "max": max(times),
        "jitter_std": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "fps_mean": 1000.0 / mean if mean > 0 else float("inf"),
    }


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
    k = min(len(s) - 1, round((p / 100) * (len(s) - 1)))
    return s[k]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

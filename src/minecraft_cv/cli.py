"""Command-line entrypoints: mcv-run, mcv-calibrate, mcv-analyze, mcv-bench.

Heavy libraries (OpenCV, MediaPipe) are imported lazily inside the command bodies so that
``import minecraft_cv.cli`` and ``--help`` stay fast and dependency-light.
"""

from __future__ import annotations

import argparse
import json
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


# --- Unified Entrypoint -------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Unified entrypoint: mcv <command> [args...]."""
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print("Usage: mcv <command> [args...]")
        print("\nCommands:")
        print("  run        Run the live gesture controller")
        print("  calibrate  Auto-calibrate spatial joysticks or pinch thresholds")
        print("  analyze    Offline clip analysis")
        print("  bench      Benchmark tracking backend latency")
        print("  doctor     Check system permissions and config")
        print("  gestures   Print the gesture reference card")
        return 0

    cmd = argv[0]
    sub_argv = argv[1:]

    if cmd == "run":
        return main_run(sub_argv)
    elif cmd == "calibrate":
        return main_calibrate(sub_argv)
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
    print("\n[Calibrated Palm-Normal Thumbsticks]")
    print("  Left palm normal x/y  -> WASD movement")
    print("    right/left -> D/A, down/up -> W/S")
    print("  Right palm normal x/y -> Mouse look")
    print("  Run `mcv calibrate --apply` before `mcv run`.")
    print("\n[Left Hand - Actions]")
    print("  Index Pinch  -> Jump (Space)")
    print("  Middle Pinch -> Inventory (E)")
    print("  Ring Pinch   -> Throw Item (Q)")
    print("  Pinky Pinch  -> Swap Offhand (F)")
    print("  Ring+Pinky Curl -> Sneak (Shift)")
    print("\n[Right Hand - Holds]")
    print("  Index Pinch  -> Attack/Mine (Left Click)")
    print("  Middle Pinch -> Use/Place (Right Click)")
    print("  Ring Pinch   -> Hotbar Scroll Up")
    print("  Pinky Pinch  -> Hotbar Scroll Down")
    print("  Ring+Pinky Curl -> Sprint (Ctrl)")
    print("\nOpen palms are neutral; legacy inventory mode is disabled by default.")
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
    except (PermissionError, RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"[mcv-run] error: {exc}", file=sys.stderr)
        return 1
    return 0


# --- mcv-calibrate ------------------------------------------------------------
def main_calibrate(argv: list[str] | None = None) -> int:
    """Guided spatial-joystick calibration wizard.

    In palm-normal mode, walks the user through neutral plus comfortable palm-normal tilts
    and writes the required calibrated neutral/gain values. ``--mode anchor`` keeps the legacy
    anchor-position calibration.

    ``--pinch`` keeps the legacy live thumb-to-fingertip distance readout for Schmitt tuning.
    """
    p = argparse.ArgumentParser(
        prog="mcv-calibrate", description="Auto-calibrate the spatial joysticks (or tune pinch)."
    )
    _add_config_arg(p)
    p.add_argument("--clip", help="calibrate from a clip instead of the live camera")
    p.add_argument("--hand", choices=["Left", "Right"], default="Right",
                   help="MediaPipe handedness label to sample (default: Right)")
    p.add_argument("--anchor", choices=["wrist", "middle_mcp"], default=None,
                   help="anchor landmark (default: from config)")
    p.add_argument("--mode", choices=["palm-normal", "anchor"], default=None,
                   help="calibration mode (default: from joystick.mode)")
    p.add_argument("--frames-per-step", type=int, default=60,
                   help="samples to collect per pose (default: 60)")
    p.add_argument("--apply", action="store_true", help="write the result back to config.yaml")
    p.add_argument("--pinch", action="store_true",
                   help="legacy mode: print live thumb-to-fingertip distances instead")
    args = p.parse_args(argv)
    settings = _load_settings(args.config)

    if args.pinch:
        return _calibrate_pinch(args, settings)
    mode = args.mode or ("palm-normal" if settings.joystick.mode == "palm_normal" else "anchor")
    if mode == "palm-normal":
        return _calibrate_palm_normals(args, settings)
    return _calibrate_joysticks(args, settings)


def _calibrate_palm_normals(args: argparse.Namespace, settings: Settings) -> int:
    """Run the guided palm-normal wizard and optionally persist calibrated settings."""
    import cv2

    from minecraft_cv.calibration import (
        PALM_NORMAL_POSES,
        compute_palm_normal_calibration,
        load_config_data,
        merge_palm_normal_calibration,
        save_config_data,
    )
    from minecraft_cv.capture.source import AVFoundationSource, ClipSource
    from minecraft_cv.tracking.tracker import HandTracker

    live = not args.clip
    try:
        source = (
            ClipSource(args.clip)
            if args.clip
            else AVFoundationSource(
                settings.camera.index, settings.camera.width,
                settings.camera.height, settings.camera.fps,
            )
        )
        tracker = HandTracker.create(settings.tracking.backend, settings.tracking.device)
    except (FileNotFoundError, RuntimeError, PermissionError) as exc:
        print(f"[mcv-calibrate] error: {exc}", file=sys.stderr)
        return 1

    steps: list[tuple[str, str]] = [
        ("neutral", "Hold BOTH hands in your comfortable resting pose."),
        *[
            (pose, f"Tilt BOTH palm normals {pose.upper()} to a comfortable full reach.")
            for pose in PALM_NORMAL_POSES
        ],
    ]
    collected: dict[str, dict[str, list[np.ndarray]]] = {
        "left": {},
        "right": {},
    }
    print("=== mcv-calibrate: palm-normal joystick wizard ===")
    print(f"  hands=both  frames/step={args.frames_per_step}")
    try:
        for name, instruction in steps:
            print(f"\n[{name}] {instruction}")
            if live:
                for c in (3, 2, 1):
                    print(f"  capturing in {c}...", end="\r", flush=True)
                    time.sleep(1.0)
            samples = _collect_palm_normal_samples(
                source,
                tracker,
                args.frames_per_step,
                settings.camera.mirror,
                settings.tracking.swap_handedness,
                cv2,
            )
            for hand in ("left", "right"):
                collected[hand][name] = samples[hand]
            print(
                f"  captured left={len(samples['left'])} right={len(samples['right'])} "
                f"samples for '{name}'.        "
            )
    except KeyboardInterrupt:
        print("\n[mcv-calibrate] aborted; nothing written.")
        return 1
    finally:
        source.release()
        tracker.close()

    try:
        result = compute_palm_normal_calibration(
            collected,
            deadzone_margin=settings.joystick.dynamic_deadzone_margin,
        )
    except ValueError as exc:
        print(f"[mcv-calibrate] error: {exc} (were both hands visible?)", file=sys.stderr)
        return 1

    overrides = result.joystick_overrides()["palm_normal"]
    print("\n=== Result ===")
    print(f"  left neutral        = {overrides['left_neutral']}")
    print(f"  right neutral       = {overrides['right_neutral']}")
    print(f"  deadzone            = {overrides['deadzone']}")
    print(f"  left sensitivity    = {overrides['left_sensitivity']}")
    print(f"  right sensitivity   = {overrides['right_sensitivity']}")

    if not args.apply:
        print(
            "\n[mcv-calibrate] preview only. Re-run with --apply to write to "
            f"{args.config}."
        )
        return 0

    config_path = args.config
    if not Path(config_path).is_file():
        print(
            f"[mcv-calibrate] error: cannot apply; config '{config_path}' does not exist.",
            file=sys.stderr,
        )
        return 1
    merged = merge_palm_normal_calibration(load_config_data(config_path), result)
    try:
        Settings(**merged)
    except Exception as exc:  # noqa: BLE001 - surface any validation failure to the user
        print(f"[mcv-calibrate] error: computed config failed validation: {exc}",
              file=sys.stderr)
        return 1
    save_config_data(config_path, merged)
    print(f"[mcv-calibrate] wrote palm-normal calibration to {config_path}.")
    return 0


def _calibrate_joysticks(args: argparse.Namespace, settings: Settings) -> int:
    """Run the guided pose wizard and optionally persist the computed joystick settings."""
    import cv2

    from minecraft_cv.calibration import (
        REACH_POSES,
        compute_calibration,
        load_config_data,
        merge_calibration,
        save_config_data,
    )
    from minecraft_cv.capture.source import AVFoundationSource, ClipSource
    from minecraft_cv.tracking.tracker import HandTracker

    anchor = args.anchor or settings.joystick.anchor
    live = not args.clip
    try:
        source = (
            ClipSource(args.clip)
            if args.clip
            else AVFoundationSource(
                settings.camera.index, settings.camera.width,
                settings.camera.height, settings.camera.fps,
            )
        )
        tracker = HandTracker.create(settings.tracking.backend, settings.tracking.device)
    except (FileNotFoundError, RuntimeError, PermissionError) as exc:
        print(f"[mcv-calibrate] error: {exc}", file=sys.stderr)
        return 1

    steps: list[tuple[str, str]] = [
        ("neutral", "Hold your hand STILL at a comfortable CENTER (neutral) position."),
        *[(d, f"Push your hand FULLY {d.upper()} and hold it there.") for d in REACH_POSES],
    ]
    collected: dict[str, list[np.ndarray]] = {}
    print("=== mcv-calibrate: spatial joystick wizard ===")
    print(f"  hand={args.hand}  anchor={anchor}  frames/step={args.frames_per_step}")
    try:
        for name, instruction in steps:
            print(f"\n[{name}] {instruction}")
            if live:
                for c in (3, 2, 1):
                    print(f"  capturing in {c}…", end="\r", flush=True)
                    time.sleep(1.0)
            samples = _collect_anchor_samples(
                source, tracker, args.hand, anchor, args.frames_per_step,
                settings.camera.mirror, cv2,
            )
            collected[name] = samples
            print(f"  captured {len(samples)} samples for '{name}'.        ")
    except KeyboardInterrupt:
        print("\n[mcv-calibrate] aborted; nothing written.")
        return 1
    finally:
        source.release()
        tracker.close()

    neutral = collected.pop("neutral", [])
    try:
        result = compute_calibration(
            neutral, collected, deadzone_margin=settings.joystick.dynamic_deadzone_margin
        )
    except ValueError as exc:
        print(f"[mcv-calibrate] error: {exc} (was a hand visible?)", file=sys.stderr)
        return 1

    print("\n=== Result ===")
    print(f"  neutral        = ({result.neutral[0]:.3f}, {result.neutral[1]:.3f})")
    print(f"  resting jitter = {result.resting_jitter:.4f}")
    print(f"  mean reach     = {result.mean_reach:.4f}")
    print(f"  -> deadzone_radius = {result.joystick_overrides()['deadzone_radius']}")
    print(f"  -> sensitivity     = {result.joystick_overrides()['sensitivity']}")

    if not args.apply:
        print("\n[mcv-calibrate] preview only. Re-run with --apply to write to "
              f"{args.config}.")
        return 0

    config_path = args.config
    if not Path(config_path).is_file():
        print(f"[mcv-calibrate] error: cannot apply; config '{config_path}' does not exist.",
              file=sys.stderr)
        return 1
    merged = merge_calibration(load_config_data(config_path), result)
    try:
        Settings(**merged)  # validate before writing — never persist an invalid config.
    except Exception as exc:  # noqa: BLE001 - surface any validation failure to the user
        print(f"[mcv-calibrate] error: computed config failed validation: {exc}",
              file=sys.stderr)
        return 1
    save_config_data(config_path, merged)
    print(f"[mcv-calibrate] wrote joystick calibration to {config_path}.")
    return 0


def _collect_anchor_samples(
    source: Any, tracker: Any, hand: str, anchor: str, n_frames: int, mirror: bool, cv2: Any
) -> list[np.ndarray]:
    """Read frames until ``n_frames`` anchor positions for ``hand`` are collected (or exhausted).

    Returns a list of ``(2,)`` normalized ``(x, y)`` anchor positions. Mirrors each frame to
    match the live pipeline's frame of reference before tracking.
    """
    from minecraft_cv.joystick.deadzone import anchor_xy

    out: list[np.ndarray] = []
    while len(out) < n_frames:
        frame = source.read()
        if frame is None:
            break
        if mirror:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for h in tracker.detect(rgb):
            if h.handedness == hand:
                out.append(anchor_xy(h.landmarks, anchor))
                break
    return out


def _collect_palm_normal_samples(
    source: Any,
    tracker: Any,
    n_frames: int,
    mirror: bool,
    swap_handedness: bool,
    cv2: Any,
) -> dict[str, list[np.ndarray]]:
    """Collect palm-normal ``(x, y)`` samples for both logical hands."""
    from minecraft_cv.joystick.palm_normal import palm_normal_xy

    out: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    while len(out["left"]) < n_frames or len(out["right"]) < n_frames:
        frame = source.read()
        if frame is None:
            break
        if mirror:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for h in tracker.detect(rgb):
            label = h.handedness
            if swap_handedness:
                label = "Right" if label == "Left" else "Left"
            key = label.lower()
            if key in out and len(out[key]) < n_frames:
                out[key].append(palm_normal_xy(h.landmarks))
    return out


def _calibrate_pinch(args: argparse.Namespace, settings: Settings) -> int:
    """Legacy live readout of normalized thumb-to-fingertip distances (Schmitt tuning)."""
    import cv2

    from minecraft_cv.capture.source import AVFoundationSource, ClipSource
    from minecraft_cv.gestures.pinch import normalized_distances
    from minecraft_cv.tracking.tracker import HandTracker

    source = (
        ClipSource(args.clip)
        if args.clip
        else AVFoundationSource(
            settings.camera.index, settings.camera.width,
            settings.camera.height, settings.camera.fps,
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
    k = min(len(s) - 1, int(round((p / 100) * (len(s) - 1))))
    return s[k]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

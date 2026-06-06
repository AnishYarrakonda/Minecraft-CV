# CLAUDE.md — minecraft_cv

Real-time webcam gesture → Minecraft keyboard/mouse controller. A Python pipeline
reads hand pose from a camera, maps two-hand spatial translation + pinch-bitmask
gestures to game input, and emits OS-level events via pynput / Quartz CGEvent.

## Quick commands

```bash
# Setup (if needed; .venv already exists)
source .venv/bin/activate

# Desktop app (recommended — Go Live toggle, Calibrate button, live HUD).
# Camera on top, compact key grid below; Pin keeps it over Minecraft, shrink it
# short to collapse into a camera-only HUD.
.venv/bin/python -m minecraft_cv.cli ui

# Dry run with camera and debug overlay (headless, no UI)
.venv/bin/python -m minecraft_cv.cli run --no-input --debug-overlay

# Live mode headless (emits real OS input)
.venv/bin/python -m minecraft_cv.cli run --input --debug-overlay

# Offline clip analysis
.venv/bin/python -m minecraft_cv.cli analyze data/clips/foo.mp4

# Fast regression test
.venv/bin/python -m pytest -k schmitt -x

# Full CI gate
.venv/bin/python -m pytest && .venv/bin/ruff check src tests && .venv/bin/mypy src
```

**For development, always use one of these patterns:**
- `mcv ui` or `--no-input --debug-overlay` with live camera to iterate safely (no real input emitted)
- `--clip data/clips/foo.mp4 --no-input --debug-overlay` for deterministic testing against recorded footage

Never debug gesture logic against live Minecraft input; it's non-deterministic and will fling your character.

## Hard invariants

1. `T_release > T_engage` strictly for every pinch. A test asserts this for all gestures.
2. Input emitter is a no-op by default. Tests/dry-runs must never move the real mouse.
3. CPU fallback always works. MPS is an accelerator, never a hard dependency.
4. `Attack+Use` are intentionally mutually exclusive. Don't fix them.

## Repo layout (brief)

```
src/minecraft_cv/
├── capture/        # VideoCapture wrapper + single-slot FrameBuffer
├── tracking/       # HandTracker ABC, MediaPipe backend, FaceLandmarker
├── gestures/       # Schmitt triggers, pinch/extension detectors, face gestures, safety
├── joystick/       # ScreenJoystick, WristTiltJoystick, One-Euro filter
├── input/          # InputEmitter ABC, NullEmitter, MacInputEmitter
├── ui/             # PySide6 desktop app (pinnable, collapsible) + Qt/pipeline bridge
├── runtime.py      # FrameProcessor: camera/clip loop
├── pipeline.py     # Pipeline: gestures + joystick → InputEmitter
└── config.py       # pydantic Settings (all tunable values via config.yaml)
cli.py              # mcv entrypoint (ui, run, analyze, bench, doctor, gestures)
tests/              # mirrors src/; gesture SM tests are pure/deterministic
```

## Context files — read when the task touches these areas

Read the relevant file(s) **before** writing any code in that area. Do not load files
for areas you're not touching.

| When working on… | Read |
|---|---|
| Gesture state machines, Schmitt triggers, pinch bitmask, face gestures, head gestures, concurrency, tracking-loss safety | `.claude/rules/gestures.md` |
| Mouse look, camera sensitivity, `ScreenJoystick`, `WristTiltJoystick`, One-Euro filter, cursor reseeding, peace-sign clutch | `.claude/rules/mouse-look.md` |
| Camera capture, OpenCV preprocessing, NumPy vectorization, MPS/PyTorch inference, frame-rate/latency, macOS camera permissions | `.claude/rules/opencv-pytorch.md` |
| Input emission — `pynput`, Quartz CGEvent, `InputEmitter`/`MacInputEmitter`, keyboard hold vs tap, scroll, Accessibility permissions | `.claude/rules/input-layer.md` |
| PySide6 UI, `PipelineWorker`, Qt threading, window pinning / collapse, `MainWindow`, `CameraView`, `KeymapPanel` | `.claude/rules/ui-qt.md` |
| Adding/removing dependencies, MediaPipe landmarks, architecture overview, code style, config system | `.claude/rules/tech-stack.md` |

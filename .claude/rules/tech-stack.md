# Rules: Tech Stack

Authoritative description of each library's role. Read this before writing any import
or reaching for a new dependency.

## MediaPipe Hands — hand pose estimation (the core intelligence)

MediaPipe is the industry-standard on-device hand tracker. It is the **only** component
that does hand tracking. It takes a raw RGB frame and outputs 21 3D landmarks per
detected hand, each as an `(x, y, z)` coordinate:

- `x`, `y` — normalized to `[0, 1]` relative to the frame width/height.
- `z` — depth relative to the wrist, in the same scale as `x`. Negative = closer to
  camera. Less reliable than `x`/`y` but usable for future depth gestures.

These 21 landmarks are the raw material for everything else:

- **Spatial joysticks:** wrist (landmark 0) or middle MCP (landmark 9) centroid
  → `(Δx, Δy)` relative to a stored neutral position.
- **Pinch-bitmask:** pairwise distances between thumb tip (4) and each fingertip
  (8, 12, 16, 20), normalized by hand scale.

MediaPipe runs on CPU only (no MPS/CUDA); it is already fast enough at its native
resolution. Do not attempt to move it to MPS — it will error. Any PyTorch/YOLO
tracker lives behind the same `HandTracker` ABC as a swappable alternative backend.

## OpenCV (`cv2`) — frame ingestion, preprocessing, and debug overlay ONLY

OpenCV's role in this project is strictly limited to three things:

1. **Raw frame ingestion** from the hardware device:
   `cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)`. Force AVFoundation backend on
   macOS; set `CAP_PROP_BUFFERSIZE=1`, explicit FPS, and explicit resolution.
   The frame comes out as a BGR `uint8` NumPy array.

2. **Basic preprocessing** before handing off to MediaPipe:
   - BGR → RGB: `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` — exactly once per frame.
   - Downscale if needed: `cv2.resize(frame, (w, h))`.
   - Optional horizontal flip: `cv2.flip(frame, 1)` for mirror mode.
   That is all. OpenCV does **not** detect hands, compute landmarks, or make gesture
   decisions.

3. **Debug overlay drawing** (gated behind `--debug-overlay`):
   `cv2.putText`, `cv2.circle`, `cv2.line`, `cv2.arrowedLine` on the annotated frame,
   displayed via `cv2.imshow` / `cv2.waitKey` on the main thread only.

OpenCV does NOT: run inference, compute distances, detect gestures, or emit input.

## NumPy — vectorized math on landmark arrays

All arithmetic on the 21-landmark arrays happens in NumPy. Never loop over landmarks
in pure Python. The canonical pattern for pinch distances:

```python
thumb = landmarks[4]                                      # (3,)
tips  = landmarks[[8, 12, 16, 20]]                        # (4, 3)
scale = np.linalg.norm(landmarks[9] - landmarks[0])       # scalar hand scale
dists = np.linalg.norm(tips - thumb, axis=1) / scale      # (4,) normalized ratios
```

One vectorized call, not four. Distances divided by hand scale are camera-distance
invariant — Schmitt thresholds are then unitless ratios that never need recalibration
just because the user moved closer to the webcam.

## PyTorch + MPS — optional YOLO tracker backend

PyTorch is only used if a YOLO-based `HandTracker` backend is enabled. MediaPipe does
not use PyTorch. When PyTorch is active:

- Use `mps` on Apple Silicon; fall back to `cpu`. One `select_device()` helper,
  no `"mps"` string literals scattered in the code.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is already set in `.claude/settings.json` `env`.
- Model weights move to device once at startup. Inputs move per frame. No weight
  transfer in the loop.
- All inference in `torch.inference_mode()` + `model.eval()`.
- MPS is async — `torch.mps.synchronize()` before any timing measurement.
- `float32` only; `float64` is not reliably supported on Metal.

## Input injection — pynput + Quartz CGEvent (macOS)

Two separate libraries handle different input types:

- **`pynput`** — keyboard key down/up (Space, Shift, E, Q, W/A/S/D) and scroll wheel
  (hotbar next/prev). Works well for discrete key events.
- **`pyobjc-framework-Quartz` (CGEvent)** — relative mouse deltas for camera look.
  Quartz gives lower latency than `pynput` mouse and sends true relative motion, which
  is what Minecraft reads. Emit small, frequent deltas; don't correct for macOS
  pointer acceleration at the injection level.

Both require **Accessibility / Input Monitoring** granted to the terminal app in
System Settings → Privacy & Security. Without it, events are silently dropped — no
error. Detect at startup by attempting a test event and checking the return code.

The entire input layer sits behind an `InputEmitter` interface:
- `NullEmitter` (default) — no-ops all calls. Used in tests and `--no-input` mode.
- `MacInputEmitter` — the real pynput + Quartz implementation.

Never emit real input in tests. Never skip the emitter interface.

## Config — pydantic-settings + config.yaml

`pydantic-settings` loads `config.yaml` into a typed `Settings` model. All tunable
values live here: `T_engage`, `T_release`, deadzone radius, sensitivity, camera index,
backend selection, device. No magic numbers in gesture or joystick code.

## Repository layout

```
src/minecraft_cv/
├── capture/        # VideoCapture wrapper, frame source interface, buffer thread
├── tracking/       # HandTracker ABC; mediapipe_backend.py; optional yolo_backend.py
├── gestures/       # SchmittTrigger, PinchStateMachine, per-hand gesture resolver
├── joystick/       # DeadzoneJoystick, accel curve, recenter macro, drift handling
├── input/          # InputEmitter ABC, NullEmitter, MacInputEmitter (pynput+Quartz)
├── pipeline.py     # wires capture → tracking → gestures → joystick → input
└── config.py       # pydantic Settings model

scripts/            # CLI entrypoints: mcv-run, mcv-calibrate, mcv-analyze, mcv-bench
tests/              # mirrors src/ structure; gesture SM tests are pure/deterministic
data/               # clips + annotations (git-ignored; large files go in data/clips/)
```

## Code style

- **PEP 8**, enforced by ruff (line length 100). `mypy --strict` on `src/`.
- Type hints mandatory on every public function/method. No bare `Any` without a comment.
- **Google-style docstrings.** Every public function that deals with coordinates must
  state units and frame of reference explicitly (normalized vs pixels, camera vs screen)
  — most bugs here are silent unit mismatches.
- No magic numbers in gesture/joystick math. All thresholds come from `Settings`.
- No `INFO`-level logging per frame. Use rate-limited debug counters.
- Tracker backends behind `HandTracker` ABC — swappable without touching the pipeline.

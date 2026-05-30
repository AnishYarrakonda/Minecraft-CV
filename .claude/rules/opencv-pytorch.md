# Rules: OpenCV + PyTorch performance on Apple Silicon macOS

Engineering rules for the real-time hand-tracking pipeline. These are loaded into
context via the `@`-import at the bottom of `CLAUDE.md`. Follow them when writing or
reviewing anything on the per-frame hot path.

The governing constraint: this is a **soft-real-time** system. End-to-end latency
(photon → input event) must stay low and *predictable*. A pipeline that averages
8 ms but spikes to 60 ms every second feels worse than a steady 16 ms one. Optimize
for tail latency (p95/p99), not just the mean.

---

## 1. NumPy / OpenCV: never iterate pixels in Python

- **No Python-level loops over pixels or landmarks.** A nested `for y: for x:` over a
  720p frame is ~920k iterations × interpreter overhead. Vectorize with NumPy slicing,
  boolean masks, and ufuncs instead. If you think you need a per-pixel loop, you need
  `np.where`, broadcasting, or an OpenCV primitive.
- **Prefer OpenCV/NumPy primitives** over hand-rolled math: `cv2.resize`,
  `cv2.cvtColor`, `cv2.threshold`, `cv2.findContours`, `np.linalg.norm`, `cv2.absdiff`.
  They're SIMD-vectorized C/C++.
- **Pinch distance = vectorized norm.** Compute all thumb-to-fingertip distances in
  one shot:
  ```python
  # landmarks: (21, 3) float32 array of (x, y, z), already on CPU
  thumb_tip = landmarks[4]
  finger_tips = landmarks[[8, 12, 16, 20]]          # index, middle, ring, pinky
  dists = np.linalg.norm(finger_tips - thumb_tip, axis=1)  # (4,) — one call
  ```
  Never compute these one fingertip at a time in a loop.
- **Normalize distances by hand scale.** Pinch thresholds must be invariant to how far
  the hand is from the camera. Divide by a reference span (e.g. wrist→middle-MCP
  distance) so `T_engage`/`T_release` are unitless ratios, not raw pixels.
- **Avoid hidden copies.** `frame[y0:y1, x0:x1]` is a view (cheap); fancy-indexing and
  `astype` copy. Reuse pre-allocated buffers with `dst=` args (`cv2.resize(..., dst=buf)`,
  `np.subtract(a, b, out=buf)`) on the hot path. Allocation = GC pressure = latency spikes.
- **Keep dtype discipline.** Images stay `uint8` until you actually need float. Convert
  once, as late as possible, and reuse the float buffer. Gratuitous `float64` doubles
  bandwidth — use `float32`.
- **Color conversion is not free.** MediaPipe wants RGB, OpenCV capture gives BGR.
  Convert exactly once per frame and pass the result around; don't re-convert per consumer.

## 2. PyTorch on MPS (Apple Silicon)

- **Single source of truth for device selection.** One helper, used everywhere:
  ```python
  import torch

  def select_device() -> torch.device:
      if torch.backends.mps.is_available():
          return torch.device("mps")
      if torch.cuda.is_available():
          return torch.device("cuda")
      return torch.device("cpu")
  ```
  Never scatter `"mps"` string literals through the codebase.
- **CPU fallback is mandatory.** Set `PYTORCH_ENABLE_MPS_FALLBACK=1` (already in
  `settings.json` `env`) so ops MPS doesn't implement fall back to CPU instead of
  crashing. But treat each fallback as a perf bug to investigate, not a free pass —
  a per-frame CPU fallback inside an otherwise-MPS graph is a latency cliff.
- **Move the model to the device once, at startup.** Never `.to(device)` inside the
  frame loop. Move input tensors per frame; keep weights resident.
- **Inference is `torch.no_grad()` + `model.eval()`.** Autograd bookkeeping is pure
  overhead at inference. Wrap the forward pass:
  ```python
  model.eval()
  with torch.inference_mode():
      out = model(x)
  ```
- **MPS is asynchronous.** GPU work is queued, not completed, when the call returns.
  Any timing measurement must `torch.mps.synchronize()` before reading the clock, or
  you'll record fictitiously fast numbers. Do **not** synchronize on the hot path for
  any reason other than benchmarking — it stalls the pipeline.
- **Minimize host↔device transfers.** `.cpu()` / `.item()` / `.numpy()` force a sync
  and a copy across the unified-memory boundary. Pull results back **once per frame**,
  as a single batched tensor, not field-by-field.
- **Watch MPS memory.** `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` (in `env`) disables the
  upper allocation limit; call `torch.mps.empty_cache()` only between phases (e.g. after
  warmup), never per frame.
- **Warm up before benchmarking.** The first few MPS forward passes pay shader
  compilation + allocation costs. Discard the first ~20 frames in any benchmark.
- **`float32`, not `float64`, on MPS.** Some `float64` ops are unsupported or silently
  slow on Metal. Keep model + inputs in `float32` (or `float16` where accuracy allows).

## 3. Frame-rate & latency engineering

- **Budget the frame.** At 60 FPS you have ~16.6 ms; at 30 FPS, ~33 ms. Capture +
  preprocess + inference + gesture logic + input emit must fit *with headroom*.
  Instrument each stage; know where the milliseconds go before optimizing.
- **Decouple capture from processing.** Read frames on a dedicated thread into a
  **single-slot** buffer (keep newest, drop stale). A queue that backs up adds latency;
  for a controller, a *dropped* old frame beats a *late* fresh action.
- **Always grab the latest frame.** With `cv2.VideoCapture`, internal buffering means a
  naive `read()` can hand you a stale frame. Set `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)`
  and/or drain to the newest frame each tick.
- **Downscale before inference.** Hand tracking rarely needs full 1080p. Resize to the
  model's native input (e.g. 256×256) once; run detection on the small frame and map
  landmarks back to full-res coords. This is usually the single biggest win.
- **Process at the rate you need, not the rate you can.** Camera at 60 FPS doesn't
  require landmark inference at 60 FPS if 30 Hz tracking + interpolation feels
  identical. Spend the saved budget on lower tail latency.
- **One-Euro / EMA filtering, not frame averaging.** Smooth landmark jitter with a
  velocity-aware filter (One-Euro) so you cut jitter when still without adding lag when
  moving. Don't average N raw frames — that's pure latency.
- **Gesture logic is O(landmarks), keep it allocation-free.** The Schmitt-trigger state
  machines and deadzone math operate on tiny arrays; they should never allocate or log
  per frame. Pre-allocate state, mutate in place.

## 4. macOS screen / camera capture constraints

- **Permissions are the #1 footgun.** Two *separate* macOS grants are involved:
  - **Camera** — for `cv2.VideoCapture` / AVFoundation. Without it, capture opens but
    yields black/empty frames (no exception). Grant to the *terminal app* running Python.
  - **Screen Recording** — required for `mss` / any screen grab. Without it you get a
    desktop-wallpaper-only image, again with no error.
  - **Accessibility / Input Monitoring** — required to *emit* synthetic input via
    `pynput` / Quartz `CGEvent`. Without it, input calls silently no-op.
  All three attach to the *binary that launched Python* (Terminal, iTerm, VS Code).
  Re-granting after upgrading that app, or after changing the Python interpreter, is
  often necessary. Toggling the permission off/on forces macOS to re-prompt.
- **Detect missing permissions early and loudly.** On startup, grab one frame and assert
  it isn't all-black/uniform; if it is, raise a clear error telling the user exactly
  which Privacy & Security pane to open. Don't let a silent black frame propagate.
- **Prefer the camera over screen capture for tracking.** `mss` screen grabs are heavier
  and Screen-Recording-gated; only use them for the debug overlay, not the tracking input.
- **AVFoundation backend.** Force it explicitly (`cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)`);
  `OPENCV_VIDEOIO_PRIORITY_AVFOUNDATION=1` is set in `env`. Set `CAP_PROP_FPS` and the
  resolution explicitly — defaults are often a low-FPS, high-res combo.
- **Continuity Camera surprises.** macOS may silently select an iPhone as camera 0.
  Enumerate devices and let the user pin the index in `config.yaml`.

## 5. OpenCV windowing / threading (HighGUI)

- **All `cv2.imshow` / `cv2.namedWindow` / `cv2.waitKey` calls must run on the main
  thread.** On macOS, Cocoa's UI event loop is main-thread-only. Calling HighGUI from a
  worker thread → frozen/blank windows, beachballs, or hard crashes.
- **Architecture:** worker thread(s) do capture + inference + gesture logic and push the
  annotated frame to a single-slot buffer; the **main thread** owns the render loop
  (`imshow` + `waitKey(1)`). Never invert this.
- **`cv2.waitKey(1)` is mandatory** to pump the GUI event loop — without it the window
  never paints. But `waitKey` also throttles; for a headless/production run, skip
  windowing entirely (`--debug-overlay` off) and save the milliseconds.
- **Prefer headless in production.** Install `opencv-python-headless` for deployment
  builds; rendering is a debug-only luxury. Drawing overlays (`cv2.putText`,
  `cv2.line`, landmark dots) is not free — gate it all behind the debug flag.
- **One window owner.** Don't create/destroy windows per frame; create once, reuse the
  handle, `destroyAllWindows()` on shutdown.

## 6. Determinism, testing & benchmarking

- **Recorded clips over live camera for tests.** A live webcam is non-deterministic and
  un-CI-able. The frame-analyzer skill + `mcv-analyze` run the full pipeline on a saved
  clip so results are reproducible. Every tracking/gesture regression gets a clip.
- **Test the state machine in isolation.** The Schmitt-trigger logic takes a sequence of
  distances and emits KEY_DOWN/KEY_UP events — pure, fast, deterministic. Unit-test it
  with synthetic jitter sequences; never require a camera to test gesture correctness.
- **Benchmark MPS vs CPU honestly:** warm up, `torch.mps.synchronize()` around the timed
  region, report p50/p95/p99 not just mean, and pin the input clip + backend in the
  benchmark name. `mcv-bench` does this; extend it rather than ad-hoc `time.time()`.
- **Profile before optimizing.** Use `cProfile` / `py-spy` for Python-side hotspots and
  per-stage timers for the pipeline. Don't "optimize" the inference if 70% of the budget
  is being spent in an accidental BGR→RGB double-convert.

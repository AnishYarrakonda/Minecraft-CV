# minecraft_cv — Realtime Latency & Reliability Audit

Scope: photon → OS input event. Default config: camera 640×480@30 (`config.py:34-36`), detect at 256×256 (`config.py:54`), MediaPipe CPU-only, `NullEmitter` in dev / `MacInputEmitter` live.

## Architecture as built

- **Capture thread** (`capture/buffer.py`): one daemon thread, single-slot newest-wins buffer. This is the *only* background thread.
- **Main thread** (`pipeline.py:600-630`): does mirror → cvtColor → resize → `tracker.detect` → `pipeline.step` → overlay. **MediaPipe inference runs on the main thread, not the capture thread.** The capture thread only does `cap.read()`.

That split is actually correct for newest-frame-wins: capture races ahead, the consumer processes the freshest frame detect-bound and drops intermediates. No unbounded queues anywhere — good. But everything that costs real time is serialized on one thread with zero pipelining and zero buffer reuse.

---

## Latency budget (per frame, no overlay, Apple Silicon estimate)

| Stage | Operation | p50 | p95 | p99 | Notes |
|---|---|---|---|---|---|
| **Capture** | `cap.read()` staleness until consumed | 5–15 ms | ~33 ms | ~33 ms | Off critical CPU path (bg thread), but adds to photon→process age. Bounded by 30 fps cadence. |
| **Preprocess** | `cv2.flip` + `cvtColor` + `resize` (3 full-frame **allocations**) | 1.0–2.0 ms | 3 ms | 6 ms | `pipeline.py:615-617`. None use `dst=` reuse → GC churn. |
| **Tracking** | `tracker.detect` (MediaPipe VIDEO, 2 hands) | 5–12 ms | 18 ms | **25–40 ms** | `mediapipe_backend.py:96-125`. p99 = palm **re-detection** spike. Dominant stage. |
| → protobuf→ndarray | 21×N Python attr reads | 0.05 ms | 0.1 ms | 0.2 ms | `mediapipe_backend.py:116-118`. Acknowledged, minor. |
| **Gesture** | `finger_extensions` + `normalized_distances` + Schmitt | 0.05–0.2 ms | 0.3 ms | 0.5 ms | Tiny NumPy. Allocation-light. Not a concern. |
| **Mapping** | joystick update + `atan2` zones + One-Euro | 0.05–0.15 ms | 0.2 ms | 0.4 ms | Fine. EMA `smoothing=0.6` adds *lag* not latency. |
| **Input** | `CGEventPost` / pynput press per event | 0.1–0.5 ms | 1 ms | 2 ms | `mac_emitter.py`. Synchronous syscalls on main thread; N events/frame. |
| **Poll gap** | `time.sleep(0.001)` waiting for fresh frame | 0–1 ms | 1–10 ms | 1–10 ms | `pipeline.py:605,608`. macOS sleep granularity can overshoot. |
| **End-to-end** | photon → input | **~12–28 ms** | ~40 ms | **~55–75 ms** | Re-detect + GC + poll-gap stack on the tail. |

The mean looks fine (~30–60 fps). The **tail is the problem**, exactly as `opencv-pytorch.md §0` warns: re-detection spikes + per-frame allocations are the p99 drivers.

---

## Critical performance issues

1. **Three full-frame heap allocations every frame** (`pipeline.py:615-617`). `cv2.flip`, `cvtColor`, and `resize` each allocate a new ~920 KB array; none use the `dst=` reuse the rules mandate (`opencv-pytorch.md §1`, "Reuse pre-allocated buffers"). The comment on line 613-614 says flip is done "in place" — it is **not**; `cv2.flip` without `dst=` returns a new array. This is steady GC pressure → periodic collection pauses → p95/p99 spikes. *Pre-allocate three buffers sized to the camera/detect resolution and pass `dst=`.*

2. **MediaPipe re-detection is the p99 cliff and is completely unmeasured.** VIDEO mode runs cheap landmark tracking most frames but re-runs full palm detection on confidence loss / hand re-entry — a 2–4× latency jump. Nothing in the codebase distinguishes a tracking frame from a re-detect frame, so you can't even see it. This is inherent to MediaPipe, but it must be *observable* to manage.

3. **Flip + convert + resize order is suboptimal.** You flip and color-convert the full 640×480 frame, then resize to 256×256. cvtColor and flip on the small frame would be ~6× cheaper. Resize first → convert/flip on 256² (mind that overlay wants the full-res frame; keep one full-res copy for overlay only, gated by the debug flag).

4. **No pipelining.** detect (the 5–40 ms stage) and `pipeline.step`+emit run strictly serially on one thread. Acceptable for a controller, but it caps throughput at `1/detect_time` even though gesture/mapping/emit are ~free.

---

## Reliability issues

1. **🔴 A single transient camera read kills the whole session.** `FrameBuffer._run` (`buffer.py:39-48`) sets `_exhausted = True` and breaks permanently the first time `source.read()` returns `None`. For `ClipSource` that's correct (clip ended). For a **live camera**, `AVFoundationSource.read()` returns `None` on any transient `ok=False` (`source.py:124-128`) — a momentary AVFoundation hiccup, USB renegotiation, or Continuity Camera handoff. The comment at `buffer.py:42-43` even admits "or a transient miss: stop reading." This directly violates the *"recovery after frame drops"* requirement: the pipeline does **not** recover from one dropped camera frame; it shuts down. Clip exhaustion and live-camera transient-miss need different handling (retry/backoff with a consecutive-failure ceiling for the camera).

2. **MediaPipe VIDEO timestamps are wall-clock-derived** (`mediapipe_backend.py:92-94`). `detect_for_video` requires *strictly* increasing timestamps. Two frames processed inside the same millisecond → equal timestamp → MediaPipe raises. Unlikely at 256² detect cost today, but it's a latent crash tied to machine speed; a monotonic frame-counter timestamp would be safer than `monotonic_ns // 1e6`.

3. **`processed % overlay_every` counter grows unbounded** (`pipeline.py:620,624`) — an `int`, so harmless in Python, but worth noting it's the only "metric" and it's never read.

State growth / leaks: **none found.** `_last_scroll_time` (popped on release), `_last_emit`, `_cal_samples` (cleared), One-Euro/joystick state are all bounded. The single-slot buffer is the right call. The one memory note is `_bench_frames` loading up to `--frames` (default 500) full frames into a list (~98 MB at 256²) — bench-only, fine.

---

## Benchmarking gaps

1. **`mcv-bench` measures only `tracker.detect`** (`cli.py:373-382`). It excludes capture, flip, cvtColor, resize, `pipeline.step`, and input emit — i.e. it does not benchmark the latency budget. The headline "FPS / PASS" is detect-only and will read healthier than reality.

2. **Synthetic frames are random noise** (`cli.py:462-464`). MediaPipe finds no hands → it never exercises the *landmark-tracking* path and instead runs palm-detection-fast-reject every frame. The benchmark's central number is unrepresentative of real tracking-mode latency *and* never triggers the re-detect spike that dominates p99. Default bench should use a real hand clip.

3. **`--device mps` + mediapipe backend is misleading** (`cli.py:467-476`). MediaPipe ignores `device` (CPU-only). `_maybe_mps_sync` will call `torch.mps.synchronize()` around a pure-CPU op, implying an MPS path that doesn't exist (YOLO backend is `NotImplementedError`, `tracker.py:75-78`). The "CPU vs MPS" comparison is currently **un-runnable** — there is no live MPS tracker.

4. **No end-to-end (photon→input) benchmark** and **no inter-frame-arrival / dropped-frame** metric. `_summarize` (`cli.py:417-439`) computes good per-stage stats (p50/95/99 + `jitter_std`) — but only ever fed detect times.

5. `_pct` uses nearest-rank with `round` (`cli.py:479-484`); fine for reporting, just not interpolated — note when comparing against other tools.

---

## Instrumentation gaps

The **live loop has zero timing instrumentation.** `run_pipeline` cannot tell you its own p95/p99, FPS, or dropped-frame count in production. `opencv-pytorch.md §3` asks to "instrument each stage; know where the milliseconds go" and §6 wants rate-limited debug counters — neither exists on the live path. You can profile a clip via `mcv-analyze --timing` (track+gesture only, `cli.py:340-343`), but never the real camera session.

## Recommended telemetry

Add a fixed-size ring-buffer profiler to `run_pipeline`, behind a `--profile` flag, rate-limited (never per-frame logging):

1. **Per-stage timers**: `convert_ms`, `resize_ms`, `detect_ms`, `step_ms`, `emit_ms` → p50/p95/p99 + `jitter_std` on a rolling window (reuse `_summarize`).
2. **Frame staleness**: stamp `perf_counter()` into `FrameBuffer` at store time; at consume, record `now − capture_time`. This is the real photon→process latency the budget table needs — currently invisible.
3. **Dropped-frame counter**: track `seq − last_seq − 1` each tick (frames the consumer skipped). A rising drop rate = detect can't keep up.
4. **Re-detect spike detector**: flag/count frames where `detect_ms > k × rolling_median` — surfaces the p99 driver.
5. **Emit-path timing** in `MacInputEmitter` (CGEventPost can stall under load).
6. **End-to-end harness**: a `mcv-bench --e2e --clip hand.mp4` that runs the *full* `Pipeline.step` with `NullEmitter` and reports the whole budget, not just detect.

---

## Performance Roadmap (ordered by ROI)

1. **Fix the transient-read shutdown** (`buffer.py:39-48`). Highest ROI — it's a one-frame-glitch-kills-session reliability bug. Distinguish clip-EOF (exhaust) from live-camera miss (retry with backoff + consecutive-failure ceiling). ~10 lines.
2. **Instrument the live loop** (telemetry #1–#4). You cannot optimize p95/p99 you can't see; everything below is guesswork until this lands. Cheap, behind a flag.
3. **Resize-before-convert/flip + preallocated `dst=` buffers** (`pipeline.py:615-617`). Kills the 3 per-frame allocations (p99/GC win) *and* cuts cvtColor/flip cost ~6×. Keep one full-res copy only when `overlay` is on.
4. **Make the bench representative** (`cli.py:373-382, 462-464`): time the full pipeline, default to a real hand clip, and split the reported `detect_ms` into tracking-frame vs re-detect-frame populations. Turns the bench from decorative into diagnostic.
5. **Re-detect-spike observability** (telemetry #4) + tune `min_tracking_confidence` (`config.py:56`) once measured — directly attacks the dominant p99 contributor.
6. **Guard the MediaPipe timestamp** (`mediapipe_backend.py:92-94`): monotonic frame counter, strictly increasing. Latent-crash insurance, low cost.
7. **(Lower)** Reconcile the dead MPS path: either implement the YOLO/MPS backend or make `mcv-bench --device mps` refuse `mediapipe` instead of faking a sync (`cli.py:467-476`). The "CPU vs MPS" comparison is moot until then.
8. **(Optional)** Pipeline detect on the capture thread so gesture/mapping/emit of frame N overlap detect of frame N+1. Only worth it if telemetry shows the serial tail matters; adds complexity to a currently-clean design.

Note on jitter/lag (not latency, but felt as lag): default joystick `smoothing=0.6` EMA (`config.py:200`) stacks on top of the One-Euro look filter — heavy uniform lag on WASD/look. Worth A/B-ing against `0.3–0.4` once #2 gives you measurement.

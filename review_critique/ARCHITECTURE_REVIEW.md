# Production-Readiness Architecture Review — `minecraft_cv`

**Reviewer role:** Principal Software Architect
**Date:** 2026-05-30
**Scope:** Backend architecture, code organization, maintainability, correctness, engineering quality.
**Explicitly out of scope:** Gesture quality, UX, ergonomics.

**Files reviewed (in full):** `README.md`, `.claude/CLAUDE.md`, `.claude/rules/*`, `pyproject.toml`,
`src/minecraft_cv/pipeline.py`, `config.py`, `recovery.py`, `calibration.py`, `cli.py`,
`tracking/**`, `gestures/**`, `joystick/**`, `input/**`, `capture/**`, `tests/conftest.py`,
plus the full inventory of test functions across `tests/**`.

**Verification:** `.venv/bin/python -m pytest -q` → **all 153 tests pass** (exit 0) at review time.

---

## Executive summary

This is a **well-architected codebase for an MVP** — markedly better layered than most
hobby real-time pipelines. The standout strengths:

- **Pure core, impure shell.** `Pipeline.step()` is pure with respect to the OS and fully
  unit-testable; `run_pipeline()` owns all I/O. This is the right seam.
- **Disciplined lazy imports.** OpenCV, MediaPipe, pynput, Quartz are imported inside function
  bodies so the deterministic test suite runs without a camera or OS-input libraries.
- **Safety-first input layer.** `InputEmitter` ABC with a `NullEmitter` default; held-key
  bookkeeping centralized in the ABC so every backend shares `release_all` semantics.
- **Config as a typed contract.** `pydantic` models centralize every threshold and enforce the
  core hysteresis invariant (`t_release > t_engage`) at construction.
- **Swappable backends** behind `HandTracker` and `FrameSource` ABCs; no dependency cycles;
  dependency direction points cleanly downward.

The gaps that keep it from *production* readiness are concentrated in three areas:
**(1) runtime failure handling of the capture thread** (a camera stall/disconnect can hang the
loop *and strand a held key* — a direct violation of the project's own safety promise),
**(2) accreted, overlapping recovery/state machinery** in the `Pipeline` (three mechanisms now
react to a missing hand), and **(3) test/observability gaps** around exactly the I/O code most
likely to fail in the field.

---

## 1. Architecture

### A1 — `Pipeline` is a god-object; `step()` carries too many responsibilities
- **Severity:** High
- **Root cause:** Every feature (handedness split, time-based recovery, inventory mode, gesture→
  emitter routing, scroll-repeat, velocity-sprint, WASD cardinal-zone math, absolute-cursor
  mapping, look filtering) was bolted directly onto `Pipeline` rather than decomposed. The class
  has a **20+ parameter** raw `__init__`.
- **Evidence:** `pipeline.py` is 653 lines; `Pipeline.__init__` takes 23 constructor args
  (`pipeline.py:53-105`); `step()` spans `pipeline.py:212-305` with recovery, inventory, event
  routing, scroll-repeat, and two joystick updates inline.
- **Recommended fix:** Extract (a) a per-hand sub-controller (see A2), and (b) a
  `GestureEventRouter` that owns the event→emitter mapping (pulse/scroll/key branching currently
  inline in `step`). Keep `from_settings` as the only public constructor; make the wide
  `__init__` private or a builder.
- **Effort:** Medium-High (1–2 days).
- **Risk if unresolved:** Change-amplification and merge risk grow with each new gesture; the
  single most-edited file is also the least decomposed.

### A2 — Left/right hand logic is duplicated instead of parameterized
- **Severity:** Medium-High
- **Root cause:** Two largely symmetric hands modeled as parallel fields/methods rather than two
  instances of one `Hand` controller.
- **Evidence:** `_update_translation` vs `_update_look` (`pipeline.py:401-452`) duplicate the
  `recenter_grace_frames` miss-counting logic; `_flush_left`/`_flush_right`
  (`pipeline.py:325-336`); paired fields `_left_miss`/`_right_miss`, `left_recovery`/
  `right_recovery`, `left_joystick`/`right_joystick`.
- **Recommended fix:** Introduce a `HandController` (joystick + recovery + miss counter + flush)
  instantiated twice; the two hands differ only in output sink (WASD vs mouse-look), injectable
  as a strategy.
- **Effort:** Medium.
- **Risk if unresolved:** Fixes applied to one hand silently miss the other (drift bugs).

### A3 — Three overlapping mechanisms now react to a missing hand
- **Severity:** High
- **Root cause:** `HandRecovery` (Task 5) was layered on top of the pre-existing
  grace-frame recenter logic without retiring the latter, and both sit beside `TrackingLossGuard`.
- **Evidence:** A missing hand is handled by **(1)** `TrackingLossGuard.process(None,…)` resetting
  the gesture SM (`safety.py:62-89`), **(2)** `HandRecovery.update` emitting a time-based `flush`
  that recenters the joystick (`recovery.py:81-125` → `pipeline.py:227-230`, `_flush_left`), and
  **(3)** `_left_miss/_right_miss` counting frames and *also* recentering after
  `recenter_grace_frames` (`pipeline.py:404-409`). The defaults nearly coincide
  (`recenter_grace_frames=3` ≈ 100 ms at 30 FPS; `dropout_flush_ms=100`), so **two independent
  triggers recenter the same joystick** with implicit precedence.
- **Recommended fix:** Make `HandRecovery` the single owner of "hand is gone" policy; delete the
  `_*_miss`/`recenter_grace_frames` path (or fold it into `HandRecovery` as the flush threshold).
- **Effort:** Medium.
- **Risk if unresolved:** Subtle, timing-dependent recenter behavior that's hard to reason about
  and test; the kind of bug that only reproduces on a real camera.

### A4 — `camera.mirror` and `tracking.swap_handedness` both affect handedness (double-negative)
- **Severity:** Medium
- **Root cause:** Two independent flags influence the same property; both default `True`.
- **Evidence:** `_split` swaps L/R labels (`pipeline.py:529-549`); `CameraSettings.mirror`
  docstring states flipping "also fixes MediaPipe handedness" (`config.py:39-44`);
  `TrackingSettings.swap_handedness` docstring describes the same correction (`config.py:58-65`).
  With both on, handedness is corrected twice.
- **Recommended fix:** Document the `(mirror × swap)` truth table in one place, or derive
  `swap_handedness` from `mirror` so there's a single knob; add one test pinning the expected
  physical-hand→logical-hand mapping for each combination.
- **Effort:** Low.
- **Risk if unresolved:** Users get inverted WASD/look depending on MediaPipe's labeling, with no
  obvious diagnosis.

### A5 — Look smoothing is applied after the acceleration non-linearity (and after EMA)
- **Severity:** Medium (design)
- **Root cause:** Filtering order. The right-hand path is
  `EMA(anchor)` → deadzone → `accel**exp` → `OneEuro(output)`. Jitter is amplified by the accel
  curve before One-Euro sees it, and there are two smoothers around one non-linearity.
- **Evidence:** `DeadzoneJoystick._smooth` EMA on input (`deadzone.py:194-210`); `_update_look`
  applies `look_filter.filter(out, now)` to the **output** (`pipeline.py:446-449`);
  `one_euro.py` docstring explicitly notes it filters the output vector, while the project rules
  prescribe smoothing *landmark* jitter.
- **Recommended fix:** Filter the anchor *position* once (One-Euro on the input), then run deadzone/
  accel un-smoothed; drop the joystick EMA for the look hand to avoid double-filtering.
- **Effort:** Medium.
- **Risk if unresolved:** Sub-optimal latency/jitter trade-off and two coupled tuning knobs.

### A6 — Hidden network dependency: model auto-download with no timeout or integrity check
- **Severity:** High
- **Root cause:** First construction of the MediaPipe backend silently downloads a model bundle
  over HTTPS and prints to stdout.
- **Evidence:** `mediapipe_backend.py:29-42` (`urllib.request.urlopen(_MODEL_URL …)`), no
  `timeout=`, no checksum/signature verification, hard-coded URL, cache path not configurable.
- **Recommended fix:** Add a connect/read `timeout`; verify a known SHA-256 of the bundle
  (supply-chain integrity); make the cache dir configurable via `Settings`; surface the network
  requirement in docs and fail with a clear message offline.
- **Effort:** Low-Medium.
- **Risk if unresolved:** First run hangs indefinitely on a flaky network; a compromised/changed
  remote artifact is executed without verification.

---

## 2. State management

### S2 — `sprint` can be driven by two state machines onto one physical key (`ctrl`)
- **Severity:** High *(only when velocity-sprint is enabled; disabled by default)*
- **Root cause:** Both the extension gesture `sprint` (middle-only → `ctrl`) and the
  `SprintVelocityTrigger` (→ `ctrl` via `bindings["sprint"]`) are independent state machines with
  no arbitration.
- **Evidence:** Default left-hand map includes `"sprint"` extension (`config.py:147`); the
  velocity path also presses `bindings["sprint"]` (`pipeline.py:359-367`). The emitter's
  idempotent `key_down`/`key_up` (`emitter.py:36-48`) means whichever machine releases first
  physically lifts `ctrl` even though the other still believes it's holding — a logical/physical
  state desync.
- **Recommended fix:** Enforce mutual exclusion (if `sprint.enabled`, reject/ignore the extension
  `sprint`, or route both through one arbiter that ref-counts holders of a shared key).
- **Effort:** Low-Medium.
- **Risk if unresolved:** Sprint that releases early/sticks when a user enables velocity-sprint
  without manually deleting the extension sprint (the config docstring *warns* but nothing
  enforces).

### S1 — Shared-key held-state is a flat set; no ref-counting for multi-source keys
- **Severity:** Medium
- **Root cause:** `InputEmitter._held_keys` is a `set[str]`; it assumes one logical owner per key.
- **Evidence:** `emitter.py:32-48`. WASD, gestures, and sprint can target overlapping keys
  (e.g. sprint forces `W` while the joystick also presses `W`: `pipeline.py:421-425`). If the
  joystick releases `W` while sprint still wants it, `W` lifts.
- **Recommended fix:** Either guarantee disjoint key ownership by construction, or ref-count holds
  per key. (Today the `W`/sprint overlap happens to be re-asserted each frame, which masks it —
  fragile.)
- **Effort:** Medium.
- **Risk if unresolved:** Latent stuck/dropped keys whenever two subsystems share a binding.

### S3 — Inventory mode suppresses left-hand gameplay but not right-hand hotbar scroll
- **Severity:** Low (spec-ambiguous)
- **Root cause:** Suppression only skips `left`-hand `KEY_DOWN` (`pipeline.py:252-253`).
- **Evidence:** In inventory mode, right-hand attack/use are intentionally kept (GUI clicks), but
  hotbar ring/pinky pinches still emit scroll ticks.
- **Recommended fix:** Decide and document whether hotbar scroll is suppressed in inventory mode;
  add a test either way.
- **Effort:** Low.

### S4 — `NullEmitter.key_tap` double-logs
- **Severity:** Low
- **Root cause:** Override appends `("key_tap", …)` *and* the underlying down/up
  (`emitter.py:136-140`).
- **Evidence:** Three log entries per pulse tap; could confuse assertions that count events.
- **Recommended fix:** Log either the tap or the down/up, not both.
- **Effort:** Trivial.

---

## 3. Error handling

### E1 — Capture thread swallows read errors; a camera stall/disconnect hangs the loop **with keys held**
- **Severity:** Critical
- **Root cause:** `FrameBuffer._run` has no `try/except` around `source.read()` and no error
  channel to the main loop; the main loop has no idle timeout and skips processing when the
  sequence number doesn't advance.
- **Evidence:** `buffer.py:39-48` (a `read()` exception kills the daemon thread silently;
  `exhausted` stays `False`); `pipeline.py:600-610` (`if seq == last_seq: time.sleep(0.001);
  continue`). If the device dies while `W` is held, `seq` freezes → `step()` is never called again
  → `TrackingLossGuard` never resets → **`W` (and any held key) stays pressed indefinitely**, and
  `pipeline.shutdown()` only runs once the loop breaks, which it never does.
- **Recommended fix:** Wrap the thread body in `try/except`, record the exception, set an
  `errored` flag; in the main loop, break (or raise) on `errored`; add a watchdog that releases
  all keys and exits if no new frame arrives within N×frame-interval.
- **Effort:** Medium.
- **Risk if unresolved:** Direct violation of the project's headline safety invariant ("a dropout
  must never leave `Space`/`Left Shift` held"). In live mode this strands real OS input.

### E2 — No max-idle / liveness timeout on the run loop
- **Severity:** High
- **Root cause:** The only exits are `KeyboardInterrupt`, `q` keypress (overlay only), or
  `buffer.exhausted` (`pipeline.py:599-636`).
- **Evidence:** With overlay off (production), there is no keypress path; combined with E1, a
  permission revocation mid-run or a stalled device yields an unbreakable spin.
- **Recommended fix:** Add a frame-arrival watchdog and a clean shutdown on timeout (reusing the
  `finally` that already calls `shutdown()`).
- **Effort:** Low-Medium.

### E3 — `_check_permissions` defaults to "trusted" when it cannot determine grant state
- **Severity:** Medium
- **Root cause:** Indeterminate state is treated as success.
- **Evidence:** `mac_emitter.py:84-100` (`except Exception: trusted = True`). If
  `ApplicationServices` import fails, input silently no-ops — the exact silent failure the rules
  call the "#1 footgun."
- **Recommended fix:** On indeterminate state, emit a loud one-time warning (and ideally a live
  test-event probe) rather than assuming granted.
- **Effort:** Low.

### E5 — No logging framework; `log_level` is configured but never wired
- **Severity:** High (observability)
- **Root cause:** Diagnostics are ad-hoc `print(...)` to stdout/stderr; `DebugSettings.log_level`
  exists but nothing consumes it.
- **Evidence:** `print` in `cli.py`, `mediapipe_backend.py:34`, `mac_emitter.py:193`;
  `config.py:320` defines `log_level` with no logger configured anywhere; the rules call for
  "rate-limited debug counters," which don't exist.
- **Recommended fix:** Stand up `logging`, honor `log_level`, replace `print`s, add the prescribed
  rate-limited per-stage counters (dropped frames, dropout/flush events, FPS).
- **Effort:** Medium.
- **Risk if unresolved:** A misbehaving live session is undiagnosable without a debugger.

---

## 4. Testing

### T2 — The failure-mode I/O paths have no automated coverage
- **Severity:** High
- **Root cause:** `run_pipeline`'s loop, the `FrameBuffer` *error* path, `AVFoundationSource`, and
  `MediaPipeHandTracker` are camera/OS-gated or untested — precisely the code most prone to the
  hangs in E1/E2. The `FrameBuffer` *happy* path **is** covered
  (`test_pipeline.py::test_frame_buffer_keeps_latest_and_exhausts` exercises latest-wins +
  exhaustion), but a `source.read()` that *raises* is not.
- **Evidence:** Test inventory has no `run_pipeline` test and no test where a `FrameSource.read()`
  raises; `AVFoundationSource`/`MediaPipeHandTracker` are exercised only indirectly. All other
  coverage is on pure logic (and is strong — see the positive note below).
- **Recommended fix:** Add a deterministic `FakeFrameSource` whose `read()` raises (assert the
  buffer surfaces the error and the loop exits releasing keys), and a `run_pipeline` test with an
  injected source + fake tracker + `NullEmitter`.
- **Effort:** Medium.

### T3 — `run_pipeline` injects `source` but constructs its own tracker/buffer/pipeline
- **Severity:** Medium
- **Root cause:** Half-finished dependency injection.
- **Evidence:** `pipeline.py:563-590` accepts `source` but always calls `HandTracker.create(...)`,
  `Pipeline.from_settings(...)`, `FrameBuffer(...)` internally — so the loop can't be tested
  without a real tracker.
- **Recommended fix:** Accept optional `tracker` (and/or `pipeline`) params mirroring `source`.
- **Effort:** Low.

### T5 — `conftest._build_extended_landmarks` is geometric and brittle
- **Severity:** Low
- **Root cause:** Extension-gesture tests depend on hand-crafted landmark geometry to hit target
  ratios; the "different directions to avoid overlap" comment signals prior fragility.
- **Evidence:** `conftest.py:66-108`.
- **Recommended fix:** Drive the extension SM by injecting a `FingerState` directly (bypass
  geometry) where the geometry itself isn't under test.
- **Effort:** Low.

**Positive coverage note:** All 153 tests pass, and the pure layers are genuinely well-tested —
Schmitt hysteresis (`test_schmitt.py`), per-hand resolvers, recovery windows (`test_recovery.py`
+ `test_pipeline_recovery.py`), dynamic deadzone, one-euro, sprint velocity, inventory toggle,
WASD cardinal zones, tracking-loss/shutdown key-release safety, and `mac_emitter` via mocks
(including `test_tracking_loss_reset_flushes_held_button` and a config invariant assertion that
`t_release > t_engage` for every pinch). The deterministic-by-design philosophy is followed well;
the gaps are confined to the OS/threading boundary (T2/T3).

---

## 5. Code organization

### O2 — `GestureEvent` is defined twice and reconciled with a `Union`
- **Severity:** Medium
- **Root cause:** The pinch and extension subsystems each define their own identical event type.
- **Evidence:** `gestures/pinch.py:65-77` and `gestures/extension.py:60-72`; `safety.py:29`
  unifies them as `Union[GestureEvent, PinchGestureEvent]`.
- **Recommended fix:** One shared `GestureEvent` (e.g. `gestures/events.py`) imported by both;
  delete the `Union`.
- **Effort:** Low.

### O3 — MediaPipe landmark indices are duplicated across modules
- **Severity:** Low
- **Root cause:** Constants re-declared per consumer despite the rules naming them a "single source
  of truth."
- **Evidence:** `pinch.py:24-30`, `finger_state.py:20-32`, `deadzone.py:24`, mirrored again in
  `conftest.py:24-33`.
- **Recommended fix:** A single `landmarks.py` with the canonical index map.
- **Effort:** Low.

### O4 — Transition tokens are bare strings across subsystems
- **Severity:** Low
- **Root cause:** `KEY_DOWN`/`KEY_UP` (gestures), `ENGAGE`/`RELEASE` (sprint), `PHASE_*`
  (recovery) are all stringly-typed; `GestureEvent.action: str`.
- **Evidence:** `schmitt.py:25-26`, `sprint_velocity.py:24-25`, `recovery.py:31-33`.
- **Recommended fix:** `enum.Enum` or `Literal[...]` types; tightens mypy `--strict`.
- **Effort:** Low-Medium.

### O5 — "Is this binding a scroll/mouse?" predicate is repeated inline
- **Severity:** Low
- **Evidence:** `binding in ("scroll_up","scroll_down")` recurs in `step`, `shutdown`,
  `_on_inventory_toggle` (`pipeline.py:267, 317, 555`); `_MOUSE_BUTTONS` membership in the emitter.
- **Recommended fix:** Centralize binding classification (typed binding objects or a small helper).
- **Effort:** Low.

### O6 — `cli.py` mixes arg-parsing, business logic, and stats for four commands
- **Severity:** Medium
- **Evidence:** 489 lines; `_calibrate_joysticks`, `main_analyze`, `main_bench` blend I/O,
  printing, and computation; `_summarize`/`_pct` are stats utilities embedded in the CLI.
- **Recommended fix:** Split per-command modules under a `cli/` package; move stats helpers to a
  reusable util shared by `analyze` and `bench`.
- **Effort:** Medium.

### O7 — Documented commands drift from actual entry points
- **Severity:** Medium
- **Root cause:** Docs reference `python -m` module paths that aren't runnable.
- **Evidence:** `CLAUDE.md`/`README` show `python -m minecraft_cv.calibration --config …`, but
  `calibration.py` is a pure library module with no `argparse`/`__main__`; the real entry is
  `mcv-calibrate = minecraft_cv.cli:main_calibrate` (`pyproject.toml:31`). Likewise there is no
  `__main__.py`, so `python -m minecraft_cv` (vs `…minecraft_cv.cli`) would fail.
- **Recommended fix:** Add a `__main__.py`, or fix the docs to the working invocations; wire
  `mcv-calibrate` to the documented path.
- **Effort:** Low.
- **Risk if unresolved:** Copy-paste commands from the docs fail on a fresh setup.

---

## 6. Dependencies

### D2 — `numpy>=1.25` is unbounded; NumPy 2.x ABI risk
- **Severity:** High (reproducibility)
- **Root cause:** No upper bound while `requires-python <3.14`.
- **Evidence:** `pyproject.toml:14`. A fresh install may resolve NumPy 2.x against
  mediapipe/opencv wheels built for `<2`, causing import-time ABI errors.
- **Recommended fix:** Constrain (`numpy>=1.25,<3`) and CI-test against the NumPy major you ship;
  or pin via a lockfile (see D5).
- **Effort:** Low.

### D5 — No committed lockfile
- **Severity:** Medium
- **Root cause:** Range specifiers only; README mentions Poetry but no `poetry.lock`/`uv.lock`.
- **Evidence:** `pyproject.toml` ranges; no lock in repo.
- **Recommended fix:** Commit a lockfile for reproducible production installs.
- **Effort:** Low.

### D1 — `certifi` used but undeclared
- **Severity:** Low
- **Evidence:** `mediapipe_backend.py:36-39` imports `certifi` (guarded by `except ImportError`).
- **Recommended fix:** Declare it (it's effectively required for the TLS download path) or keep the
  fallback and document it.
- **Effort:** Trivial.

### D4 — `torch` extra exists for a backend that's `NotImplementedError`
- **Severity:** Low
- **Evidence:** `pyproject.toml:25`; `tracker.py:75-78`.
- **Recommended fix:** Keep but label clearly as V2 (already documented); no action required now.

### D6 — `pytest-cov` present but no coverage gate
- **Severity:** Low
- **Evidence:** dev extras include `pytest-cov`; no coverage config or threshold.
- **Recommended fix:** Add a coverage config and a modest CI floor once T1/T2 land.
- **Effort:** Low.

**mypy note:** `StepResult.events: list` and `left/right_output` use bare/loose types
(`pipeline.py:43-45`); under the configured `mypy --strict` (`pyproject.toml:62-67`),
`list` without a parameter trips `disallow_any_generics`. Tighten to `list[GestureEvent]` /
`npt.NDArray[np.float64]`.

---

## Cross-cutting correctness notes

- **CLAUDE.md invariant #4 wording overstates the code.** "Attack+Use / Jump+Sneak are
  intentionally mutually exclusive" — the code does **not** enforce exclusion; these are
  independent Schmitt triggers and *can* fire together (the exclusivity is anatomical/game-level,
  matching the `gestures.md` "Blocked — game-logic conflict" rationale). `test_pinch.py`'s
  `…blocked_note` appears to be a documentation assertion, not an enforcement test. Reword the
  invariant to "not co-asserted by design," or enforce it if truly required.
- **`min_emit_confidence` gates on handedness score, not detection score.** `_split` filters on
  `HandResult.score` (`pipeline.py:540`), which `MediaPipeHandTracker` populates from the
  *handedness* category score (`mediapipe_backend.py:119-123`). Conflating handedness confidence
  with presence confidence is a minor semantic mismatch worth a comment.

---

## Top 10 Architecture Priorities (ranked by impact)

1. **E1 + E2 — Capture-thread failure hangs the loop with keys held.** Fix the thread's exception
   handling and add a frame-arrival watchdog. This is the only finding that breaks the project's
   own safety guarantee in live mode. *(Critical)*
2. **A3 — Consolidate the three overlapping hand-loss recovery mechanisms** into a single owner
   (`HandRecovery`); remove the redundant grace-frame recenter. *(High)*
3. **A1 + A2 — Decompose the `Pipeline` god-object** and unify the duplicated per-hand logic into
   one `HandController`. *(High / Medium-High)*
4. **S2 + S1 — Arbitrate shared physical keys** (sprint↔`ctrl`, sprint↔`W`); ref-count or
   guarantee disjoint ownership to prevent state desync. *(High when sprint enabled)*
5. **E5 — Stand up real logging/observability** (honor `log_level`, rate-limited counters);
   replace scattered `print`s. *(High)*
6. **A6 — Harden the model download** (timeout, checksum, configurable cache, offline message).
   *(High)*
7. **D2 + D5 — Pin/bound dependencies and commit a lockfile** (NumPy 2.x ABI risk). *(High/Medium)*
8. **T2 + T3 — Cover the failure-mode I/O paths** (a raising `FrameSource.read()`; inject the
   tracker into `run_pipeline` for a loop test). These guard the exact code behind priority #1.
   *(High/Medium)*
9. **O2 + O3 + O4 — De-duplicate `GestureEvent`, landmark indices, and stringly-typed tokens**;
   tighten the `mypy --strict` surface. *(Medium/Low)*
10. **O7 + A4 — Fix doc-vs-code command drift** (`python -m minecraft_cv.calibration` has no entry
    point) and **clarify the `mirror × swap_handedness` matrix** so a fresh setup works from the
    docs. *(Medium/Low)*

---

*End of review.*

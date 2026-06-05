# Rules: Gesture System Design

The contract between the CV tracking layer and the OS input layer. Read this when
writing or reviewing anything in `gestures/`, `joystick/`, or `input/`.

## Input paradigm

Three decoupled sub-systems run concurrently per frame:

1. **Pinch bitmask (left hand)** — thumb-to-finger pinches → WASD movement keys.
2. **Pinch bitmask + cursor (right hand)** — thumb-to-finger pinches → combat buttons; index-MCP screen position → mouse-look deltas.
3. **Face + head** — MediaPipe FaceLandmarker blendshapes → action keys; head-roll angle → hotbar scroll.

The `WristTiltJoystick` is still wired to the left hand but produces **HUD-only output** — it no longer presses WASD keys. WASD comes entirely from left-hand pinch events.

---

## Left hand: Pinch-bitmask WASD

The default pose is an **open or relaxed hand**. WASD keys are driven by thumb-to-fingertip pinches, each on an independent Schmitt trigger. Two simultaneous pinches produce a diagonal (e.g. index + middle → W+D).

### Gesture map

| Gesture          | Finger       | Key/Event      |
|------------------|--------------|----------------|
| Move right       | Thumb→Index  | `D`            |
| Move forward     | Thumb→Middle | `W`            |
| Move left        | Thumb→Ring   | `A`            |
| Move back        | Thumb→Pinky  | `S`            |

All four are `hold` mode: key held while pinch is engaged, released when the distance rises above `T_release`.

No conflict groups on the left hand — diagonal movement is intentional.

---

## Right hand: Pinch-bitmask + cursor

### Gesture map

| Gesture           | Detector          | Key/Event      |
|-------------------|-------------------|----------------|
| Attack / Break    | Thumb→Index       | Left click     |
| Use / Interact    | Thumb→Middle      | Right click    |
| Jump              | Thumb→Ring        | Space          |
| Swap Offhand      | Thumb→Pinky       | F              |
| Recenter (clutch) | Peace sign¹       | (no key)       |

¹ Peace sign = `extension_combo`: index + middle extended above `T_engage`, ring + pinky curled below `T_release`. While held, it suppresses attack, use, jump, and swap_offhand; freezes mouse output; and seeds the cursor anchor at the current index-MCP position (mouse-lifted clutch). On release, look resumes from the new anchor — no camera snap.

**Conflict group `primary_click`**: Attack and Use are mutually exclusive. The stronger pinch wins; the weaker is force-released. This reflects Minecraft's game-logic constraint.

### Mouse-look (cursor signal)

Camera look uses the **index-MCP position** (landmark 5) tracked frame-to-frame. The right joystick is a `ScreenJoystick` with a fixed neutral anchor at `(0.75, 0.5)` by default. The per-frame delta is:

```
mouse_delta = (current_index_mcp - prev_index_mcp) * right_sensitivity
```

filtered by a **One-Euro velocity-adaptive filter** (`joystick.look_filter: one_euro`). A cursor re-seed (via peace-sign or on re-entry) sets `_right_cursor_prev` to the current position without emitting movement.

---

## Face gestures

MediaPipe FaceLandmarker provides 52 blendshape scores per frame. Each face gesture has a Schmitt-trigger state machine with frame-count debounce.

### Blendshape gesture map

| Gesture       | Blendshape      | Key/Event    | Mode  |
|---------------|-----------------|--------------|-------|
| Inventory     | `browInnerUp`   | `E`          | Pulse |
| Throw Item    | `jawOpen`       | `Q`          | Pulse |

Face gesture semantics: **`t_engage > t_release`** (higher score = more engaged). `engage_frames` consecutive frames above `t_engage` fires KEY_DOWN; `release_frames` consecutive frames below `t_release` fires KEY_UP. Debounce absorbs single-frame noise.

---

## Head gestures: sneak and hotbar scroll

The head-roll angle is derived from the outer eye-corner line (FaceMesh landmarks 33 → 263). Roll is `atan2(dy, dx)` of that vector in normalized image space.

Two mutually-exclusive Schmitt states share the same angle signal:

```
roll > +engage_deg  →  hotbar_next (scroll up)
roll < -engage_deg  →  hotbar_prev (scroll down)
```

Each releases when the angle returns inside `±release_deg`. `engage_deg > release_deg` strictly. A held tilt re-emits scroll ticks at `scroll_repeat_rate_hz`.

The head-pitch (nodding down) gesture uses the 2D vertical ratio of (chin to nose) / (nose to nasion). The ratio drops when looking down, driving the `sneak` gesture via a Schmitt trigger.

---

## Schmitt trigger (hysteresis gate)

Two sign conventions used across the project:

**Lower-is-engaged** (pinch detectors):
- `T_release > T_engage` strictly.
- Signal = normalized thumb-to-fingertip distance; engage when it drops below `T_engage`.

**Higher-is-engaged** (extension_combo, face blendshapes):
- `T_engage > T_release` strictly.
- Signal = extension ratio or blendshape score; engage when it rises above `T_engage`.

Equal or inverted thresholds reintroduce the chatter the gate exists to prevent. A config validator (`@model_validator`) and a test both assert the correct ordering for all configured gestures.

---

## Concurrency model

| Action combo           | Feasibility | Reasoning |
|------------------------|-------------|-----------|
| Move + Look            | Seamless    | Independent hands |
| Move + Jump            | Seamless    | LH WASD pinch + RH ring pinch |
| Move + Attack          | Seamless    | LH pinch + RH pinch — different hands |
| Move + Jump + Attack   | Seamless    | All independent |
| Attack + Use           | Blocked     | Conflict group `primary_click` (game-logic) |
| Recenter + Attack      | Blocked     | `recenter` suppresses `attack`, `use`, `jump` |
| Sneak (face) + WASD    | Seamless    | Face stream is fully independent |
| Head tilt + anything   | Seamless    | Face/head stream is fully independent |

---

## Failure modes and mitigations

| Failure | Mitigation |
|---------|------------|
| Hand occlusion mid-gesture | `TrackingLossGuard` releases all held keys on dropout |
| Camera drift | Peace-sign recenter reseeds cursor anchor without moving camera |
| Pinch jitter at threshold | Schmitt hysteresis band swallows sub-threshold oscillation |
| Face blendshape noise spike | Frame-count debounce (`engage_frames`) absorbs single-frame noise |
| Head tracking lost | `HeadRollDetector.reset()` force-releases held scroll direction |
| Stuck key on crash | `Pipeline.shutdown()` releases all held keys + WASD + emitter |

### Tracking loss safety

If MediaPipe returns no hand for a given side:
1. Emit KEY_UP for every currently-held gesture via `TrackingLossGuard`.
2. Zero the joystick output for that hand.
3. After `dropout_flush_ms`, hard-flush: reset the cursor anchor and look filter.

A crash or dropout must never leave `Space` held (bunny hopping) or a click stuck down. This is a hard requirement enforced in tests.

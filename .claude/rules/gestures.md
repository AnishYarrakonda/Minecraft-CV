# Rules: Gesture System Design

The contract between the CV tracking layer and the OS input layer. Read this when
writing or reviewing anything in `gestures/`, `joystick/`, or `input/`.

## Input paradigm

Two decoupled sub-systems run concurrently per frame:

1. **Spatial joysticks (continuous)** — palm/wrist position → WASD + camera look.
2. **Discrete gestures** — finger extensions (left hand) + thumb-to-finger pinches (right hand) → button events.

They share the same landmark stream but are completely independent state machines.

---

## Spatial joysticks

Each hand's **wrist (landmark 0)** or **middle MCP (landmark 9)** is the anchor point.
Do NOT use the bounding-box center — pinching shifts it, corrupting the joystick vector.

```
NEUTRAL: pos inside deadzone sphere  → output (0, 0)
ACTIVE:  pos outside deadzone sphere → output = (pos − deadzone_edge) * sensitivity
```

- The deadzone is a **sphere** (not a box) so diagonal directions aren't biased.
- Output is continuous at the sphere boundary — no step discontinuity.
- Apply an **exponential acceleration curve** so large physical movements map to fast
  in-game camera/movement without requiring the user to travel far. This directly
  mitigates Gorilla Arm syndrome.
- **Left hand** → WASD translation. **Right hand** → mouse look (camera rotation).

### Cardinal zones

WASD output uses angular cardinal zones instead of independent axis checks:
- Each axis direction has a pure zone of ±`cardinal_half_width` degrees (default 35°).
- Between zones (20° gaps), both adjacent keys are pressed (diagonal movement).
- This ensures the user can achieve pure W, A, S, or D without always getting diagonals.

### Recenter / drift macro

When both hands leave the frame and re-enter, the new entry coordinates become the
fresh `(0,0,0)` neutral. This is the drift/recenter macro — no button press required.

---

## Handedness swap

MediaPipe's handedness labels may be inverted when using a mirrored camera feed.
The `tracking.swap_handedness` config flag (default: `true`) inverts the L/R labels
so the user's physical left hand drives left-hand gestures/WASD and the physical
right hand drives right-hand gestures/mouse-look.

---

## Left hand: Extension-based gestures

The default pose is a **relaxed closed fist**. Gestures are triggered by extending
specific fingers. Extension is measured as a continuous ratio: `dist(wrist, tip) /
dist(wrist, PIP)` — values > ~1.15 indicate extension; < ~1.0 indicate curled.

### Gesture map

| Gesture          | Finger Pattern         | Key/Event      | Mode  |
|------------------|------------------------|----------------|-------|
| Jump             | Thumb extended outward | `Space`        | Hold  |
| Sneak            | Index extended only    | `Left Shift`   | Hold  |
| Sprint           | Middle extended only   | `Ctrl`         | Hold  |
| Inventory (E)    | Index + Middle (peace) | `E`            | Pulse |
| Throw Item (Q)   | Ring extended only     | `Q`            | Pulse |
| Switch Offhand   | Pinky extended only    | `F`            | Pulse |

### Exclusion logic

Single-finger "only" gestures include exclusion checks: if other non-required
fingers are also extended above the engage threshold, the gesture is suppressed.
This prevents a fully open hand from triggering every gesture simultaneously.

The thumb is independent — `thumb_out` has no exclusion fingers.

### Pulse gestures

Inventory, Throw Item, and Switch Offhand use **pulse mode**: a single key tap
(key_down + immediate key_up) on engage. No repeat while held, no key_up event
on release. This is appropriate for toggle/one-shot actions.

---

## Right hand: Pinch-bitmask

Each finger's pinch state is an independent Schmitt trigger. All four can be active
simultaneously (subject to anatomical constraints below).

### Gesture map

| Gesture          | Finger       | Key/Event      |
|------------------|--------------|----------------|
| Attack / Break   | Thumb→Index  | Left click     |
| Use / Interact   | Thumb→Middle | Right click    |
| Hotbar Next      | Thumb→Ring   | Scroll up      |
| Hotbar Prev      | Thumb→Pinky  | Scroll down    |

### Schmitt trigger (hysteresis gate)

The most important correctness invariant in the project. Each pinch gesture has two
thresholds operating on the **normalized** thumb-to-fingertip distance (divided by
wrist→middle-MCP span):

```
STATE: RELEASED
  if distance < T_engage  → KEY_DOWN(action); STATE = HOLDING

STATE: HOLDING
  if distance > T_release → KEY_UP(action);   STATE = RELEASED
```

**`T_release` must be strictly greater than `T_engage`.** For extension gestures,
the invariant is inverted: **`T_engage` must be strictly greater than `T_release`.**

### Hotbar scroll — momentary pulse with repeat

Ring and pinky anatomical coupling means Hotbar Next/Prev use a **repeat-rate model**:
- Engage → emit one scroll tick.
- Hold the pinch → re-emit at `scroll_repeat_rate_hz` (default 8 Hz).
- Release → stop.

---

## Concurrency model

| Action combo       | Feasibility | Reasoning |
|--------------------|-------------|-----------|
| Move + Look        | Seamless    | Independent hands |
| Move + Jump        | Seamless    | LH translation + LH thumb out |
| Jump + Attack      | Seamless    | LH thumb + RH index — different hands |
| Move+Jump+Attack   | Seamless    | All independent |
| Jump + Sneak       | Possible    | LH thumb + LH index — biomechanically feasible |
| Sneak + Sprint     | Blocked     | Mutually exclusive by design |
| Attack + Use       | Blocked     | RH index + RH middle — usually a game-logic conflict |
| Move + Inventory   | Works       | Peace sign + WASD still tracks wrist position |

---

## Failure modes and mitigations

| Failure | Mitigation |
|---------|------------|
| Hand occlusion (fingers hide thumb) | Mount camera at ~45° looking down; enforce physical operating space |
| Drifting neutral (user shifts in chair) | Recenter macro on re-entry; dynamic deadzones in V2 |
| Pinky pinch drags ring finger | Hotbar = momentary pulse + repeat rate, not hold-to-repeat |
| Jitter at engage threshold | Widen Schmitt band; add One-Euro smoothing upstream |
| Tracking lost mid-gesture | State machines must fail-safe: release all held keys on dropout |
| False positive from open hand | Exclusion logic: single-finger gestures rejected if others also extended |

### Tracking loss safety

If MediaPipe returns no hand for a given side:
1. Emit KEY_UP for any currently-held gesture on that hand.
2. Zero the joystick output for that hand.
3. Do NOT leave any key stuck down.

This is a hard requirement. A crash or dropout must never leave `Space` held (bunny
hopping forever) or `Left Shift` held (sneak-locked).

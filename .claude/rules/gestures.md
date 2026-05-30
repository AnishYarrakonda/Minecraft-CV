# Rules: Gesture System Design

The contract between the CV tracking layer and the OS input layer. Read this when
writing or reviewing anything in `gestures/`, `joystick/`, or `input/`.

## Input paradigm

Two decoupled sub-systems run concurrently per frame:

1. **Spatial joysticks (continuous)** — palm/wrist position → WASD + camera look.
2. **Pinch-bitmask (discrete)** — thumb-to-finger distances → button events.

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
- **Sprint** is derived from left-hand translation *velocity* (derivative), not a
  dedicated gesture. Fast left-hand movement → synthesize double-tap `W`. This saves
  a finger gesture slot.

### Recenter / drift macro

When both hands leave the frame and re-enter, the new entry coordinates become the
fresh `(0,0,0)` neutral. This is the drift/recenter macro — no button press required.
Dynamic deadzones (V2) should follow slow drift but not chase fast intentional motion.

---

## Pinch-bitmask

Each finger's pinch state is an independent Schmitt trigger. All four can be active
simultaneously (subject to anatomical constraints below).

### Gesture map

| Hand  | Gesture          | Finger       | Key/Event      |
|-------|------------------|--------------|----------------|
| Left  | Jump             | Thumb→Index  | `Space`        |
| Left  | Sneak            | Thumb→Middle | `Left Shift`   |
| Left  | Inventory (mode) | Full fist    | `E`            |
| Right | Attack / Break   | Thumb→Index  | Left click     |
| Right | Use / Interact   | Thumb→Middle | Right click    |
| Right | Hotbar Next      | Thumb→Ring   | Scroll up      |
| Right | Hotbar Prev      | Thumb→Pinky  | Scroll down    |

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

**`T_release` must be strictly greater than `T_engage`.** A test asserts this for
every configured gesture. The gap between them is the hysteresis band; it swallows
CV frame jitter so a finger hovering near the engage point doesn't chatter.

Tuning rules:
- Too narrow (T_release ≈ T_engage) → chatter (rapid KEY_DOWN/KEY_UP).
- Too wide → gesture feels unresponsive to engage or sticky to release.
- Diagnose chattering with the `frame-analyzer` skill on a jittery clip, not by
  guessing. The correct fix is almost always **widen the band**, not lower T_engage.
- Complement with **One-Euro or EMA smoothing** upstream on the landmark stream to
  reduce jitter amplitude before it reaches the trigger.

### Inventory mode switch

Clenching a full fist on the Left Hand is a **mode switch**, not a momentary action:

- On engage: emit `E` (open inventory), suspend LH WASD tracking, repurpose LH
  spatial position as a UI mouse cursor.
- On release (fist opens): emit `E` again (close inventory), restore LH WASD,
  re-capture new wrist neutral.
- Any movement keys held at fist-engage must be released cleanly before the switch.

### Hotbar scroll — momentary pulse only

Ring and pinky anatomical coupling (pinching the pinky drags the ring finger) means
Hotbar Next/Prev must be **momentary pulses** with debouncing, not continuous holds:

- Engage → emit one scroll tick, start a short cooldown.
- Hold the pinch → re-emit at a low repeat rate (not every frame).
- Release → stop. Do not accumulate scroll ticks while held.

---

## Concurrency model

| Action combo       | Feasibility | Reasoning |
|--------------------|-------------|-----------|
| Move + Look        | Seamless    | Independent hands |
| Move + Jump        | Seamless    | LH translation + LH index pinch |
| Jump + Attack      | Seamless    | LH index + RH index — different hands |
| Move+Jump+Attack   | Seamless    | All independent |
| Jump + Sneak       | Blocked     | LH index + LH middle — biomechanically awkward; rarely needed |
| Attack + Use       | Blocked     | RH index + RH middle — usually a game-logic conflict anyway |
| Move + Inventory   | Mode switch | Fist suspends WASD; hand becomes cursor |

`Attack+Use` and `Jump+Sneak` blocks are **intentional design**, not bugs. Do not
implement them as allowed combinations.

---

## Failure modes and mitigations

| Failure | Mitigation |
|---------|------------|
| Hand occlusion (fingers hide thumb) | Mount camera at ~45° looking down; enforce physical operating space |
| Drifting neutral (user shifts in chair) | Recenter macro on re-entry; dynamic deadzones in V2 |
| Pinky pinch drags ring finger | Hotbar = momentary pulse + cooldown, not hold-to-repeat |
| Jitter at engage threshold | Widen Schmitt band; add One-Euro smoothing upstream |
| Tracking lost mid-gesture | State machines must fail-safe: release all held keys on dropout |

### Tracking loss safety

If MediaPipe returns no hand for a given side:
1. Emit KEY_UP for any currently-held gesture on that hand.
2. Zero the joystick output for that hand.
3. Do NOT leave any key stuck down.

This is a hard requirement. A crash or dropout must never leave `Space` held (bunny
hopping forever) or `Left Shift` held (sneak-locked).

---

## Implementation roadmap

**MVP (current priority):**
- Wrist-anchor spatial joysticks (LH → WASD, RH → mouse look).
- Index + middle pinch only: Jump, Sneak, Attack, Use.
- Hotbar via physical mouse scroll wheel (bypass ring/pinky gestures until tuned).
- Schmitt trigger with threshold assertions.
- NullEmitter default; MacInputEmitter opt-in.

**V2:**
- Dynamic deadzones (slow drift following).
- Sprint via LH translation velocity derivative.
- Inventory mode switch (fist → cursor repurpose).
- Ring/pinky hotbar with pulse debouncing.

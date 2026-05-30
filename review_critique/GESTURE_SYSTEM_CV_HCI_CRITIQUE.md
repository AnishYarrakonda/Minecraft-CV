# minecraft_cv — Gesture System Critique (CV + HCI)

*Specialist review of the gesture detection, Schmitt-trigger, joystick, and tracking-loss
layers. Read against `.claude/rules/gestures.md`, `config.yaml`, `src/minecraft_cv/gestures/**`,
`src/minecraft_cv/joystick/**`, `src/minecraft_cv/recovery.py`, and the gesture tests.*

## TL;DR verdict

The **Schmitt-trigger primitive is correct and well-tested**, and the joystick math (spherical
deadzone, ease-in curve, One-Euro on look) is genuinely good HCI. But the **gesture *vocabulary*
is biomechanically naive**, the **discrete state machines have no temporal debounce** (a single
bad CV frame can latch or drop an action), and the **tracking-loss layer releases held keys on a
single dropped frame** while *over-suppressing* discrete re-engage for 500 ms. Several mapped
gestures (`ring_only`, `middle_only`-as-hold, `thumb_out`) will fail or fatigue in real play.

---

## 1. Gesture accuracy

### 1.1 No temporal debounce → single-frame latch (false positive) — **HIGH likelihood, HIGH impact**

`SchmittTrigger.update` (`schmitt.py:69`) and `_GestureTrigger.update` (`extension.py:131`)
engage on the **first** frame that crosses the threshold. There is no "N consecutive frames"
requirement — unlike `SprintVelocityTrigger` (`trigger_frames`), `InventoryModeToggle`
(`hold_frames`), and `HandRecovery`, which all debounce.

**Why it bites:** MediaPipe periodically emits a single garbage frame (a fingertip landmark
snaps across the hand during fast motion or partial occlusion). One frame with `attack` distance
< 0.30 fires `KEY_DOWN`, and because the trigger is now `HOLDING`, it **stays latched** until
distance climbs back above `t_release` (0.45). So one bad frame → an attack that sticks ON for
several frames. In Minecraft that's an accidental block break / mob hit.

- **Likelihood:** High — attack is the most-used action and MediaPipe jitter is constant.
- **Impact:** High — wrong block destroyed, wrong entity attacked, griefing on servers.
- **Fix:** Require `engage_frames` (≈2–3) consecutive sub-threshold frames before `KEY_DOWN`.
  Cheap, deterministic, testable with the existing synthetic-sequence harness.

### 1.2 `thumb_out` (Jump) threshold likely unreachable — **MED-HIGH likelihood, HIGH impact**

`thumb_ext = dist(thumb_tip, index_mcp) / hand_scale` where `hand_scale = dist(wrist, middle_mcp)`
(`finger_state.py:74-77`). Jump needs `thumb_ext > 1.2` (`config.yaml:32`).

**Why it bites:** In a relaxed fist the thumb tip rests *on* the index MCP → ratio ≈ 0. A
maximally abducted thumb reaches roughly palm-length away, i.e. `dist(thumb_tip, index_mcp) ≈
hand_scale` → ratio ≈ 0.8–1.0 for most people. **1.2× palm length is anatomically near the
ceiling.** Jump will frequently *false-negative*, and since jump is high-frequency (parkour,
combat, mining up), this is brutal. Worse, this metric is **not normalized for thumb length**
(children, short thumbs), so the failure is user-dependent.

- **Fix:** Drop the engage threshold to ~0.7–0.8, or measure thumb *abduction angle*
  (thumb-CMC→thumb-tip vs index-MCP→pinky-MCP) instead of a raw distance ratio. Angle is
  length-invariant.

### 1.3 Extension ratio is orientation-dependent — **HIGH likelihood (with recommended mount), MED impact**

`index_ext = dist(wrist,tip)/dist(wrist,pip)` uses 3D landmarks, but the recommended camera mount
is "~45° looking down" (gestures.md failure table). At that angle a finger pointing toward/away
from the camera **foreshortens** — the projected tip-to-wrist distance shrinks even though the
finger is fully extended → ratio drops below `t_engage` → false negative; or a curled finger
viewed edge-on reads as extended → false positive.

**Why:** `z` is "the least reliable landmark axis" (your own tech-stack rule), so the 3D distance
is dominated by noisy z whenever the finger axis aligns with the optical axis.

- **Fix:** Compute extension from **joint angles** (MCP→PIP→DIP→TIP vector angles) rather than
  wrist-to-tip distance ratios. Angles are far more pose-invariant than radial distances under
  perspective.

### 1.4 Attack+Use are *not* actually mutually exclusive in code — **MED likelihood, HIGH impact**

CLAUDE.md invariant #4 and gestures.md call `Attack+Use` "Blocked." Nothing in code blocks it.
`PinchStateMachine.update` (`pinch.py:138`) runs **independent** triggers for `index` and
`middle`. When the thumb tip lands in the gap between index and middle tips (a very natural
pinch), **both** normalized distances drop below 0.30 → `attack` *and* `use` both fire →
simultaneous left+right click.

**Why it bites:** Thumb-to-index and thumb-to-middle are the two *easiest, most adjacent*
pinches; cross-firing is the default failure, not an edge case.

- **Likelihood:** Medium-High.
- **Impact:** High — left+right click together in Minecraft is a real conflict (place + break
  races).
- **Note:** Your docs *assert* exclusivity but the implementation provides none. This is a
  doc/code contradiction worth resolving (the instruction "don't fix them" refers to *not* trying
  to make them co-usable — but the silent *cross-firing* is a different bug).

### 1.5 Exclusion gray band — **LOW-MED likelihood, LOW impact**

`_check_exclusion` (`extension.py:163`) suppresses a single-finger gesture only if a non-required
finger exceeds **the gesture's own `t_engage`** (1.15), while the gesture *holds* down to
`t_release` (1.05). A neighbor sitting in 1.05–1.15 neither excludes nor reads as curled. Mostly
benign, but it means "index_only" tolerates a half-extended middle finger — sneak can stay
engaged with a sloppy hand. The exclusion threshold should be the *neighbor's* curled threshold,
not the active gesture's engage threshold.

---

## 2. Schmitt-trigger design

**What's right:** Construction-time `t_release > t_engage` enforcement (`schmitt.py:52`),
idempotent `reset()`, the 0.15-wide pinch band (0.30/0.45) is comfortably wider than typical
MediaPipe jitter (~0.02–0.05 normalized) — good tremor headroom. Tests cover hover-in-band,
jittery sequences, and inverted thresholds.

### 2.1 Invariant #1 is unenforced for extension gestures — **MED likelihood (config error), MED impact**

`_GestureTrigger` (`extension.py:108`) has **no `__post_init__` check**. The inverted invariant
("`t_engage > t_release` for extension gestures") is documented but never validated in code. A
typo in `config.yaml` (e.g. `t_engage: 1.05, t_release: 1.15` for sneak) would silently produce
an *un-releasable or chattering* gesture. The `SchmittTrigger` class guards this for pinch; the
extension path doesn't even reuse `SchmittTrigger` — it reimplements the comparison inline. Add
the symmetric assertion.

### 2.2 Hysteresis band is static and uniform — **MED likelihood, MED impact for accessibility**

Band width is fixed in config (0.15 pinch, 0.10 extension). Tremor amplitude varies hugely per
user; a fixed band that's comfortable for one user chatters for a person with essential tremor
and feels mushy/laggy for a steady user. The dynamic-deadzone idea (joystick) is exactly the
right instinct — **but it isn't applied to the discrete Schmitt bands.** Per-user band
auto-sizing from resting jitter would help (see V2).

### 2.3 No trigger-level rate limit

Threshold-crossing relies on `key_repeat_guard_ms` downstream. Fine for keys, but the *latch*
problem (1.1) is upstream of that guard and unaffected by it.

---

## 3. Biomechanics

This is the weakest area. The finger-independence assumptions are wrong for several gestures.

| Gesture | Pose | Independence reality | Verdict |
|---|---|---|---|
| `ring_only` → Throw (Q) | ring extended, index/middle/pinky curled | **Worst case.** Ring finger shares the flexor/extensor digitorum and juncturae tendinum with middle+pinky. Isolated ring extension is the single hardest digit pose; many people physically can't do it cleanly. | **Will fail.** False-negative + fatigue. Reassign. |
| `pinky_only` → Switch offhand (F) | pinky only | Moderate independence; harder than index but doable. | Marginal; fatiguing if frequent. |
| `middle_only` → Sprint (Ctrl), **held** | middle extended, others curled, **sustained** | Middle has OK independence, but sprint in Minecraft is held for *minutes* of travel. Sustained isolated middle extension = forearm fatigue + it's the obscene gesture (social cost). | **Fatiguing.** Velocity-sprint (already in config, disabled) is the right answer — enable it. |
| `index_only` → Sneak, **held** | index extended, sustained | Index is independent (good accuracy), but sneak is held a long time (cliff edges, building). Sustained hold = fatigue. | Accuracy fine, ergonomics poor for long holds. Consider a toggle. |
| `thumb_out` → Jump | see 1.2 | Threshold likely unreachable for many hands. | Accuracy poor. |
| `index_middle` → Inventory | peace sign | Easy, well-chosen. | Good. |

**Hand-size / proportion variation:** Distance-ratio metrics (extension) are *partially*
proportion-invariant, but `thumb_ext`'s absolute 1.2 threshold is **not** (thumb-length/palm-length
ratio varies widely). Children, and adults with shorter thumbs/pinkies, get systematically worse
jump and offhand detection. No per-user calibration exists for the discrete thresholds (only the
joystick deadzone calibrates).

**Mobility / Gorilla Arm:** The spatial joystick's ease-in curve genuinely mitigates large-travel
fatigue — good. But the *discrete left hand* must simultaneously hold a fist AND extend a finger
AND keep the wrist in the WASD joystick's active zone. Holding a fist while also being the WASD
anchor means the hand is **never relaxed** during movement. That's the dominant fatigue source
and it's structural, not tunable.

---

## 4. Accessibility

- **Tremor:** Static Schmitt bands + 0.05 static deadzone will chatter for tremor users. The
  One-Euro look filter and *optional* dynamic deadzone are the right tools but **off by default**
  (`dynamic_deadzone: false`). Default-on adaptive sizing would dramatically widen the usable
  population. Also, tremor amplitude *during motion* isn't handled — One-Euro helps look, but
  WASD uses plain EMA only.
- **Reduced range of motion / injury:** The vocabulary *requires* a full closed fist as the
  resting pose plus isolated finger extensions plus a maximally abducted thumb. Anyone with
  arthritis, a splint, partial amputation, Dupuytren's contracture, or limited thumb abduction is
  locked out of multiple core actions with no remapping path. Bindings are configurable but
  **gesture *types* are not** — you can't say "use a pinch for jump instead of thumb-out."
- **Non-standard anatomy:** Missing/fused fingers break the `_FINGER_ORDER` assumptions and the
  exclusion logic outright. No graceful degradation.
- **One-handed users:** The two-hand split (left=WASD+extensions, right=look+pinch) is mandatory.
  No single-hand mode exists.

**Impact:** This is open-source-facing software; the current design implicitly assumes a
neurotypical adult with five fully independent fingers on each hand. That's an accessibility gap
worth a roadmap item, not a footnote.

---

## 5. Tracking loss

**What's right:** `safety.py` + `TrackingLossGuard` release held keys on absence (no stuck
Space/Shift), and the `HandRecovery` flush+stabilization design is thoughtful for joystick snap.

### 5.1 Held keys release on a *single* dropped frame — **HIGH likelihood, HIGH impact**

`HandRecovery.update` returns `emit=False` the instant a hand is absent (`recovery.py`, the
`if not present` branch), and the pipeline passes `None` to `guard.process` whenever `emit` is
False (`pipeline.py:244-246`), which calls `machine.reset()` → immediate `KEY_UP`.

**There is no grace period for discrete held gestures.** The joystick has
`recenter_grace_frames: 3`, but that grace **does not protect held keys** — it only affects
neutral recentering. So one MediaPipe miss (common during fast hand motion or when fingers
occlude each other — your own "Hand occlusion" failure mode) → sneak releases for one frame → you
step off the cliff you were sneaking on; or your held attack stutters mid-combat.

- **Fix:** Give the discrete guard the same N-frame grace the joystick has: keep holding through
  `release_grace_frames` of absence before emitting `KEY_UP`. This is the inverse of the
  engage-debounce in 1.1 and equally important.

### 5.2 Stabilization window over-suppresses discrete gestures — **MED likelihood, MED impact**

After a >100 ms dropout, `stabilization_ms: 500` blocks **all** emit for that hand
(`emit=False, track=True`). That's correct for the *joystick* (the neutral needs re-seeding to
avoid a snap). But discrete pinches/extensions **don't need a neutral** — they're scale-invariant
and stateless per frame. Coupling them to the joystick's stabilization means: your right hand
briefly leaves frame, comes back, and **attack is dead for half a second** even though the hand
is perfectly tracked. In combat that's a death.

- **Fix:** Split the emit gate — suppress *joystick* output during stabilization, but allow
  discrete gestures to re-engage immediately (they have their own safety via reset).
  Architecturally, `RecoveryDecision` should carry `emit_joystick` vs `emit_discrete`.

### 5.3 Partial visibility / camera-angle changes

Extension ratios degrade silently (see 1.3). There's no confidence gate per *landmark*, only per-
hand (`min_emit_confidence`). A hand that's present but with a poorly-localized ring fingertip
will produce confident-but-wrong gestures.

---

## 6. Minecraft-specific usability

- **Action frequency mismatch:** The most frequent actions get the most fragile/fatiguing
  gestures. Attack (constant) = a sustained pinch hold (fatigue + latch risk). Jump (very
  frequent) = the hardest-to-reach `thumb_out`. Sprint (held for minutes) = sustained isolated-
  middle extension. The frequency/effort curve is inverted — rare actions (offhand, throw) are
  *also* hard (`ring_only`/`pinky_only`), so there's no easy/hard tradeoff being made
  deliberately.
- **Mining/attacking** is *held* in Minecraft (continuous left-click to break). A pinch-hold for
  5–10 seconds repeatedly is fatiguing and prone to the latch/release jitter at the band edges.
  Consider an auto-repeat or "lock" affordance.
- **Naturalness:** Peace-sign→inventory and pinch→click are intuitive. Thumb-out→jump and
  middle→sprint are not discoverable and one is socially awkward.
- **Hotbar ring/pinky pinch:** ring and pinky pinches **drag each other** (your own failure table
  admits this). `hotbar_next` (ring) and `hotbar_prev` (pinky) will misclassify into each other —
  scrolling the wrong way. The repeat-rate model addresses *spam* but not *misclassification*.

---

# Gesture System V2

*Goal: fix correctness and the worst biomechanics without changing the two-hand paradigm. All
changes are deterministic and unit-testable with the existing synthetic-sequence harness.*

1. **Temporal debounce on every discrete trigger.** Add `engage_frames` (default 2) and
   `release_grace_frames` (default 3) to `SchmittTrigger` and `_GestureTrigger`. Engage only after
   N consecutive crossing frames; release only after M consecutive absent/over-threshold frames.
   Kills both the single-frame latch (1.1) and the single-frame dropout release (5.1). One config
   block, ~15 lines, fully testable.

2. **Enforce the inverted invariant in code for extension gestures.** Add `__post_init__` to
   `_GestureTrigger` asserting `t_engage > t_release`, mirroring `SchmittTrigger`. Extend the
   existing "invariant holds for all gestures" test to the extension set.

3. **Decouple discrete emit from joystick stabilization.** Split `RecoveryDecision.emit` into
   `emit_joystick` and `emit_discrete`; let pinches/extensions re-engage immediately on hand
   return while the joystick still stabilizes (5.2).

4. **Reassign the biomechanically broken gestures:**
   - Throw (Q): move off `ring_only` → a **thumb→ring pinch** *or* a quick downward flick; ring
     extension is unreliable.
   - Sprint: **enable velocity-sprint by default** (it's already implemented) and retire
     `middle_only`-as-hold, or make sprint a *toggle* not a hold.
   - Jump: fix the `thumb_out` metric — lower engage to ~0.7 and/or switch to thumb-abduction
     *angle* (length-invariant). Re-tune against recorded clips.

5. **Block attack+use cross-fire deliberately.** Add a single arbitration rule in
   `PinchStateMachine`: if both index and middle distances are sub-engage in the same frame, hold
   whichever engaged first and suppress the other (matches the documented "mutually exclusive"
   intent instead of silently double-firing).

6. **Default-on adaptive thresholds.** Turn `dynamic_deadzone: true` by default and add an
   analogous resting-jitter auto-sizing pass for the discrete Schmitt bands (measure per-finger
   jitter during the same calibration window; set band = base + margin·jitter). This is the single
   biggest accessibility win for tremor.

7. **Per-landmark confidence gate** before computing extension ratios, so a present-but-poorly-
   localized fingertip doesn't produce confident wrong gestures (5.3).

---

# Gesture System V3

*Goal: rethink the vocabulary and accessibility model. Bigger lifts; data- and ergonomics-driven.*

1. **Angle-based pose features instead of distance ratios.** Replace `wrist→tip / wrist→pip` and
   raw thumb distance with **joint-angle vectors** (per-finger MCP/PIP flexion angles + thumb
   abduction angle). Angles are length- and perspective-invariant — fixes hand-size variation
   (1.2, §3) and 45°-mount foreshortening (1.3) in one move. Feed angles into the same Schmitt
   machinery.

2. **Configurable gesture *types*, not just key bindings.** Promote the gesture→pose mapping into
   config so a user can assign *any* action to *any* feasible pose (pinch, extension, flick,
   dwell). This is the foundation for accessibility: a user with limited ring mobility remaps
   Throw to a thumb pinch; a one-handed user maps everything to right-hand pinch-bitmask + dwell.

3. **Per-user gesture calibration wizard.** Extend `calibration.py` beyond the joystick: have the
   user perform each gesture 5× and each "rest"; fit per-user `t_engage`/`t_release`/band-width
   and per-finger independence (auto-disable gestures the user can't reliably perform, suggesting
   remaps). This directly addresses non-standard anatomy and removes the magic 1.15/1.2 constants.

4. **A small learned classifier as an optional backend.** Instead of hand-tuned per-finger
   thresholds, train a tiny MLP / random forest on the 21-landmark vector → discrete pose class,
   per-user-fine-tuned from the calibration samples. Keep the Schmitt/hysteresis layer on the
   *class confidence* for temporal stability, but get robustness to anatomy and orientation that
   hand-tuned ratios can't. Lives behind the existing swappable interface; CPU-cheap.

5. **Ergonomic remapping by action frequency.** Re-derive the gesture map from a frequency/effort
   model: highest-frequency actions (attack, jump, move) get the lowest-effort, most-independent
   poses; rare actions tolerate harder poses. Add **hold-to-toggle** affordances for long-duration
   actions (sneak, sprint, continuous mining) so users aren't holding a pose for minutes.

6. **Accessibility profiles + single-hand mode.** Ship presets (tremor, reduced-ROM, one-handed)
   that pre-select feasible gestures, widen bands, and enable dwell-based activation as an
   alternative to precise poses. A one-handed mode driving WASD via dwell-zones and actions via
   right-hand-only pinch-bitmask makes the project usable by a far wider audience.

---

## Highest-priority fixes

Do these first — they're cheap and correctness-critical:

1. Temporal engage-debounce (1.1)
2. Held-key release grace (5.1)
3. Extension-invariant enforcement (2.1)
4. `thumb_out` jump threshold (1.2)

The first two are the difference between "demo that works on video" and "controller you can
actually play with."

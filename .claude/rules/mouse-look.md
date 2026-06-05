# Rules: Mouse Look & Joystick

Read this when working on `joystick/`, mouse-look deltas, cursor filtering, or
camera-look feel (sensitivity, lag, drift, recentering).

## Signal source

Camera look is driven by the **right-hand index-MCP position** (landmark 5) tracked
frame-to-frame. The landmark is in normalized `[0, 1]` screen space.

The per-frame delta:
```
mouse_delta = (current_index_mcp - prev_index_mcp) * right_sensitivity
```

This is a **relative** delta emitted each frame, not an absolute pointer position.
Minecraft reads raw relative motion — do not convert to absolute screen coords.

## ScreenJoystick

`joystick/screen.py` — `ScreenJoystick` owns the right-hand cursor state:

- Holds `_right_cursor_prev`: the landmark position from the previous frame.
- On first entry or after a re-seed, sets `_right_cursor_prev = current` and emits
  **no delta** (avoids the snap-to-new-position jump on hand re-entry).
- Scales delta by `settings.joystick.right_sensitivity` before emission.

**Never compute raw pixel deltas.** The signal stays in normalized `[0, 1]` space
all the way to `MacInputEmitter`, which scales to OS units there.

## One-Euro filter

`joystick/one_euro.py` — velocity-adaptive low-pass filter.

- When the hand moves slowly: high filtering (cuts jitter).
- When the hand moves fast: low filtering (preserves responsiveness).
- Config key: `joystick.look_filter: one_euro` (alternatives: `none`, `ema`).
- Filter parameters in `config.yaml`: `one_euro_min_cutoff`, `one_euro_beta`.
- **Do not substitute a rolling average** — that adds fixed lag proportional to
  window size, which makes fast flicks feel mushy.

## Peace-sign recenter (mouse-lifted clutch)

The right-hand peace sign (`extension_combo`) acts as a mouse-lift clutch:

1. On **engage**: freeze all mouse output; record `current_index_mcp` as the new
   `_right_cursor_prev` (seeds the anchor without emitting movement).
2. While **held**: suppress `attack`, `use`, `jump`, `swap_offhand`; emit zero delta.
3. On **release**: look resumes from the new anchor — **no camera snap**.

This lets the user reposition their hand without the camera jerking. It is the
primary mechanism for dealing with the bounded physical range of hand motion.

## Cursor re-seed on tracking dropout

`TrackingLossGuard` + `dropout_flush_ms`:
- At dropout: zero the joystick delta immediately.
- After `dropout_flush_ms` ms: hard-flush — call `ScreenJoystick.reseed(current_pos)`
  so re-entry doesn't snap the camera to the hand's new position.

## Sensitivity tuning

All sensitivity values live in `config.yaml` under `joystick:`. No literals in code.
The sensitivity knob is a multiplier on the normalized delta; keep it unitless so
it stays valid across different screen resolutions.

## WristTiltJoystick (HUD only)

`joystick/wrist_tilt.py` produces a tilt vector from the wrist→index-MCP angle.
This feeds the **HUD debug overlay only** — it does **not** press WASD or move the
mouse. Do not wire it to any input action.

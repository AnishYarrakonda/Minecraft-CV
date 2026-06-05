# Rules: Input Layer

Read this when working on `input/`, synthetic key/mouse emission, or anything that
touches `InputEmitter`, `MacInputEmitter`, pynput, or Quartz CGEvent.

## Interface

`input/emitter.py` — `InputEmitter` ABC. **All input goes through this interface.**

```
InputEmitter
├── NullEmitter   — no-ops everything; used in tests and --no-input mode
└── MacInputEmitter — real pynput + Quartz implementation
```

Never bypass the interface. Never call pynput or Quartz directly from pipeline code.

## Two libraries, two jobs

| Library | What it handles | Why |
|---------|-----------------|-----|
| `pynput` | Keyboard key down/up (W/A/S/D, Space, Shift, E, Q, F) and scroll wheel | Clean cross-platform key event API |
| `pyobjc-framework-Quartz` (CGEvent) | Relative mouse deltas for camera look | Lower latency than pynput mouse; sends true relative motion that Minecraft reads |

**Do not use pynput for mouse look.** It goes through the macOS pointer acceleration
stack and introduces variable lag. CGEvent relative deltas bypass that.

## Building the emitter

`MacInputEmitter` **must be constructed on the GUI main thread** (or any non-Qt-worker
thread). Building it on a QThread worker causes a SIGTRAP on first use because pynput
internally touches Cocoa APIs that require the main run loop.

See: `project-golive-sigtrap-root-cause` memory — this was a real incident.

## Permissions

Both pynput and CGEvent require **Accessibility / Input Monitoring** granted to the
*terminal app* (or app bundle) running Python in:

> System Settings → Privacy & Security → Accessibility / Input Monitoring

Without the grant, events are **silently dropped** — no error, no exception. Detect at
startup by attempting a test CGEvent and checking the return code. If it fails, print
an actionable error pointing at the exact pane.

## Emitting mouse look

Emit **small, frequent deltas** — one per pipeline frame. Never accumulate and send a
large batch; Minecraft's camera integrates each delta immediately, so large deltas
produce visible jerks.

Do not correct for macOS pointer acceleration at the injection level. The CGEvent
relative motion API bypasses acceleration; correcting for it double-counts.

## Keyboard hold vs tap

- `hold` gestures: emit KEY_DOWN on engage, KEY_UP on release. The key stays held
  for the entire duration of the pinch. WASD works this way.
- `pulse` gestures (face actions): emit KEY_DOWN then KEY_UP in the same frame.
  Inventory (`E`) and throw (`Q`) work this way.

## Safety: never emit in tests

`NullEmitter` is the default. Tests **must not** construct a `MacInputEmitter`. The
`--no-input` flag ensures `NullEmitter` is used in dev runs. A test that accidentally
emits real input will move the mouse or press keys on the CI machine.

## Scroll wheel (hotbar)

Hotbar next/prev uses pynput scroll. One scroll tick per `scroll_repeat_rate_hz`
interval while the head-roll trigger is held. The scroll direction maps:

```
roll > +engage_deg  →  scroll up   (hotbar_next)
roll < -engage_deg  →  scroll down (hotbar_prev)
```

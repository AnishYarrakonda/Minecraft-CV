# minecraft_cv

Real-time webcam gesture -> Minecraft keyboard/mouse controller. A Python pipeline reads
hand pose from a camera, maps two-hand spatial translation + pinch-bitmask gestures to game
input, and emits OS-level events via pynput / Quartz CGEvent.

See [`.claude/CLAUDE.md`](.claude/CLAUDE.md) and [`.claude/rules/`](.claude/rules/) for the
authoritative design contract (gesture map, Schmitt-trigger invariants, perf rules).

## Setup

The `.venv` is already initialized. If you need to reinstall:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Commands

**Use with the venv:**

```bash
# Dry run with camera and debug overlay
.venv/bin/python -m minecraft_cv.cli --config config.yaml --no-input --debug-overlay

# Live mode (emits real OS input to Minecraft)
.venv/bin/python -m minecraft_cv.cli --config config.yaml --input --debug-overlay

# Calibrate spatial joystick (guided wizard)
.venv/bin/python -m minecraft_cv.calibration --config config.yaml --apply

# Offline analysis on a recorded clip
.venv/bin/python -m minecraft_cv.cli --config config.yaml --clip data/clips/foo.mp4 --debug-overlay

# Run tests
.venv/bin/python -m pytest                                 # all tests
.venv/bin/python -m pytest -k schmitt -x                  # fast gesture regression

# Full CI gate
.venv/bin/python -m pytest && .venv/bin/ruff check src tests && .venv/bin/mypy src
```

**Or with Poetry (if installed):**

```bash
poetry install
poetry run mcv-run --config config.yaml --no-input --debug-overlay
```

## Safety invariants

1. `T_release > T_engage` strictly for every pinch gesture (asserted in tests).
2. The input emitter is a **no-op by default** (`NullEmitter`). Tests/dry-runs never move
   the real mouse or press real keys.
3. CPU fallback always works; MPS is an accelerator, never a hard dependency.
4. Tracking loss releases every held key (no stuck `Space`/`Shift`).

## Status

MVP: wrist-anchor spatial joysticks (LH -> WASD, RH -> mouse look), index/middle pinch
(Jump, Sneak, Attack, Use), Schmitt triggers, NullEmitter default. Inventory mode switch,
ring/pinky hotbar gestures, dynamic deadzones, and sprint-via-velocity are V2.

# CLAUDE.md — minecraft_cv

Real-time webcam gesture → Minecraft keyboard/mouse controller. A Python pipeline
reads hand pose from a camera, maps two-hand spatial translation + pinch-bitmask
gestures to game input, and emits OS-level events via pynput / Quartz CGEvent.

## Quick commands

```bash
# Setup (if needed; .venv already exists)
source .venv/bin/activate

# Desktop app (recommended — Go Live toggle, Calibrate button, live HUD)
.venv/bin/python -m minecraft_cv.cli ui

# Compact always-on-top overlay (for use alongside Minecraft)
.venv/bin/python -m minecraft_cv.cli overlay --live

# Dry run with camera and debug overlay (headless, no UI)
.venv/bin/python -m minecraft_cv.cli run --no-input --debug-overlay

# Live mode headless (emits real OS input)
.venv/bin/python -m minecraft_cv.cli run --input --debug-overlay

# Offline clip analysis
.venv/bin/python -m minecraft_cv.cli analyze data/clips/foo.mp4

# Fast regression test
.venv/bin/python -m pytest -k schmitt -x

# Full CI gate
.venv/bin/python -m pytest && .venv/bin/ruff check src tests && .venv/bin/mypy src
```

**For development, always use one of these patterns:**
- `mcv ui` or `--no-input --debug-overlay` with live camera to iterate safely (no real input emitted)
- `--clip data/clips/foo.mp4 --no-input --debug-overlay` for deterministic testing against recorded footage

Never debug gesture logic against live Minecraft input; it's non-deterministic and will fling your character.

## Hard invariants

1. `T_release > T_engage` strictly for every pinch. A test asserts this for all gestures.
2. Input emitter is a no-op by default. Tests/dry-runs must never move the real mouse.
3. CPU fallback always works. MPS is an accelerator, never a hard dependency.
4. `Attack+Use` and `Jump+Sneak` are intentionally mutually exclusive. Don't fix them.

## Rules (loaded into context)

@.claude/rules/tech-stack.md
@.claude/rules/gestures.md
@.claude/rules/opencv-pytorch.md

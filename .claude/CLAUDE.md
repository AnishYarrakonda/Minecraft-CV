# CLAUDE.md — minecraft_cv

Real-time webcam gesture → Minecraft keyboard/mouse controller. A Python pipeline
reads hand pose from a camera, maps two-hand spatial translation + pinch-bitmask
gestures to game input, and emits OS-level events via pynput / Quartz CGEvent.

## Quick commands

```bash
poetry install                                                   # set up env
poetry run mcv-run --config config.yaml --no-input --debug-overlay  # dry run
poetry run mcv-run --config config.yaml                         # live (emits input)
poetry run mcv-calibrate                                         # tune thresholds
poetry run mcv-analyze data/clips/foo.mp4 --backend mediapipe   # offline debug
poetry run mcv-bench --backend mediapipe --device mps --frames 2000
poetry run pytest -k schmitt -x                                  # fast regression
poetry run pytest && ruff check src tests && mypy src            # full CI gate
```

**Always use `--no-input` + a recorded clip when iterating.** Never debug gesture
logic against live Minecraft — it's non-deterministic and will fling your character.

## Hard invariants

1. `T_release > T_engage` strictly for every pinch. A test asserts this for all gestures.
2. Input emitter is a no-op by default. Tests/dry-runs must never move the real mouse.
3. CPU fallback always works. MPS is an accelerator, never a hard dependency.
4. `Attack+Use` and `Jump+Sneak` are intentionally mutually exclusive. Don't fix them.

## Rules (loaded into context)

@.claude/rules/tech-stack.md
@.claude/rules/gestures.md
@.claude/rules/opencv-pytorch.md

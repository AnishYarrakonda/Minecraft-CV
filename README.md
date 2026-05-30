# minecraft_cv

Real-time webcam gesture -> Minecraft keyboard/mouse controller. A Python pipeline reads
hand pose from a camera, maps two-hand spatial translation + pinch-bitmask gestures to game
input, and emits OS-level events via pynput / Quartz CGEvent.

See [`.claude/CLAUDE.md`](.claude/CLAUDE.md) and [`.claude/rules/`](.claude/rules/) for the
authoritative design contract (gesture map, Schmitt-trigger invariants, perf rules).

## Setup

```bash
# Poetry (preferred once installed):
poetry install

# Or plain venv + pip (works with the PEP 621 pyproject):
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Commands

```bash
mcv-run --config config.yaml --no-input --debug-overlay   # dry run (no OS input)
mcv-run --config config.yaml --input                      # live (emits input)
mcv-calibrate --config config.yaml                        # tune thresholds
mcv-analyze data/clips/foo.mp4 --backend mediapipe        # offline debug
mcv-bench --backend mediapipe --device mps --frames 2000  # benchmark

pytest -k schmitt -x                                       # fast regression
pytest && ruff check src tests && mypy src                 # full CI gate
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

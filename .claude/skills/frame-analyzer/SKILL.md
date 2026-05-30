---
name: frame-analyzer
description: >-
  Run deterministic, offline analysis of a recorded clip or single image through the
  hand-tracking pipeline. Use when debugging gesture jitter, false pinch triggers,
  Schmitt-trigger thresholds, dropped frames, MPS-vs-CPU divergence, FPS/latency
  regressions, or "why did Attack fire when I didn't pinch". Produces per-frame
  landmark distances, detected pinch state transitions, and timing stats — no live
  camera and no OS input required.
allowed-tools:
  - Bash
  - Read
  - Edit
---

# Frame Analyzer

A debugging skill for the `minecraft_cv` real-time pipeline. It runs the **full
tracking + gesture stack on saved frames** so behavior is reproducible and CI-able —
no webcam, no Minecraft, no synthetic input. This is the first tool to reach for when
a gesture misfires.

## When to use this

- A pinch fires when the user didn't pinch (or fails to fire when they did).
- Suspected Schmitt-trigger problem: chattering KEY_DOWN/KEY_UP near a threshold.
- Landmark jitter / tracking dropout on specific hand poses or lighting.
- FPS or latency regression — locate which stage blew the frame budget.
- MPS vs CPU backend producing different gesture decisions.
- Validating new `T_engage` / `T_release` values against a known-tricky clip.

## How to run it

The helper script is `frame_analyzer.py` (next to this file). Invoke it via the
project interpreter so it imports `minecraft_cv`:

```bash
# Analyze a clip with the default (MediaPipe / auto-device) backend
poetry run python .claude/skills/frame-analyzer/frame_analyzer.py data/clips/pinch_jitter.mp4

# Focus on one hand + one gesture, dump every state transition
poetry run python .claude/skills/frame-analyzer/frame_analyzer.py \
    data/clips/pinch_jitter.mp4 --hand right --gesture attack --transitions

# Sweep candidate thresholds to find a hysteresis band with zero chatter
poetry run python .claude/skills/frame-analyzer/frame_analyzer.py \
    data/clips/pinch_jitter.mp4 --gesture attack \
    --engage 0.30 --release 0.45 --transitions

# Compare backends / devices on the same clip (timing + decision divergence)
poetry run python .claude/skills/frame-analyzer/frame_analyzer.py \
    data/clips/move_look.mp4 --backend mediapipe --device mps --timing
poetry run python .claude/skills/frame-analyzer/frame_analyzer.py \
    data/clips/move_look.mp4 --backend mediapipe --device cpu --timing

# A single still image instead of a clip
poetry run python .claude/skills/frame-analyzer/frame_analyzer.py data/frames/fist.png
```

## Output

- **Per-frame table** (or `--transitions` for state-change rows only): frame index,
  normalized thumb→fingertip distances, and the resolved pinch state per gesture.
- **Chatter report:** counts rapid RELEASED→HOLDING→RELEASED flips inside a short
  window — the signature of a Schmitt band that's too narrow for the clip's jitter.
- **Timing summary** (`--timing`): per-stage p50/p95/p99 and effective FPS, MPS-synced.

## Notes for Claude

- This script is **read-only w.r.t. the OS** — it never emits keyboard/mouse input.
  Safe to run freely while debugging.
- If `minecraft_cv` isn't importable yet (early scaffold), the script falls back to a
  self-contained reference implementation of the Schmitt-trigger + normalized-distance
  math so threshold sweeps still work. Treat that path as a design reference for the
  real `src/minecraft_cv/gestures/` module.
- When a chatter problem is found, the fix is almost always **widen the hysteresis band
  (`T_release` − `T_engage`)** or **add One-Euro smoothing upstream** — not lower the
  engage threshold. See `@.claude/rules/opencv-pytorch.md` §3.

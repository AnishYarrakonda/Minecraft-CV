---
name: debug-on-recorded-clips
description: Reproduce gesture/tracking issues on saved clips with --no-input before ever touching live camera + live Minecraft.
metadata:
  type: feedback
---

Preferred debugging workflow for this project: reproduce on a **recorded clip** with
the input emitter disabled (`--no-input`) before going live.

**Why:** A live webcam is non-deterministic and un-CI-able, and live OS-input emission
during debugging literally moves the character / flings the camera, corrupting the very
thing you're observing (and occasionally walking the player off a cliff). Saved clips
make tracking + gesture behavior reproducible and let regression tests exist at all.

**How to apply:** Use the `frame-analyzer` skill / `mcv-analyze` on a clip to inspect
per-frame distances and Schmitt transitions. Keep `--no-input` until the pipeline is
verified; only then enable real input with `--input`. Capture a new clip for every
gesture/tracking bug so it becomes a permanent test case. The OS-input emitter must be
a no-op by default so this stays safe. See [[schmitt-trigger-chatter-fix]].

---
name: project-overview
description: What minecraft_cv is, the gesture model, and its non-negotiable invariants.
metadata:
  type: project
---

minecraft_cv is a webcam → Minecraft input controller. Two hands act as spatial
joysticks (LH=WASD, RH=camera/mouse look) gated by a deadzone sphere; thumb-to-finger
pinches are discrete buttons via a pinch-bitmask. LH: index=Jump, middle=Sneak,
fist=Inventory mode-switch. RH: index=Attack, middle=Use, ring=Hotbar+, pinky=Hotbar−.

Priorities, in order: low-latency deterministic output → simultaneous-action concurrency
→ accessibility/ergonomics.

Hard invariants (do not violate):
1. Every pinch uses a Schmitt trigger with `T_release > T_engage` strictly — see
   [[schmitt-trigger-chatter-fix]].
2. OS input emitter is a no-op unless explicitly enabled; tests/dry-runs never emit —
   see [[debug-on-recorded-clips]].
3. CPU fallback always works; MPS is an accelerator only — see [[pytorch-mps-setup]].
4. Attack+Use and Jump+Sneak are intentionally mutually exclusive.

Stack: Python 3.11 + Poetry, MediaPipe Hands (PyTorch/YOLO backend optional behind a
`HandTracker` ABC), OpenCV, MPS on Apple Silicon, pynput + Quartz for input.

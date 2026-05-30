---
name: continuity-camera-wrong-device
description: cv2.VideoCapture(0) silently grabbed a nearby iPhone via Continuity Camera; pin the camera index in config.
metadata:
  type: project
---

Tracking suddenly broke for no code reason — the feed showed a weird angle/zoom.
`cv2.VideoCapture(0)` had silently selected a nearby **iPhone** as device 0 via macOS
Continuity Camera instead of the mounted webcam.

Root cause: macOS reorders/inserts camera devices; index 0 is not stable. Continuity
Camera can preempt the intended webcam whenever an iPhone is nearby and unlocked.

Fix: enumerate available cameras at startup, let the user pin an explicit camera index
in `config.yaml`, and force `cv2.CAP_AVFOUNDATION`. Log which device was actually
opened so a surprise is visible immediately.

**Why:** Silent device substitution looks identical to a tracking-model regression and
wastes debugging time — same silent-failure family as [[macos-camera-permission-black-frames]].

**How to apply:** Never trust hardcoded index 0. Pin the camera in config, verify the
opened device on startup, and surface a clear log line naming it.

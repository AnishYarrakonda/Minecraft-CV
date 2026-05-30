---
name: macos-camera-permission-black-frames
description: All-black/uniform frames on macOS mean a missing Camera grant, not a code bug — no exception is raised.
metadata:
  type: project
---

Symptom seen early in MVP: `cv2.VideoCapture(0)` opened successfully (`isOpened()`
true) and `read()` returned `ok=True`, but every frame was uniformly black. Hand
detection silently found nothing. Hours lost suspecting the model.

Root cause: macOS **Camera** privacy permission was not granted to the terminal app
running Python (Terminal/iTerm/VS Code). macOS yields black frames instead of raising.
The grant attaches to the *launching binary*, and had to be re-granted after the
terminal app updated and after switching the Python interpreter (Poetry venv).

**Why:** macOS Privacy & Security failures are silent at the OpenCV layer — no
exception, just empty pixels. This same class of silent failure also hits **Screen
Recording** (mss → wallpaper-only) and **Accessibility/Input Monitoring** (input
no-ops). See [[continuity-camera-wrong-device]] for the related wrong-device trap.

**How to apply:** On startup, grab one frame and assert it isn't all-black/uniform;
if it is, raise a clear error naming the exact pane (Privacy & Security → Camera) and
the app to grant. Never debug tracking logic until a known-good frame is confirmed.
Toggling the permission off/on forces macOS to re-prompt.

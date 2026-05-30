---
name: stale-frame-buffer-lag
description: Input lag traced to VideoCapture internal buffering; fixed with BUFFERSIZE=1 and latest-frame-wins.
metadata:
  type: project
---

Camera ran at a healthy FPS but the controller felt laggy — actions registered a
fraction of a second after the gesture. Profiling showed inference wasn't the
bottleneck.

Root cause: `cv2.VideoCapture` buffers frames internally, and the processing loop was
also feeding a growing queue. A naive `read()` handed back an *old* frame, and the
queue backlog added latency that grew over time.

Fix: `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)` plus a dedicated capture thread writing into
a **single-slot** buffer (newest frame wins, stale frames dropped). For a controller, a
*dropped old* frame beats a *late fresh* action.

**Why:** Tail latency, not mean FPS, determines how the controller feels. A backing-up
queue silently trades latency for completeness — exactly the wrong trade here. Pairs
with the main-thread render rule in [[opencv-window-main-thread]].

**How to apply:** Decouple capture from processing; always grab the latest frame; never
let a queue back up. Also force the AVFoundation backend and set FPS/resolution
explicitly (defaults are often low-FPS high-res).

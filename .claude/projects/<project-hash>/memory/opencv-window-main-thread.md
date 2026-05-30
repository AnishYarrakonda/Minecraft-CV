---
name: opencv-window-main-thread
description: OpenCV HighGUI (imshow/waitKey) must run on the main thread on macOS or windows freeze/crash.
metadata:
  type: project
---

When we moved capture + inference onto a worker thread for throughput and left the
debug `cv2.imshow` call inside that worker, the preview window rendered blank, beach-
balled, and occasionally hard-crashed the process on macOS.

Root cause: macOS Cocoa requires all UI/event-loop work on the **main thread**.
OpenCV HighGUI (`namedWindow`, `imshow`, `waitKey`) is UI work. Calling it from a
worker thread is undefined behavior on macOS.

Fix / architecture: worker thread(s) do capture + inference + gesture logic and push
the annotated frame into a single-slot buffer; the **main thread** owns the render
loop (`imshow` + `cv2.waitKey(1)`). `waitKey(1)` is mandatory to pump the GUI event
loop — without it the window never paints.

**Why:** This is a hard macOS platform constraint, not a tuning choice — getting it
wrong looks like a flaky GPU/driver bug but is purely a threading issue. Related to the
capture-threading design in [[stale-frame-buffer-lag]].

**How to apply:** Never call any `cv2` window/`waitKey` function off the main thread.
In production prefer headless (`opencv-python-headless`, debug overlay off) and skip
windowing entirely to save the milliseconds. Gate all overlay drawing behind the debug
flag.

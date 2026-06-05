# Rules: UI & Qt (PySide6)

Read this when working on `ui/`, `cli.py` UI/overlay subcommands, or anything that
touches PySide6 widgets, the pipeline ↔ Qt bridge, or window management.

## Threading model (critical)

```
Main thread        — Qt event loop, all widget creation/updates, cv2 HighGUI
PipelineWorker     — QThread; runs camera capture + inference + gesture logic
                     communicates back via Qt signals (never direct widget access)
```

**Never touch a Qt widget from `PipelineWorker`.** Use signals:
- `frame_ready(np.ndarray)` → `CameraView` updates the preview
- `gesture_state(dict)` → `KeymapPanel` lights up key indicators
- `pipeline_error(str)` → `MainWindow` shows error dialog

**Never run `cv2.imshow` / `cv2.waitKey` from PipelineWorker.** HighGUI must stay
on the main thread (Cocoa constraint — same rule as headless mode).

## Entry points

| Command | Function | File |
|---------|----------|------|
| `mcv ui` | `run_app()` | `ui/app.py` |
| `mcv overlay --live` | `run_overlay()` | `ui/overlay.py` |

Both build `MacInputEmitter` on the **main thread before** starting `PipelineWorker`.
This is mandatory — see `input-layer.md` for why.

## PipelineWorker (`ui/worker.py`)

- Subclasses `QThread`; override `run()`.
- On `go_live` signal: constructs and starts the pipeline, then enters the frame loop.
- On `stop` signal: calls `pipeline.shutdown()` (releases held keys) then exits.
- `shutdown()` is called on the worker thread; Qt cleanup happens after `finished` signal.

## MainWindow (`ui/app.py`)

- `HeaderBar` contains **Go Live** toggle and **Calibrate** button (`ui/panels.py`).
- `CameraView` renders the annotated frame as a `QLabel` pixmap (`ui/camera_view.py`).
- `KeymapPanel` shows per-binding key indicators that light up with gesture state
  (`ui/panels.py`, `ui/keymap.py`).
- `SkeletonOverlay` draws MediaPipe hand landmarks on top of `CameraView`
  (`ui/skeleton.py`).

## Overlay (`ui/overlay.py`)

- Compact always-on-top window for use alongside Minecraft.
- Only shows key state HUD — no camera feed (saves GPU/CPU).
- Uses `keep_window_in_front()` from `ui/macos_window.py` for native NSWindow pinning
  (stays above fullscreen games via `NSFloatingWindowLevel` + `canJoinAllSpaces`).

## Window pinning (`ui/macos_window.py`)

`keep_window_in_front(win: QMainWindow)` — call once after `show()`:
- Sets `NSWindowLevel` to `NSFloatingWindowLevel` (above normal apps).
- Sets `NSWindowCollectionBehavior` to `canJoinAllSpaces` (visible on all Spaces).
- Prevents App Nap (`setInhibitsSystemSleep_` / background activity token).

Only call this for the overlay, not the main debug window.

## Theme (`ui/theme.py`)

`apply_theme(app: QApplication)` — zinc dark-mode palette. Call once at startup
before any windows are shown. Don't set per-widget stylesheets that fight the theme.

## Widgets (`ui/widgets.py`)

Shared reusable primitives. Add new reusable widgets here; keep app-specific logic
in the files that own the concept (panels, keymap, etc.).

## Do not mix Qt and HighGUI

If the Qt UI is running, do **not** also open a `cv2.imshow` window. Use
`CameraView` for the frame preview. `--debug-overlay` / HighGUI is for the
headless (`mcv run`) mode only.

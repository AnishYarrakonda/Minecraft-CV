# Rules: UI & Qt (PySide6)

Read this when working on `ui/`, the `cli.py` `ui` subcommand, or anything that
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

`mcv ui` builds `MacInputEmitter` on the **main thread before** starting `PipelineWorker`.
This is mandatory — see `input-layer.md` for why. (The old standalone `mcv overlay` window was
removed; the main window is now pinnable + collapsible and serves as the in-game HUD.)

## PipelineWorker (`ui/worker.py`)

- Subclasses `QThread`; override `run()`.
- On `go_live` signal: constructs and starts the pipeline, then enters the frame loop.
- On `stop` signal: calls `pipeline.shutdown()` (releases held keys) then exits.
- `shutdown()` is called on the worker thread; Qt cleanup happens after `finished` signal.

## MainWindow (`ui/app.py`)

- **Vertical layout (top → bottom):** `CameraView`, then `KeymapPanel` (the compact key-cap
  grid grouped **MOVE / COMBAT / FACE**), then `HeaderBar` as a bottom control bar. Narrow +
  tall by default to save horizontal space alongside Minecraft.
- `CameraView` is **aspect-locked** (`heightForWidth`): its height follows its width, so the
  feed fills the widget with no black letterbox bands and a vertical resize never shrinks it
  (`ui/camera_view.py`). Annotated frame painted as a pixmap with a glowing skeleton.
- `KeymapPanel` fills the space under the camera; its key-caps light up with gesture state via
  `update_state()` (`ui/panels.py`, `ui/keymap.py`), wrap via `FlowLayout`, and the
  mouse-sensitivity slider sits below the grid. It scrolls when squeezed.
- `HeaderBar` (named for history; now the **bottom** bar) holds the status pill, L/R/F chips,
  and **Start / Go Live / Calibrate / Pin**, laid out with `FlowLayout` so they wrap to extra
  rows instead of squishing in a narrow window.
- **Pin** floats the window over fullscreen Minecraft (`keep_window_in_front`); **unpin** calls
  `reset_window_level`. Shrinking the window below `_COLLAPSE_HEIGHT` hides the key grid
  (camera + control bar only) via `resizeEvent`.
- `SkeletonOverlay` draws MediaPipe hand landmarks on top of `CameraView` (`ui/skeleton.py`).

## Window pinning (`ui/macos_window.py`)

`keep_window_in_front(win)` — pins the native `NSWindow` above other apps and fullscreen Spaces
(`NSStatusWindowLevel` + `canJoinAllSpaces | FullScreenAuxiliary | Stationary`, no hide-on-
deactivate). Call after the window is realized (e.g. from `_on_pin`). `reset_window_level(win)`
undoes it (normal level + default behavior) on unpin. Both are macOS-only no-ops elsewhere.

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

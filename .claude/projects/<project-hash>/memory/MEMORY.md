# Memory — minecraft_cv

Index of persistent project memories. One line per memory; full content lives in the
linked file. Newest/most-important first.

## Project
- [project-overview](project-overview.md) — what minecraft_cv is and its hard invariants.

## Solved issues / hard-won lessons
- [macos-camera-permission-black-frames](macos-camera-permission-black-frames.md) — black frames = missing Camera grant for the terminal app, no exception thrown.
- [pytorch-mps-setup](pytorch-mps-setup.md) — MPS device selection + `PYTORCH_ENABLE_MPS_FALLBACK=1`, CPU fallback mandatory.
- [opencv-window-main-thread](opencv-window-main-thread.md) — HighGUI must run on the main thread or windows freeze/crash on macOS.
- [schmitt-trigger-chatter-fix](schmitt-trigger-chatter-fix.md) — pinch chatter fixed by widening the hysteresis band, not lowering engage.
- [stale-frame-buffer-lag](stale-frame-buffer-lag.md) — input lag traced to VideoCapture buffering; `BUFFERSIZE=1` + latest-frame-wins.
- [mps-benchmark-sync](mps-benchmark-sync.md) — async MPS made benchmarks lie; must `torch.mps.synchronize()` before timing.
- [continuity-camera-wrong-device](continuity-camera-wrong-device.md) — index 0 silently grabbed an iPhone; pin camera index in config.

## Feedback / working preferences
- [debug-on-recorded-clips](debug-on-recorded-clips.md) — always reproduce on saved clips with `--no-input` before live camera/Minecraft.

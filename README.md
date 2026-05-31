# minecraft_cv

Real-time webcam gesture -> Minecraft keyboard/mouse controller. A Python pipeline reads hand pose from a camera, maps two-hand spatial translation + pinch-bitmask gestures to game input, and emits OS-level events via pynput / Quartz CGEvent.

**Note:** This project currently only supports **macOS** because it relies on AVFoundation and Quartz CGEvent.

## The App

`minecraft_cv` ships with a polished desktop app (PySide6). The camera feed is framed as a clean
"painting" with a subtle glowing hand skeleton — no clutter drawn over your hands — while a
dark-glass sidebar shows every key mapping lighting up live as you gesture, your WASD / look
state, per-hand tracking health, and Start / Go-Live / Calibrate controls. It opens in safe
**Dry-Run** (no real input) by default.

```bash
pip install -e .      # pulls in PySide6 and the rest
python main.py        # opens the app
```

<!-- TODO: add docs/demo.gif here — a short clip of the app in action -->
<!-- ![minecraft_cv desktop app](docs/demo.gif) -->

Prefer the headless / OpenCV-overlay controller? See `mcv run` under **Quick Start** below.

## Prerequisites
- macOS 13+
- Python 3.11+
- **Camera Permission:** Must be granted to your terminal to read camera frames.
- **Accessibility Permission:** Must be granted to your terminal (System Settings → Privacy & Security → Accessibility) to emit keyboard and mouse events.

## Setup

```bash
git clone <repository_url> && cd minecraft_cv
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

Launch the desktop app (recommended):
```bash
.venv/bin/python main.py
# or:
.venv/bin/python -m minecraft_cv.cli ui
```
The app opens in Dry-Run. Click **Go Live** to emit real input (you'll be prompted to grant
Accessibility if needed), **Calibrate** to recenter the joysticks at your hand's rest pose, and
**Pin** to keep the window above Minecraft.

Check if your camera and permissions are set up correctly:
```bash
.venv/bin/python -m minecraft_cv.cli doctor
```

Run in dry-run mode (does not emit OS input, shows camera overlay):
```bash
.venv/bin/python -m minecraft_cv.cli run --no-input --debug-overlay
```

Run in live mode with the camera overlay while playing:
```bash
.venv/bin/python -m minecraft_cv.cli run --input --debug-overlay
```

## Gesture & Control Reference

Here is how you control the game using gestures with your hands:

### 🎮 Screen Joysticks

*   **Movement (WASD) - Left Hand**: Move your left hand away from its neutral anchor to hold `W/A/S/D`.
*   **Camera Look / Cursor - Right Hand**: The right thumb tip is treated like the mouse. Each frame's thumb movement becomes the mouse movement directly, with no right-hand deadzone or velocity hold. Tune the gain with `joystick.right_sensitivity` in `config.yaml`; the project default is `40.0`.
*   **Relocalize / mouse-lift clutch**: Hold a peace sign (**Index + Middle** extended, **Ring + Pinky** curled). While held, that hand resets the thumb cursor point and the right hand sends no look movement.

---

### 🖐️ Left-Hand Actions
All actions are holds. A quick hold/release acts like a tap in Minecraft.

*   **Jump (`Space`)**: Pinch **Thumb** to **Index Finger**.
*   **Inventory (`E`)**: Pinch **Thumb** to **Middle Finger**.
*   **Throw Item (`Q`)**: Pinch **Thumb** to **Ring Finger**.
*   **Sneak (`Shift`)**: Pinch **Thumb** to **Pinky Finger**.
*   **Relocalize movement**: Peace sign (**Index + Middle** extended, **Ring + Pinky** curled).

---

### 🤏 Right-Hand Holds
Triggered by pinching your thumb to specific fingertips:
*   **Attack/Mine (`Left Click`)**: Pinch **Thumb** to **Index Finger**.
*   **Use/Place (`Right Click`)**: Pinch **Thumb** to **Middle Finger**.
*   **Hotbar Scroll Up (`Scroll Up`)**: Pinch **Thumb** to **Ring Finger**.
*   **Hotbar Scroll Down (`Scroll Down`)**: Pinch **Thumb** to **Pinky Finger**.
*   **Relocalize look / cursor clutch**: Peace sign (**Index + Middle** extended, **Ring + Pinky** curled). The right hand sends no look motion while this is held.

---

### 💼 Inventory
There is no separate inventory control mode. The left-hand thumb-to-middle pinch holds `E`, and right-hand look/cursor motion plus click holds continue to work while Minecraft's inventory UI is open.

## Commands

| Command | Description |
|---------|-------------|
| `mcv run` | Run the gesture controller (use `--input` for live mode, `--no-input` for testing). |
| `mcv calibrate` | Calibrate joysticks or pinch thresholds. |
| `mcv analyze` | Offline clip analysis. |
| `mcv bench` | Benchmark tracking backend latency. |
| `mcv doctor` | Check system permissions and camera. |
| `mcv gestures` | Print the gesture reference card. |

## Status

Fully implemented: screen-space movement joystick (left hand), thumb-tip look/cursor control (right hand), detector-backed hold gestures for both hands, pinch/curl-combo detectors with suppression, Schmitt triggers, hotbar gestures, relocalization, and smooth continuous mouse emission.

## Safety Invariants

1. `T_release > T_engage` strictly for every pinch gesture (asserted in tests).
2. The input emitter is a **no-op by default** (`NullEmitter`). Tests/dry-runs never move the real mouse or press real keys.
3. CPU fallback always works; MPS is an accelerator, never a hard dependency.
4. Tracking loss releases every held key (no stuck `Space`/`Shift`).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

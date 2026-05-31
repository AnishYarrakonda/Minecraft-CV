# minecraft_cv

Real-time webcam gesture -> Minecraft keyboard/mouse controller. A Python pipeline reads hand pose from a camera, maps two-hand spatial translation + pinch-bitmask gestures to game input, and emits OS-level events via pynput / Quartz CGEvent.

**Note:** This project currently only supports **macOS** because it relies on AVFoundation and Quartz CGEvent.

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

Easy launcher:
```bash
.venv/bin/python main.py
```
Press `Enter` for the default `Y` answers. The launcher can run real input, show the
camera overlay while playing, and do quick setup if palm-normal calibration is missing.

Check if your camera and permissions are set up correctly:
```bash
.venv/bin/python -m minecraft_cv.cli doctor
```

Run in dry-run mode (does not emit OS input, shows camera overlay):
```bash
.venv/bin/python -m minecraft_cv.cli run --no-input --debug-overlay
```

Quick one-pose setup before live gameplay:
```bash
.venv/bin/python -m minecraft_cv.cli calibrate --quick-neutral --apply
```

Full palm-normal calibration if you want tuned sensitivity:
```bash
.venv/bin/python -m minecraft_cv.cli calibrate --apply
```

Run in live mode with the camera overlay while playing:
```bash
.venv/bin/python -m minecraft_cv.cli run --input --debug-overlay
```

## Gesture & Control Reference

Here is how you control the game using gestures with your hands:

### 🎮 Calibrated Palm-Normal Thumbsticks
Run `.venv/bin/python -m minecraft_cv.cli calibrate --quick-neutral --apply` before gameplay for the easiest setup. Dry-run overlay can launch before calibration with temporary first-visible-hand neutrals, but `mcv run --input` refuses to start until neutral values exist in `config.yaml`.

*   **Movement (WASD) - Left Hand**: Palm-normal tilt drives movement from calibrated rest.
    *   **Forward (`W`)**: Tilt the palm normal down.
    *   **Back (`S`)**: Tilt the palm normal up.
    *   **Left (`A`)**: Tilt the palm normal left.
    *   **Right (`D`)**: Tilt the palm normal right.
*   **Camera Look (Mouse Move) - Right Hand**: The right palm normal uses the same `x/y` axes for look. Output is zero inside the calibrated deadzone and scales linearly as the palm normal tilts farther from neutral.

---

### 🖐️ Left-Hand Actions
All actions are holds. A quick hold/release acts like a tap in Minecraft.

*   **Jump (`Space`)**: Pinch **Thumb** to **Index Finger**.
*   **Inventory (`E`)**: Pinch **Thumb** to **Middle Finger**.
*   **Throw Item (`Q`)**: Pinch **Thumb** to **Ring Finger**.
*   **Switch Offhand (`F`)**: Pinch **Thumb** to **Pinky Finger**.
*   **Sneak (`Shift`)**: Curl **Ring + Pinky** on the left hand. This suppresses the left ring/pinky pinches while held, but index/middle pinches remain available.

---

### 🤏 Right-Hand Holds
Triggered by pinching your thumb to specific fingertips:
*   **Attack/Mine (`Left Click`)**: Pinch **Thumb** to **Index Finger**.
*   **Use/Place (`Right Click`)**: Pinch **Thumb** to **Middle Finger**.
*   **Hotbar Scroll Up (`Scroll Up`)**: Pinch **Thumb** to **Ring Finger**.
*   **Hotbar Scroll Down (`Scroll Down`)**: Pinch **Thumb** to **Pinky Finger**.
*   **Sprint (`Ctrl`)**: Curl **Ring + Pinky** on the right hand. This suppresses right ring/pinky hotbar pinches while held, but attack/use remain available.

---

### 💼 Special Modes
The older two-open-palms inventory cursor mode is disabled by default because open palms are now the neutral gameplay pose. It remains available in configuration for experiments, but the default inventory action is the left-hand thumb-to-middle pinch (`E` hold).

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

Fully implemented: calibrated palm-normal virtual thumbsticks (LH -> WASD, RH -> mouse look), detector-backed hold gestures for both hands, pinch/curl-combo detectors with suppression, Schmitt triggers, hotbar gestures, sprint-via-velocity, and optional legacy inventory cursor mode.

## Safety Invariants

1. `T_release > T_engage` strictly for every pinch gesture (asserted in tests).
2. The input emitter is a **no-op by default** (`NullEmitter`). Tests/dry-runs never move the real mouse or press real keys.
3. CPU fallback always works; MPS is an accelerator, never a hard dependency.
4. Tracking loss releases every held key (no stuck `Space`/`Shift`).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

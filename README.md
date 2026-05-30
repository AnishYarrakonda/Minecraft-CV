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

Check if your camera and permissions are set up correctly:
```bash
mcv doctor
```

Run in dry-run mode (does not emit OS input, shows camera overlay):
```bash
mcv run --no-input --debug-overlay
```

Run in live mode (controls your mouse and keyboard):
```bash
mcv run --input
```

## Gesture & Control Reference

Here is how you control the game using gestures with your hands:

### 🎮 Joysticks (Movement & Camera)
*   **Movement (WASD) - Left Hand**: Moving your hand relative to its initial center point (neutral pose) drives translation.
    *   **Forward (`W`)**: Move hand **up** in the camera frame.
    *   **Back (`S`)**: Move hand **down** in the camera frame.
    *   **Left (`A`)**: Move hand **left** in the camera frame.
    *   **Right (`D`)**: Move hand **right** in the camera frame.
*   **Camera Look (Mouse Move) - Right Hand**: Moving your hand relative to its initial center point drives the camera look direction.

---

### 🖐️ Left-Hand Extension Gestures (Discrete Actions)
Triggered by extending specific fingers from a closed fist:
*   **Jump (`Space`)**: Extend your **Thumb** outward (`thumb_out`).
*   **Sneak (`Shift`)**: Extend your **Index Finger** only (`index_only`).
*   **Sprint (`Ctrl`)**: Extend your **Middle Finger** only (`middle_only`).
*   **Inventory (`E`)**: Extend both your **Index and Middle Fingers** (Peace Sign, `index_middle`). *[Pulse/One-shot]*
*   **Throw Item (`Q`)**: Extend your **Ring Finger** only (`ring_only`). *[Pulse/One-shot]*
*   **Switch Offhand (`F`)**: Extend your **Pinky Finger** only (`pinky_only`). *[Pulse/One-shot]*

---

### 🤏 Right-Hand Pinch Gestures (Interactions)
Triggered by pinching your thumb to specific fingertips:
*   **Attack/Mine (`Left Click`)**: Pinch **Thumb** to **Index Finger**.
*   **Use/Place (`Right Click`)**: Pinch **Thumb** to **Middle Finger**.
*   **Hotbar Scroll Up (`Scroll Up`)**: Pinch **Thumb** to **Ring Finger**.
*   **Hotbar Scroll Down (`Scroll Down`)**: Pinch **Thumb** to **Pinky Finger**.

---

### 💼 Special Modes
*   **Inventory Mode Toggle**: Fully open **both hands** (spread all fingers) to toggle menu navigation mode. When active:
    *   WASD movement is paused.
    *   Moving your right hand drives the macOS mouse cursor in absolute screen coordinates.
    *   Right-hand index pinch acts as a Left Click (and held pinch acts as click-and-drag).
    *   Right-hand middle pinch acts as a Right Click.
    *   Fully open both hands again to return to normal gameplay control.

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

Fully implemented: wrist-anchor spatial joysticks (LH -> WASD, RH -> mouse look), extension gestures for left hand, pinch gestures for right hand, Schmitt triggers, dynamic deadzones, sprint-via-velocity, hotbar gestures, and inventory mode.

## Safety Invariants

1. `T_release > T_engage` strictly for every pinch gesture (asserted in tests).
2. The input emitter is a **no-op by default** (`NullEmitter`). Tests/dry-runs never move the real mouse or press real keys.
3. CPU fallback always works; MPS is an accelerator, never a hard dependency.
4. Tracking loss releases every held key (no stuck `Space`/`Shift`).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

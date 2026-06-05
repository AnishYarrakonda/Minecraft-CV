# minecraft_cv — Control Minecraft with hand gestures

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Platform: macOS 13+](https://img.shields.io/badge/platform-macOS%2013%2B-lightgrey)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

> Play vanilla Minecraft — no mods, no plugins. Just your webcam and your hands.

![minecraft_cv in action](docs/demo.gif)

Your webcam reads your hand pose in real time. Your hands walk the character, aim the camera, mine blocks, place items, jump, sneak, and scroll the hotbar — all at full speed using real OS-level input injection. No Minecraft mod required; it works with any version, Java or Bedrock.

**Genuinely playable.** Not a proof of concept — low latency, Schmitt-trigger hysteresis on every pinch, and a calibrated deadzone so your hands stay comfortable.

> **macOS only** — relies on AVFoundation for camera capture and Quartz CGEvent for input injection.

---

## The App

![minecraft_cv desktop app](docs/screenshot.png)

`minecraft_cv` ships with a polished desktop app. The camera feed shows your hand skeleton live against a clean dark background, while a sidebar lights up each key mapping as you gesture and displays WASD / look state per hand. Opens in **Dry-Run** by default — no real input is emitted until you click **Go Live**.

```bash
pip install -e .
python main.py
```

---

## How to control Minecraft

### Left hand — movement & actions

| Gesture | Action |
|---|---|
| Tilt hand forward / back / left / right | Walk (W / S / A / D) |
| Pinch thumb → index finger | Jump (Space) |
| Pinch thumb → middle finger | Inventory (E) |
| Pinch thumb → ring finger | Throw item (Q) |
| Pinch thumb → pinky finger | Sneak (Shift) |
| Peace sign (index + middle out, ring + pinky curled) | Pause movement / relocalize |

### Right hand — camera & combat

| Gesture | Action |
|---|---|
| Move hand | Aim camera / move cursor |
| Pinch thumb → index finger | Attack / mine (left click) |
| Pinch thumb → middle finger | Place / interact (right click) |
| Pinch thumb → ring finger | Hotbar next |
| Pinch thumb → pinky finger | Hotbar prev |
| Peace sign (index + middle out, ring + pinky curled) | Pause look / relocalize |

Both hands work at the same time — walk, jump, and mine simultaneously with no conflicts.

---

## Setup

**Prerequisites**
- macOS 13+
- Python 3.11+
- Camera and Accessibility permissions (the `doctor` command checks both)

**Install**

```bash
git clone https://github.com/AnishYarrakonda/Minecraft-CV && cd Minecraft-CV
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

**First run**

```bash
# Verify camera and permissions are ready
.venv/bin/python -m minecraft_cv.cli doctor

# Open the desktop app (recommended)
python main.py

# Or run headless with camera overlay (no real input, safe to try first)
.venv/bin/python -m minecraft_cv.cli run --no-input --debug-overlay
```

Click **Calibrate** in the app (or run `mcv calibrate --apply`) to store your hands' resting position as the movement neutral. Do this once from your normal playing posture.

---

## Commands

| Command | Description |
|---|---|
| `mcv run` | Run the gesture controller (`--input` for live, `--no-input` for dry-run) |
| `mcv calibrate` | Recenter joystick neutral at your hands' resting pose |
| `mcv doctor` | Check camera, permissions, and system health |
| `mcv analyze` | Analyze a recorded clip offline |
| `mcv bench` | Benchmark tracking backend latency |
| `mcv gestures` | Print the full gesture reference card |

---

<details>
<summary>How it works</summary>

Two independent signal streams run concurrently per camera frame:

**Spatial joystick** — MediaPipe estimates 21 hand landmarks per frame. The wrist-to-knuckle tilt vector is compared against a calibrated resting neutral. Deviation outside a sphere deadzone maps to WASD (left hand) or mouse-look deltas (right hand), with exponential acceleration and a One-Euro velocity-adaptive filter for stable-at-rest / snappy-in-motion feel.

**Pinch bitmask** — Thumb-to-fingertip distances are computed in one vectorized NumPy call and normalized by hand scale so thresholds never need recalibration just because you moved closer to the camera. Each pinch goes through a Schmitt-trigger hysteresis gate (`T_engage < T_release`) to prevent jitter at the threshold from causing rapid key up/down chatter.

Both paths run on CPU, processing at 30 fps on Apple Silicon with no GPU required.

</details>

---

## Safety

1. Input emitter is a **no-op by default** — dry-run and tests never move your real mouse or press real keys.
2. Tracking loss releases every held key immediately — no stuck jumps, no infinite sneaking.
3. CPU fallback always works; MPS acceleration is optional.
4. `T_release > T_engage` strictly for every pinch — asserted in tests.

---

## License

MIT — see [LICENSE](LICENSE).

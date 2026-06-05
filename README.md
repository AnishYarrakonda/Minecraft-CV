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
mcv ui          # or: python main.py
```

---

## How to control Minecraft

### Left hand — WASD movement

| Gesture | Action |
|---|---|
| Pinch thumb → index finger | Move right (D) |
| Pinch thumb → middle finger | Move forward (W) |
| Pinch thumb → ring finger | Move left (A) |
| Pinch thumb → pinky finger | Move back (S) |

Hold two pinches at once for diagonal movement (e.g. index + middle = forward-right).

### Right hand — camera & combat

| Gesture | Action |
|---|---|
| Move hand across frame | Aim camera (mouse look) |
| Pinch thumb → index finger | Attack / mine (left click) |
| Pinch thumb → middle finger | Place / interact (right click) |
| Pinch thumb → ring finger | Jump (Space) |
| Pinch thumb → pinky finger | Swap offhand (F) |
| Peace sign (index + middle extended, ring + pinky curled) | Relocalize camera neutral |

### Face — actions

| Gesture | Action |
|---|---|
| Raise eyebrows | Inventory (E) |
| Open mouth | Throw item (Q) |

### Head — actions

| Gesture | Action |
|---|---|
| Nod head down | Sneak (Shift, hold) |

| Gesture | Action |
|---|---|
| Roll head toward left shoulder | Hotbar next (scroll up) |
| Roll head toward right shoulder | Hotbar prev (scroll down) |

All inputs work at the same time — walk, jump, attack, and look simultaneously with no conflicts.

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
.venv/bin/python -m minecraft_cv.cli ui

# Or run headless with debug overlay (no real input, safe to try first)
.venv/bin/python -m minecraft_cv.cli run --no-input --debug-overlay
```

The app opens in **Dry-Run** mode — no real input is emitted until you click **Go Live**.

---

## Commands

| Command | Description |
|---|---|
| `mcv ui` | Launch the polished desktop app (recommended) |
| `mcv overlay` | Compact always-on-top overlay that stays in front of Minecraft (even fullscreen) |
| `mcv run` | Headless gesture controller (`--input` for live, `--no-input` for dry-run) |
| `mcv doctor` | Check camera, permissions, and system health |
| `mcv analyze` | Analyze a recorded clip offline |
| `mcv bench` | Benchmark tracking backend latency |
| `mcv gestures` | Print the full gesture reference card |

---

<details>
<summary>How it works</summary>

Three independent signal streams run concurrently per camera frame:

**Pinch bitmask (both hands)** — Thumb-to-fingertip distances are computed in one vectorized NumPy call and normalized by hand scale so thresholds never need recalibration just because you moved closer to the camera. Each pinch passes through a Schmitt-trigger hysteresis gate (`T_engage < T_release`) to prevent threshold jitter from causing rapid key chattering. Left-hand pinches drive WASD; right-hand pinches drive attack, use, jump, and swap-offhand.

**Cursor look (right hand)** — The index-finger MCP position on screen is tracked frame-to-frame. Deltas are passed through a One-Euro velocity-adaptive filter (stable at rest, snappy in motion) then sent as relative mouse moves via Quartz CGEvent. A peace-sign gesture acts as a "mouse lifted" clutch, resetting the cursor anchor without moving the camera.

**Face + head** — MediaPipe FaceLandmarker runs alongside hand tracking. Blendshape scores feed per-gesture Schmitt triggers for inventory (raise eyebrows) and throw (open mouth). The eye-corner angle drives a head-roll detector for hotbar scroll (tilt left = next, tilt right = prev), and a head-pitch ratio drives a nod-down detector for sneak.

All paths run on CPU at 30 fps on Apple Silicon with no GPU required.

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

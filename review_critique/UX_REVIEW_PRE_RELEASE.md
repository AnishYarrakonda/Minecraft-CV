# minecraft_cv — Pre-Release UX Review

*Senior product-design review of the brand-new-user journey (install → permit → launch → calibrate → play), grounded in `README.md`, `config.yaml`, and `src/minecraft_cv/cli.py`.*

## The core problem in one sentence

This is an engineering artifact, not a product. Every capability a player needs (install, permit, launch, calibrate, play) exists, but each is exposed as a **developer command with a flag**, and the failure modes that a first-timer will absolutely hit (black camera, wrong device, no Accessibility grant, untuned thresholds) are handled as *stderr strings*, not *guided recovery*. A new user has a ~0% chance of reaching "playing Minecraft" without reading the source.

---

## Walkthrough: what actually happens to a new user

**1. Install** — README says `.venv is already initialized` (true for the author, meaningless to a downloader). The real path is `pip install -e ".[dev]"` — which installs the **dev** toolchain (pytest, ruff, mypy) on a *player* who will never run a test. There's no `mcv` binary on PATH; everything is `.venv/bin/python -m minecraft_cv.cli`. README offers Poetry as an alternative, but Poetry isn't even installed — so the two halves of the doc disagree.

**2. Permissions** — The single hardest macOS footgun (Camera + Accessibility + Screen Recording, all attached to the *terminal binary*, all failing **silently**) is documented only in `.claude/rules/`, which a user never sees. The CLI has **no preflight check**. `cli.py:75` catches `PermissionError`, but Quartz/pynput don't raise it — they no-op. So "I launched live mode and nothing happens in Minecraft" has no diagnostic at all.

**3. Launch** — `main_run` prints one line (`cli.py:69`): `DRY-RUN (no input); overlay=False; source=camera:0`. If camera 0 is a Continuity iPhone (config.yaml:6 even admits this), the user gets a black window or their phone's view and no explanation. There is no device enumeration command.

**4. Calibrate** — This is the *best* part of the tool (the wizard at `cli.py:113` is genuinely good: countdown, per-pose instructions, preview-before-apply, validate-before-write). But: it's undiscoverable, the README command (`python -m minecraft_cv.calibration`) **doesn't match the CLI entrypoint** (`main_calibrate` / `mcv-calibrate`) — likely a broken invocation. And it only calibrates the **joystick**; pinch tuning is a separate hidden `--pinch` legacy mode (`cli.py:233`) that dumps raw floats to a `\r` line and expects the user to mentally pick thresholds.

**5. Play** — Even fully calibrated, the user must mentally hold a 12-gesture map (config.yaml:26–74) with no in-app reference card. The debug overlay exists (`--debug-overlay`) but defaults off, and from the code it draws landmarks/distances — engineer telemetry, not "you are holding W / Attack armed."

---

## Friction inventory (every point found)

| # | Area | Friction | Where |
|---|------|----------|-------|
| 1 | Install | No player install path; `.[dev]` forces test tooling; no `mcv` console script on PATH | README:14–18 |
| 2 | Install | README Poetry path is dead (not installed); two doc halves disagree | README:45–50 |
| 3 | Permissions | Zero preflight; 3 silent-failure grants documented only in internal rules | cli.py (absent) |
| 4 | Permissions | `--input` live mode no-ops with no error if Accessibility ungranted | cli.py:75 |
| 5 | Camera | No `--list-cameras`; Continuity Camera steals index 0; black frame not detected | config.yaml:6 |
| 6 | First-run | One-line status print; no "is this working?" signal | cli.py:69 |
| 7 | Calibrate | README invocation ≠ actual entrypoint (broken command) | README:32 vs cli.py:82 |
| 8 | Calibrate | Pinch tuning is hidden `--pinch` legacy float-dump; user picks thresholds by eye | cli.py:233 |
| 9 | Calibrate | Joystick wizard is camera-only & undiscoverable; no "calibrate everything" flow | cli.py:113 |
| 10 | Discoverability | No gesture cheat-sheet anywhere a player sees; 12 gestures in YAML comments | config.yaml:26 |
| 11 | Feedback | Per-frame state (which key is held, which pinch armed) invisible without reading code | overlay |
| 12 | Debugging | Errors are bare stderr strings; no remediation, no "open this Settings pane" | cli.py:76,140,176 |
| 13 | CLI | Mode flags (`--input`/`--no-input`) are subtle; safe default is good but easy to mis-set live | cli.py:42 |
| 14 | CLI | Four entrypoints (run/calibrate/analyze/bench) with no unifying `mcv` command or `--help` index | cli.py |
| 15 | Config | 146-line YAML mixes V1, V2, and "Task N" internal labels; user can't tell what's safe to touch | config.yaml |
| 16 | Config | Threshold semantics (`t_engage=0.30` normalized ratio) are unexplained to a non-engineer | config.yaml:59 |
| 17 | Config | Editing YAML by hand is the *only* way to remap keys or change sensitivity short of re-calibrating | config.yaml:123 |

---

## Prioritized UX Roadmap

Severity: 🔴 blocker / 🟠 major / 🟡 minor. Effort: S (<1d) / M (1–3d) / L (week+).

### P0 — A new user literally cannot succeed without these

**R1. Permission & camera preflight ("Doctor") — 🔴 / Effort: M**
*Impact:* Eliminates the #1 and #2 silent-failure classes (black frame, dead input). Without this, ~every first run fails mysteriously.
*Implementation:* Add `mcv doctor` (and run it automatically on first `mcv run`). On startup: (a) grab one frame, assert it isn't uniformly black → if it is, print the exact pane: *"Camera access blocked. Open System Settings → Privacy & Security → Camera, enable [Terminal/iTerm/VS Code], relaunch."* (b) emit one test CGEvent and check the return code → if dropped, point to Accessibility. (c) Enumerate AVFoundation devices with names and flag if index 0 is a Continuity Camera. Output a green/red checklist.

**R2. `mcv list-cameras` + named device selection — 🔴 / Effort: S**
*Impact:* Fixes the Continuity-Camera-steals-index-0 trap that config.yaml:6 openly admits exists.
*Implementation:* Enumerate and print `0: FaceTime HD  1: iPhone (Continuity)  2: USB Webcam`. Let `camera.index` accept a name substring, not just an int. Surface this in `doctor`.

**R3. Player install path + `mcv` console script — 🔴 / Effort: S**
*Impact:* Removes the `.venv/bin/python -m minecraft_cv.cli --config config.yaml …` wall of ceremony and the dead Poetry path.
*Implementation:* Add `[project.scripts] mcv = "minecraft_cv.cli:main"` with a subcommand dispatcher (`mcv run`, `mcv calibrate`, `mcv doctor`). Split extras: `pip install minecraft-cv` for players, `.[dev]` only for contributors. Default `--config` already resolves (cli.py:21), so `mcv run` with no args should Just Work.

**R4. Fix the broken calibrate command in README — 🔴 / Effort: S**
*Impact:* The one genuinely good onboarding tool is currently invoked with a command that doesn't match the entrypoint.
*Implementation:* Reconcile README:32 with the actual `mcv-calibrate`/`main_calibrate` entry; add a smoke test that every README command line parses.

### P1 — Needed for the experience to feel like a product

**R5. First-run onboarding wizard — 🟠 / Effort: M**
*Impact:* Converts "read three rule files" into a 90-second guided start.
*Implementation:* On first launch (no calibrated config detected), run a linear flow: **Welcome → Doctor (R1) → pick camera (R2) → live-preview mirror check ("wave — do you see yourself? y/n") → run guided calibration (R6) → show gesture cheat-sheet (R8) → "Launch live? This will control Minecraft."** Persist a `calibrated: true` marker so it doesn't repeat. Keep every step abortable (the wizard already handles `KeyboardInterrupt` cleanly at cli.py:163).

**R6. Unify & complete guided calibration — 🟠 / Effort: M**
*Impact:* Today joystick calibration is great but pinch tuning is a raw float-dump (cli.py:233). A user can't tune Attack without understanding normalized ratios.
*Implementation:* Extend the existing wizard with a pinch stage: "Pinch thumb-to-index and hold" / "Open your hand" → auto-derive `t_engage`/`t_release` with the safety margin and the `T_release > T_engage` invariant enforced before write (you already validate via `Settings(**merged)` at cli.py:198 — reuse it). Add `mcv calibrate --all`. Retire `--pinch` to a `--raw` expert flag.

**R7. Live status HUD (not just landmark dots) — 🟠 / Effort: M**
*Impact:* Closes the "is it even working?" gap during play and tuning.
*Implementation:* In the debug overlay, render a player-facing panel, not engineer telemetry: armed/held state per gesture (W A S D lit up, Attack/Use indicators), a deadzone ring with the live joystick vector, FPS, and a **mode banner** (`DRY-RUN` green vs `LIVE` red — reuse the string already built at cli.py:68). Gate cost behind the existing `overlay_every` decimation (config.yaml:145).

**R8. In-app gesture cheat-sheet — 🟠 / Effort: S**
*Impact:* The 12-gesture map currently lives only in YAML comments and internal rules.
*Implementation:* `mcv gestures` prints a formatted L-hand/R-hand reference card; show it at end of onboarding and bind a key (e.g. `?`) to overlay it during a run.

**R9. Actionable error messages — 🟠 / Effort: S**
*Impact:* Turns dead-ends into recoveries. Current handlers (cli.py:76, 140, 176) print the raw exception only.
*Implementation:* Map the common failures to remediation text: camera-busy → "another app is using the camera, quit Zoom/FaceTime"; no-hand-during-calibration (cli.py:176 already hints this) → "move into frame, ensure good lighting"; config-not-found → offer to generate a default.

### P2 — Polish that raises perceived quality

**R10. Config generation + safe-subset UI — 🟡 / Effort: M**
*Impact:* Stops hand-editing a 146-line file that interleaves V1/V2/"Task N" internals (config.yaml). Most users should never open it.
*Implementation:* `mcv config init` writes a minimal player config (camera, sensitivity, bindings) — defaults for everything else come from the model. A `mcv config set joystick.sensitivity 2.5` setter avoids YAML entirely. Reorganize the shipped file into **"You can change these"** vs **"Advanced / experimental (V2)"** sections, and strip internal "Task N" labels from user-facing comments.

**R11. Key-rebind flow — 🟡 / Effort: S**
*Impact:* `bindings:` (config.yaml:123) is the one thing players most want to change; today it's raw YAML.
*Implementation:* Interactive `mcv bind jump` → "press the key to bind" → writes back validated.

**R12. Troubleshooting/record-clip tool surfaced to users — 🟡 / Effort: S**
*Impact:* When gestures misfire, "record a clip and analyze it offline" is the right answer — but `mcv-analyze` (cli.py:277) is framed as a dev tool.
*Implementation:* `mcv record 10s` + a one-liner "having misfires? run `mcv doctor --replay last`" that runs the analyzer and reports false-trigger transitions in plain language.

---

## What's already good (keep it)

- **Safe-by-default input** (NullEmitter; `--input` opt-in, mutually-exclusive group at cli.py:42) — excellent and rare. Preserve it; just make the LIVE/DRY-RUN banner (R7) impossible to miss.
- **Calibration wizard UX** (countdown, preview-then-`--apply`, validate-before-write at cli.py:198) — this is the template; extend it to pinch and onboarding rather than reinventing.
- **Lazy heavy imports** (cli.py:1–5) keep `--help` instant — good CLI hygiene.

---

## The single highest-leverage move

If you ship one thing before release: **R1 (preflight doctor) + R3 (the `mcv` command)**. Together they convert the most common first-run outcome from *"black window, no input, no idea why"* into *"here's exactly which Settings pane to open."* Everything else improves a journey the user can already complete; these two make the journey completable at all.

One factual flag to fix regardless of roadmap priority: **README:32's calibrate command doesn't match the actual entrypoint** — verify and correct it, since it breaks the one tool most likely to give a new user a good first impression.

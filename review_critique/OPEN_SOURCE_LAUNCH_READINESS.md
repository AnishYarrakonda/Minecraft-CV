# minecraft_cv — Open Source Launch Readiness Review

**Repo:** `github.com/AnishYarrakonda/Minecraft-CV` · version `0.1.0` · license declared MIT (in metadata only)

## The 15-minute test: **FAIL**

A random visitor cannot install and run this in 15 minutes. Three hard stops:

1. **README opens with `The .venv is already initialized.`** — false for anyone who just cloned. The first thing a new user reads is an instruction written for *your* machine, not theirs. The actual install steps are buried as a parenthetical "if you need to reinstall."
2. **No platform warning.** The project is macOS-only (Quartz CGEvent, AVFoundation, `pyobjc` pinned to `sys_platform == 'darwin'`), but nothing says so until you hit an import error. A Linux/Windows visitor wastes their whole 15 minutes before discovering it can't work.
3. **No license file, no CI, no contribution path** — the repo *looks* unfinished, which kills trust before install even starts.

---

## Launch blockers

These must be fixed before going public.

| # | Blocker | Evidence | Why it blocks |
|---|---------|----------|---------------|
| 1 | **No `LICENSE` file** | `pyproject.toml` declares `license = MIT` but no `LICENSE`/`COPYING` exists | Without the actual file, the code is legally **all-rights-reserved** regardless of the metadata. Nobody can legally use or contribute. This is the single biggest blocker. |
| 2 | **README install path is broken for fresh clones** | README line 12: "The `.venv` is already initialized" | New users follow step 1 and it makes no sense. The working `python3 -m venv` steps are framed as optional. |
| 3 | **macOS-only not disclosed up front** | `pyobjc-framework-Quartz ... sys_platform=='darwin'`, AVFoundation, CGEvent | Sets false expectations; non-Mac users hit silent failures. Needs a prominent "Requirements: macOS + Apple Silicon recommended" banner. |
| 4 | **Committed Claude Code session transcript** | `.claude/projects/<hash>/session-id.jsonl` is git-tracked | A raw session log is noise at best and may leak paths/prompts. Should not ship in a public repo. |
| 5 | **No CI workflow** | No `.github/workflows/` | The README advertises a "Full CI gate" but nothing enforces it. PRs can't be validated; `main` is already red on ruff, and nothing catches that. |

---

## High-priority improvements

Strongly recommended before or immediately at launch.

1. **Missing community health files** — none of these exist:
   - `CONTRIBUTING.md` (how to set up, run the gate, the safety invariants contributors must not break)
   - `CODE_OF_CONDUCT.md`
   - `SECURITY.md` (this project emits OS-level synthetic input and requests Accessibility/Input Monitoring — it genuinely needs a security/responsible-use note)
   - `.github/ISSUE_TEMPLATE/` (bug + feature)
   - `.github/PULL_REQUEST_TEMPLATE.md`
   - `CHANGELOG.md`

2. **No screenshots or GIF.** This is a *gesture-controlled* project — the entire value proposition is visual. A README for a webcam→Minecraft controller with zero imagery will not get stars or contributors. A 10-second GIF of a hand pinching → Minecraft attacking is the highest-ROI launch asset you can make.

3. **Poetry instructions are misleading.** README offers a "Or with Poetry" path and `poetry run mcv-run`, but there's no committed `poetry.lock` (it's `.gitignore`d, line 31) and poetry isn't the supported path. Either commit the lockfile and support it, or delete the Poetry section. Right now it's a trap.

4. **README "Status" contradicts the shipped code.** README (lines 62-64) says inventory mode, hotbar gestures, dynamic deadzones, and sprint-by-velocity are "V2" / not done — but `config.yaml` and `src/` clearly implement all of them (`inventory.py`, `hotbar`, `dynamic_deadzone`, `sprint_velocity.py`). Stale status section undersells the project and confuses users about what works.

5. **No quickstart for the actual entrypoints.** The `mcv-run` / `mcv-calibrate` / `mcv-analyze` / `mcv-bench` console scripts (defined in `pyproject.toml`) are barely mentioned. After `pip install -e .` these are the natural UX, yet the README leads with the verbose `.venv/bin/python -m minecraft_cv.cli` form.

6. **Permissions setup isn't in the README.** The single most common footgun (Camera + Accessibility/Input Monitoring grants) is documented thoroughly in `.claude/rules/` but invisible to a normal user who'll never open those files. This belongs in a "First run / macOS permissions" section.

---

## Nice-to-have improvements

1. **Badges** — CI status, license, Python version, platform. Cheap credibility signal.
2. **`.github/FUNDING.yml`** if you want sponsors.
3. **Architecture diagram** — capture → tracking → gestures → joystick → input is a clean pipeline that diagrams well.
4. **PyPI publish workflow** (`release.yml` on tag) once the package stabilizes; `mediapipe`/`pyobjc` make this Mac-only on PyPI, so document that.
5. **A recorded demo clip** checked into `data/clips/` (currently only `.gitkeep`) so `mcv-analyze` works out of the box and the offline-testing story is real for newcomers.
6. **`Dependabot`/`renovate`** config for the heavy CV/torch deps.
7. **Decide on `.claude/` scope.** The agents/rules/skills are genuinely useful project artifacts and fine to ship; the `projects/*/memory/*` and session `.jsonl` are not. Consider `.gitignore`ing `.claude/projects/`.
8. **Move design docs out of `.claude/`.** The gesture contract in `.claude/rules/gestures.md` is excellent user-facing documentation hidden in a tool-specific folder. Surface a `docs/` version.

---

# Open Source Launch Checklist

### Blockers (do all before making the repo public)
- [ ] Add a real `LICENSE` file at repo root (MIT, matching `pyproject.toml`; year + "anish").
- [ ] Rewrite the README "Setup" section to assume a **fresh clone**: `git clone` → `python3 -m venv .venv` → `source .venv/bin/activate` → `pip install -e ".[dev]"`. Delete "the .venv is already initialized."
- [ ] Add a **Requirements** banner at the top of the README: macOS (Apple Silicon recommended), Python 3.11–3.13, a webcam, Minecraft (Java).
- [ ] `git rm --cached .claude/projects/<hash>/session-id.jsonl` and add `.claude/projects/` to `.gitignore`.
- [ ] Add `.github/workflows/ci.yml` running `pytest`, `ruff check src tests`, and (optionally) `mypy src` on push/PR. Confirm it's green — fix the existing ruff failures on `main` first, or scope the gate so it passes honestly.

### High priority (at launch)
- [ ] Add `CONTRIBUTING.md` — dev setup, the CI gate command, and the four **hard invariants** (`T_release > T_engage`, NullEmitter-by-default, CPU fallback, release-on-tracking-loss) that PRs must not violate.
- [ ] Add `CODE_OF_CONDUCT.md` (Contributor Covenant).
- [ ] Add `SECURITY.md` — reporting channel + a responsible-use note (this tool injects synthetic keyboard/mouse and needs Accessibility permission).
- [ ] Add `.github/ISSUE_TEMPLATE/bug_report.md` and `feature_request.md` (ask for macOS version, camera model, Python version, config diff).
- [ ] Add `.github/PULL_REQUEST_TEMPLATE.md` (checklist: tests pass, invariants intact, no real input emitted in tests).
- [ ] Add `CHANGELOG.md` seeded with `0.1.0`.
- [ ] Record and embed a **demo GIF** of a gesture driving Minecraft in the README.
- [ ] Add a **"First run & macOS permissions"** README section (Camera + Accessibility/Input Monitoring, granted to the terminal app).
- [ ] Fix or delete the **Poetry** section (commit `poetry.lock` or remove the path).
- [ ] Update the README **Status** section to reflect that inventory/hotbar/dynamic-deadzone/sprint-velocity are implemented.
- [ ] Lead the usage examples with the `mcv-run` console scripts; keep the `-m` form as the dev/no-install alternative.

### Nice-to-have (post-launch polish)
- [ ] Add README badges (CI, license, Python, platform).
- [ ] Add an architecture diagram of the pipeline.
- [ ] Commit a small sample clip to `data/clips/` so `mcv-analyze` runs out of the box.
- [ ] Add a `release.yml` tag-triggered workflow (document Mac-only PyPI constraints).
- [ ] Add Dependabot/Renovate config.
- [ ] Surface the gesture-map design doc into a public `docs/` folder.
- [ ] Add `.github/FUNDING.yml` if desired.

---

**Bottom line:** the *code* and internal docs are in good shape — this reads like a real, well-architected project. The gap is entirely **launch packaging**: a missing license, a README written for your laptop instead of a stranger's, no platform disclosure, no CI, and no visual demo. Fix the five blockers and the GIF, and the 15-minute test flips to a pass.

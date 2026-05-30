---
name: feedback-setup-commands
description: .venv is pre-initialized; always use .venv/bin/python -m minecraft_cv.cli, not poetry
metadata:
  type: feedback
---

**Use `.venv/bin/python -m minecraft_cv.cli` for all commands, not `poetry run`.**

**Why:** Poetry is not installed in the user's environment. The .venv is already initialized and functional. All CLI entrypoints are accessible via `.venv/bin/python -m minecraft_cv.cli` which maps to the setup.py script entrypoints (mcv-run, mcv-calibrate, etc.).

**How to apply:**
- All command examples in README.md, CLAUDE.md, and docs should prioritize `.venv/bin/python -m minecraft_cv.cli [args]`
- Only mention `poetry run` as a secondary option "if poetry is installed"
- Never assume poetry is available; it adds a setup step that isn't necessary

**Common patterns:**
- Dry run: `.venv/bin/python -m minecraft_cv.cli --config config.yaml --no-input --debug-overlay`
- Live: `.venv/bin/python -m minecraft_cv.cli --config config.yaml --input --debug-overlay`
- Tests: `.venv/bin/python -m pytest`
- Linting: `.venv/bin/ruff check src tests`
- Type checking: `.venv/bin/mypy src`

Updated 2026-05-30 after doc review; commands are now consistent with actual environment setup.

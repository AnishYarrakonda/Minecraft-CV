"""Friendly launcher for minecraft_cv — opens the polished desktop app.

Run with:

    .venv/bin/python main.py

This just bootstraps the source path and launches the GUI (``mcv ui``), which opens in safe
Dry-Run mode. Flip to Live, calibrate, and start/stop from inside the app. The headless
controller and clip tools still live behind ``mcv run`` / ``mcv analyze``.
"""

from __future__ import annotations

# ruff: noqa: E402,I001

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minecraft_cv.cli import main as cli_main  # noqa: E402


def main() -> int:
    """Launch the desktop app, forwarding any extra CLI args (e.g. ``--config``)."""
    return cli_main(["ui", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())

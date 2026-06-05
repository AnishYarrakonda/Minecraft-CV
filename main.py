"""Friendly launcher for minecraft_cv — opens the polished desktop app.

Run with:

    .venv/bin/python main.py

This just bootstraps the source path and launches the GUI (``mcv ui``), which opens in safe
Dry-Run mode. Flip to Live, calibrate, and start/stop from inside the app. The headless
controller and clip tools still live behind ``mcv run`` / ``mcv analyze``.
"""

from __future__ import annotations

# ruff: noqa: E402,I001

import os
import sys
from pathlib import Path

# Must be set before Qt (PySide6) initialises OpenGL. Qt's GL init makes mediapipe
# detect GPU as available and take the GPU landmark-projection path, which then crashes
# (SIGTRAP) when the graph runs on a non-main thread. Disabling GPU forces the CPU path.
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from minecraft_cv.cli import main as cli_main  # noqa: E402


def main() -> int:
    """Ask the user which mode to launch, then start it."""
    try:
        import PySide6  # noqa: F401
    except ImportError:
        print(
            "[main.py] PySide6 not found — launching headless mode (--no-input --debug-overlay).\n"
            "          To get the full desktop app:  pip install PySide6\n",
            file=sys.stderr,
        )
        return cli_main(["run", "--no-input", "--debug-overlay", *sys.argv[1:]])

    print("minecraft_cv — choose a launch mode:")
    print("  1  Full app    (camera + HUD sidebar)")
    print("  2  Overlay     (compact always-on-top window)")
    print()
    try:
        choice = input("Enter 1 or 2 [default: 1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return 0

    if choice == "2":
        return cli_main(["overlay", *sys.argv[1:]])
    return cli_main(["ui", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())

"""Friendly all-in-one terminal launcher for minecraft_cv.

Run with:

    .venv/bin/python main.py

The real app logic still lives in ``minecraft_cv.cli``; this file just asks which mode you
want, runs calibration for you when it's needed (or when you ask), and then launches the
controller — so you never have to remember the separate ``mcv calibrate`` step.
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


def yes_no(prompt: str, default: bool = True) -> bool:
    """Ask a yes/no terminal question. Empty input returns the default."""
    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        answer = input(prompt + suffix).strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please enter y or n.")


def main() -> int:
    """Prompt for debug or real mode, then launch the controller."""
    print("minecraft_cv launcher")
    real_input = yes_no("Run with real Minecraft keyboard/mouse input?", default=True)
    show_overlay = yes_no("Show camera overlay?", default=True)

    args = ["run", "--input" if real_input else "--no-input"]
    if show_overlay:
        args.append("--debug-overlay")
    if not real_input:
        print("\nStarting debug mode with no real input.")
    return cli_main(args)


if __name__ == "__main__":
    raise SystemExit(main())

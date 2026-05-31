"""Friendly terminal launcher for minecraft_cv.

Run with:

    .venv/bin/python main.py

The real app logic still lives in ``minecraft_cv.cli``; this file just asks which mode you
want and passes through to the existing commands.
"""

from __future__ import annotations

# ruff: noqa: E402,I001

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minecraft_cv.config import Settings  # noqa: E402
from minecraft_cv.cli import DEFAULT_CONFIG, main as cli_main  # noqa: E402


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


def palm_normal_calibration_missing(config_path: str = DEFAULT_CONFIG) -> bool:
    """Return True when live palm-normal mode needs calibration before it can run."""
    path = config_path if Path(config_path).is_file() else None
    settings = Settings.load(path)
    if settings.joystick.mode != "palm_normal":
        return False
    palm = settings.joystick.palm_normal
    return palm.left_neutral is None or palm.right_neutral is None


def main() -> int:
    """Prompt for debug or real mode, then launch the existing CLI."""
    print("minecraft_cv launcher")
    real_input = yes_no("Run with real Minecraft keyboard/mouse input?", default=True)
    show_overlay = yes_no("Show camera overlay?", default=True)

    if real_input:
        if palm_normal_calibration_missing():
            print(
                "\nPalm-normal calibration is missing, so live input cannot start yet."
                "\nQuick setup only captures your resting hand pose and keeps the default "
                "sensitivity."
            )
            if yes_no("Run quick setup now?", default=True):
                code = cli_main(["calibrate", "--quick-neutral", "--apply"])
                if code != 0:
                    return code
            else:
                print("Cancelled. Use debug mode or run calibration before live input.")
                return 1
        args = ["run", "--input"]
        if show_overlay:
            args.append("--debug-overlay")
        return cli_main(args)

    print("\nStarting debug mode with no real input.")
    args = ["run", "--no-input"]
    if show_overlay:
        args.append("--debug-overlay")
    return cli_main(args)


if __name__ == "__main__":
    raise SystemExit(main())

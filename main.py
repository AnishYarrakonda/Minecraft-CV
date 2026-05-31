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


def calibration_state(config_path: str = DEFAULT_CONFIG) -> tuple[str, bool]:
    """Return ``(mode, missing)`` for the active joystick mode.

    ``missing`` is True when the active mode needs a calibrated neutral before live input can
    start. The position-based ``wrist_rotation`` / ``anchor`` modes never need one.
    """
    path = config_path if Path(config_path).is_file() else None
    settings = Settings.load(path)
    mode = settings.joystick.mode
    if mode == "palm_tilt":
        block = settings.joystick.tilt
    elif mode == "palm_normal":
        block = settings.joystick.palm_normal
    else:
        return mode, False
    return mode, block.left_neutral is None or block.right_neutral is None


def run_calibration(quick: bool) -> int:
    """Run the guided joystick calibration for the active mode and write it to config."""
    args = ["calibrate", "--apply"]
    if quick:
        args.append("--quick-neutral")
    else:
        print(
            "\nGuided calibration: hold each pose for a couple seconds when prompted —\n"
            "  1) rest both hands comfortably (neutral)\n"
            "  2) tilt both hands UP, then DOWN, then LEFT, then RIGHT, to a comfy reach.\n"
        )
    return cli_main(args)


def maybe_calibrate(mode: str, missing: bool, *, required: bool) -> int | None:
    """Calibrate when needed or when the user asks. Returns an exit code on failure, else None.

    Args:
        mode: Active joystick mode name (for messaging).
        missing: Whether the active mode currently lacks a calibrated neutral.
        required: True for live input (calibration must exist before launch); False for the
            debug preview, which runs fine uncalibrated.
    """
    if required and missing:
        print(f"\n{mode} calibration is missing — let's calibrate now so live input works.")
        do_it = True
    else:
        prompt = "Recalibrate the joysticks first?"
        do_it = yes_no(prompt, default=False)
    if not do_it:
        return None
    quick = not yes_no(
        "Full guided calibration (best feel)?  (n = quick resting-pose only)", default=True
    )
    code = run_calibration(quick)
    if code != 0:
        print("\nCalibration failed or was cancelled; not launching.")
        return code
    return None


def main() -> int:
    """Prompt for debug or real mode, calibrate if needed, then launch the controller."""
    print("minecraft_cv launcher")
    real_input = yes_no("Run with real Minecraft keyboard/mouse input?", default=True)
    show_overlay = yes_no("Show camera overlay?", default=True)

    mode, missing = calibration_state()
    code = maybe_calibrate(mode, missing, required=real_input)
    if code is not None:
        return code

    args = ["run", "--input" if real_input else "--no-input"]
    if show_overlay:
        args.append("--debug-overlay")
    if not real_input:
        print("\nStarting debug mode with no real input.")
    return cli_main(args)


if __name__ == "__main__":
    raise SystemExit(main())

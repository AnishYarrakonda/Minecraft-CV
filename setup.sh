#!/usr/bin/env bash
# One-command setup for minecraft_cv (macOS).
# Creates a virtualenv, installs the package, and runs the permission/health check.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

echo "==> Creating virtual environment (.venv)"
"$PYTHON" -m venv .venv

echo "==> Installing minecraft_cv"
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip >/dev/null
pip install -e ".[dev]"

echo "==> Checking camera + permissions"
.venv/bin/python -m minecraft_cv.cli doctor || true

cat <<'DONE'

==> Done. To play:

    source .venv/bin/activate
    mcv ui

The app opens in Dry-Run mode — click "Go Live" when you're ready to send
real input to Minecraft.
DONE

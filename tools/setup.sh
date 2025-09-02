#!/usr/bin/env bash
set -euo pipefail

# Create a virtual environment under tools/.venv and install requirements.
# macOS/pyenv friendly. Optionally set PYTHON_BIN to choose interpreter.

cd "$(dirname "$0")"

VENV_DIR=".venv"

# Choose interpreter in this order:
# 1) $PYTHON_BIN if provided
# 2) pyenv's current python (if pyenv is available)
# 3) system/default python3
if [ -n "${PYTHON_BIN:-}" ]; then
  PYTHON_BIN="$PYTHON_BIN"
elif command -v pyenv >/dev/null 2>&1; then
  PYTHON_BIN="$(pyenv which python)"
else
  PYTHON_BIN="python3"
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "[setup] Creating venv at tools/$VENV_DIR using $PYTHON_BIN"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "[setup] Reusing existing tools/$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
pip install -r ../requirements.txt

echo
echo "[ok] Environment ready. Activate with:"
echo "     source tools/.venv/bin/activate"
echo "Or run once without activating:"
echo "     tools/.venv/bin/python tools/split_cards.py --help"

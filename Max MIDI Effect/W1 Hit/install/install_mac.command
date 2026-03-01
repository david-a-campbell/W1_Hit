#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$DIR/.venv"

echo "W1 Hit installer"
echo "Device dir: $DIR"

if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "ERROR: python3 not found. Install Python 3 first."
  exit 1
fi

echo "Using: $($PY --version)"

if [ ! -d "$VENV" ]; then
  $PY -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"

echo "✅ Done. You can now run the device."
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$DIR/.venv"
REQ="$SCRIPT_DIR/requirements.txt"

echo "W1 Hit installer"
echo "Device dir: $DIR"

if [ ! -f "$REQ" ]; then
  echo "ERROR: requirements.txt not found at: $REQ"
  exit 1
fi

# PyTorch and python-rtmidi may not publish wheels for the newest Python yet.
# Avoid Python 3.13+ because pip will report: "No matching distribution found for torch".
MIN_MINOR=10
MAX_MINOR=12

version_ok() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
major, minor = sys.version_info[:2]
raise SystemExit(0 if major == 3 and 10 <= minor <= 12 else 1)
PY
}

python_version() {
  "$1" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
PY
}

PY=""

# Prefer known-good Python versions first.
for candidate in \
  python3.11 \
  python3.10 \
  python3.12 \
  /opt/homebrew/bin/python3.11 \
  /opt/homebrew/bin/python3.10 \
  /opt/homebrew/bin/python3.12 \
  /usr/local/bin/python3.11 \
  /usr/local/bin/python3.10 \
  /usr/local/bin/python3.12 \
  /opt/anaconda3/envs/python3_10/bin/python \
  python3 \
  python
 do
  if command -v "$candidate" >/dev/null 2>&1; then
    candidate_path="$(command -v "$candidate")"
    if version_ok "$candidate_path"; then
      PY="$candidate_path"
      break
    fi
  elif [ -x "$candidate" ]; then
    if version_ok "$candidate"; then
      PY="$candidate"
      break
    fi
  fi
done

if [ -z "$PY" ]; then
  echo "ERROR: Could not find a compatible Python."
  echo "W1 Hit needs Python 3.10, 3.11, or 3.12 for PyTorch wheels."
  echo "Your default Python is probably too new, such as Python 3.13."
  echo ""
  echo "Install Python 3.11, then run this installer again:"
  echo "  brew install python@3.11"
  echo ""
  echo "Or create/use your existing Conda env with Python 3.10/3.11."
  exit 1
fi

echo "Using Python: $PY ($(python_version "$PY"))"

# Recreate the venv if it was made with an incompatible Python.
if [ -d "$VENV" ]; then
  if ! version_ok "$VENV/bin/python"; then
    echo "Existing venv uses incompatible Python: $($VENV/bin/python --version 2>&1 || true)"
    echo "Recreating venv..."
    rm -rf "$VENV"
  fi
fi

if [ ! -d "$VENV" ]; then
  "$PY" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV/bin/python" -m pip install -r "$REQ"

echo "✅ Done. You can now run the device."

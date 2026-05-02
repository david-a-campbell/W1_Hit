#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$DIR/.venv"
REQ="$SCRIPT_DIR/requirements.txt"
CONSTRAINTS="$SCRIPT_DIR/.w1_hit_constraints.txt"
PREFERRED_MINOR=11

BREW_PATHS=("/opt/homebrew/bin/brew" "/usr/local/bin/brew" "brew")
PYTHON_CANDIDATES=(
  "/opt/homebrew/bin/python3.11" "/usr/local/bin/python3.11" "python3.11"
  "/opt/homebrew/opt/python@3.11/bin/python3.11" "/usr/local/opt/python@3.11/bin/python3.11"
  "/opt/homebrew/opt/python@3.11/libexec/bin/python3" "/usr/local/opt/python@3.11/libexec/bin/python3"
  "/opt/homebrew/bin/python3.10" "/usr/local/bin/python3.10" "python3.10"
  "/opt/homebrew/opt/python@3.10/bin/python3.10" "/usr/local/opt/python@3.10/bin/python3.10"
  "/opt/homebrew/opt/python@3.10/libexec/bin/python3" "/usr/local/opt/python@3.10/libexec/bin/python3"
  "/opt/homebrew/bin/python3.12" "/usr/local/bin/python3.12" "python3.12"
  "/opt/homebrew/opt/python@3.12/bin/python3.12" "/usr/local/opt/python@3.12/bin/python3.12"
  "/opt/homebrew/opt/python@3.12/libexec/bin/python3" "/usr/local/opt/python@3.12/libexec/bin/python3"
  "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3" "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3" "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
  "python3"
)

pause_on_error() {
  local code=$?
  if [ "$code" -ne 0 ]; then
    echo
    echo "Installation did not complete. See the error above."
    echo "Press Return to close this window."
    read -r _ || true
  fi
  exit "$code"
}
trap pause_on_error EXIT

print_header() {
  echo "W1 Hit installer"
  echo "Device dir: $DIR"
  echo
}

is_blocked_python() {
  local p="${1:-}"
  case "$p" in
    *conda*|*Conda*|*anaconda*|*Anaconda*|*miniconda*|*Miniconda*|*mambaforge*|*Mambaforge*|*miniforge*|*Miniforge*) return 0 ;;
    *) return 1 ;;
  esac
}

version_ok() {
  "$1" - <<'PYCODE' >/dev/null 2>&1
import sys
major, minor = sys.version_info[:2]
raise SystemExit(0 if major == 3 and 10 <= minor <= 12 else 1)
PYCODE
}

python_version() {
  "$1" - <<'PYCODE'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
PYCODE
}

python_executable() {
  "$1" - <<'PYCODE'
import sys
print(sys.executable)
PYCODE
}

find_python() {
  local candidate candidate_path resolved

  if [ "${PYTHON:-}" != "" ]; then
    if [ ! -x "$PYTHON" ]; then
      echo "ERROR: PYTHON was set, but it is not executable: $PYTHON" >&2
      return 1
    fi
    resolved="$(python_executable "$PYTHON" 2>/dev/null || echo "$PYTHON")"
    if is_blocked_python "$resolved"; then
      echo "ERROR: PYTHON points to an unsupported Python install: $resolved" >&2
      return 1
    fi
    if version_ok "$PYTHON"; then
      echo "$PYTHON"
      return 0
    fi
    echo "ERROR: PYTHON must be Python 3.10, 3.11, or 3.12: $PYTHON" >&2
    return 1
  fi

  for candidate in "${PYTHON_CANDIDATES[@]}"; do
    if command -v "$candidate" >/dev/null 2>&1; then
      candidate_path="$(command -v "$candidate")"
    elif [ -x "$candidate" ]; then
      candidate_path="$candidate"
    else
      continue
    fi

    resolved="$(python_executable "$candidate_path" 2>/dev/null || echo "$candidate_path")"
    if ! is_blocked_python "$resolved" && version_ok "$candidate_path"; then
      echo "$candidate_path"
      return 0
    fi
  done

  return 1
}

find_brew() {
  local brew_candidate
  for brew_candidate in "${BREW_PATHS[@]}"; do
    if command -v "$brew_candidate" >/dev/null 2>&1; then
      command -v "$brew_candidate"
      return 0
    elif [ -x "$brew_candidate" ]; then
      echo "$brew_candidate"
      return 0
    fi
  done
  return 1
}

install_homebrew() {
  echo "Homebrew was not found. Installing Homebrew first..." >&2
  echo "macOS may ask for your password or Command Line Tools permission." >&2
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" >&2

  if [ -x "/opt/homebrew/bin/brew" ]; then
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$HOME/.zprofile" 2>/dev/null || true
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [ -x "/usr/local/bin/brew" ]; then
    echo 'eval "$(/usr/local/bin/brew shellenv)"' >> "$HOME/.zprofile" 2>/dev/null || true
    eval "$(/usr/local/bin/brew shellenv)"
  fi
}

install_python_with_brew() {
  local BREW p

  if ! BREW="$(find_brew)"; then
    install_homebrew
    BREW="$(find_brew)" || {
      echo "ERROR: Homebrew installation finished, but brew was not found."
      echo "Install Python 3.11 from python.org, then run this installer again."
      return 1
    }
  fi

  echo "Installing Python 3.$PREFERRED_MINOR..." >&2
  "$BREW" update >&2 || true
  "$BREW" install "python@3.$PREFERRED_MINOR" >&2 || "$BREW" upgrade "python@3.$PREFERRED_MINOR" >&2 || true

  # Homebrew sometimes installs versioned Python only under the formula's opt path
  # instead of linking /opt/homebrew/bin/python3.11. Check both places.
  for p in "/opt/homebrew/bin/python3.$PREFERRED_MINOR" "/usr/local/bin/python3.$PREFERRED_MINOR" \
           "/opt/homebrew/opt/python@3.$PREFERRED_MINOR/bin/python3.$PREFERRED_MINOR" "/usr/local/opt/python@3.$PREFERRED_MINOR/bin/python3.$PREFERRED_MINOR" \
           "/opt/homebrew/opt/python@3.$PREFERRED_MINOR/libexec/bin/python3" "/usr/local/opt/python@3.$PREFERRED_MINOR/libexec/bin/python3" \
           "python3.$PREFERRED_MINOR"; do
    if command -v "$p" >/dev/null 2>&1; then
      p="$(command -v "$p")"
      if ! is_blocked_python "$p" && version_ok "$p"; then
        echo "$p"
        return 0
      fi
    elif [ -x "$p" ] && version_ok "$p"; then
      echo "$p"
      return 0
    fi
  done

  echo "ERROR: Python 3.$PREFERRED_MINOR was installed, but the executable was not found." >&2
  echo "Homebrew may have installed it without linking it into your PATH." >&2
  echo "Try running this installer again, or install Python 3.11 from python.org." >&2
  return 1
}

print_header

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE _CE_CONDA _CE_M PYTHONHOME || true
export PIP_DISABLE_PIP_VERSION_CHECK=1

if [ ! -f "$REQ" ]; then
  echo "ERROR: requirements.txt not found at: $REQ"
  echo "Make sure this installer is in the same folder as requirements.txt."
  exit 1
fi

if PY="$(find_python)"; then
  :
else
  PY="$(install_python_with_brew)"
fi

if is_blocked_python "$(python_executable "$PY" 2>/dev/null || echo "$PY")" || ! version_ok "$PY"; then
  echo "ERROR: Selected Python is not compatible: $PY"
  echo "W1 Hit needs Python 3.10, 3.11, or 3.12."
  exit 1
fi

echo "Using Python: $PY ($(python_version "$PY"))"

if [ -d "$VENV" ]; then
  VENV_EXE="$($VENV/bin/python -c 'import sys; print(sys.executable)' 2>/dev/null || echo invalid)"
  if [ ! -x "$VENV/bin/python" ] || ! version_ok "$VENV/bin/python" || is_blocked_python "$VENV_EXE"; then
    echo "Existing virtual environment is incompatible. Recreating it..."
    rm -rf "$VENV"
  fi
fi

if [ ! -d "$VENV" ]; then
  echo "Creating virtual environment..."
  "$PY" -m venv "$VENV"
fi

cat > "$CONSTRAINTS" <<'CONSTRAINTS'
numpy<2
CONSTRAINTS

echo "Upgrading pip tools..."
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel

echo "Installing requirements..."
"$VENV/bin/python" -m pip install --force-reinstall "numpy<2"
"$VENV/bin/python" -m pip install -c "$CONSTRAINTS" -r "$REQ"
"$VENV/bin/python" -m pip install --force-reinstall "numpy<2"

echo "Verifying install..."
"$VENV/bin/python" - <<'PYCODE'
import importlib
import sys
required = ["numpy", "torch", "mido"]
missing = []
for name in required:
    try:
        mod = importlib.import_module(name)
        print(f"{name} {getattr(mod, '__version__', 'installed')}")
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    print("Install verification failed:")
    for item in missing:
        print(" -", item)
    sys.exit(1)
PYCODE

echo
echo "Done. You can now run the device."
trap - EXIT
exit 0

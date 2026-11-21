#!/usr/bin/env bash
# AutoScout one-click start (macOS / Linux): ./start.sh
# Finds Python 3.10+, creates a private environment (.venv) the first time,
# then launcher.py updates AutoScout, installs packages and opens the dashboard.
# Options are passed on: ./start.sh --no-update --no-browser --port 8766
#
# Everything runs inside main(), called on the last line together with exit,
# so bash has read the whole file before an update can change it.

main() {
    cd "$(dirname "$0")" || exit 1

    find_python() {
        for cmd in python3.13 python3.12 python3.11 python3.10 python3 python; do
            if command -v "$cmd" >/dev/null 2>&1 &&
               "$cmd" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
                echo "$cmd"
                return 0
            fi
        done
        return 1
    }

    # (re)create .venv when it is missing or its Python no longer runs
    # (e.g. after Python was upgraded or removed)
    if ! .venv/bin/python -c 'import sys' >/dev/null 2>&1; then
        PY=$(find_python) || {
            echo "[AutoScout] Python 3.10 or newer is needed. Install it from https://www.python.org/downloads/"
            echo "            (macOS: brew install python   Ubuntu/Debian: sudo apt install python3 python3-venv)"
            return 1
        }
        if [ -d .venv ]; then
            echo "[AutoScout] The Python environment in .venv is broken (Python was upgraded or removed?), rebuilding it..."
        else
            echo "[AutoScout] First start: creating the Python environment in .venv (one time)..."
        fi
        if ! "$PY" -m venv --clear .venv; then
            rm -rf .venv
            echo "[AutoScout] Could not create .venv. On Ubuntu/Debian install it with: sudo apt install python3-venv"
            return 1
        fi
    fi

    .venv/bin/python launcher.py "$@" || {
        code=$?
        echo "[AutoScout] Stopped with an error - see the messages above. If it keeps happening after a Python update, delete the .venv folder and start again."
        return $code
    }
}

main "$@"; exit $?

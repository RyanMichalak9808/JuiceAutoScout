"""
One-click start for the AutoScout dashboard (run by start.bat / start.sh).

Every start:
  1. update: if this folder is a git clone, pull the latest version
     (skipped with a warning when you changed tracked files yourself, when
     offline, or with --no-update),
  2. packages: install requirements.txt into .venv when it changed, and keep
     yt-dlp current (YouTube breaks old versions every few weeks),
  3. check that ffmpeg is installed (needed for downloads and stream splitting),
  4. start the dashboard and open it in the browser (if it is already
     running, just open it).

Your settings (dashboard_settings.json) and results (output_dashboard/) are
not part of the repository, so updates never touch them.

Options: --no-update  --no-browser  --port N
"""
import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / ".venv" / "autoscout_launcher.json"
YTDLP_REFRESH_S = 24 * 3600
MIN_PYTHON = (3, 10)


def say(msg: str) -> None:
    print("[AutoScout] " + msg, flush=True)


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError:
        pass


def git(*args, timeout=60):
    return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)


def update_code() -> bool:
    """git pull --ff-only when it is safe. Returns True if new code arrived."""
    if not (ROOT / ".git").exists():
        say("Not a git clone (downloaded as ZIP?): to update, download the latest version again. "
            "Your settings and results are kept if you copy them over.")
        return False
    if shutil.which("git") is None:
        say("git is not installed, skipping the update check.")
        return False
    try:
        if git("status", "--porcelain", "--untracked-files=no").stdout.strip():
            say("You changed files in this folder, so it was NOT updated automatically. "
                "Commit or undo your changes (git stash), then start again to update.")
            return False
        branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        upstream = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
        if branch == "HEAD" or upstream.returncode != 0:
            say("This checkout is not on a branch that tracks GitHub (pinned to a version?), not updating.")
            return False
        say("Checking for updates...")
        fetch = git("fetch", "--quiet", timeout=30)
        if fetch.returncode != 0:
            say("Could not reach GitHub (offline?), starting the current version.")
            return False
        behind = git("rev-list", "--count", "HEAD..@{u}").stdout.strip()
        if not behind or behind == "0":
            say("Up to date ({}).".format(version_text()))
            return False
        pull = git("pull", "--ff-only", "--quiet", timeout=120)
        if pull.returncode != 0:
            say("Update failed, starting the current version:\n" + (pull.stderr or pull.stdout).strip())
            return False
        say("Updated: {} new change(s), now at {}.".format(behind, version_text()))
        return True
    except (OSError, subprocess.TimeoutExpired) as exc:
        say("Update check skipped ({}).".format(exc))
        return False


def version_text() -> str:
    try:
        out = git("describe", "--tags", "--always", "--dirty").stdout.strip()
        return out or "unknown version"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown version"


def pip(*args) -> int:
    return subprocess.call([sys.executable, "-m", "pip", "--disable-pip-version-check", *args], cwd=str(ROOT))


def ensure_packages(state: dict) -> bool:
    req = ROOT / "requirements.txt"
    digest = hashlib.sha256(req.read_bytes()).hexdigest() if req.exists() else ""
    if digest and state.get("requirements_sha256") != digest:
        say("Installing Python packages (first start or after an update, takes a few minutes)...")
        if pip("install", "--upgrade", "pip") != 0 or pip("install", "-r", str(req)) != 0:
            say("Installing packages failed - check your internet connection and the messages above, then start again.")
            return False
        state["requirements_sha256"] = digest
        state["ytdlp_checked_at"] = time.time()
        save_state(state)
    elif time.time() - float(state.get("ytdlp_checked_at") or 0) > YTDLP_REFRESH_S:
        say("Updating yt-dlp (YouTube changes often)...")
        if pip("install", "--quiet", "--upgrade", "yt-dlp") == 0:
            state["ytdlp_checked_at"] = time.time()
            save_state(state)
        else:
            say("Could not update yt-dlp (offline?), using the installed version.")
    return True


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    if sys.platform.startswith("win"):
        hint = "winget install Gyan.FFmpeg   (then close this window and start again)"
    elif sys.platform == "darwin":
        hint = "brew install ffmpeg"
    else:
        hint = "sudo apt install ffmpeg   (or your distribution's package manager)"
    say("ffmpeg is not installed. Downloading matches and splitting streams need it. Install it with:\n    " + hint)


def dashboard_running(port: int) -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:{}/api/health".format(port), timeout=2) as resp:
            return json.loads(resp.read().decode("utf-8")).get("ok") is True
    except (OSError, ValueError):
        return False


def port_free(port: int) -> bool:
    # something already answering on the port = taken
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        if s.connect_ex(("127.0.0.1", port)) == 0:
            return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if not sys.platform.startswith("win"):
            # (on Windows SO_REUSEADDR would allow binding a port that is in use)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def open_when_ready(url: str, port: int) -> None:
    for _ in range(120):
        if dashboard_running(port):
            webbrowser.open(url)
            return
        time.sleep(0.5)


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the AutoScout dashboard (updates first).")
    parser.add_argument("--no-update", action="store_true", help="do not pull the latest version")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    os.chdir(str(ROOT))
    if sys.version_info < MIN_PYTHON:
        say("Python {}.{} or newer is needed (this is {}).".format(*MIN_PYTHON, sys.version.split()[0]))
        return 1

    state = load_state()
    if not args.no_update and update_code():
        # new code may need new packages and a new launcher: run the updated launcher
        say("Restarting with the updated version...")
        return subprocess.call([sys.executable, str(ROOT / "launcher.py"), "--no-update",
                                *(["--no-browser"] if args.no_browser else []), "--port", str(args.port)])
    if not ensure_packages(state):
        return 1
    check_ffmpeg()

    port = args.port
    url = "http://127.0.0.1:{}/".format(port)
    if dashboard_running(port):
        say("The dashboard is already running: {}".format(url))
        if not args.no_browser:
            webbrowser.open(url)
        return 0
    if not port_free(port):
        for candidate in range(port + 1, port + 20):
            if port_free(candidate):
                say("Port {} is used by another program, using {} instead.".format(port, candidate))
                port = candidate
                break
        else:
            say("No free port found near {}. Close the program using it or start with --port.".format(port))
            return 1
        url = "http://127.0.0.1:{}/".format(port)

    say("Starting the dashboard at {}  (close this window or press Ctrl+C to stop it)".format(url))
    if not args.no_browser:
        threading.Thread(target=open_when_ready, args=(url, port), daemon=True).start()
    try:
        return subprocess.call([sys.executable, str(ROOT / "dashboard_server.py"), "--port", str(port)], cwd=str(ROOT))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())

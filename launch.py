#!/usr/bin/env python3
"""launch.py — cross-platform launcher for MindTune-OS.

Works on macOS, Linux, and Windows without any extra tools.
Usage:
    python launch.py          # start both processes
    python launch.py --stop   # stop both processes
"""

import os
import sys
import time
import signal
import subprocess
import webbrowser
import argparse
from pathlib import Path
from dotenv import dotenv_values

ROOT = Path(__file__).parent.resolve()
PIDS_FILE = ROOT / ".pids"
LOGS_DIR = ROOT / "logs"
ENV_FILE = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"
DASHBOARD_URL = "http://127.0.0.1:5050"

REQUIRED_VARS = [
    "SPOTIFY_CLIENT_ID",
    "SPOTIFY_CLIENT_SECRET",
    "SPOTIFY_REDIRECT_URI",
    "GROQ_API_KEY",
]

IS_WINDOWS = sys.platform == "win32"


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_python():
    """Return the Python executable that has this project's dependencies."""
    for candidate in [sys.executable, "python", "python3"]:
        try:
            result = subprocess.run(
                [candidate, "-c", "import joblib"],
                capture_output=True,
            )
            if result.returncode == 0:
                return candidate
        except FileNotFoundError:
            continue
    return None


def is_running(pid):
    """Return True if a process with the given PID is alive."""
    if IS_WINDOWS:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True, text=True,
        )
        return str(pid) in result.stdout
    else:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def kill_pid(pid, name="process"):
    """Terminate a process by PID."""
    if not is_running(pid):
        return False
    if IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/F", "/PID", str(pid)],
            capture_output=True,
        )
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return False
    print(f"  Stopped {name} (PID {pid})")
    return True


def kill_by_script(script_rel_path):
    """Kill any processes matching a script path (fallback for stray processes)."""
    script = str(ROOT / script_rel_path)
    if IS_WINDOWS:
        result = subprocess.run(
            ["wmic", "process", "where",
             f"CommandLine like '%{script_rel_path}%'",
             "get", "ProcessId"],
            capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                pid = int(line)
                kill_pid(pid, script_rel_path)
    else:
        try:
            result = subprocess.run(
                ["pgrep", "-f", script],
                capture_output=True, text=True,
            )
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.isdigit():
                    kill_pid(int(line), script_rel_path)
        except FileNotFoundError:
            pass  # pgrep not available on all Unix systems


def kill_port(port):
    """Kill any process listening on the given port (macOS/Linux only)."""
    if IS_WINDOWS:
        return  # handled implicitly by kill_by_script on Windows
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                pid = int(line)
                try:
                    os.kill(pid, signal.SIGKILL)
                    print(f"  Killed process on port {port} (PID {pid})")
                except ProcessLookupError:
                    pass
    except FileNotFoundError:
        pass  # lsof not available


def load_pids():
    """Read {name: pid} from .pids file."""
    pids = {}
    if PIDS_FILE.exists():
        for line in PIDS_FILE.read_text().splitlines():
            if "=" in line:
                name, _, pid_str = line.partition("=")
                try:
                    pids[name.strip()] = int(pid_str.strip())
                except ValueError:
                    pass
    return pids


def save_pids(pids):
    """Write {name: pid} to .pids file."""
    PIDS_FILE.write_text(
        "\n".join(f"{name}={pid}" for name, pid in pids.items()) + "\n"
    )


# ── Stop ──────────────────────────────────────────────────────────────────────

def do_stop():
    stopped = 0

    # Kill by saved PIDs
    pids = load_pids()
    if pids:
        for name, pid in pids.items():
            if kill_pid(pid, name):
                stopped += 1
        PIDS_FILE.unlink(missing_ok=True)

    # Fallback: kill any stray processes
    for script in ("src/main_loop.py", "src/dashboard.py"):
        kill_by_script(script)

    if stopped == 0:
        print("  No processes were running.")
    else:
        print("Done.")


# ── Start ─────────────────────────────────────────────────────────────────────

def do_start():
    # ── Check .env exists ─────────────────────────────────────────────────────
    if not ENV_FILE.exists():
        if ENV_EXAMPLE.exists():
            print("ERROR: .env file not found.")
            print(f"       Copy .env.example to .env and fill in your API keys:")
            print(f"       Windows: copy .env.example .env")
            print(f"       Mac/Linux: cp .env.example .env")
        else:
            print("ERROR: .env file not found. Create it with your API keys.")
        sys.exit(1)

    # ── Validate required env vars ────────────────────────────────────────────
    env = dotenv_values(ENV_FILE)
    missing = [v for v in REQUIRED_VARS if not env.get(v)]
    if missing:
        print(f"ERROR: The following .env variables are empty: {', '.join(missing)}")
        print("       Edit .env and fill in all values, then run again.")
        sys.exit(1)

    # ── Find Python ───────────────────────────────────────────────────────────
    py = find_python()
    if py is None:
        print("ERROR: Cannot find a Python interpreter with project dependencies.")
        print("       Run:  python setup.py   (or: pip install -r requirements.txt)")
        sys.exit(1)

    # ── Kill stale processes (always, not just when .pids exists) ────────────
    print("Clearing any stale processes...")
    do_stop()           # kills by .pids if present + stray script processes
    kill_port(5050)     # force-kill anything still on the port (e.g. after crash)
    time.sleep(0.5)

    # ── Clear logs and temporary state ────────────────────────────────────────
    LOGS_DIR.mkdir(exist_ok=True)
    for log_file in LOGS_DIR.glob("*.log"):
        try:
            log_file.unlink()
        except Exception:
            pass
    
    state_file = ROOT / "state.json"
    if state_file.exists():
        try:
            state_file.unlink()
        except Exception:
            pass

    # Set PYTHONPATH so src/ modules can import each other
    env_with_path = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    # Also inject .env vars so child processes can read them
    env_with_path.update(env)

    # ── Start dashboard ───────────────────────────────────────────────────────
    print("Starting dashboard...", end=" ", flush=True)
    dash_log = open(LOGS_DIR / "dashboard.log", "a")
    dash_proc = subprocess.Popen(
        [py, str(ROOT / "src" / "dashboard.py")],
        cwd=str(ROOT / "src"),
        stdout=dash_log,
        stderr=dash_log,
        env=env_with_path,
    )
    time.sleep(1.5)  # Give Flask a moment to bind the port

    if dash_proc.poll() is not None:
        print("FAILED")
        print("ERROR: Dashboard failed to start. Check logs/dashboard.log")
        sys.exit(1)
    print(f"OK (PID {dash_proc.pid})")

    # ── Start main loop ───────────────────────────────────────────────────────
    print("Starting main loop...", end=" ", flush=True)
    loop_log = open(LOGS_DIR / "main_loop.log", "a")
    loop_proc = subprocess.Popen(
        [py, str(ROOT / "src" / "main_loop.py")],
        cwd=str(ROOT / "src"),
        stdout=loop_log,
        stderr=loop_log,
        env=env_with_path,
    )
    time.sleep(0.5)

    if loop_proc.poll() is not None:
        print("FAILED")
        print("ERROR: Main loop failed to start. Check logs/main_loop.log")
        kill_pid(dash_proc.pid, "dashboard")
        sys.exit(1)
    print(f"OK (PID {loop_proc.pid})")

    # ── Save PIDs ─────────────────────────────────────────────────────────────
    save_pids({"dashboard": dash_proc.pid, "main_loop": loop_proc.pid})

    print()
    print("System running.")
    print(f"  Dashboard : {DASHBOARD_URL}")
    print(f"  Logs      : logs/dashboard.log  |  logs/main_loop.log")
    print(f"  Stop      : python stop.py   (or Ctrl+C here)")
    print()

    # ── Open browser ──────────────────────────────────────────────────────────
    time.sleep(0.5)
    webbrowser.open(DASHBOARD_URL)

    # ── Wait and handle Ctrl+C ────────────────────────────────────────────────
    def _shutdown(sig=None, frame=None):
        print("\nShutting down...")
        do_stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    if not IS_WINDOWS:
        signal.signal(signal.SIGTERM, _shutdown)

    try:
        # Poll until both children exit (they normally run forever)
        while True:
            if dash_proc.poll() is not None or loop_proc.poll() is not None:
                print("\nA process exited unexpectedly. Shutting down...")
                _shutdown()
            time.sleep(1)
    except KeyboardInterrupt:
        _shutdown()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MindTune-OS launcher")
    parser.add_argument("--stop", action="store_true", help="Stop running processes")
    args = parser.parse_args()

    os.chdir(ROOT)

    if args.stop:
        do_stop()
    else:
        do_start()

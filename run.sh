#!/usr/bin/env bash
# run.sh — start dashboard.py + main_loop.py as background processes,
#           write PIDs to .pids, open browser, trap Ctrl+C for clean shutdown.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Load credentials ──────────────────────────────────────────────────────────
# shellcheck source=.env
set +u
source .env 2>/dev/null || true
set -u

# Validate required keys are non-empty
missing=()
for var in SPOTIFY_CLIENT_ID SPOTIFY_CLIENT_SECRET SPOTIFY_REDIRECT_URI GROQ_API_KEY; do
    val="${!var:-}"
    if [ -z "$val" ]; then
        missing+=("$var")
    fi
done

if [ "${#missing[@]}" -gt 0 ]; then
    echo "ERROR: The following .env variables are empty: ${missing[*]}"
    echo "       Edit .env and fill in all values, then run again."
    exit 1
fi

# ── Detect the right Python interpreter ──────────────────────────────────────
# Prefers "python" (Anaconda/conda) over "python3" (system), using joblib
# import as the test — matching the same logic in the Makefile.
if python -c "import joblib" 2>/dev/null; then
    PY=python
elif python3 -c "import joblib" 2>/dev/null; then
    PY=python3
else
    echo "ERROR: Cannot find a Python interpreter with project dependencies."
    echo "       Run: make setup  (or: pip install -r requirements.txt)"
    exit 1
fi

# ── Kill stale processes from a previous run ──────────────────────────────────
if [ -f .pids ]; then
    echo "→ Stopping previous session..."
    bash stop.sh || true
fi

mkdir -p logs

# ── Start dashboard ───────────────────────────────────────────────────────────
echo "→ Starting dashboard..."
$PY src/dashboard.py >> logs/dashboard.log 2>&1 &
DASH_PID=$!
echo "dashboard=$DASH_PID" > .pids

# Give Flask a moment to bind the port
sleep 1

if ! kill -0 "$DASH_PID" 2>/dev/null; then
    echo "ERROR: Dashboard failed to start."
    echo "       Check logs/dashboard.log for details."
    rm -f .pids
    exit 1
fi

# ── Start main loop ───────────────────────────────────────────────────────────
echo "→ Starting main loop..."
$PY src/main_loop.py >> logs/main_loop.log 2>&1 &
LOOP_PID=$!
echo "main_loop=$LOOP_PID" >> .pids

echo ""
echo "✓ System running"
echo "  Dashboard : http://127.0.0.1:5050"
echo "  Logs      : tail -f logs/main_loop.log logs/dashboard.log"
echo "  Stop      : make stop   (or Ctrl+C here)"
echo ""

# ── Open browser ──────────────────────────────────────────────────────────────
if command -v open &>/dev/null; then
    open "http://127.0.0.1:5050"
elif command -v xdg-open &>/dev/null; then
    xdg-open "http://127.0.0.1:5050"
fi

# ── Trap Ctrl+C / SIGTERM → clean shutdown ────────────────────────────────────
cleanup() {
    echo ""
    echo "→ Shutting down..."
    bash stop.sh
    exit 0
}
trap cleanup INT TERM

# Keep the shell alive so the trap can fire
wait "$LOOP_PID" "$DASH_PID" 2>/dev/null || true

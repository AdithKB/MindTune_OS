#!/usr/bin/env bash
# stop.sh — kill main_loop.py and dashboard.py
# Kills by saved PID first, then by process name as a fallback so that
# manually-started processes (no .pids file) are also cleaned up.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

stopped=0

# ── Kill by saved PIDs ────────────────────────────────────────────────────────
if [ -f .pids ]; then
    while IFS='=' read -r name pid; do
        [ -z "$pid" ] && continue
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "→ Stopped $name (PID $pid)"
            stopped=$((stopped + 1))
        fi
    done < .pids
    rm -f .pids
fi

# ── Kill any stray processes started manually (not via make run) ──────────────
for script in src/main_loop.py src/dashboard.py; do
    pids=$(pgrep -f "$script" 2>/dev/null || true)
    for pid in $pids; do
        kill "$pid" 2>/dev/null && \
            echo "→ Stopped stray $script (PID $pid)" && \
            stopped=$((stopped + 1))
    done
done

if [ "$stopped" -eq 0 ]; then
    echo "→ No processes were running"
else
    echo "✓ Done"
fi

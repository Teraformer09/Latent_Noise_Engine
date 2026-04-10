#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Latent Noise Engine — Frontend Launcher
# Usage:
#   bash frontend/launch.sh          # start the dashboard
#   bash frontend/launch.sh test     # run full test suite
#   bash frontend/launch.sh test --fast    # fast test mode
#   bash frontend/launch.sh test --verbose # verbose test output
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

cd "$REPO_DIR"

if [[ "$1" == "test" ]]; then
    shift
    echo "Running test suite..."
    python3 frontend/tests/run_tests.py "$@"
else
    echo "Starting Latent Noise Engine Frontend..."
    echo "Open http://localhost:5006 in your browser"
    python3 frontend/app.py
fi

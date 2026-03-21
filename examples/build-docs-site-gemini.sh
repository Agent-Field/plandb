#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# PlanDB Example: Build Documentation Site (Gemini CLI)
#
# Gemini CLI is a single-agent worker like Codex. It uses plandb
# go/done sequentially. YOLO mode auto-approves all actions.
#
# Usage:
#   ./examples/build-docs-site-gemini.sh             # interactive
#   ./examples/build-docs-site-gemini.sh --headless   # non-interactive
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLANDB_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_DIR="$PLANDB_ROOT/experiments/docs-site-gemini"

command -v plandb >/dev/null 2>&1 || { echo "error: plandb not found. Run: cargo install --path $PLANDB_ROOT"; exit 1; }
command -v gemini >/dev/null 2>&1 || { echo "error: gemini not found. Install: https://github.com/google-gemini/gemini-cli"; exit 1; }

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PlanDB Example: Docs Site (Gemini CLI)                     ║"
echo "║  Gemini uses PlanDB to plan, decompose, and execute.        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Load shared prompt
SHARED_PROMPT=$(cat "$SCRIPT_DIR/PLANDB_PROMPT.md")

PROMPT="${SHARED_PROMPT}

The environment variable PLANDB_DB is already set.
Start by running plandb init, then decompose the work into tasks and execute them.
Use plandb go / plandb done --next as your work loop."

export PLANDB_DB="$WORK_DIR/.plandb.db"
MODE="${1:---interactive}"

cd "$WORK_DIR"

if [[ "$MODE" == "--headless" ]]; then
  echo "Running in headless mode..."
  gemini -y -p "$PROMPT"
else
  echo "Launching Gemini interactive session..."
  echo ""
  gemini -y "$PROMPT"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Build Complete                                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
plandb status --full 2>/dev/null || echo "(no plandb project found)"
echo ""
echo "Output: $WORK_DIR"
echo "Serve:  cd $WORK_DIR && python3 -m http.server 8080"

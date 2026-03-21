#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# PlanDB Example: Build Documentation Site (Codex CLI)
#
# Codex is a single-agent worker — it uses plandb go/done sequentially.
# The task graph still enables parallelism for multi-agent setups.
#
# Usage:
#   ./examples/build-docs-site.sh          # interactive (see the TUI)
#   ./examples/build-docs-site.sh --exec   # non-interactive (headless)
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLANDB_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_DIR="$PLANDB_ROOT/experiments/docs-site-codex"

command -v plandb >/dev/null 2>&1 || { echo "error: plandb not found. Run: cargo install --path $PLANDB_ROOT"; exit 1; }
command -v codex  >/dev/null 2>&1 || { echo "error: codex not found. Install: https://github.com/openai/codex"; exit 1; }

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PlanDB Example: Docs Site (Codex)                          ║"
echo "║  Codex uses PlanDB to plan, decompose, and execute.         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Load shared prompt
SHARED_PROMPT=$(cat "$SCRIPT_DIR/PLANDB_PROMPT.md")

PROMPT="${SHARED_PROMPT}

The environment variable PLANDB_DB is already set.
Start by running plandb init, then decompose the work into tasks and execute them."

export PLANDB_DB="$WORK_DIR/.plandb.db"
MODE="${1:---interactive}"

if [[ "$MODE" == "--exec" ]]; then
  echo "Running in non-interactive mode..."
  codex exec --full-auto -C "$WORK_DIR" "$PROMPT"
else
  echo "Launching Codex interactive session..."
  echo ""
  codex --full-auto -C "$WORK_DIR" "$PROMPT"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Build Complete                                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
plandb status --full 2>/dev/null || echo "(no plandb project found)"
echo ""
echo "Output: $WORK_DIR"
echo "Serve:  cd $WORK_DIR && python3 -m http.server 8080"

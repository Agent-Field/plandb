#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# PlanDB Example: Build Documentation Site (Claude Code)
#
# Claude Code CAN spawn parallel sub-agents via the Agent tool.
# The prompt instructs it to check plandb list --status ready and
# dispatch parallel agents for independent tasks.
#
# Usage:
#   ./examples/build-docs-site-claude.sh              # interactive TUI
#   ./examples/build-docs-site-claude.sh --headless    # non-interactive
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLANDB_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_DIR="$PLANDB_ROOT/experiments/docs-site-claude"

command -v plandb >/dev/null 2>&1 || { echo "error: plandb not found. Run: cargo install --path $PLANDB_ROOT"; exit 1; }
command -v claude >/dev/null 2>&1 || { echo "error: claude not found. Install: https://docs.anthropic.com/en/docs/claude-code"; exit 1; }

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PlanDB Example: Docs Site (Claude Code)                    ║"
echo "║  Claude uses PlanDB + parallel sub-agents.                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Load shared prompt + Claude-specific parallelization instructions
SHARED_PROMPT=$(cat "$SCRIPT_DIR/PLANDB_PROMPT.md")

SYSTEM_PROMPT="${SHARED_PROMPT}

## Claude Code: Parallel Execution (you MUST do this)

You have the Agent tool. After creating the task graph, follow this loop:

1. Run \`plandb list --status ready\` to find all tasks with no unmet dependencies
2. If multiple tasks are ready, spawn one Agent per ready task IN PARALLEL (single message, multiple Agent tool calls)
3. Each agent's prompt should include:
   - Run \`PLANDB_AGENT=worker-N plandb go\` to atomically claim a task
   - Run \`plandb show <task-id>\` to read the full description
   - Execute the work described in the description
   - Run \`PLANDB_AGENT=worker-N plandb done\` to complete
4. After all parallel agents finish, run \`plandb status --detail\` and repeat from step 1
5. Continue until \`plandb status\` shows 100%

DO NOT work tasks sequentially when they could be parallel. The graph tells you what is safe to parallelize."

export PLANDB_DB="$WORK_DIR/.plandb.db"
MODE="${1:---interactive}"

if [[ "$MODE" == "--headless" ]]; then
  echo "Running in headless mode..."
  claude -p \
    --allowedTools "Bash Agent Read Write Edit Grep Glob" \
    --permission-mode "auto" \
    --model sonnet \
    "$SYSTEM_PROMPT"
else
  echo "Launching Claude Code interactive session..."
  echo ""
  cd "$WORK_DIR"
  KICKOFF="Start now. Working directory: $(pwd). PLANDB_DB=$PLANDB_DB. Run plandb commands directly. Begin with plandb init, decompose into tasks with descriptions, then execute using parallel sub-agents for all ready tasks."
  claude \
    --permission-mode auto \
    --allowedTools "Bash Edit Write Read Grep Glob Agent" \
    --system-prompt "$SYSTEM_PROMPT" \
    "$KICKOFF"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Build Complete                                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
plandb status --full 2>/dev/null || echo "(no plandb project found)"
echo ""
echo "Output: $WORK_DIR"
echo "Serve:  cd $WORK_DIR && python3 -m http.server 8080"

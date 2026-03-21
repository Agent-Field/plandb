#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# PlanDB Example: Build Mini GPT from Scratch in Rust
#
# Claude Code builds a working transformer language model in pure Rust
# (no ML framework deps) using PlanDB for task orchestration.
#
# This is a hard test: genuine 3+ level hierarchy, cross-level deps,
# parallel implementation tracks, quality gates, mid-flight adaptation.
#
# Usage:
#   ./examples/build-mini-gpt-claude.sh             # interactive
#   ./examples/build-mini-gpt-claude.sh --headless   # non-interactive
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLANDB_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_DIR="$PLANDB_ROOT/experiments/mini-gpt-rust"

command -v plandb >/dev/null 2>&1 || { echo "error: plandb not found. Run: cargo install --path $PLANDB_ROOT"; exit 1; }
command -v claude >/dev/null 2>&1 || { echo "error: claude not found. Install: https://docs.anthropic.com/en/docs/claude-code"; exit 1; }

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PlanDB Example: Mini GPT in Rust (Claude Code)            ║"
echo "║  From-scratch transformer with PlanDB task orchestration.   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Load shared PlanDB reference
PLANDB_REF=$(cat "$SCRIPT_DIR/PLANDB_PROMPT.md")

SYSTEM_PROMPT="${PLANDB_REF}

## Claude Code: Parallel Execution

You have the Agent tool. After creating the task graph:

1. Run \`plandb list --status ready\` to find all parallelizable tasks
2. If multiple are ready, spawn one Agent per task IN PARALLEL (single message, multiple Agent tool calls)
3. Each agent: \`PLANDB_AGENT=worker-N plandb go\` → \`plandb show <id>\` → implement → \`PLANDB_AGENT=worker-N plandb done\`
4. After agents finish, \`plandb status --detail\` + \`plandb critical-path\` → repeat
5. Continue until 100%

DO NOT work sequentially when tasks could be parallel."

export PLANDB_DB="$WORK_DIR/.plandb.db"
MODE="${1:---interactive}"

KICKOFF=$(cat <<KICKOFF_EOF
## Task

Build a working GPT-style transformer language model from scratch in Rust.

**The goal**: \`cargo run\` trains on text and generates coherent English output. That's it.

**Constraints**:
- Pure Rust. No ML framework crates (no tch, candle, burn, ndarray). \`rand\` is fine.
- Single binary. Training data bundled in the source (Shakespeare or similar public domain).
- Must finish training in under 2 minutes on a laptop.
- Smallest, cleanest implementation you can design. Minimize code, maximize clarity.

**Exit condition**: You are done when \`cargo run\` produces generated text with recognizable English words and patterns. Not when the code is written — when it WORKS. If it fails, debug and fix. Keep iterating.

**Using PlanDB**: You are in $(pwd). PLANDB_DB=$PLANDB_DB is set. Use plandb to plan, decompose, and track this project. Design the architecture and decomposition yourself — figure out the most efficient approach, what can be parallelized, what's on the critical path.

Start now.
KICKOFF_EOF
)

if [[ "$MODE" == "--headless" ]]; then
  echo "Running in headless mode..."
  claude -p \
    --allowedTools "Bash Agent Read Write Edit Grep Glob" \
    --permission-mode "auto" \
    --model sonnet \
    --system-prompt "$SYSTEM_PROMPT" \
    "$KICKOFF"
else
  echo "Launching Claude Code interactive session..."
  echo ""
  cd "$WORK_DIR"
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
echo "Run:    cd $WORK_DIR && cargo run"

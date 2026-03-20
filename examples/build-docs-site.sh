#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# PlanDB Example: Build Documentation Site + Interactive Playground
#
# Gives Codex CLI a task and access to plandb. Codex decides on its own
# how to decompose the work, what tasks to create, when to split, etc.
# We provide zero pre-built task graph — that's the whole point.
#
# Usage:
#   ./examples/build-docs-site.sh          # interactive (see the TUI)
#   ./examples/build-docs-site.sh --exec   # non-interactive (headless)
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLANDB_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_DIR="$PLANDB_ROOT/docs-site"

# Check dependencies
command -v plandb >/dev/null 2>&1 || { echo "error: plandb not found in PATH. Run: cargo install --path $PLANDB_ROOT"; exit 1; }
command -v codex  >/dev/null 2>&1 || { echo "error: codex not found in PATH. Install: https://github.com/openai/codex"; exit 1; }

# Clean slate
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PlanDB Example: Docs Site + Playground                     ║"
echo "║  Codex will use PlanDB to plan and build the project.       ║"
echo "║  We give it zero pre-built tasks — it figures it out.       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Build the prompt ─────────────────────────────────────────────────
# Just the task + plandb reference. No pre-seeded graph.

PROMPT=$(cat <<'PROMPT_EOF'
## Task

Build a documentation site and interactive playground for PlanDB — a CLI task graph tool for AI agents.

Deliverables (all as static HTML/CSS/JS in the current directory):

1. **Landing page** — hero section, what PlanDB is, why it exists (compound graph for AI agent task orchestration)
2. **Getting Started guide** — walkthrough of: init → add tasks with deps → go → done --next → split → status
3. **CLI Reference** — run `plandb --help`, `plandb task --help`, `plandb split --help` etc. and build a complete reference page from the actual output
4. **Architecture page** — explain the compound graph model (place graph for containment + link graph for dependencies, orthogonal structures)
5. **Interactive playground** — a terminal-like UI in the browser where users can type PlanDB commands and see simulated output. Include a guided tutorial mode that walks through the core workflow step by step. Should feel like a real terminal.
6. **Navigation + polish** — consistent styling, responsive layout, all pages linked, deploy-ready for GitHub Pages

Design: clean, minimal, professional. No heavy JS frameworks — vanilla HTML/CSS/JS or a tiny bundler at most.

## Tool: PlanDB

You have `plandb` installed. Use it to plan and track your own work on this project. This is the demo — the tool eating its own dogfood.

Quick reference:
```
plandb init "project-name"              # create a project
plandb add "task title"                 # add a task
plandb add "task" --dep t-xxx           # add with dependency
plandb add "task" --as my-id            # custom ID → t-my-id
plandb go                               # claim next ready task
plandb done --next                      # complete current + claim next
plandb split --into "A, B, C"           # split current task into independent parts
plandb split --into "A > B > C"         # split into dependency chain
plandb status --detail                  # see the full task graph
plandb show <task-id>                   # see task details
```

Start by running `plandb init` and decomposing the work into tasks with dependencies.
Then use `plandb go` / `plandb done --next` as you work through them.
Split tasks that turn out to be complex. Check `plandb status --detail` periodically.

The environment variable PLANDB_DB is already set to point to the right database.
PROMPT_EOF
)

# ── Launch Codex ─────────────────────────────────────────────────────
MODE="${1:---interactive}"

export PLANDB_DB="$WORK_DIR/.plandb.db"

if [[ "$MODE" == "--exec" ]]; then
  echo "Running in non-interactive (exec) mode..."
  codex exec --full-auto -C "$WORK_DIR" "$PROMPT"
else
  echo "Launching interactive Codex session..."
  echo "(Watch Codex use PlanDB to plan, decompose, and execute)"
  echo ""
  codex --full-auto -C "$WORK_DIR" "$PROMPT"
fi

# ── Show final status ────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Build Complete                                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
plandb status --detail 2>/dev/null || echo "(no plandb project found)"
echo ""
echo "Output: $WORK_DIR"
echo "Serve:  cd $WORK_DIR && python3 -m http.server 8080"

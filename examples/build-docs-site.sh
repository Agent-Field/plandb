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

You have `plandb` installed. Use it to plan and track your own work on this project.

Quick reference:
```
plandb init "project-name"                                  # create a project
plandb add "short title" --description "detailed spec..."   # add a task WITH description
plandb add "task" --dep t-xxx                               # add with dependency
plandb add "task" --as my-id                                # custom ID → t-my-id
plandb add "task" --kind code                               # kinds: generic, code, research, review, test, shell (NO other values)
plandb go                                                    # claim next ready task
plandb done --next                                           # complete current + claim next
plandb split --into "A, B, C"                                # split into independent parts
plandb split --into "A > B > C"                              # split into dependency chain
plandb status --detail                                       # see the full task graph
plandb show <task-id>                                        # see task details + description
plandb task add-dep --after t-upstream t-downstream          # add dependency AFTER creation (--after = must complete first)
```

IMPORTANT constraints:
- --kind ONLY accepts: generic, code, research, review, test, shell. No other values (not "docs", not "design", etc). Use "generic" if unsure.
- --dep references must point to task IDs that already exist. Add tasks in dependency order (create upstream tasks first).
- To add a dependency after both tasks exist: `plandb task add-dep --after t-upstream t-downstream` (NOT positional — use --after flag).
- --description should be a single string (quote it). Newlines inside are fine.

## CRITICAL: How to create tasks

Every task MUST have a --description that contains the full specification of the work.
The title is a short label. The description is the actual prompt — detailed enough that
you (or a sub-agent) can pick up the task later with `plandb go` and `plandb show <id>`
and know EXACTLY what to build without any other context.

Think of each task description as a self-contained work order:
- What files to create or modify
- What the output should look like
- Acceptance criteria
- Any technical constraints or decisions
- References to upstream task outputs if relevant

Example:
```
plandb add "Build landing page" --as landing --kind code \
  --description "Create index.html with:
- Hero section: h1 'PlanDB', tagline 'Task graph primitive for AI agents', brief description
- Feature highlights: compound graph model, recursive decomposition, zero-friction CLI, multi-agent support
- Code snippet showing the 2-command core loop (plandb go / plandb done --next)
- Call-to-action linking to getting-started.html
- Use shared styles from styles.css (created by design task)
- Responsive layout, works on mobile
- No JS frameworks, vanilla HTML/CSS only"
```

## Workflow

1. Run `plandb init` to create the project
2. Decompose ALL the work upfront into tasks with dependencies and detailed descriptions
3. Use `plandb status --detail` to verify the graph looks right
4. Work through tasks: `plandb go` → read description with `plandb show <id>` → do the work → `plandb done --next`
5. If a task turns out to be complex while working on it, split it: `plandb split --into "Part A, Part B"`
6. When splitting, each new subtask also needs a proper description — use `plandb show` after split to verify

If you could spawn sub-agents, each would claim a task with `plandb go`, read its
full description, execute it independently, and complete with `plandb done --next`.
The descriptions must be complete enough for that — no implicit context.

The environment variable PLANDB_DB is already set.
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

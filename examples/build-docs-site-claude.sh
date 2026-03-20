#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# PlanDB Example: Build Documentation Site + Interactive Playground
# Using Claude Code with parallel sub-agents via the Agent tool
#
# Unlike the Codex version, Claude Code CAN spawn parallel sub-agents.
# The prompt instructs it to check `plandb list --status ready` and
# dispatch parallel Agent tool calls for independent tasks.
#
# Usage:
#   ./examples/build-docs-site-claude.sh              # interactive TUI
#   ./examples/build-docs-site-claude.sh --headless    # non-interactive (print mode)
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLANDB_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_DIR="$PLANDB_ROOT/docs-site-claude"

# Check dependencies
command -v plandb >/dev/null 2>&1 || { echo "error: plandb not found in PATH. Run: cargo install --path $PLANDB_ROOT"; exit 1; }
command -v claude >/dev/null 2>&1 || { echo "error: claude not found in PATH. Install: https://docs.anthropic.com/en/docs/claude-code"; exit 1; }

# Clean slate
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PlanDB Example: Docs Site + Playground (Claude Code)       ║"
echo "║  Claude will use PlanDB + parallel sub-agents.              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Build the prompt ─────────────────────────────────────────────────

PROMPT=$(cat <<'PROMPT_EOF'
## Task

Build a documentation site and interactive playground for PlanDB — a CLI task graph tool for AI agents.

Deliverables (all as static HTML/CSS/JS files):

1. **Landing page** — hero section, what PlanDB is, why it exists (compound graph for AI agent task orchestration)
2. **Getting Started guide** — walkthrough of: init → add tasks with deps → go → done --next → split → status
3. **CLI Reference** — run `plandb --help`, `plandb task --help`, `plandb split --help` etc. and build a complete reference page from the actual output
4. **Architecture page** — explain the compound graph model (place graph for containment + link graph for dependencies, orthogonal structures)
5. **Interactive playground** — a terminal-like UI in the browser where users can type PlanDB commands and see simulated output. Include a guided tutorial mode. Should feel like a real terminal.
6. **Navigation + polish** — consistent styling, responsive layout, all pages linked, deploy-ready for GitHub Pages

Design: clean, minimal, professional. No heavy JS frameworks — vanilla HTML/CSS/JS only.

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
plandb list --status ready                                   # list all tasks that can run NOW
plandb task add-dep --after t-upstream t-downstream          # add dependency AFTER creation
```

IMPORTANT constraints:
- --kind ONLY accepts: generic, code, research, review, test, shell. No other values. Use "generic" if unsure.
- --dep references must point to task IDs that already exist. Create upstream tasks first.
- To add a dependency after both tasks exist: `plandb task add-dep --after t-upstream t-downstream` (use --after flag, NOT positional).
- --description should be a single string (quote it). Newlines inside are fine.

## CRITICAL: How to create tasks

Every task MUST have a --description that contains the full specification of the work.
The title is a short label. The description is the actual prompt — detailed enough that
a sub-agent can pick up the task with `plandb go` + `plandb show <id>` and know EXACTLY
what to build without any other context.

Each description is a self-contained work order:
- What files to create or modify
- What the output should look like
- Acceptance criteria
- Technical constraints
- References to upstream task outputs if relevant

Example:
```
plandb add "Build landing page" --as landing --kind code \
  --description "Create index.html with:
- Hero section: h1 'PlanDB', tagline 'Task graph primitive for AI agents'
- Feature highlights: compound graph, recursive decomposition, zero-friction CLI
- Code snippet showing the core loop (plandb go / plandb done --next)
- Call-to-action linking to getting-started.html
- Use shared styles from styles.css (created by t-foundation)
- Responsive layout, vanilla HTML/CSS only
- Output: index.html"
```

## Workflow

1. Run `plandb init` to create the project
2. Decompose ALL the work upfront into tasks with dependencies and detailed descriptions
3. Use `plandb status --detail` to verify the graph looks right
4. Work through tasks using the parallelization strategy below

## PARALLELIZATION (you MUST do this)

You have the Agent tool and can spawn sub-agents. USE THEM for parallel execution.

After creating the task graph, follow this loop:

1. Run `plandb list --status ready` to find all tasks with no unmet dependencies
2. If multiple tasks are ready, spawn one Agent per ready task IN PARALLEL (single message, multiple Agent tool calls)
3. Each agent's prompt should be:
   - Run `PLANDB_AGENT=worker-N plandb go` to atomically claim a task
   - Run `plandb show <task-id>` to read the full description
   - Execute the work described
   - Run `PLANDB_AGENT=worker-N plandb done` to complete
4. After all parallel agents finish, run `plandb status --detail` and repeat from step 1
5. Continue until `plandb status` shows 100%

PlanDB handles coordination: atomic claiming prevents double-assignment, dependencies
are enforced automatically. The graph tells you exactly what is safe to run concurrently.

DO NOT work tasks sequentially when they could be parallel. Check ready tasks and dispatch.

The environment variable PLANDB_DB is already set.
PROMPT_EOF
)

# ── Launch Claude Code ───────────────────────────────────────────────
MODE="${1:---interactive}"

export PLANDB_DB="$WORK_DIR/.plandb.db"

if [[ "$MODE" == "--headless" ]]; then
  echo "Running in headless (print) mode..."
  claude -p \
    --allowedTools "Bash Agent Read Write Edit Grep Glob" \
    --permission-mode "auto" \
    --model sonnet \
    "$PROMPT"
else
  echo "Launching interactive Claude Code session..."
  echo "(Watch Claude use PlanDB with parallel sub-agents)"
  echo ""
  cd "$WORK_DIR"
  claude \
    --permission-mode auto \
    --allowedTools "Bash Edit Write Read Grep Glob Agent" \
    "$PROMPT"
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

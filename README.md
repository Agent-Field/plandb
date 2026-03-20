# PlanDB

Task graph primitive for AI agents. Compound graph (recursive hierarchy + cross-level dependencies) in SQLite with CLI, MCP, and HTTP interfaces.

## Install

```bash
cargo install --path .
```

## Quick Start

```bash
plandb init "my-project"
plandb add "Design the API"
plandb add "Implement backend" --dep t-abc
plandb add "Write tests" --dep t-abc
plandb go                    # claim next ready task
# ... do work ...
plandb done --next           # complete + claim next
```

## Core Loop

Two commands. No IDs to remember. No flags required.

```bash
plandb go          # claim + start next ready task
plandb done --next # complete current + claim next
```

- `done` without a task ID completes your current running task
- `go` delivers upstream context automatically (results from completed dependencies)
- Agent identity defaults to `"default"` — set `PLANDB_AGENT` env var for multi-agent

## Adding Tasks

```bash
plandb add "Task title"                          # positional title
plandb add "Task title" --dep t-abc              # with dependency
plandb add "Task title" --as api                 # custom ID → t-api
plandb add "Task title" --kind code --priority 5 # with metadata
plandb add "Task title" --dep t-abc:blocks       # dep type: feeds_into|blocks|suggests
plandb add "Task title" --tag backend --tag auth # with tags
```

## Decomposition

Split any task into subtasks. Works at any depth (recursive — subtasks can be split further).

```bash
# Comma-separated (independent subtasks)
plandb split t-abc --into "Design, Implement, Test"

# Chain with > (linear dependencies: Design → Implement → Test)
plandb split t-abc --into "Design > Implement > Test"

# Omit task ID to split your current running task
plandb split --into "Part A, Part B"

# JSON for full control
plandb task split t-abc --into '[{"title":"A","deps_on":[]},{"title":"B","deps_on":["A"]}]'

# From YAML file
plandb task decompose t-abc --file subtasks.yaml

# Cancel pending subtasks and recreate
plandb task replan t-abc --file revised.yaml
```

Composite tasks auto-complete when all children finish. This bubbles up recursively.

## Scope

Zoom into a composite task to work within its subtree.

```bash
plandb use t-abc     # scope into composite task
plandb list          # shows children of t-abc only
plandb go            # claims from this scope
plandb add "Subtask" # creates as child of t-abc
plandb use ..        # zoom out one level
plandb use --clear   # back to project root
```

Cross-level dependencies work transparently — a subtask at any depth can depend on any other task.

## Status & Inspection

```bash
plandb status              # progress summary
plandb status --detail     # per-task breakdown
plandb list                # all tasks
plandb list --status ready # filter by status
plandb show t-abc          # full task details
plandb ahead               # what becomes ready next
```

## Plan Adaptation

```bash
plandb task insert --after t-a --before t-b --title "New step"  # insert between
plandb task amend t-abc --prepend "NOTE: edge case found"       # annotate future task
plandb task pivot t-abc --file new-plan.yaml                    # replace subtree
plandb what-if cancel t-abc                                     # preview effects
```

## Multi-Agent

```bash
# Set agent identity via env var
PLANDB_AGENT=worker-1 plandb go
PLANDB_AGENT=worker-1 plandb done --next

# Or use --agent flag
plandb go --agent worker-2
plandb done --next --agent worker-2
```

Atomic claiming prevents double-assignment. Dependencies enforced across agents.

## Batch Creation

```yaml
# tasks.yaml
tasks:
  - title: "Design API"
    kind: code
    priority: 10
  - title: "Implement"
    deps: [{ from: "Design API", kind: feeds_into }]
  - title: "Write tests"
    deps: [{ from: "Implement", kind: feeds_into }]
```

```bash
plandb task create-batch --file tasks.yaml
```

## Task States

```
pending → ready → claimed → running → done
                                    → failed
                                    → cancelled
```

Tasks become `ready` when all `feeds_into` and `blocks` dependencies complete. `suggests` dependencies don't block.

## Dependency Types

| Type | Meaning | Blocks? |
|------|---------|---------|
| `feeds_into` | Result data flows downstream (default) | Yes |
| `blocks` | Must complete first, no data flow | Yes |
| `suggests` | Nice to have first, doesn't block | No |

## IDs

Short IDs: `t-k3m` (tasks), `p-abc` (projects). Fuzzy-matched on typos.

Custom IDs: `plandb add "Design" --as design` → `t-design`

## Output Modes

```bash
plandb status              # human-readable (default)
plandb --json status       # structured JSON
plandb --json -c status    # compact JSON (optimized for LLM context)
```

## Interfaces

| Interface | Command | Use Case |
|-----------|---------|----------|
| CLI | `plandb <command>` | Direct agent use |
| MCP | `plandb mcp` | Claude Code, Cursor, Windsurf |
| HTTP | `plandb serve --port 8484` | Custom agents, webhooks |

Generate integration config: `plandb prompt --for mcp|cli|http`

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `PLANDB_DB` | SQLite database path | `.plandb.db` (walks up dirs) |
| `PLANDB_AGENT` | Agent identity | `default` |
| `NO_COLOR` | Disable colored output | unset |

## Architecture

PlanDB uses a **compound graph** model — two independent structures composed together:

- **Place graph** (containment): tasks can contain subtasks recursively, forming a forest
- **Link graph** (dependencies): DAG edges can cross containment boundaries freely

This is more general than a hypergraph — nesting and flow are orthogonal. A subtask at depth 3 can depend on a task at depth 0 in a different branch.

## License

Apache License 2.0

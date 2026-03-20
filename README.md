# PlanDB

Task graph primitive for AI agents. Compound graph (recursive hierarchy + cross-level dependencies) in SQLite with CLI, MCP, and HTTP interfaces.

## Install

```bash
cargo install --path .
```

## Quick Start

```bash
plandb init "my-project"
plandb add "Design the API" --as design --description "Define REST endpoints, auth strategy, response schemas"
plandb add "Implement backend" --dep t-design --description "Build Express server implementing the API spec from t-design"
plandb add "Write tests" --dep t-design --description "Integration tests for all endpoints defined in t-design"
plandb go                    # claim next ready task
plandb show t-design         # read the full description
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

Every task needs a `--description` — the detailed spec of what to do. The title is a short label. The description is the actual work order.

```bash
plandb add "Task title" --description "Full spec of what to build..."   # ALWAYS include description
plandb add "Task title" --dep t-abc                                     # with dependency (upstream must exist first)
plandb add "Task title" --as api                                        # custom ID → t-api
plandb add "Task title" --kind code                                     # kind: generic, code, research, review, test, shell
plandb add "Task title" --dep t-abc:blocks                              # dep type: feeds_into (default), blocks, suggests
plandb add "Task title" --tag backend --tag auth                        # with tags
```

### Writing Good Descriptions

Each description should be a self-contained work order — detailed enough that an agent can pick it up with `plandb go` + `plandb show <id>` and execute without any other context:

```bash
plandb add "Build landing page" --as landing --kind code \
  --description "Create index.html with:
- Hero section: h1 'PlanDB', tagline, brief description
- Feature highlights: compound graph, recursive decomposition, zero-friction CLI
- Code snippet showing the core loop (plandb go / plandb done --next)
- Call-to-action linking to getting-started.html
- Responsive layout, vanilla HTML/CSS only
- Output: index.html"
```

### Constraints

- `--kind` only accepts: `generic`, `code`, `research`, `review`, `test`, `shell`
- `--dep` references must point to task IDs that already exist — create upstream tasks first
- To add a dependency after both tasks exist: `plandb task add-dep --after t-upstream t-downstream`

## Decomposition

Split any task into subtasks. Works at any depth (recursive — subtasks can be split further).

```bash
# Comma-separated (independent subtasks)
plandb split t-abc --into "Design, Implement, Test"

# Chain with > (linear dependencies: Design → Implement → Test)
plandb split t-abc --into "Design > Implement > Test"

# Omit task ID to split your current running task
plandb split --into "Part A, Part B"

# From YAML file (full control)
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
plandb show t-abc          # full task details + description
plandb ahead               # what becomes ready next
```

## Plan Adaptation

```bash
plandb task insert --after t-a --before t-b --title "New step"   # insert between
plandb task amend t-abc --prepend "NOTE: edge case found"        # annotate future task
plandb task pivot t-abc --file new-plan.yaml                     # replace subtree
plandb task add-dep --after t-upstream t-downstream              # add dependency edge
plandb what-if cancel t-abc                                      # preview effects
```

## Parallelization

When `plandb list --status ready` returns multiple tasks, they have no unmet dependencies and can run concurrently. **If you can spawn sub-agents, you should.**

Each worker runs the same core loop independently — PlanDB coordinates via atomic claiming:

```bash
# Worker 1                                    # Worker 2
PLANDB_AGENT=w1 plandb go                     PLANDB_AGENT=w2 plandb go
# ... work ...                                # ... work ...
PLANDB_AGENT=w1 plandb done --next            PLANDB_AGENT=w2 plandb done --next
```

The graph tells you exactly what is safe to run concurrently. No two agents claim the same task. Dependencies are enforced automatically — downstream tasks become `ready` only when upstream completes.

## Batch Creation

```yaml
# tasks.yaml
tasks:
  - title: "Design API"
    kind: code
    priority: 10
    description: "Define REST endpoints, auth strategy, response format"
  - title: "Implement"
    description: "Build the server implementing the API spec"
    deps: [{ from: "Design API", kind: feeds_into }]
  - title: "Write tests"
    description: "Integration tests for all endpoints"
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

Short IDs: `t-k3m9` (tasks), `p-abcd` (projects). Fuzzy-matched on typos.

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

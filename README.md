# PlanDB

Task graph primitive for AI agents. Manages dependency-aware task graphs in SQLite with CLI, MCP, and HTTP interfaces.

## Install

```bash
cargo install --path .
```

Or download a prebuilt binary from [GitHub Releases](https://github.com/Agent-Field/plandb/releases).

## Quick Start

```bash
plandb init "my-project"
plandb add --title "Design the API"
plandb add --title "Implement backend" --dep t-<id>
plandb add --title "Write tests" --dep t-<id>
plandb go --agent worker-1        # claim next ready task
# ... do work ...
plandb done t-<id> --result "done" --next --agent worker-1  # complete + claim next
```

## Core Loop

Two commands. That's it.

```bash
plandb go --agent worker-1                                    # claim + start next ready task
plandb done t-<id> --result "implemented auth" --next --agent worker-1  # complete + claim next
```

`go` delivers upstream context automatically — results from completed dependencies.
`done --next` completes the task and claims the next ready one in a single call.

## Adding Tasks

```bash
plandb add --title "Task title"                                # minimal
plandb add --title "Task title" --dep t-abc                    # with dependency (feeds_into)
plandb add --title "Task title" --dep t-abc:blocks             # explicit dep type
plandb add --title "Task title" --kind code --priority 5       # with metadata
plandb add --title "Task title" --dep t-abc --dep t-def        # multiple dependencies
plandb add --title "Task title" --tag api --tag auth            # with tags
plandb add --title "Task title" --description "Full details"   # with description
```

Dependency types: `feeds_into` (default), `blocks`, `suggests`.

## Decomposition

Split a task into subtasks or decompose from a YAML file.

```bash
# Split with JSON spec
plandb task split t-abc --into '[{"title":"Design"},{"title":"Implement"},{"title":"Test"}]'

# Decompose from YAML file (full control over deps, kinds, priorities)
plandb task decompose t-abc --file subtasks.yaml

# Cancel pending subtasks and recreate from YAML
plandb task replan t-abc --file revised.yaml
```

## Status & Inspection

```bash
plandb status              # progress summary
plandb status --detail     # per-task breakdown with DAG tree
plandb status --full       # all tasks + dependency edges
plandb list                # all tasks
plandb list --status ready # filter by status
plandb show t-abc          # full task details
plandb ahead               # what becomes ready after current tasks complete
```

## Plan Adaptation

```bash
plandb task insert --after t-a --before t-b --title "New step"  # insert between
plandb task amend t-abc --prepend "NOTE: edge case found"       # annotate
plandb task pivot t-parent --file new-plan.yaml                 # replace subtree
plandb task pivot t-parent --keep-done --file plan.yaml         # replace, keep done tasks
plandb what-if cancel t-abc                                     # preview effects
```

Every mutation response includes effect analysis: which tasks got delayed, accelerated, or unblocked.

## Multi-Agent

```bash
# Agent 1
plandb go --agent worker-1
plandb done t-<id> --result "implemented" --next --agent worker-1

# Agent 2
plandb go --agent worker-2
plandb done t-<id> --result "tested" --next --agent worker-2
```

Atomic claiming prevents double-assignment. Dependencies enforced across agents.

## Default Project

```bash
plandb use p-abc       # set default project (avoids --project on every command)
plandb use             # show current default
plandb use --clear     # unset default
```

When only one project exists in the database, it is used automatically.

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

Tasks become `ready` when all `feeds_into` and `blocks` dependencies are `done`. `suggests` dependencies don't block.

## Dependency Types

| Type | Meaning | Blocks? |
|------|---------|---------|
| `feeds_into` | Result data flows downstream (default) | Yes |
| `blocks` | Must complete first, no data flow | Yes |
| `suggests` | Nice to have first, doesn't block | No |

## IDs

Short IDs: `t-a1b2c3d4` (tasks), `p-a1b2c3d4` (projects). Fuzzy-matched on typos.

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
| HTTP | `plandb serve --port 8080` | Custom agents, webhooks |

MCP config for Claude Code / Cursor / Windsurf:

```json
{
  "mcpServers": {
    "plandb": {
      "command": "plandb",
      "args": ["mcp"]
    }
  }
}
```

Generate integration prompts: `plandb prompt --for mcp|cli|http`

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `PLANDB_DB` | SQLite database path | `.plandb.db` (walks up dirs) |
| `NO_COLOR` | Disable colored output | unset |

## License

Apache License 2.0

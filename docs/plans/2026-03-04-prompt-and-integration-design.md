# Planq Prompt & Platform Integration — Design Spec

**Date**: 2026-03-04  
**Status**: Approved  
**Principle**: Behavioral prompt + self-generating integration snippets

---

## The Problem

Agents need to know Planq exists and how to use it. Most tools solve this with a giant system prompt listing every command. This wastes tokens and teaches commands, not workflow.

**Target**: A ~60 token behavioral prompt that teaches the agent a workflow pattern. The agent discovers specific commands via `--help` and MCP tool descriptions.

---

## Core Behavioral Prompt (~60 tokens)

```
Decompose work into a planq DAG before coding. Adapt the plan as you learn.
When multiple tasks are ready, work them in parallel.
  planq go          — claim next ready task  
  planq done --next — complete + claim next
  planq task insert — add a missed step
  planq ahead       — see what's coming
Run planq --help to discover more.
```

### Design Choices

- **4 commands shown**: the essential loop (go, done --next) + 2 adaptation commands (insert, ahead)
- **"Run planq --help"**: delegates discovery to the tool. No stale docs in the prompt.
- **"Adapt the plan as you learn"**: behavioral trigger for JIT planning
- **"When multiple tasks are ready, work them in parallel"**: parallelization hint without prescribing how
- **What's absent**: what-if, pivot, split, amend — agent discovers via --help when needed

---

## `planq prompt` Command

### Usage

```bash
planq prompt --for mcp     # Claude Code, Cursor, Windsurf, any MCP client
planq prompt --for cli     # Codex, Aider, any CLI-based agent
planq prompt --for http    # OpenRouter, custom agents, any HTTP client
planq prompt --list        # Show available platforms
```

### Output: MCP mode

```
# MCP Config (Claude Code, Cursor, Windsurf, or any MCP client):

{
  "mcpServers": {
    "planq": {
      "command": "planq",
      "args": ["mcp"]
    }
  }
}

# Add to project instructions (CLAUDE.md, .cursorrules, etc.):

Use planq to plan and track work. Decompose into tasks with dependencies before coding.
When multiple tasks are ready, work them in parallel.
- `planq go` — claim next ready task
- `planq done --next` — complete and advance
- Adapt: `planq task insert`, `planq ahead`, `planq what-if`
```

### Output: CLI mode

```
# System prompt / AGENTS.md addition:

You have planq for task DAG management (binary: planq).
Before coding: planq project create "name" then create tasks with --dep for dependencies.
Work loop: planq go --agent $AGENT → implement → planq done --next.
When parallel tasks are ready, spawn separate agents for each.
Adapt mid-flight: planq task insert, planq ahead, planq what-if.
planq --help for full reference.
```

### Output: HTTP mode

```
# Start the Planq server:

planq serve --port 8080

# System prompt addition:

You have a task management API at http://localhost:8080.
POST /projects — create project. POST /tasks — create tasks with deps.
POST /go — claim next ready task. POST /tasks/:id/done — complete.
POST /tasks/insert — add steps. GET /ahead — preview upcoming.
Decompose work into tasks with dependencies. Adapt as you learn.
When parallel tasks are ready, work them in parallel.
```

### Implementation Notes

- No DB access needed (pure output command)
- Handle in `main.rs` before `init_db`, alongside Mcp/Serve
- `--for` is required. `--list` shows available options.
- If `PLANQ_DB` is set, include the path in CLI mode output

---

## README Integration Section

Replace current "MCP Config" section with unified setup:

```markdown
## Setup

cargo install --path .

### MCP (Claude Code, Cursor, Windsurf)

    planq prompt --for mcp

### CLI (Codex, Aider)

    planq prompt --for cli

### HTTP (OpenRouter, custom agents)

    planq prompt --for http
```

12 lines. Setup is always: run `planq prompt`, paste the output.

---

## Token Budget

| Component | Tokens |
|-----------|--------|
| Core behavioral prompt | ~60 |
| MCP tool descriptions (auto-discovered) | 0 (free via MCP) |
| Full command reference | 0 (agent uses --help) |

Total agent overhead: **~60 tokens**.

# Planq

The task graph primitive for AI coding agents.

![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Rust](https://img.shields.io/badge/rust-stable-orange)
![Binary Size](https://img.shields.io/badge/binary-~2.6MB-informational)

`ai-agents` `task-management` `task-graph` `mcp` `rust` `sqlite` `orchestration` `ai-native` `jit-planning`

---

## The Problem

AI coding agents (Claude Code, Codex, Cursor, Aider) work on tasks sequentially. When a task is complex, the natural move is to break it into subtasks and run agents in parallel. But how do you coordinate them?

**The common workaround**: Create GitHub Issues for each subtask, define dependencies manually, dispatch agents to each issue, manage PRs, merge conflicts, repeat. GitHub becomes a state management proxy for something it was never designed for.

This works, but:
- **Agents become stateless** — each agent session starts fresh, can't see what other agents did
- **Dependencies are implicit** — nothing enforces "don't start B until A is done"
- **Plans can't change** — halfway through, the agent discovers the approach is wrong. Reorganizing 12 GitHub issues and their dependencies takes 3,000+ tokens
- **Token overhead** — issue descriptions, API responses, and PR metadata are optimized for humans, not 8K context windows
- **Single point of coupling** — offline? Rate-limited? Everything stops

## What Planq Does

Planq is one binary (~2.6MB) that manages a task dependency graph in SQLite. Three interfaces: CLI, MCP server, HTTP API. Local-first. No daemon. No external services.

```bash
planq project create "build-auth-system"

planq task create --project p-a1b2c3 --title "Design JWT schema"
planq task create --project p-a1b2c3 --title "Implement middleware" --dep t-d4e5f6
planq task create --project p-a1b2c3 --title "Write tests" --dep t-g7h8i9

planq go --agent claude-1          # claim next ready task
# ... agent works ...
planq done --next                  # complete + claim next
```

The entire agent work loop is two commands. The graph handles the rest.

## Why Not Just Let Each Agent Track Its Own Plan?

Every AI coding agent already has internal task management (todos, checklists, scratchpads). So why does Planq exist?

| Problem | Internal plan | Planq |
|---------|--------------|-------|
| **Multi-agent coordination** | Each agent has its own plan. They can't see each other's work, claim tasks, or avoid conflicts. | Shared task graph with atomic claim protocol. Agent 2 sees that Agent 1 already claimed the auth task. |
| **Session continuity** | Agent dies, context is lost. New session starts from scratch. | Graph persists in SQLite. New agent picks up where the last one stopped, with handoff context. |
| **Plan adaptation** | Changing a plan mid-execution means the agent rewrites its internal state. No way to preview consequences. | `planq what-if cancel t-abc1` shows ripple effects before committing. `planq task insert` rewires the graph atomically. |
| **Dependency enforcement** | Agent decides order by vibes. Might start the test task before the implementation task. | Topological ordering. `planq go` only returns tasks whose dependencies are complete. |
| **Parallelization** | One agent, sequential execution. | Graph shows 3 tasks are ready simultaneously. Harness spawns 3 agents. |
| **Observability** | "How far along is the agent?" — no answer until it finishes. | `planq status` at any time: `5/12 done, 2 running, 1 ready, 4 pending`. |

## What Makes It AI-Specific

These aren't features you'd build for human project managers. They exist because LLMs have specific constraints:

**Short IDs** — Task IDs are 8 characters (`t-a1b2c3`), not UUIDs. Every token matters when your context window is 8K-200K.

**Compact output** — Default responses are terse. `planq done` returns `{"id":"t-a1b2c3","status":"done","next":"t-d4e5f6"}`, not a 50-field JSON blob.

**Compound commands** — `planq go` claims the next ready task in one call. `planq done --next` completes the current task and claims the next. What takes 4-5 API calls with GitHub Issues takes 1.

**Fuzzy ID resolution** — Agent misspells a task ID? Planq suggests the closest match instead of erroring. LLMs make typos. The tool should handle it.

**`planq --help` as documentation** — No need to paste a command reference into the system prompt. The agent runs `--help` and discovers commands itself.

**Effect analysis on every mutation** — When an agent modifies the plan (insert a task, cancel a branch), the response includes what got delayed, what's now ready, and the new critical path. The agent never needs a separate "what happened?" call.

**Handoff protocol** — When Agent 1 finishes a task, its result is automatically available as context for Agent 2's downstream task. Knowledge propagates through the graph.

**JIT plan adaptation** — Plans change. Planq handles it in 1-3 commands:

```bash
planq ahead                          # see what's coming (lookahead buffer)
planq what-if cancel t-abc1          # preview effects before acting
planq task insert --after A --before B --title "Add auth"   # add a missed step
planq task amend t-xyz --prepend "NOTE: use JWT, not sessions"  # annotate future tasks
planq task pivot t-parent --keep-done --file new-plan.yaml  # replace a subtree
planq task split t-big --into '[...]'                       # decompose mid-execution
```

## Setup

```bash
cargo install --path .
```

Pick your platform:

```bash
planq prompt --for mcp     # Claude Code, Cursor, Windsurf
planq prompt --for cli     # Codex, Aider
planq prompt --for http    # OpenRouter, custom agents
```

Paste the output into your agent's config. That's the entire setup.

### MCP Config (for reference)

```json
{
  "mcpServers": {
    "planq": {
      "command": "planq",
      "args": ["mcp"]
    }
  }
}
```

Works in Claude Code, Cursor, Windsurf, or any MCP client.

### HTTP Mode

```bash
planq serve --port 8080
```

Full REST API + SSE event stream. Use from any language, any agent framework.

## How Agents Use It

### Single Agent

```
1. planq project create "my-project"
2. planq task create ... (define tasks + dependencies)
3. planq go --agent my-agent
4. ... work on the task ...
5. planq done --next
6. goto 4
```

### Multiple Agents in Parallel

```
Harness checks: planq status → "3 tasks ready"
Harness spawns 3 agents, each runs:
  planq go --agent agent-N     → gets a different ready task (atomic claim)
  ... work ...
  planq done --next            → completes, claims next ready task
```

The graph ensures no two agents claim the same task. When Agent 1 finishes task A, tasks that depended on A become ready for other agents to claim.

### Adapting the Plan Mid-Flight

Agent is on task 5 of 12 and discovers the approach is wrong:

```bash
planq what-if cancel t-old          # preview: "would delay 3 tasks, orphan 0"
planq task insert --after t-d4 --before t-e5 --title "Add auth layer"
# → response includes: effect.delayed, effect.ready_now, effect.critical_path, project_state
```

Every mutation response tells the agent exactly what changed in the graph. No guessing.

## Comparison

| | Planq | GitHub Issues | Linear | TaskMaster AI |
|---|---|---|---|---|
| Local-first | Yes | No | No | No |
| MCP server | Yes | No | No | Partial |
| CLI | Yes | Limited | Limited | Yes |
| HTTP API | Yes | Yes | Yes | No |
| Dependency graph | Yes | No | No | Partial |
| Token-optimized | Yes | No | No | Partial |
| JIT plan adaptation | Yes | No | No | No |
| Single binary | Yes | No | No | No |
| Multi-agent claim protocol | Yes | No | No | No |

## All Features

- **Three interfaces**: CLI, MCP server (stdio JSON-RPC), HTTP API (REST + SSE)
- **Dependency graph**: `feeds_into`, `blocks`, `validates`, `informs`
- **Claim protocol**: atomic claim + heartbeat + timeout reclaim
- **Compound commands**: `go`, `done --next`, `next --claim`
- **JIT planning**: `what-if`, `insert`, `ahead`, `amend`, `pivot`, `split`
- **Effect analysis**: every mutation returns delayed/accelerated/ready_now/critical_path
- **Short IDs**: 8-char task/project IDs
- **Compact output**: terse defaults for context windows
- **Handoff protocol**: results propagate to downstream tasks
- **Fuzzy ID resolution**: typo recovery + closest-match suggestions
- **File tracking**: attach files to tasks, detect conflicts between agents
- **Inter-agent signals**: notes and event stream for coordination
- **Pause/resume**: partial completion with progress tracking
- **Sticky project**: `planq use <id>` sets default, fewer flags per command
- **Progressive status**: one-liner, `--detail`, or `--full`

## Philosophy

Planq is a primitive, not a platform. It does one thing: coordinate dependent tasks across agents with minimal overhead. It doesn't schedule agents, doesn't pick models, doesn't manage git. It's the SQLite of agent task management — embed it, script it, build on it.

## License

Apache License 2.0

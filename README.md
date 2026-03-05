<h1 align="center">plandb</h1>

<p align="center">
<strong>SQLite for agent task management.</strong><br>
A single binary (~2.6 MB) that manages task dependency graphs for AI agents. Local-first. Zero config.
</p>

<p align="center">
  <a href="https://github.com/Agent-Field/plandb/actions/workflows/release.yml"><img src="https://github.com/Agent-Field/plandb/actions/workflows/release.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-stable-orange" alt="Rust"></a>
  <a href="https://github.com/Agent-Field/plandb/releases"><img src="https://img.shields.io/badge/binary-~2.6MB-informational" alt="Binary Size"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> · <a href="#install">Install</a> · <a href="#mcp-setup">MCP Setup</a> · <a href="#why-plandb">Why Plandb?</a> · <a href="#comparison">Comparison</a> · <a href="#all-features">Features</a> · <a href="#contributing">Contributing</a>
</p>

---

## Quick Start

Create a project and add tasks with dependencies:

```bash
plandb init "build-auth-system"
# created p-a1b2c3 (build-auth-system)
# next: plandb add --title "First task"
# tip:  start with 1-2 tasks. add more as you learn things.

plandb add --title "Design JWT schema"
plandb add --title "Implement middleware" --dep t-d4e5f6
plandb add --title "Write tests" --dep t-g7h8i9
```

Claim a task — plandb delivers upstream context automatically:

```bash
plandb go --agent claude-1
# → t-d4e5f6 "Implement middleware" [1/3 · 1 ready · 1 blocked]
#
# upstream:
#   t-a1b2c3 → "JWT schema: RS256, 15min access, 7d refresh"
#
# downstream: t-g7h8i9 "Write tests" (receives YOUR result)
```

Complete it — see what you unlocked and claim the next task:

```bash
plandb done t-d4e5f6 --result "middleware on /api/* routes" --next --agent claude-1
# ✓ t-d4e5f6 done → claimed t-g7h8i9 "Write tests" [2/3 · 1 ready]
#   upstream: t-d4e5f6 → middleware on /api/* routes
```

See the dependency graph:

```bash
plandb status --detail
# p-a1b2c3 build-auth-system: 2/3 done (66%)
# ✓ t-a1b2c3 Design JWT schema
#   └─✓ t-d4e5f6 Implement middleware
#     └─◉ t-g7h8i9 Write tests @claude-1
```

Two commands per task. Results flow through the graph — each task sees what its dependencies produced.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/Agent-Field/plandb/main/install.sh | sh
```

<details>
<summary>Other install methods</summary>

**Direct download:**
[GitHub Releases](https://github.com/Agent-Field/plandb/releases) — prebuilt binaries for Linux and macOS (x86_64 + ARM).

**From source:**
```bash
git clone https://github.com/Agent-Field/plandb.git && cd plandb
cargo install --path .
```

</details>

## MCP Setup

Works with Claude Code, Cursor, Windsurf, or any MCP client:

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

<details>
<summary>CLI and HTTP setup</summary>

```bash
plandb prompt --for mcp     # Claude Code, Cursor, Windsurf
plandb prompt --for cli     # Codex, Aider — paste into system prompt
plandb prompt --for http    # OpenRouter, custom agents
```

**HTTP mode:**
```bash
plandb serve --port 8080    # REST API + SSE event stream
```
</details>

---

## Why Plandb?

AI agents (Claude Code, Codex, Cursor, Aider, custom LLM pipelines) hit the same coordination wall: how do you manage subtasks with dependencies across sessions, agents, and time?

The common workaround — GitHub Issues, Notion boards, ad-hoc JSON files — was designed for humans, not agents:

- **Agents become stateless** — each session starts fresh, can't see what other agents did
- **Dependencies are implicit** — nothing enforces "don't start B until A is done"
- **Plans can't change** — reorganizing 12 GitHub issues mid-execution costs 3,000+ tokens
- **Token overhead** — API responses optimized for humans, not 8K context windows
- **Single point of coupling** — offline or rate-limited? Everything stops

Plandb is one binary (~2.6MB) that manages a task dependency graph in SQLite. Three interfaces: CLI, MCP server, HTTP API. No daemon. No external services.

### Why not just let each agent track its own plan?

Every AI coding agent already has internal task management. So why external state?

| Problem | Internal plan | Plandb |
|---------|--------------|-------|
| **Multi-agent coordination** | Each agent has its own plan. Can't see each other's work. | Shared graph with atomic claim protocol. No double-claims. |
| **Session continuity** | Agent dies, context is lost. | Graph persists. New agent picks up with handoff context. |
| **Plan adaptation** | Rewriting internal state. No consequence preview. | `what-if cancel` shows ripple effects. `task insert` rewires atomically. |
| **Dependency enforcement** | Agent decides order by vibes. | Topological ordering. `go` only returns tasks with all deps complete. |
| **Parallelization** | One agent, sequential. | Graph shows 3 tasks ready → harness spawns 3 agents. |
| **Observability** | No answer until it finishes. | `plandb status` anytime: `5/12 done, 2 running, 1 ready`. |

## How It Works

### The Agent Loop

```
1. plandb init "my-project"
2. plandb add --title "..." (define tasks + dependencies)
3. plandb go --agent my-agent       → shows upstream context
4. ... work on the task ...
5. plandb done --result "..." --next → shows unlocked tasks, claims next
6. goto 4
```

### Multiple Agents in Parallel

```
Harness checks: plandb status → "3 tasks ready"
Harness spawns 3 agents, each runs:
  plandb go --agent agent-N                → gets a different ready task (atomic claim)
  ... work ...
  plandb done ID --next --agent agent-N    → completes, claims next ready task
```

No two agents claim the same task. When Agent 1 finishes task A, tasks that depended on A become ready for other agents.

### Adapting the Plan Mid-Flight

Agent is on task 5 of 12 and discovers the approach is wrong:

```bash
plandb ahead                          # see what's coming (lookahead buffer)
plandb what-if cancel t-abc1          # preview: "would delay 3 tasks, orphan 0"
plandb task insert --after A --before B --title "Add auth"   # add a missed step
plandb task amend t-xyz --prepend "NOTE: use JWT, not sessions"  # annotate
plandb task pivot t-parent --keep-done --file new-plan.yaml  # replace a subtree
plandb task split t-big --into '[...]'                       # decompose mid-execution
```

Every mutation response includes `effect.delayed`, `effect.ready_now`, and `project_state`. No guessing.

## What Makes It AI-Specific

These aren't features you'd build for human project managers:

**Dynamic context delivery** — `go` shows upstream results from completed dependencies. `done` shows which tasks were unlocked and nudges the agent to build downstream connections. Plans evolve naturally.

**Short IDs** — 8-character task IDs (`t-a1b2c3`), not UUIDs. Every token matters.

**Compound commands** — `go` = claim + start. `done --next` = complete + claim next. What takes 4-5 API calls with GitHub Issues takes 1.

**Fuzzy ID resolution** — Agent misspells an ID? Plandb suggests the closest match. LLMs make typos.

**`--help` as documentation** — No system prompt needed. The agent runs `--help` and discovers commands itself.

**Effect analysis** — Every mutation returns what got delayed, what's now ready, and the new critical path.

**Handoff protocol** — Agent 1's result is automatically context for Agent 2's downstream task.

## Comparison

| | Plandb | GitHub Issues | Linear | TaskMaster AI |
|---|---|---|---|---|
| Local-first | Yes | No | No | No |
| MCP server | Yes | No | No | Partial |
| CLI + HTTP API | Yes | Limited | Limited | Partial |
| Dependency graph | Yes | No | No | Partial |
| Token-optimized | Yes | No | No | Partial |
| JIT plan adaptation | Yes | No | No | No |
| Single binary | Yes | No | No | No |
| Multi-agent claim | Yes | No | No | No |
| Dynamic context delivery | Yes | No | No | No |

## All Features

- **Three interfaces** — CLI, MCP server (stdio JSON-RPC), HTTP API (REST + SSE)
- **Dependency graph** — `feeds_into`, `blocks`, `validates`, `informs`
- **Claim protocol** — atomic claim + heartbeat + timeout reclaim
- **Compound commands** — `go`, `done --next`, `next --claim`, `init`
- **Dynamic context delivery** — `go` surfaces upstream results, `done` shows unlocked tasks
- **JIT planning** — `what-if`, `insert`, `ahead`, `amend`, `pivot`, `split`
- **Effect analysis** — every mutation returns delayed/accelerated/ready_now/critical_path
- **DAG tree view** — `status --detail` renders the dependency graph with unicode tree characters
- **Short IDs** — 8-char task/project IDs
- **Compact output** — terse defaults, `--json` for machine parsing
- **Handoff protocol** — results propagate through dependency edges
- **Fuzzy ID resolution** — typo recovery + closest-match suggestions
- **File tracking** — attach files to tasks, detect conflicts between agents
- **Notes + events** — inter-agent signals and real-time event stream
- **Pause/resume** — partial completion with progress tracking
- **Sticky project** — `plandb use <id>` sets default, fewer flags per command
- **Progressive status** — one-liner, `--detail` (DAG tree), or `--full`

## Beyond Code

Plandb coordinates any work with dependencies — not just software:

- **Research pipelines** — parallel literature searches, synthesize when all complete
- **Data workflows** — ETL stages with fan-out/fan-in dependency patterns
- **Content production** — research → outline → draft → review → publish
- **Infrastructure** — provision resources in dependency order
- **Multi-model chains** — route subtasks to specialized models, aggregate results

## Philosophy

Plandb is a primitive, not a platform. It does one thing: coordinate dependent tasks across agents with minimal overhead. It doesn't schedule agents, doesn't pick models, doesn't manage git. It's the SQLite of agent task management — embed it, script it, build on it.

## Contributing

Contributions welcome! Plandb is written in Rust with zero runtime dependencies beyond SQLite (bundled).

```bash
git clone https://github.com/Agent-Field/plandb.git
cd plandb
cargo build
cargo test
bash tests/functional_test.sh    # 113 functional assertions
```

Check out the [open issues](https://github.com/Agent-Field/plandb/issues) or open a new one if you have ideas.

## License

Apache License 2.0

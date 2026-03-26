<div align="center">

<img src="assets/hero.png" alt="PlanDB — Task graph primitive for AI agents" width="100%" />

# PlanDB

### **The missing primitive for agentic coordination.**

*One binary. Zero infra. Your agent gets a graph it can think with.*

[![Stars](https://img.shields.io/github/stars/Agent-Field/plandb?style=flat&logo=github&logoColor=white&color=7c3aed&labelColor=1e1e2e)](https://github.com/Agent-Field/plandb/stargazers)
[![License](https://img.shields.io/badge/license-Apache%202.0-7c3aed.svg?style=flat&labelColor=1e1e2e)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/Agent-Field/plandb?style=flat&logo=git&logoColor=white&color=7c3aed&labelColor=1e1e2e)](https://github.com/Agent-Field/plandb/commits/main)

**[Copy-Paste Prompt](#paste-this-into-your-agent)** · **[Demo](#see-it-work)** · **[Architecture](docs/ARCHITECTURE.md)** · **[Examples](examples/)**

</div>

---

PlanDB is a task graph database for AI agents. Agents decompose work into tasks with dependencies, execute in dependency order, parallelize what's safe, adapt when plans break, and persist knowledge across sessions — all through a single binary backed by SQLite.

Not a to-do list. Not a project management tool. A **structured reasoning surface** that agents read from and write to, the way git is a structured history that tools read from and write to.

Tested with Claude Code, Codex CLI, Gemini CLI, and Cursor. Agents adopt it from the prompt alone — no training, no fine-tuning, no custom integrations.

## Install

```bash
curl -fsSL https://github.com/Agent-Field/plandb/releases/latest/download/plandb-$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m) -o /usr/local/bin/plandb && chmod +x /usr/local/bin/plandb
```

<details>
<summary>From source</summary>

```bash
cargo install --path .
```
</details>

## Paste This Into Your Agent

This is the entire integration. Copy into your system prompt, `CLAUDE.md`, `.cursorrules`, or MCP config:

```
You have plandb installed for task planning. Use it to decompose work and track progress.

Core loop:    plandb go → work → plandb done --next
Add tasks:    plandb add "title" --description "detailed spec" --dep t-xxx
Split:        plandb split --into "A, B, C" (independent) or "A > B > C" (chain)
Context:      plandb context "what you discovered" --kind discovery
Search:       plandb search "query" (BM25 across context + tasks)
Introspect:   plandb critical-path | plandb bottlenecks | plandb what-unlocks <id>
Status:       plandb status --detail

Record discoveries and decisions with plandb context as you work.
plandb go auto-surfaces relevant context — no need to search manually.
After each completion, reassess: plandb status --detail + plandb critical-path.
Plans are hypotheses — adapt as you learn.
When plandb list --status ready shows multiple tasks, parallelize them.
```

That's it. The agent handles the rest. For richer prompts: `plandb prompt --for cli`, `plandb prompt --for mcp`, or `plandb prompt --for http`.

## See It Work

```bash
plandb init "auth-system"
plandb add "Design schema" --as schema --kind research \
  --description "Define user/session tables, auth flows, token format"
plandb add "Build API" --as api --kind code --dep t-schema \
  --description "Implement endpoints: register, login, refresh, logout"
plandb add "Write tests" --as tests --kind test --dep t-schema \
  --description "Integration tests for all auth endpoints"
plandb add "Deploy" --as deploy --kind shell --dep t-api --dep t-tests \
  --description "Docker build, push, deploy to staging"

plandb go            # → claims "Design schema" (the only task with no blockers)
plandb done --next   # → completes it, claims "Build API" — and "Write tests" is now ready too
```

Two tasks are ready simultaneously. If you have two agents, they work in parallel — atomic claiming prevents double-assignment. The graph is the coordination layer.

## Why This Exists

Ask an AI agent to "build an auth system." It will start coding immediately — no plan, no structure, no idea what depends on what. It might build the API before the schema exists. It will definitely forget to write tests. And if you give the same task to two agents, they'll duplicate each other's work.

**The root cause:** agents have no working memory for plans. They operate in a single context window with no structured state. Every task is independent. Every session starts from zero.

PlanDB fixes this with three ideas:

**1. Plans are dependency graphs, not checklists.** A task knows what it depends on and what depends on it. The graph determines execution order, parallelization, and critical path — automatically. No coordinator agent needed.

**2. Plans are hypotheses that evolve.** You don't know the real plan until you start executing. An agent discovers a task is harder than expected — it `split`s it. Finds a missing step — `insert`s one. A whole approach fails — `pivot`s the subtree. Dependencies rewire. Ready tasks recalculate. The plan adapts to reality.

**3. Agents accumulate knowledge in the graph.** When an agent discovers something ("JWT tokens conflict with the session cache"), it records it with `plandb context`. When a future agent claims a related task, that knowledge surfaces automatically via BM25 search — no one has to remember to look for it. The graph becomes institutional memory.

## The Architecture: A Compound Graph

Most task systems are flat lists or simple DAGs. PlanDB is a **compound graph** — two independent structures composed together:

```
                    ┌─────────────────────────────────┐
                    │          Project                 │
                    │  ┌──────────┐  ┌──────────┐     │
  Containment       │  │ Backend  │  │ Frontend │     │   ← hierarchy
  (place graph)     │  │ ┌──┐┌──┐│  │ ┌──┐     │     │     (what contains what)
                    │  │ │DB││API││  │ │UI│     │     │
                    │  │ └──┘└──┘│  │ └──┘     │     │
                    │  └──────────┘  └──────────┘     │
                    └─────────────────────────────────┘

  Dependencies       DB ──→ API ──→ UI                     ← flow
  (link graph)            ╲─────────→ UI                     (what depends on what)
```

**Dependencies cross containment boundaries.** A subtask at depth 3 can depend on a task at depth 0 in a completely different branch. The hierarchy and the flow are orthogonal — like a filesystem overlaid with a build graph.

This gives agents things flat DAGs can't:
- **Recursive decomposition.** Split tasks to any depth. Split the subtasks. Composites auto-complete when children finish.
- **Scoped reasoning.** `plandb use t-backend` zooms into a subtree. The agent works there without seeing the whole project.
- **Cross-level coordination.** Frontend UI can depend on a Backend DB subtask directly. No artificial flattening.

## What This Unlocks

**Plans that adapt mid-flight.**
An agent plans 6 tasks. During execution, it discovers the problem is harder than expected. It splits, inserts, pivots — the graph grows to 20 tasks. Dependencies rewire automatically. This isn't failure. This is how real planning works.

**Cross-session memory.**
Agent A discovers "the API uses non-standard date formats" and records it. Three days later, Agent B claims a task to build the frontend. `plandb go` auto-surfaces that discovery via BM25 matching. Agent B never searched for it — it just appeared. The graph remembers what agents learn.

**Multi-agent coordination without a coordinator.**
No orchestrator agent deciding who works on what. The graph structure IS the coordinator: `ready` means all dependencies are met. Atomic claiming means no double-assignment. `critical-path` shows what to prioritize. The data structure does what would otherwise require a meta-agent.

**Quality gates agents can't skip.**
`--pre` conditions shown when claiming a task: *"schema must define all endpoints."* `--post` conditions shown when completing: *"verify all routes return valid JSON."* The agent sees these as part of the task — not as an afterthought.

## Showcase: Autonomous ML Research

We gave a Claude Code instance one sentence: *"Build a GPT from scratch in Rust, then train it to do tool calling."*

No human intervention. No step-by-step instructions. Just PlanDB and a prompt.

The agent built a **3,769-line transformer with backpropagation** in pure Rust (zero ML frameworks), then designed and ran a **7-method RL experiment**. The task graph evolved from 6 planned tasks to 20 — the agent split tasks that proved complex, parallelized independent experiments, and pivoted when REINFORCE catastrophically collapsed:

```
t-design ──▶ t-sft ─────┬──▶ t-reinforce ──┐
                         ├──▶ t-dpo ─────────┤
                         ├──▶ t-custom ──────┼──▶ t-compare ──▶ t-final
                         ├──▶ t-sft-v2 ──────┤      (fan-in)
                         ├──▶ t-sft-v3 ──────┤
                         └──▶ t-reject ──────┘
```

| Method | Format Acc | Tool Acc | Composite |
|--------|-----------|----------|-----------|
| **Rejection Sampling** | **71.3%** | **70.0%** | **0.601** |
| SFT Baseline | 66.3% | 63.8% | 0.577 |
| DPO | 65.0% | 62.5% | 0.570 |
| REINFORCE | 0.0% | 0.0% | 0.090 |

Pre-trained weights included: `cd experiments/mini-gpt-rust && cargo run --release -- --demo`

> More in [`experiments/`](experiments/) — docs sites built autonomously by Codex, Claude Code, and Gemini CLI.

## Interfaces

| | Command | For |
|---|---|---|
| **CLI** | `plandb <command>` | Claude Code, Codex, Gemini, any shell agent |
| **MCP** | `plandb mcp` | Claude Code, Cursor, Windsurf (native tool integration) |
| **HTTP** | `plandb serve --port 8484` | Custom agents, webhooks, dashboards |

## Part of AgentField

PlanDB is the task planning layer for [**AgentField**](https://github.com/Agent-Field/agentfield) — the open-source AI backend for building and running AI agents. [**SWE-AF**](https://github.com/Agent-Field/SWE-AF) uses PlanDB internally to orchestrate parallel agent workstreams.

**[Architecture Docs](docs/ARCHITECTURE.md)** · **[Examples](examples/)** · **[CLI Reference: `plandb --help`](#)**

## License

Apache License 2.0

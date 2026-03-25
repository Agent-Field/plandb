# PlanDB Product Iteration — Learnings & Philosophy

Accumulated from user conversations during the product iteration loop.
Read this before making any product decisions.

## Core Philosophy

### 1. The Agent is the Primary User

PlanDB's CLI, prompts, and output are designed for **AI agents**, not humans.
The human's only job is to paste the prompt and give a task. Everything else
is the agent's responsibility.

- Agents should auto-discover all features from the prompt + CLI help
- Agents should dynamically create their own task graphs
- Agents should know when to use context, split, search, adapt — without being told
- CLI output should be self-documenting with contextual hints at every step

### 2. The Prompt IS the Product

The prompt (`plandb prompt --for cli`) is the entire product surface.
Quality of agent behavior is 100% determined by prompt quality.

- Example-first beats documentation-heavy (A/B tested: 51% smaller prompt → better behavior)
- Leading with a concrete dependency example makes agents create proper graphs
- The prompt is the only thing the human touches — everything else is agent-driven

### 3. CLI Output Teaches the Agent

Every response should hint at what the agent can do next:

- `init` → shows how to add tasks with deps, the core loop
- `go` → shows context, split, search, and done commands
- `done` → shows what unlocked, how to reassess, how to continue
- `status` → surfaces actionable intelligence (not just numbers)

If an agent struggles with a feature, the product/prompt is wrong, not the agent.

### 4. Test with Real Agents, Not Humans

- Agent behavior differs from human expectations
- Test every feature with Codex, Gemini CLI, or Claude Code as subprocesses
- The test output IS the example — don't create synthetic demos
- Capture real agent sessions (plandb status, task graphs, context entries) as examples

## Feature Design Principles

### 5. Templates are Extracted, Not Written

Templates are NOT hand-written YAML. They're extracted from real agent runs:

1. Agent works with PlanDB, creates tasks, adds context, adapts mid-flight
2. `plandb export` captures the entire evolved graph — tasks, deps, context, decisions
3. Someone else imports it — the template provides structure AND institutional knowledge
4. The next agent can still adapt — the template is a starting point, not a constraint

Templates should include:
- Task graph (tasks, deps, pre/post conditions, hooks)
- Context entries (decisions, discoveries, patterns — the "clues")
- Metadata about what adapted (so the next agent knows where flexibility is)

### 6. Everything Must Be General-Purpose

- Features are domain-agnostic, not tied to specific applications
- Freeform `--kind` is better than fixed enums (agents pick their own taxonomy)
- Universal context types, not application-specific ones
- No bloated features — each feature should be minimal but composable

### 7. Zero Infrastructure, Always

Like SQLite eliminated database servers, PlanDB eliminates orchestration servers.
- Single binary, single file, no daemon, no setup
- No vector DB, no embeddings, no external services
- BM25 via SQLite FTS5 — zero extra dependencies
- `curl | sh` install — one command

### 8. Lazy Over Explicit

- Lazy recall > explicit search (context auto-surfaces in `plandb go`)
- Auto-link to running task > manual `--task` flags
- Auto-detect agent > require `--agent` flag
- The agent should be able to "just use it" without reading docs

## Technical Learnings

### 9. BM25 Search Design

- Query expansion with stop-word filtering works well for task title → context matching
- Use OR queries for broader recall (agents benefit from more hits, not precision)
- Field-aware scoring (title > description > content) would improve relevance
- Keep queries focused — too many terms dilute BM25 results (cap at 6 tokens)

### 10. Hook Design

- Hooks are advisory — failures never block operations (print warning, continue)
- Set environment variables: PLANDB_TASK_ID, PLANDB_PROJECT_ID, PLANDB_TASK_TITLE, PLANDB_AGENT_ID
- Hooks enable replayable procedures: template + hooks = automated workflow
- pre_hook at go (task starts), post_hook at done (task completes)

### 11. Dependency Validation

- Always validate dep targets exist BEFORE creating the task
- Previously: create_task succeeded then add_dependency failed on FK → orphaned task
- Now: pre-validate all deps, fail fast with friendly error + fuzzy match suggestion

## Product Metrics (from experiments)

| Metric | Value |
|--------|-------|
| Prompt size reduction | 51% (199→110 lines, 9.5KB→4.7KB) |
| Agent dependency usage (original prompt) | None |
| Agent dependency usage (improved prompt) | Full graph with custom IDs |
| Context entries per agent session | ~6 (Codex URL shortener test) |
| Context types used by agents | discovery, decision, pattern (freeform) |
| Test pass rate | 128/128 functional tests |

### 12. Templates as a Learning Framework

Templates are NOT written by hand — they're learned by running agents:

1. **Seed**: Start with a minimal template (or no template)
2. **Run**: Agent works through the task, records context, adapts
3. **Export**: `plandb export` captures the evolved graph + all context
4. **Distribute**: The evolved template is MORE valuable than the original
5. **Repeat**: Next agent imports, adapts further, re-exports

Each cycle compounds — templates get better with every run because
they carry real-world findings, not theoretical patterns.

Example: Security Audit template evolved from 7 hand-seeded tasks with
5 pattern entries to 6 agent-optimized tasks with 6 real finding entries.

### 13. Templates Must Be Explicitly Forced in Prompts

Agents prefer creating their own task structure over importing templates.
To get template-driven behavior:
- The prompt must say "import the template FIRST"
- `plandb init` hints should lead with templates
- `plandb go` should show "(from template: X)" when following one
- After import, hints guide adaptation (insert, split, context, export)

The agent will still adapt the template (which is good), but it needs to
START from the template to benefit from institutional knowledge.

## Anti-Patterns Discovered

1. **Don't create examples by hand** — the test output IS the example
2. **Don't use fixed enums** — agents are better judges of what types they need
3. **Don't require explicit flags** — auto-detection reduces friction
4. **Don't make agents search explicitly** — lazy recall surfaces context proactively
5. **Don't assume the human sets up the plan** — agents decompose their own work
6. **Don't track tasks in markdown files** — SQL-based persistence is the product differentiator

# PlanDB Product Iteration — Findings

## Phase 1: Existing Feature Validation

### Test 1: Core Loop (Codex)
**Agent**: Codex 0.116.0 | **Status**: PASS | **Tokens**: 40,945

Codex successfully used plandb go/done loop, completing 3/3 tasks. However, it created no dependencies between tasks — all were independent. Root cause: the prompt didn't demonstrate dependencies as essential.

### Test 2: Decomposition (Gemini)
**Agent**: Gemini CLI 0.32.1 | **Status**: PARTIAL PASS (2/12 complete)

Gemini created an excellent 12-task graph with custom IDs, proper dependencies, and used split for decomposition. Only completed 2 tasks before stopping (likely context/timeout).

### Test 3: Edge Cases (Manual)
**Status**: PASS with 1 bug found

Bug: `plandb add --dep t-nonexistent` created the task first, then failed on FK constraint, leaving an orphaned task. **Fixed**: now validates deps before creating task with friendly error message.

### Test 4: Feature Completeness (Manual)
All features validated: init, add, go, done, split, status, critical-path, bottlenecks, what-unlocks, insert, amend, export/import.

## Phase 2: Prompt Optimization

### A/B Test: Original vs Improved Prompt
| Metric | Original (199 lines) | Improved (110 lines) |
|--------|---------------------|---------------------|
| Token cost | ~2,400 tokens | ~1,200 tokens |
| Dependencies used | No | Yes (full graph) |
| Custom IDs | No | Yes |
| Task completion | 3/3 | 4/4 |
| Dependency ordering | N/A | Correct |

**Key insight**: Example-first prompts produce better agent behavior than documentation-heavy prompts. Leading with a concrete dependency example made Codex create proper dependency graphs.

## Phase 3: New Features

### Context Store with BM25 Search

**Agent validation (Codex)**:
- Created 6 context entries while building a URL shortener
- Types used: discovery (3), decision (2), pattern (1)
- Auto-linking to running task worked correctly
- Used context to record research findings, design decisions, environment constraints

**Feature summary**:
- `plandb context` — freeform typed context entries on the project graph
- `plandb search` — BM25 search via SQLite FTS5 (zero extra deps)
- Auto-link to current running task (no --task needed)
- Freeform --kind (not a fixed enum — agent decides the taxonomy)

### Lazy Recall (Context in `plandb go`)

**Validation**: Manual testing with realistic multi-task project

When an agent claims a task, BM25 automatically searches for relevant context entries
using key terms extracted from the task title and description.

Example: Claiming "Implement auth endpoints" auto-surfaced:
- `[decision] Use argon2id for password hashing — bcrypt has 72-byte limit`

Applied to all three interfaces: CLI, MCP, HTTP.

### Task Lifecycle Hooks

**Validation**: Manual testing

- `--pre-hook` fires when task transitions to running (via `plandb go`)
- `--post-hook` fires when task transitions to done (via `plandb done`)
- Sets PLANDB_TASK_ID, PLANDB_PROJECT_ID, PLANDB_TASK_TITLE, PLANDB_AGENT_ID env vars
- Hooks are advisory — failures warn but never block
- Included in export/import templates for replayable procedures

### Contextual Action Hints in CLI Output

Every lifecycle response now teaches the agent what it can do:
- `init`: dependency syntax, core loop
- `go`: context recording, split, search, done commands
- `done`: reassess, next task, insert, parallel options

This makes the CLI self-documenting — agents auto-discover features without needing
the full prompt reference.

### Session Continuity (plandb resume)

**Validation**: Manual testing

`plandb resume` provides everything an agent needs to pick up from a previous session:
- Progress summary (done/total/running/ready/blocked)
- Currently running tasks with agent IDs
- Ready-to-claim tasks
- Recent context entries (last 5)
- Contextual action hints

Replaces the need for a separate session start/end/resume system.

### Search Quality (FTS5 on tasks)

Tasks are now indexed via FTS5 with weighted fields:
- Title matches weighted 10x over description matches
- Proper BM25 ranking instead of LIKE search
- Combined results: context entries + tasks, sorted by relevance

## Phase 4: Templates with Institutional Knowledge

Templates now include context entries (institutional knowledge), so imported templates
carry not just the task structure but also the discoveries/decisions from the original run.

Shipped two template recipes:
- `templates/pr-review.yaml` — 7-task fan-out pipeline (parse → 4 parallel reviewers → synthesize → report)
- `templates/epic-decomposition.yaml` — 7-task waterfall with parallel test/docs/edge-case phase

### Template Import + Execution Flow

Validated end-to-end template workflow:
1. `plandb import templates/pr-review.yaml` creates 7-task fan-out graph
2. Template context entries available via `plandb search`
3. `plandb go` shows downstream tasks and research-specific hints
4. Post-conditions from template preserved on tasks
5. Template context surfaces in lazy recall for relevant tasks

### Gemini CLI Cross-Agent Validation

Gemini CLI completed 4/4 tasks (100%) with proper dependencies, 3 context entries,
and custom IDs. Validates PlanDB works across different AI agents.

### Agent Validation (Codex, Updated Prompt v2)

Re-tested with the updated prompt (120 lines, includes all new features):
- Agent created 4-task dependency chain for Flask URL shortener
- 5 context entries: 3 decisions, 1 discovery, 1 constraint
- Used structured `--result` JSON to hand off design decisions
- Used custom IDs throughout
- Completed 2/4 tasks before timeout

## Meta-Findings

1. **The prompt IS the product** — agents' behavior is shaped entirely by the prompt quality
2. **Test with real agents, not humans** — agent behavior differs from human expectations
3. **Smaller prompts > larger prompts** — 51% reduction produced better results
4. **Freeform > fixed taxonomies** — agents are better judges of what types they need
5. **Auto-detection > explicit flags** — auto-linking to running task reduces friction
6. **Lazy > explicit** — auto-surfacing context in `go` is better than requiring search
7. **CLI output is a teaching surface** — every response should hint at available actions
8. **Templates are extracted, not written** — export from real runs, not hand-crafted YAML
9. **The test IS the example** — real agent session output becomes the documentation

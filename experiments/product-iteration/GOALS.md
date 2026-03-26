# PlanDB Product Iteration — Goals & Ideation

## Philosophy

PlanDB is the **agent-native database for autonomous harnesses** — Claude Code, Codex, Gemini CLI, and custom agents. It replaces ad-hoc task tracking with a compound task graph that agents can query, mutate, and coordinate through.

The key insight: **PlanDB should be the single source of truth for long-running autonomous work** — not just task decomposition, but the evolving knowledge, context, and learned patterns that accumulate across sessions.

## Phase 1: Validate Existing Features

Verify every existing feature works correctly when used by real AI agents (Codex/Gemini/Claude). Focus:

1. **Core loop**: `init → add → go → done --next` — does an agent naturally adopt this?
2. **Decomposition**: `split --into` with commas and chains — does recursive decomposition work?
3. **Dependencies**: cross-level deps, feeds_into/blocks/suggests — correct enforcement?
4. **Introspection**: `critical-path`, `bottlenecks`, `what-unlocks` — useful output?
5. **Adaptation**: `insert`, `amend`, `pivot`, `replan` — does mid-flight adaptation work?
6. **Multi-agent**: Atomic claiming, PLANDB_AGENT, parallel execution — race conditions?
7. **Templates**: `export → import` round-trip — lossless?
8. **Prompt quality**: Is the `plandb prompt --for cli` prompt sufficient for an agent to use PlanDB effectively without human guidance?

## Phase 2: Optimize for Agent Ergonomics

Based on Phase 1 findings, improve:

- Prompt conciseness (minimal tokens for maximum agent comprehension)
- Error messages (agent-friendly, not human-friendly)
- Output formats (compact mode truly minimal?)
- Common failure modes (what confuses agents?)

## Phase 3: New Features — Task-Level Knowledge Store

### Core Idea: Contextual Knowledge Accumulation

PlanDB tasks accumulate **context entries** — structured observations, decisions, and discoveries that persist across sessions. Unlike ephemeral agent memory, these are tied to specific tasks and queryable.

Think of it as: **git commits are to code as context entries are to planning**.

#### Proposed Features:

1. **Context Entries** (not "memories" — more precise)
   - Attach structured context to any task: findings, decisions, blockers, patterns
   - Queryable: `plandb context search "authentication"` uses BM25-style ranking
   - Hierarchical: context on parent tasks is visible to children
   - Types: `discovery`, `decision`, `blocker`, `pattern`, `reference`

2. **Semantic Search (BM25)**
   - Lightweight, no embeddings, no external dependencies
   - Index task titles + descriptions + context entries + notes
   - `plandb search "rate limiting patterns"` → ranked results
   - Useful for agents to recall what was learned in previous sessions

3. **Pre/Post Hooks**
   - `plandb add "task" --pre-hook "plandb search 'related work'" --post-hook "plandb context add --type decision"`
   - Hooks are shell commands executed before claiming / after completing
   - Enable: auto-context-gathering, validation, notifications, sub-agent spawning

4. **Session Continuity**
   - `plandb session start` / `plandb session end` — track work sessions
   - `plandb session resume` — show what was last worked on, context from last session
   - Session metadata: agent ID, duration, tasks completed, context added

5. **Project-Level Patterns**
   - `plandb patterns` — extract recurring decomposition patterns from completed projects
   - Auto-suggest: "This task looks like X pattern from project Y"
   - Feed into template library automatically

6. **Agent Coordination Primitives**
   - `plandb broadcast "message"` — notify all active agents
   - `plandb inbox` — check for messages from other agents
   - Lock-free coordination beyond atomic claiming

## Experiment Structure

```
experiments/product-iteration/
├── GOALS.md                 ← this file
├── FINDINGS.md              ← accumulated findings from experiments
├── phase-1-validation/      ← existing feature validation
│   ├── test-core-loop/      ← each test scenario is a directory
│   ├── test-decomposition/
│   ├── test-multi-agent/
│   └── ...
├── phase-2-optimization/    ← agent ergonomics improvements
└── phase-3-features/        ← new feature prototypes
    ├── context-store/
    ├── bm25-search/
    ├── hooks/
    └── session-continuity/
```

## Testing Methodology

Each experiment:
1. Create a fresh PlanDB database in the experiment directory
2. Launch an AI agent (prefer Codex/Gemini for cost, Claude for complex validation)
3. Give it a task that exercises the feature being tested
4. Capture the agent's PlanDB interactions (commands, outputs, errors)
5. Analyze: Did the agent use PlanDB effectively? What confused it? What's missing?
6. Record findings in FINDINGS.md
7. If changes needed: implement, re-test, iterate

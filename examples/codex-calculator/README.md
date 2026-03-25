# Example: Codex builds a CLI calculator using PlanDB (full feature test)

This example validates PlanDB's complete feature set — dependencies, context store,
lazy recall, search, custom IDs — all auto-discovered by the agent from the prompt alone.

## Prompt

```
You have plandb installed for task planning. Use it to decompose work and track progress.

Core loop:    plandb go → work → plandb done --next
Add tasks:    plandb add "title" --description "detailed spec" --dep t-xxx
Split:        plandb split --into "A, B, C" (independent) or "A > B > C" (chain)
Context:      plandb context "what you learned" --kind discovery
Search:       plandb search "query" (BM25 across context + tasks)
Introspect:   plandb critical-path | plandb bottlenecks | plandb what-unlocks <id>
Status:       plandb status --detail

After each completion, reassess: plandb status --detail + plandb critical-path.
Record discoveries: plandb context "what you found" --kind discovery
Plans are hypotheses — adapt as you learn.

TASK: Build a Python CLI calculator that supports add, subtract, multiply, divide.
Create a dependency graph with at least 3 tasks. Use context to record design decisions.
After completing the first task, check plandb search to see if any relevant context exists.
```

## What the Agent Did

### 1. Created a dependency chain

```
t-calc-structure ──▶ t-calc-impl ──▶ t-calc-verify
```

| Task | Kind | Description |
|------|------|-------------|
| `t-calc-structure` | research | Choose file layout, CLI interface, invocation pattern |
| `t-calc-impl` | code | Implement arithmetic + CLI + error handling |
| `t-calc-verify` | test | Run CLI, verify errors, add usage text |

### 2. Recorded context while working

```
c-oyr8 [discovery] Design decision: keep the calculator as a single-file Python CLI
                    using argparse in calculator.py so the project stays minimal...
c-q7a3 [discovery] Scaffolded calculator.py with argparse subcommands for add,
                    subtract, multiply, and divide. Deferred implementation...
c-r02l [discovery] Searched plandb after completing the first task and found no
                    prior calculator-specific context...
```

### 3. Used search proactively

The agent used `plandb search` after completing the first task — exactly as instructed
in the prompt. Finding no prior context, it recorded what it discovered.

## Features Validated

| Feature | Agent behavior |
|---------|---------------|
| **Dependencies** | Created 3-task chain with `feeds_into` |
| **Custom IDs** | Used `--as calc-structure`, `--as calc-impl`, `--as calc-verify` |
| **Context store** | Created 3 context entries recording design decisions |
| **Search** | Used `plandb search` after first task completion |
| **Core loop** | `init → add → go → done → context → go` |
| **Descriptions** | Self-contained work orders with acceptance criteria |

## Files Created

- [`calculator.py`](calculator.py) — CLI calculator scaffold with argparse subcommands

## Reproduce

```bash
cd /tmp/calculator-test && git init
codex exec --full-auto "$(plandb prompt --for cli)

TASK: Build a Python CLI calculator that supports add, subtract, multiply, divide.
Create a dependency graph with at least 3 tasks. Use context to record design decisions."
```

# Improved CLI Prompt v1

## Design Principles (based on Phase 1 findings)

1. **Lead with a working example** — agents learn by imitation, not documentation
2. **Show dependencies as essential** — the first example must use `--dep`
3. **Emphasize the continuous loop** — go/done/reassess is the core behavior
4. **Minimize token usage** — ~150 lines max, prioritize signal over noise
5. **Make PLANDB_DB prominent** — avoid --db repetition
6. **Show parallelization** — make it obvious when multiple tasks are ready

## Proposed Prompt

```
## PlanDB — Task Graph Database

You have `plandb` for managing task dependencies. ALWAYS use PlanDB for task planning.
Use `export PLANDB_DB=<path>` to set the database path.

### Example: 4-task project with dependencies

```bash
plandb init "auth-system"
plandb add "Design schema" --as schema --kind research \
  --description "Define user/session tables, auth flows, token format"
plandb add "Build API" --as api --kind code --dep t-schema \
  --description "Implement endpoints from schema: register, login, refresh, logout"
plandb add "Write tests" --as tests --kind test --dep t-schema \
  --description "Integration tests for all auth endpoints"
plandb add "Deploy" --as deploy --kind shell --dep t-api --dep t-tests \
  --description "Docker build, push, deploy to staging"
plandb status --detail
plandb critical-path
```

### Execute: go → work → done → reassess → repeat

```bash
plandb go                 # Claim next ready task (shows pre-conditions + upstream context)
# ... do the work described in the task ...
plandb done --next        # Complete + claim next (shows post-conditions to verify)
plandb status --detail    # Reassess: does the plan still make sense?
plandb critical-path      # What should be prioritized?
```

After each task: reassess. Add tasks, split complex ones, amend descriptions.
Plans are hypotheses — the graph should evolve as you learn.

### Key Commands

```bash
# Adding tasks (ALWAYS include --description)
plandb add "title" --description "detailed spec" --dep t-upstream [--as custom-id] [--kind code]

# Decomposition
plandb split --into "A, B, C"           # Independent subtasks (parallel)
plandb split --into "A > B > C"         # Dependency chain (sequential)

# Introspection
plandb status --detail                   # Dependency tree with progress
plandb critical-path                     # Longest chain to completion
plandb bottlenecks                       # What's blocking the most work
plandb what-unlocks t-xxx                # Impact of completing a task
plandb list --status ready               # Tasks safe to parallelize NOW

# Plan adaptation
plandb task insert --after t-a --before t-b --title "Missed step"
plandb task amend t-xxx --prepend "NOTE: discovered X, use approach Y"
plandb what-if cancel t-xxx              # Preview before destructive action
```

### Rules
- `--dep` upstream must exist first — create tasks in dependency order
- `--description` is the work order, not the title. Make it self-contained.
- `--kind`: generic, code, research, review, test, shell
- When `plandb list --status ready` shows multiple tasks → parallelize them
- Composite tasks auto-complete when all children finish
- IDs: auto-generated (t-k3m9) or custom (--as api → t-api)
```

## Token Count Comparison

- Original prompt: ~2,400 tokens (199 lines)
- Proposed prompt: ~800 tokens (~60 lines)
- Reduction: ~67%

## Key Changes

1. Moved from documentation style to example-first style
2. First example shows 4 tasks WITH dependencies (not optional)
3. Added "ALWAYS use PlanDB" instruction
4. Added "reassess after each task" as core loop step
5. Made PLANDB_DB env var the primary config mechanism
6. Cut decomposition philosophy (agents learn this by doing, not reading)
7. Cut cross-level dependency explanation (too abstract for first-time use)

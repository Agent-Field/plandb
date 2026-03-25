# PlanDB Agent Guide — Neural Genesis Experiments

PlanDB is NOT a simple hierarchical task list. It is a **complex directed acyclic graph (DAG)** with typed edges, atomic multi-agent claiming, mid-flight adaptation, and recursive decomposition. You must use it as your primary coordination and logging tool.

## Core Mental Model

Think of PlanDB as a **computational graph** where:
- **Nodes** = tasks (with state: pending → ready → running → done/failed)
- **Edges** = typed dependencies (feeds_into, blocks, suggests)
- **State transitions** are atomic (prevents race conditions between parallel agents)
- **Ready** = all dependencies satisfied (all upstream tasks are done)
- Tasks can be **composite** (auto-complete when all children finish)

## Dependency Types (Critical to Understand)

| Type | Meaning | When to Use |
|------|---------|-------------|
| `feeds_into` | Data flows from A → B. B needs A's result. | Task B uses output of Task A (e.g., trained model → evaluation) |
| `blocks` | Ordering only. B waits for A but doesn't use A's data. | Task B needs a side-effect of A (e.g., file must exist) |
| `suggests` | Soft dep. B can proceed without A, but A helps. | Nice-to-have ordering, non-blocking |

```bash
plandb add "Evaluate model" --dep t-train           # default: feeds_into
plandb add "Cleanup logs" --dep t-train:blocks       # ordering only
plandb add "Read paper" --dep t-implement:suggests   # soft, non-blocking
```

## Your Workflow — The Experiment Loop

### 1. Decompose Your Task

When you start an experiment, IMMEDIATELY decompose it:

```bash
# Sequential chain (each step depends on previous)
plandb split t-your-task --into "Step A > Step B > Step C > Step D"

# Independent parallel subtasks
plandb split t-your-task --into "Part A, Part B, Part C"

# Mix: some parallel, some sequential
plandb add "Implement FFT" --parent t-your-task --as sub-fft --kind code
plandb add "Implement model struct" --parent t-your-task --as sub-model --kind code
plandb add "Implement backward" --parent t-your-task --as sub-backward --kind code --dep t-sub-model
plandb add "Train and evaluate" --parent t-your-task --as sub-train --kind code --dep t-sub-fft --dep t-sub-backward
```

Children of a split can themselves be split further — any depth. The parent auto-completes when all children finish.

### 2. Claim → Work → Complete (The Core Loop)

```bash
plandb go                           # claim next ready task (atomic, prevents conflicts)
# ... do the work ...
plandb done t-xxx --result '{"loss": 1.23, "accuracy": 0.67}'  # complete with structured data
```

### 3. Log Observations As You Go

```bash
plandb task note t-xxx "FFT implementation: using radix-2 Cooley-Tukey, O(n log n)"
plandb task note t-xxx "DISCOVERY: loss plateaus at 2.1 — may need learning rate warmup"
plandb task note t-xxx "DECISION: using leapfrog integrator over RK4 for symplectic property"
plandb task progress t-xxx --pct 60 --note "model compiles, starting training loop"
```

### 4. Adapt Mid-Flight (THIS IS THE KEY FEATURE)

Plans are hypotheses. When you discover something unexpected, ADAPT:

```bash
# Insert a missing step between existing tasks
plandb task insert --after t-implement --before t-train --title "Fix numerical instability in FFT"

# Amend a future task with new info
plandb task amend t-train --prepend "NOTE: learning rate must be < 0.0003 or loss diverges"

# A task turns out to be more complex than expected — split it
plandb split t-train --into "Pretrain base model > Fine-tune on tool data > Evaluate"

# Something failed — pivot the approach
plandb task pivot t-old-approach --to "New approach after discovering X" --reason "Original method diverges"

# Preview effects before acting
plandb what-if cancel t-xxx     # what tasks would be affected?
```

### 5. Cross-Level Dependencies

Subtasks can depend on tasks ANYWHERE in the graph, not just siblings:

```bash
# A subtask of EXP1 depends on a subtask of EXP3
plandb add "Compare spectral vs RG efficiency" --dep t-exp1-train --dep t-exp3-train
```

### 6. Introspection — Use These Regularly

```bash
plandb status --detail          # overview of all tasks and their states
plandb critical-path            # what's the longest chain blocking completion?
plandb bottlenecks              # what task is blocking the most downstream work?
plandb what-unlocks t-xxx       # what becomes ready if I finish this task?
plandb ahead                    # what's coming next in the graph?
plandb task notes t-xxx         # read all notes on a task
plandb list --status ready      # what can I work on right now?
plandb list --status failed     # what needs attention?
```

### 7. Structured Results (feeds downstream)

When you complete a task, pass structured results. Downstream tasks can reference these:

```bash
plandb done t-train --result '{"final_loss": 1.45, "format_acc": 0.65, "tool_acc": 0.60, "params": 45000, "inference_speed": 2100, "training_time_sec": 120}'
```

## Experiment-Specific Patterns

### Pattern: Hyperparameter Search
```bash
plandb add "Train with lr=0.001" --parent t-hparam-search --as hp-lr1 --kind code
plandb add "Train with lr=0.0005" --parent t-hparam-search --as hp-lr2 --kind code
plandb add "Train with lr=0.0001" --parent t-hparam-search --as hp-lr3 --kind code
plandb add "Pick best lr and retrain" --parent t-hparam-search --as hp-best --dep t-hp-lr1 --dep t-hp-lr2 --dep t-hp-lr3
```

### Pattern: Iterative Refinement
```bash
plandb add "Train v1" --as iter-v1 --kind code
plandb add "Analyze v1 results" --as iter-analyze1 --kind research --dep t-iter-v1
plandb add "Train v2 with fixes" --as iter-v2 --kind code --dep t-iter-analyze1
# If v2 still needs work, add more iterations dynamically
```

### Pattern: Ablation Study
```bash
plandb add "Full model" --parent t-ablation --as abl-full
plandb add "Without component A" --parent t-ablation --as abl-no-a
plandb add "Without component B" --parent t-ablation --as abl-no-b
plandb add "Compare ablation results" --parent t-ablation --dep t-abl-full --dep t-abl-no-a --dep t-abl-no-b
```

### Pattern: Discovery-Driven Branching
When you find something unexpected during training:
```bash
# Log the discovery
plandb task note t-current "UNEXPECTED: spectral filters converge to band-pass patterns"

# Create a new investigation branch
plandb add "Investigate band-pass convergence" --as investigate-bandpass --dep t-current:suggests --kind research
plandb add "Test pre-initialized band-pass filters" --as test-bandpass --dep t-investigate-bandpass --kind code
```

## Key Rules

1. **ALWAYS decompose before coding** — split your task into 3-7 subtasks
2. **Log observations with `task note`** — every discovery, decision, and surprise
3. **Report progress with `task progress`** — so the orchestrator can track you
4. **Adapt the graph when plans change** — insert, split, amend, pivot
5. **Use structured results** — pass metrics as JSON when completing tasks
6. **Check `critical-path` after completing tasks** — reassess the plan
7. **Never silently fail** — use `plandb task fail t-xxx --error "reason"` if something breaks

## Quick Reference

```bash
plandb add "title" [--as ID] [--kind code|test|research] [--dep TASK:TYPE] [--parent TASK] [--pre "condition"] [--post "condition"]
plandb go [--agent NAME]                    # claim next ready task
plandb done TASK [--result 'JSON']          # complete task
plandb task fail TASK --error "reason"       # mark failed
plandb task note TASK "message"             # add observation
plandb task notes TASK                       # read notes
plandb task progress TASK --pct N --note ""  # progress update
plandb split TASK --into "A > B > C"        # sequential decomposition
plandb split TASK --into "A, B, C"          # parallel decomposition
plandb task insert --after A --before B --title "new step"
plandb task amend TASK --prepend "new info"
plandb task pivot TASK --to "new title" --reason "why"
plandb what-if cancel TASK                   # preview effects
plandb critical-path                         # longest blocking chain
plandb bottlenecks                           # most-blocking task
plandb what-unlocks TASK                     # what completing this enables
plandb ahead                                 # upcoming tasks
plandb status --detail                       # full overview
plandb list --status ready|running|done|failed
plandb project dag                           # visual dependency graph
```

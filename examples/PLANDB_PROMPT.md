## Task

Build a documentation site and interactive playground for PlanDB — a CLI task graph tool for AI agents.

Deliverables (all as static HTML/CSS/JS in the current directory):

1. **Landing page** — hero section, what PlanDB is, why it exists (compound graph for AI agent task orchestration)
2. **Getting Started guide** — walkthrough of: init → add tasks with deps → go → done --next → split → status
3. **CLI Reference** — run `plandb --help`, `plandb task --help`, `plandb split --help` etc. and build a complete reference page from the actual output
4. **Architecture page** — explain the compound graph model (place graph for containment + link graph for dependencies, orthogonal structures)
5. **Interactive playground** — a terminal-like UI in the browser where users can type PlanDB commands and see simulated output. Include a guided tutorial mode. Should feel like a real terminal.
6. **Navigation + polish** — consistent styling, responsive layout, all pages linked, deploy-ready for GitHub Pages

Design: clean, minimal, professional. No heavy JS frameworks — vanilla HTML/CSS/JS only.

## Tool: PlanDB

You have `plandb` installed. Use it to plan and track your own work on this project.

PlanDB is a **compound graph** — two orthogonal structures composed together:
- **Containment (place graph)**: tasks contain subtasks recursively, to any depth — like a filesystem
- **Dependencies (link graph)**: edges between tasks at ANY level — like a build graph

These are independent. Dependencies do NOT follow the containment tree.
A subtask at depth 3 can depend on a task at depth 0 in a different branch.

### Quick reference
```
plandb init "project-name"                                  # create a project
plandb add "short title" --description "detailed spec..."   # add a task WITH description
plandb add "task" --dep t-xxx                               # with dependency (upstream must exist)
plandb add "task" --as my-id                                # custom ID → t-my-id
plandb add "task" --kind code                               # kinds: generic, code, research, review, test, shell (NO other values)
plandb go                                                    # claim next ready task
plandb done --next                                           # complete current + claim next
plandb split --into "A, B, C"                                # split into independent subtasks
plandb split --into "A > B > C"                              # split into dependency chain
plandb status --detail                                       # dependency tree view
plandb status --full                                         # containment tree + dependency edges
plandb show <task-id>                                        # full task details + description
plandb list --status ready                                   # all tasks that can run NOW
plandb task add-dep --after t-upstream t-downstream          # add dependency (use --after flag, NOT positional)
plandb add "task" --pre "precondition text" --post "postcondition text"  # quality gates
plandb critical-path                                        # longest chain — prioritize this
plandb bottlenecks                                          # tasks blocking the most work
plandb what-unlocks <task-id>                               # what becomes ready on completion
plandb watch                                                # live-updating dashboard
plandb export > template.yaml                               # save decomposition as template
plandb import template.yaml                                 # apply template to project
```

### Constraints
- `--kind` ONLY accepts: generic, code, research, review, test, shell. Use "generic" if unsure.
- `--dep` upstream tasks must already exist — create in dependency order.
- `--dep` can reference ANY task at any depth — dependencies cross containment boundaries.
- `--description` is a single quoted string. Newlines inside are fine.

## CRITICAL: Task descriptions

Every task MUST have a `--description` with the full spec of the work. The title is a short label.
The description is the actual work order — self-contained enough that any agent can pick it up
with `plandb go` + `plandb show <id>` and execute without any other context.

Include: files to create/modify, expected output, acceptance criteria, constraints, references to upstream outputs.

## When to decompose: flat task vs hierarchy

**Keep it flat when:**
- A single agent can complete it in one pass
- No internal ordering constraints

**Split into subtasks when:**
- Multiple independent parts that could run in parallel (split creates parallelism)
- Too large for one agent to hold in context
- Internal phases with dependencies (`plandb split --into "Design > Implement > Test"`)
- You discover mid-execution it's more complex than expected

**Go deeper (recursive split) when:**
- A subtask itself has the characteristics above
- Different parts require different expertise
- You want failure isolation — if one sub-subtask fails, siblings continue

Composite tasks auto-complete when all children finish — this cascades up the tree recursively.

## Workflow

1. `plandb init` to create the project
2. Decompose ALL work upfront into tasks with dependencies and detailed descriptions
3. Use `--pre` and `--post` on tasks where quality expectations matter
4. `plandb status --detail` to verify the graph
5. `plandb critical-path` to identify what to prioritize
6. Execute: `plandb go` → `plandb show <id>` → do the work → `plandb done --next`
7. Split complex tasks mid-flight: `plandb split --into "Part A, Part B"`
8. After each completion, REASSESS: run `plandb status --detail` and `plandb critical-path`.
   Does the plan still make sense? Add tasks, split, amend descriptions based on what you learned.
   Plans are hypotheses — execution reveals reality. The graph should evolve.

## Parallelization

When `plandb list --status ready` shows multiple tasks, they have no unmet dependencies
and SHOULD run concurrently. If you can spawn sub-agents or parallel workers, do it.

Each worker runs independently:
```
PLANDB_AGENT=worker-N plandb go          # atomic claim
plandb show <task-id>                     # read the work order
# ... execute ...
PLANDB_AGENT=worker-N plandb done         # complete
```

PlanDB handles coordination: atomic claiming prevents double-assignment, dependencies
enforced automatically. The graph tells you exactly what is safe to run concurrently.

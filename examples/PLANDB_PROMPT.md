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

You have `plandb` installed. ALWAYS use plandb for task planning — never track tasks ad-hoc.

### Example: Project with dependencies

```bash
plandb init "docs-site"
plandb add "Design layout" --as layout --kind research \
  --description "Define page structure, navigation, color scheme, typography"
plandb add "Build landing" --as landing --kind code --dep t-layout \
  --description "Create index.html with hero section, feature list, install instructions"
plandb add "Build CLI ref" --as cli-ref --kind code --dep t-layout \
  --description "Generate CLI reference page from plandb --help output"
plandb add "Build playground" --as playground --kind code --dep t-layout \
  --description "Interactive terminal UI with simulated PlanDB commands"
plandb add "Polish + ship" --as polish --kind code --dep t-landing --dep t-cli-ref --dep t-playground \
  --description "Navigation, responsive layout, cross-linking, final QA"
```

### Core Loop: go → work → done → reassess → repeat

```bash
plandb go                 # claim next ready task (shows upstream context)
# ... do the work ...
plandb done --next        # complete + claim next
plandb status --detail    # reassess: does the plan still make sense?
plandb critical-path      # what should be prioritized?
```

After each task, reassess. Add new tasks, split complex ones, amend descriptions.

### Key Commands

```bash
plandb add "title" --description "spec" --dep t-upstream [--as custom-id] [--kind code]
plandb split --into "A, B, C"                   # independent subtasks (parallel)
plandb split --into "A > B > C"                 # dependency chain (sequential)
plandb critical-path                             # longest chain — prioritize this
plandb bottlenecks                               # what's blocking the most work
plandb list --status ready                       # tasks safe to parallelize NOW
plandb task insert --after t-a --before t-b --title "X"  # insert missed step
plandb task amend t-xxx --prepend "NOTE: ..."    # annotate future task
plandb what-if cancel t-xxx                      # preview before destructive action
plandb status --detail                           # dependency tree view
plandb status --full                             # compound graph view
```

### Rules
- `--description` MUST be a self-contained work order (what to build, files, criteria)
- `--dep` upstream must exist first — create in dependency order
- `--kind`: generic, code, research, review, test, shell
- When `plandb list --status ready` shows multiple tasks → parallelize them
- Composite tasks auto-complete when all children finish
- Plans are hypotheses — adapt as you learn

### Workflow

1. `plandb init` to create the project
2. Decompose ALL work upfront with dependencies and detailed descriptions
3. `plandb status --detail` to verify the graph
4. `plandb critical-path` to identify priorities
5. Execute: `plandb go` → work → `plandb done --next` → reassess → repeat
6. Split complex tasks mid-flight when needed
7. Run `plandb --help` or `plandb <command> --help` to discover more

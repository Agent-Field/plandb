use clap::{CommandFactory, Parser};
use plandb::cli::{Cli, Commands};
use plandb::db::init_db;

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Mcp) => {
            if let Err(err) = plandb::mcp::run_mcp_server(&cli.db) {
                eprintln!("error: {err}");
                std::process::exit(1);
            }
        }
        Some(Commands::Serve { port }) => {
            let db_path = cli.db.clone();
            let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
            if let Err(err) = rt.block_on(plandb::server::run_server(&db_path, port)) {
                eprintln!("error: {err}");
                std::process::exit(1);
            }
        }
        Some(Commands::Prompt { r#for, list }) => {
            if list || r#for.is_none() {
                println!("Available platforms:");
                println!("  mcp   — Claude Code, Cursor, Windsurf, any MCP client");
                println!("  cli   — Codex, Aider, any CLI-based agent");
                println!("  http  — OpenRouter, custom agents, any HTTP client");
                println!();
                println!("Usage: plandb prompt --for <platform>");
                return;
            }
            match r#for.as_deref().unwrap() {
                "mcp" => print_prompt_mcp(),
                "cli" => print_prompt_cli(&cli.db),
                "http" => print_prompt_http(),
                _ => unreachable!(),
            }
        }
        None => match init_db(&cli.db) {
            Ok(db) => {
                if let Ok(Some(project_id)) = plandb::db::get_meta(&db, "current_project") {
                    if let Err(err) = plandb::cli::project::status_cmd(
                        &db,
                        Some(&project_id),
                        false,
                        false,
                        false,
                        cli.json,
                        cli.compact,
                    ) {
                        eprintln!("error: {err}");
                        std::process::exit(1);
                    }
                } else {
                    let _ = Cli::command().print_help();
                    println!();
                }
            }
            Err(_) => {
                let _ = Cli::command().print_help();
                println!();
            }
        },
        Some(command) => match init_db(&cli.db)
            .and_then(|db| plandb::cli::run(&db, command, cli.json, cli.compact))
        {
            Ok(()) => {}
            Err(err) => {
                eprintln!("error: {err}");
                std::process::exit(1);
            }
        },
    }
}

fn print_prompt_mcp() {
    println!(
        r#"# ─── MCP Config ───────────────────────────────────────────────
# Add to your MCP settings (Claude Code, Cursor, Windsurf, any MCP client):

{{
  "mcpServers": {{
    "plandb": {{
      "command": "plandb",
      "args": ["mcp"]
    }}
  }}
}}

# ─── Paste into project instructions (CLAUDE.md, .cursorrules, etc.) ───

## Plandb — Task Graph for Agent Coordination

You have `plandb` available as an MCP server for managing task dependency graphs.
Use it to decompose complex work into tasks with dependencies, then execute them
in dependency order. The graph enforces ordering — you only see tasks whose
prerequisites are complete.

### When to Use Plandb
- Any task with 3+ steps that have ordering constraints
- Work that could be parallelized across agents
- Plans that might need mid-flight adaptation

### Core Workflow
1. Create a project: `plandb_project_create` with a name
2. Add tasks with dependencies — each task declares which tasks must finish first
3. Claim work: `plandb_go` returns the next ready task with handoff context from completed upstream tasks
4. Complete + advance: `plandb_done` marks complete, `plandb_go` gets the next one
5. Check progress: `plandb_status` shows done/total/ready/running counts

### Context Store (Project Knowledge)
- Record discoveries: `plandb_context_create` with kind (discovery, decision, pattern, etc.)
- Search knowledge: `plandb_search` — BM25 across context entries and task descriptions
- Context is auto-linked to your running task and auto-recalled when claiming related tasks
- Kind is freeform — use whatever labels fit (discovery, decision, constraint, bug, etc.)

### Plan Adaptation (mid-flight)
- `plandb_task_insert` — add a missed step between existing tasks
- `plandb_task_amend` — prepend notes to a future task ("use JWT not sessions")
- `plandb_what_if_cancel` — preview what happens before cancelling
- `plandb_ahead` — see what tasks are coming next

### Key Concepts
- Tasks flow: pending → ready (when deps done) → claimed → running → done/failed
- Dependency types: `feeds_into` (default), `blocks`, `suggests`
- Task kinds: `generic`, `code`, `research`, `review`, `test`, `shell`
- `plandb_go` returns relevant context entries automatically (lazy recall)
- Tasks can have `pre_hook`/`post_hook` — shell commands at start/completion
- Templates (`plandb export/import`) carry context entries as institutional knowledge
- Use `--compact` flag on tools for token-efficient output"#
    );
}

fn print_prompt_cli(db_path: &str) {
    println!(
        r#"# ─── Paste into system prompt, AGENTS.md, or project instructions ───

## PlanDB — Task Graph for Agent Coordination

You have `plandb` (binary at PATH, DB: {db_path}) for dependency-aware task planning.
ALWAYS use plandb for task management — never track tasks ad-hoc.
Every command shows hints about what you can do next — read them.

### Example: Project with dependencies

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
plandb status --detail    # verify the dependency graph
plandb critical-path      # see what to prioritize
```

### Core Loop: go → work → done → reassess → repeat

```bash
plandb go                 # claim next ready task — shows:
                          #   upstream handoff data, relevant context (auto-recalled),
                          #   downstream tasks, pre-conditions, and action hints
# ... do the work described in the task ...
plandb done --next        # complete current + claim next — shows:
                          #   what unlocked, post-conditions, next steps
plandb status --detail    # reassess: does the plan still make sense?
```

After each task completion, REASSESS. Add new tasks, split complex ones, amend
descriptions with discoveries. Plans are hypotheses — the graph evolves as you learn.

### Adding Tasks

```bash
plandb add "title" --description "detailed spec" --dep t-upstream [--as custom-id] [--kind code]
```

- `--description` is the work order (REQUIRED). Must be self-contained: what to build,
  files to create/modify, acceptance criteria. Another agent should execute it with no other context.
- `--dep` upstream must exist first — create in dependency order. Types: feeds_into (default), blocks, suggests
- `--kind`: generic, code, research, review, test, shell
- `--as`: custom ID (plandb add "X" --as foo → t-foo). Otherwise auto-generated (t-k3m9).
- `--pre-hook` / `--post-hook`: shell commands that run at task start/completion

### Decomposition

```bash
plandb split --into "Design, Implement, Test"   # independent subtasks (creates parallelism)
plandb split --into "Design > Implement > Test"  # dependency chain (sequential)
```

Split when: task has independent parts (parallel), is too large for one pass, or
proves more complex than expected mid-execution. Subtasks can be split further (any depth).
Use `plandb use t-xxx` to zoom into a subtree, `plandb use ..` to zoom out.

Cross-level dependencies: a subtask can depend on ANY task at ANY level:
```bash
plandb task add-dep --after t-backend-api t-frontend   # subtask → top-level
```

### Context Store (Project Knowledge)

Record what you discover while working. Context persists across sessions and is
automatically recalled when claiming related tasks.

```bash
plandb context "JWT expiry conflicts with session cache" --kind discovery
plandb context "use token bucket for rate limiting" --kind decision
plandb search "rate limiting"                      # BM25 search across context + tasks
plandb contexts --kind decision                    # list all decisions
```

Context is auto-linked to your current running task. --kind is freeform (discovery,
decision, pattern, constraint, bug — use whatever fits). When you `plandb go`,
relevant context entries are surfaced automatically (lazy recall).

### Plan Adaptation

```bash
plandb task insert --after t-a --before t-b --title "Missed step"   # rewires deps
plandb task amend t-xxx --prepend "NOTE: use JWT, not sessions"     # annotate future task
plandb what-if cancel t-xxx                                         # preview before acting
plandb critical-path                                                 # longest chain
plandb bottlenecks                                                   # what blocks the most work
```

### Multi-Agent Parallelism

When `plandb list --status ready` returns multiple tasks, they CAN run concurrently:
```bash
PLANDB_AGENT=worker-1 plandb go && PLANDB_AGENT=worker-2 plandb go
```
Atomic claiming prevents double-assignment. The graph IS the coordination layer.

### Quality Gates & Hooks

```bash
plandb add "Implement" --dep t-schema \
  --pre "schema defines all endpoints" --post "all routes return valid JSON" \
  --pre-hook 'echo "starting $PLANDB_TASK_TITLE"' \
  --post-hook 'pytest tests/' \
  --description "..."
```

### Templates (Replayable Procedures)

```bash
plandb export > template.yaml   # save structure + context as reusable template
plandb import template.yaml     # apply template (tasks, deps, hooks, context)
```

### Reference
- **States**: pending → ready (deps done) → claimed → running → done/failed/cancelled
- **Handoff**: `--result '{{"key":"val"}}'` on `done` passes data to downstream via `go`
- **Lazy recall**: `go` auto-surfaces relevant context entries for the claimed task
- **Scope**: `plandb use t-xxx` zooms into subtree, `plandb use ..` zooms out
- **Status**: `plandb status [--detail|--full|--full --verbose]`
- Run `plandb --help` or `plandb <command> --help` to discover all commands"#
    );
}

fn print_prompt_http() {
    println!(
        r#"# ─── HTTP Mode Setup ──────────────────────────────────────────
# Start the server first:
#   plandb serve --port 8080
#
# ─── Paste into system prompt or agent config ───

## Plandb — Task Graph REST API

You have a task graph API at http://localhost:8080 for managing dependencies between tasks.
Use it to decompose complex work, enforce ordering, and coordinate multiple agents.

### API Reference

PROJECT MANAGEMENT:
  POST   /projects                   Create project. Body: {{"name": "...", "description": "..."}}
  GET    /projects                   List all projects
  GET    /projects/:id               Get project details

TASK MANAGEMENT:
  POST   /tasks                      Create task. Body: {{"project_id": "...", "title": "...", "deps": ["t-xxx"], "kind": "code"}}
  GET    /tasks?project_id=X         List tasks (filter: status, kind, agent, tag)
  GET    /tasks/:id                  Get task details
  PATCH  /tasks/:id                  Update task fields

WORK LOOP:
  POST   /go                         Claim + start next ready task. Body: {{"project_id": "...", "agent_id": "..."}}
                                     Returns: task, handoff context, file conflicts, remaining counts
  POST   /tasks/:id/done             Complete task. Body: {{"result": ..., "files": ["src/x.rs"]}}
  POST   /tasks/:id/fail             Fail task. Body: {{"error": "..."}}
  POST   /tasks/:id/claim            Claim specific task. Body: {{"agent_id": "..."}}
  POST   /tasks/:id/heartbeat        Update heartbeat (proves agent alive)
  POST   /tasks/:id/progress         Report progress. Body: {{"percent": 50, "note": "..."}}
  POST   /tasks/:id/pause            Pause task

PLAN ADAPTATION:
  POST   /tasks/insert               Insert between tasks. Body: {{"after": "t-a", "before": "t-b", "title": "...", "project_id": "..."}}
  POST   /tasks/:id/amend            Prepend context. Body: {{"prepend": "NOTE: use JWT"}}
  POST   /what-if/cancel/:id         Preview cancel effects (read-only)
  GET    /ahead?project_id=X&depth=2 Lookahead buffer

STATUS:
  GET    /status?project_id=X        Project progress summary
  GET    /tasks/:id/notes            List notes on task
  POST   /tasks/:id/notes            Add note. Body: {{"content": "...", "agent_id": "..."}}

EVENTS (real-time):
  GET    /events?project_id=X        SSE stream of task state changes

### Key Concepts
- Task states: pending → ready (deps done) → claimed → running → done/failed
- Dependency types: `feeds_into` (default), `blocks`, `suggests`
- Task kinds: `generic`, `code`, `research`, `review`, `test`, `shell`
- IDs are short 8-char strings (e.g. `t-a1b2c3d4`)
- Add `?compact=true` to any GET for token-efficient responses
- POST /go is the preferred agent entry point — returns task + upstream context
- POST /tasks/:id/done with result data enables handoff to downstream tasks"#
    );
}

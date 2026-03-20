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

### Plan Adaptation (mid-flight)
- `plandb_task_insert` — add a missed step between existing tasks
- `plandb_task_amend` — prepend notes to a future task ("use JWT not sessions")
- `plandb_what_if_cancel` — preview what happens before cancelling
- `plandb_ahead` — see what tasks are coming next

### Key Concepts
- Tasks flow: pending → ready (when deps done) → claimed → running → done/failed
- Dependency types: `feeds_into` (default), `blocks`, `suggests`
- Task kinds: `generic`, `code`, `research`, `review`, `test`, `shell`
- IDs are short 8-char strings (e.g. `t-a1b2c3d4`)
- Fuzzy matching: misspell a task ID and plandb suggests the closest match
- Use `--compact` flag on tools for token-efficient output"#
    );
}

fn print_prompt_cli(db_path: &str) {
    println!(
        r#"# ─── Paste into system prompt, AGENTS.md, or project instructions ───

## Plandb — Task Graph for Agent Coordination

You have `plandb` (binary in PATH, DB: {db_path}) for dependency-aware task graphs.
The graph enforces ordering — `plandb go` only returns tasks whose dependencies are done.

### Core Loop (2 commands, no IDs needed)
```bash
plandb go          # claim + start next ready task (delivers upstream context)
plandb done --next # complete current task + claim next
```

`done` without a task ID completes your current running task.
Agent identity defaults to "default". Set `PLANDB_AGENT=name` for multi-agent.

### Setup
```bash
plandb init "my-project"
```

### Adding Tasks
```bash
plandb add "Design API"                          # positional title
plandb add "Implement" --dep t-abc               # with dependency
plandb add "Implement" --as impl                 # custom ID → t-impl
plandb add "Auth" --kind code --priority 10      # with metadata
plandb add "Tests" --dep t-abc --dep t-def       # multiple deps
```

Dep types: `feeds_into` (default), `blocks`, `suggests`. Example: `--dep t-abc:blocks`

### Decomposition (recursive)
```bash
plandb split --into "Design, Implement, Test"          # split current task (comma = independent)
plandb split --into "Design > Implement > Test"         # chain with > (linear deps)
plandb split t-abc --into "Part A, Part B"              # split specific task
plandb task decompose t-abc --file subtasks.yaml        # from YAML
plandb task replan t-abc --file revised.yaml            # cancel + recreate subtasks
```

Subtasks can be split further (any depth). Composite tasks auto-complete when all children finish.

### Scope (zoom into subtrees)
```bash
plandb use t-abc     # scope into composite task
plandb list          # shows only children of t-abc
plandb go            # claims from this scope
plandb use ..        # zoom out one level
plandb use --clear   # back to project root
```

### Status
```bash
plandb status              # progress summary
plandb status --detail     # per-task breakdown
plandb list --status ready # filter tasks
plandb show t-abc          # task details
plandb ahead               # what's next
plandb --json -c status    # compact JSON for LLM context
```

### Plan Adaptation
```bash
plandb task insert --after t-a --before t-b --title "Add validation"
plandb task amend t-abc --prepend "NOTE: use JWT"
plandb task pivot t-parent --file new-plan.yaml
plandb what-if cancel t-abc                        # preview effects (safe, read-only)
```

### Multi-Agent
```bash
PLANDB_AGENT=worker-1 plandb go && PLANDB_AGENT=worker-1 plandb done --next
PLANDB_AGENT=worker-2 plandb go && PLANDB_AGENT=worker-2 plandb done --next
```

### Reference
- **States**: pending → ready (deps done) → claimed → running → done/failed/cancelled
- **Dep types**: `feeds_into` (data flows), `blocks` (ordering), `suggests` (soft)
- **Kinds**: generic, code, research, review, test, shell
- **IDs**: short (`t-k3m`), fuzzy-matched on typos, custom via `--as`
- **Output**: `--json` for structured, `-c` for compact, default human-readable
- **Handoff**: `--result` on `done` passes data to downstream tasks via `go`"#
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

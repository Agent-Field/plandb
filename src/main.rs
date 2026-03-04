use clap::Parser;
use planq::cli::{Cli, Commands};
use planq::db::init_db;

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Mcp => {
            if let Err(err) = planq::mcp::run_mcp_server(&cli.db) {
                eprintln!("error: {err}");
                std::process::exit(1);
            }
        }
        Commands::Serve { port } => {
            let db_path = cli.db.clone();
            let port = *port;
            let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
            if let Err(err) = rt.block_on(planq::server::run_server(&db_path, port)) {
                eprintln!("error: {err}");
                std::process::exit(1);
            }
        }
        Commands::Prompt { r#for, list } => {
            if *list || r#for.is_none() {
                println!("Available platforms:");
                println!("  mcp   — Claude Code, Cursor, Windsurf, any MCP client");
                println!("  cli   — Codex, Aider, any CLI-based agent");
                println!("  http  — OpenRouter, custom agents, any HTTP client");
                println!();
                println!("Usage: planq prompt --for <platform>");
                return;
            }
            match r#for.as_deref().unwrap() {
                "mcp" => print_prompt_mcp(),
                "cli" => print_prompt_cli(&cli.db),
                "http" => print_prompt_http(),
                _ => unreachable!(),
            }
        }
        _ => match init_db(&cli.db).and_then(|db| planq::cli::run(&db, cli)) {
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
    "planq": {{
      "command": "planq",
      "args": ["mcp"]
    }}
  }}
}}

# ─── Paste into project instructions (CLAUDE.md, .cursorrules, etc.) ───

## Planq — Task Graph for Agent Coordination

You have `planq` available as an MCP server for managing task dependency graphs.
Use it to decompose complex work into tasks with dependencies, then execute them
in dependency order. The graph enforces ordering — you only see tasks whose
prerequisites are complete.

### When to Use Planq
- Any task with 3+ steps that have ordering constraints
- Work that could be parallelized across agents
- Plans that might need mid-flight adaptation

### Core Workflow
1. Create a project: `planq_project_create` with a name
2. Add tasks with dependencies — each task declares which tasks must finish first
3. Claim work: `planq_go` returns the next ready task with handoff context from completed upstream tasks
4. Complete + advance: `planq_done` marks complete, `planq_go` gets the next one
5. Check progress: `planq_status` shows done/total/ready/running counts

### Plan Adaptation (mid-flight)
- `planq_task_insert` — add a missed step between existing tasks
- `planq_task_amend` — prepend notes to a future task ("use JWT not sessions")
- `planq_what_if_cancel` — preview what happens before cancelling
- `planq_ahead` — see what tasks are coming next

### Key Concepts
- Tasks flow: pending → ready (when deps done) → claimed → running → done/failed
- Dependency types: `feeds_into` (default), `blocks`, `suggests`
- Task kinds: `generic`, `code`, `research`, `review`, `test`, `shell`
- IDs are short 8-char strings (e.g. `t-a1b2c3d4`)
- Fuzzy matching: misspell a task ID and planq suggests the closest match
- Use `--compact` flag on tools for token-efficient output"#
    );
}

fn print_prompt_cli(db_path: &str) {
    println!(
        r#"# ─── Paste into system prompt, AGENTS.md, or project instructions ───

## Planq — Task Graph for Agent Coordination

You have `planq` (binary in PATH, DB: {db_path}) for managing task dependency graphs.
Use it to decompose complex work into tasks with dependencies, then execute them in
dependency order. The graph enforces ordering — `planq go` only returns tasks whose
prerequisites are complete.

### When to Use Planq
- Any task with 3+ steps that have ordering constraints
- Work that could be parallelized across agents
- Plans that might need mid-flight adaptation

### Setup (once per project)
```bash
planq project create "my-project"
# Automatically set as default — no --project needed on subsequent commands
```

### Adding Tasks
```bash
# Simple task (no dependencies — becomes immediately ready)
planq task create --title "Design API schema" --kind research

# Task that depends on another (stays pending until dep completes)
planq task create --title "Implement endpoints" --dep t-a1b2c3d4

# Multiple dependencies
planq task create --title "Integration tests" --dep t-a1b2c3d4 --dep t-e5f6g7h8

# With description, priority, tags
planq task create --title "Auth middleware" --description "JWT-based, refresh tokens" \
  --kind code --priority 10 --tag auth --tag backend --dep t-a1b2c3d4

# Bulk create from YAML file
planq task create-batch --file tasks.yaml
```

### The Work Loop (2 commands)
```bash
# Claim + start next ready task (preferred entry point)
planq go --agent my-agent
# Returns: task details, handoff context from upstream tasks, file conflicts, progress

# ... do the work ...

# Complete + claim next in one command
planq done t-TASKID --result '{{\"summary\": \"implemented auth\"}}' --next --agent my-agent
# --result passes data to downstream tasks via handoff protocol
# --files "src/auth.rs,src/middleware.rs" enables conflict detection
```

### Checking Status
```bash
planq status                   # One-line: "5/12 done (42%) | ready: t-xx,t-yy | running: t-zz@agent-1"
planq status --detail          # Per-task breakdown with status icons
planq status --full            # All tasks + dependency edges
planq --json -c status         # Compact JSON (token-efficient for LLM consumption)
planq project dag              # Tree view of the dependency graph
planq task overview -c --json  # Full task list + deps + summary in compact JSON
```

### Plan Adaptation (change plans mid-flight)
```bash
# See what's coming next
planq ahead --depth 3

# Preview effects before acting (read-only, safe)
planq what-if cancel t-abc123

# Insert a missed step between two existing tasks (rewires dependencies)
planq task insert --after t-a1 --before t-b2 --title "Add input validation"

# Annotate a future task with new context
planq task amend t-future123 --prepend "NOTE: use JWT, not sessions"

# Replace an entire subtree with new plan
planq task pivot t-parent --keep-done --file new-plan.yaml

# Split one task into multiple sub-tasks
planq task split t-big --into '[{{"title":"Part 1"}},{{"title":"Part 2"}}]'

# Decompose into subtasks from YAML (with internal dependencies)
planq task decompose t-big --file subtasks.yaml

# Cancel + replan: cancel pending subtasks and create fresh ones
planq task replan t-parent --file revised-plan.yaml
```

### Inter-Agent Communication
```bash
# Leave a note on a task (visible to all agents)
planq task note t-abc123 "Found edge case: handle null emails" --agent agent-1

# Read notes left by other agents
planq task notes t-abc123
```

### Key Concepts
- **Task states**: pending → ready (when deps complete) → claimed → running → done/failed
- **Dependency types**: `feeds_into` (default, result passed downstream), `blocks` (ordering only), `suggests` (soft)
- **Task kinds**: `generic`, `code`, `research`, `review`, `test`, `shell`
- **IDs**: short 8-char strings like `t-a1b2c3d4` — every token matters
- **Fuzzy matching**: misspell a task ID and planq suggests the closest match
- **Default project**: `planq use <id>` sets default, no --project needed per command
- **Output modes**: human default, `--json` for structured, `-c`/`--compact` for token-efficient
- **Handoff protocol**: when you complete a task with --result, that data is available to the agent working on downstream tasks via `planq go`
- **Effect analysis**: insert/pivot/split responses include which tasks got delayed/accelerated/unblocked

### Multi-Agent Pattern
When `planq status` shows multiple ready tasks, a harness can spawn parallel agents:
```
Agent 1: planq go --agent agent-1 → work → planq done ID --next --agent agent-1
Agent 2: planq go --agent agent-2 → work → planq done ID --next --agent agent-2
```
Atomic claim protocol prevents two agents from claiming the same task."#
    );
}

fn print_prompt_http() {
    println!(
        r#"# ─── HTTP Mode Setup ──────────────────────────────────────────
# Start the server first:
#   planq serve --port 8080
#
# ─── Paste into system prompt or agent config ───

## Planq — Task Graph REST API

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

pub mod artifact;
pub mod events;
pub mod project;
pub mod task;

use crate::db::Database;
use crate::models::{DependencyKind, EventType, ProjectStatus, Task, TaskKind, TaskStatus};
use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use serde_json::{json, Value};
use std::str::FromStr;

// Re-export arg structs used by top-level aliases
pub use task::{CreateTaskArgs, DoneArgs, GetTaskArgs, GoArgs, ListTasksArgs, SplitTaskArgs};

#[derive(Parser, Debug)]
#[command(
    name = "plandb",
    version,
    infer_subcommands = true,
    about = "The agent-native database for task planning.\n\n\
        PlanDB is a compound graph — two orthogonal structures:\n\
        \x20 • Containment (place graph): tasks contain subtasks recursively, to any depth\n\
        \x20 • Dependencies (link graph): edges between tasks at ANY level, crossing containment boundaries\n\
        Dependencies do NOT follow the hierarchy. A subtask at depth 3 can depend on\n\
        a task at depth 0 in a different branch. More general than a DAG or hypergraph.\n\n\
        CORE LOOP:\n\
        \x20 plandb go                                              Claim next ready task\n\
        \x20 plandb done --next                                     Complete + claim next\n\n\
        ADDING TASKS:\n\
        \x20 plandb add \"title\" --description \"detailed spec\"        Every task needs a description\n\
        \x20 plandb add \"title\" --dep t-xxx --as my-id               Dependencies + custom IDs\n\
        \x20 plandb add \"title\" --pre \"precondition\" --post \"verify\" Quality gates\n\n\
        RECURSIVE DECOMPOSITION:\n\
        \x20 plandb split --into \"A, B, C\"     Split into independent subtasks (creates parallelism)\n\
        \x20 plandb split --into \"A > B > C\"   Split with dependency chain\n\
        \x20 plandb use <task-id>              Zoom into composite task scope\n\
        \x20 plandb use ..                     Zoom out one level\n\
        \x20 Subtasks can be split further — any depth. Composite tasks auto-complete when children finish.\n\n\
        GRAPH INTROSPECTION:\n\
        \x20 plandb critical-path              Longest dependency chain — prioritize this\n\
        \x20 plandb bottlenecks                Tasks blocking the most downstream work\n\
        \x20 plandb what-unlocks <task-id>     What becomes ready when a task completes\n\
        \x20 plandb watch                      Live-updating dashboard\n\
        \x20 plandb status --full              Compound graph view (containment + dependency edges)\n\n\
        DYNAMIC PLAN ADAPTATION:\n\
        \x20 plandb task insert                Add a step between existing tasks (rewires deps)\n\
        \x20 plandb task amend                 Annotate a future task with new context\n\
        \x20 plandb task pivot                 Replace a subtree with new tasks\n\
        \x20 plandb what-if cancel             Preview effects before committing\n\
        \x20 plandb export / import            Save and reuse decomposition patterns\n\n\
        MULTI-AGENT PARALLELISM:\n\
        \x20 When plandb list --status ready returns multiple tasks, they can run concurrently.\n\
        \x20 Each agent: plandb go → work → plandb done --next. Atomic claiming prevents conflicts.\n\n\
        CONCEPTS:\n\
        \x20 States:      pending → ready (deps done) → claimed → running → done/failed\n\
        \x20 Dep types:   feeds_into (data flows), blocks (ordering), suggests (soft)\n\
        \x20 Kinds:       generic, code, research, review, test, shell\n\
        \x20 IDs:         short (t-k3m9) or custom (--as api → t-api). Fuzzy-matched.\n\
        \x20 Description: ALWAYS use --description. It's the work order, not the title.\n\n\
        ENVIRONMENT:\n\
        \x20 PLANDB_AGENT  Default agent ID (avoids --agent on every command)\n\
        \x20 PLANDB_DB     Path to SQLite database (default: .plandb.db)",
    after_help = "EXAMPLES:\n\
        \x20 plandb init \"auth-system\"                                   Create project\n\
        \x20 plandb add \"Design schema\" --kind research                  Add a task\n\
        \x20 plandb add \"Implement\" --dep t-a1b2c3 --as impl             Add dependent task with custom ID\n\
        \x20 plandb go                                                    Claim + start next ready\n\
        \x20 plandb done --result '{\"api\":\"done\"}' --next               Complete current + claim next\n\
        \x20 plandb split --into \"A, B, C\"                               Split current task\n\
        \x20 plandb split --into \"A > B > C\"                             Split with chain\n\
        \x20 plandb use t-a1b2c3                                          Scope into composite task\n\
        \x20 plandb use ..                                                Go up one level\n\
        \x20 plandb task insert --after t-a1 --before t-b2 --title \"Add validation\"\n\
        \x20 plandb what-if cancel t-a1b2c3                               Preview cancel effects\n\
        \x20 plandb status --detail                                       Per-task breakdown\n\
        \x20 plandb --json -c status                                      Compact JSON for LLMs"
)]
pub struct Cli {
    #[arg(long, default_value_t = default_db_path(), global = true, help = "Path to SQLite database file")]
    pub db: String,

    #[arg(long, global = true, help = "Output as structured JSON")]
    pub json: bool,

    #[arg(
        long,
        short = 'c',
        global = true,
        help = "Compact output optimized for LLM context windows"
    )]
    pub compact: bool,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

fn default_db_path() -> String {
    // 1. Explicit env var takes priority
    if let Ok(path) = std::env::var("PLANDB_DB") {
        return path;
    }
    // 2. Walk up from CWD looking for .plandb.db
    if let Ok(mut dir) = std::env::current_dir() {
        loop {
            let candidate = dir.join(".plandb.db");
            if candidate.exists() {
                return candidate.to_string_lossy().into_owned();
            }
            if !dir.pop() {
                break;
            }
        }
    }
    // 3. Fall back to CWD (will be created by init, or error on other commands)
    ".plandb.db".to_string()
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    #[command(about = "Manage projects (create, list, status, dag)")]
    Project(project::ProjectCommand),
    #[command(about = "Manage tasks (create, claim, complete, adapt)")]
    Task(task::TaskCommand),
    #[command(about = "Preview effects of mutations without applying them")]
    WhatIf(task::WhatIfCommand),
    #[command(about = "Attach/read artifacts (files, outputs) on tasks")]
    Artifact(artifact::ArtifactCommand),
    #[command(about = "List or watch project events in real-time")]
    Events(events::EventsCommand),
    #[command(
        about = "Show upcoming tasks after current running tasks complete.\n\n\
                  Returns the lookahead buffer: currently running tasks and the next N layers\n\
                  of tasks that will become ready as current tasks complete.\n\
                  Useful for agents to anticipate what's coming and prepare."
    )]
    Ahead {
        #[arg(
            long,
            default_value_t = 2,
            help = "Number of dependency layers to look ahead"
        )]
        depth: usize,
        #[arg(long, help = "Project ID (uses default if not set)")]
        project: Option<String>,
    },
    #[command(
        about = "Set scope: project ('plandb use <project>'), composite task ('plandb use <task>'), or parent ('plandb use ..')"
    )]
    Use {
        #[arg(help = "Project ID, task ID, or '..' to go up")]
        target: Option<String>,
        #[arg(long, help = "Clear scope to project root")]
        clear: bool,
    },
    #[command(
        about = "Show project progress: done/total, ready tasks, running agents.\n\n\
                  Three detail levels:\n\
                  \x20 plandb status              One-line summary with counts\n\
                  \x20 plandb status --detail      Per-task breakdown with status icons\n\
                  \x20 plandb status --full        Compound graph: containment tree + dependencies\n\
                  \x20 plandb status --full --verbose  Everything: descriptions, notes, results, conditions"
    )]
    Status {
        #[arg(long, help = "Project ID (uses default if not set)")]
        project: Option<String>,
        #[arg(long, help = "Show per-task breakdown")]
        detail: bool,
        #[arg(long, help = "Show all tasks and dependencies")]
        full: bool,
        #[arg(long, short = 'v', help = "With --full: include descriptions, notes, results, conditions")]
        verbose: bool,
    },
    #[command(about = "Claim + start next ready task (shortcut for 'plandb task go')")]
    Go(GoArgs),
    #[command(about = "Complete a task, optionally claim next (shortcut for 'plandb task done')")]
    Done(DoneArgs),
    #[command(about = "List tasks with optional filters (shortcut for 'plandb task list')")]
    List(ListTasksArgs),
    #[command(about = "Create a new task (shortcut for 'plandb task create')")]
    Add(CreateTaskArgs),
    #[command(about = "Show full details of a task (shortcut for 'plandb task get')")]
    Show(GetTaskArgs),
    #[command(about = "Split a task into sub-tasks (shortcut for 'plandb task split')")]
    Split(SplitTaskArgs),
    #[command(hide = true, about = "Alias for 'done'")]
    Complete(DoneArgs),
    #[command(hide = true, about = "Alias for 'done'")]
    Finish(DoneArgs),
    #[command(hide = true, about = "Alias for 'done'")]
    Update(DoneArgs),
    #[command(hide = true, about = "Alias for 'task list'")]
    Tasks(ListTasksArgs),
    #[command(hide = true, about = "Alias for 'task list'")]
    Ls(ListTasksArgs),
    #[command(about = "Show the critical path — longest dependency chain to completion")]
    CriticalPath {
        #[arg(long, help = "Project ID (uses default if not set)")]
        project: Option<String>,
    },
    #[command(about = "Find bottleneck tasks — tasks blocking the most downstream work")]
    Bottlenecks {
        #[arg(long, help = "Project ID (uses default if not set)")]
        project: Option<String>,
        #[arg(long, default_value_t = 5, help = "Number of bottlenecks to show")]
        limit: i64,
    },
    #[command(about = "Show what tasks become ready when a specific task completes")]
    WhatUnlocks {
        #[arg(help = "Task ID to check")]
        task_id: String,
    },
    #[command(about = "Export project graph as a reusable template (YAML)")]
    Export {
        #[arg(long, help = "Project ID (uses default if not set)")]
        project: Option<String>,
        #[arg(long, help = "Template name")]
        name: Option<String>,
        #[arg(long, help = "Template description")]
        description: Option<String>,
    },
    #[command(about = "Import a template to create tasks from a saved graph pattern")]
    Import {
        #[arg(help = "Path to template YAML file")]
        file: String,
        #[arg(long, help = "Project ID (uses default if not set)")]
        project: Option<String>,
    },
    #[command(about = "Live-watch project progress (refreshes every N seconds)")]
    Watch {
        #[arg(long, help = "Project ID (uses default if not set)")]
        project: Option<String>,
        #[arg(long, default_value_t = 2, help = "Refresh interval in seconds")]
        interval: u64,
    },
    #[command(hide = true, about = "Alias for 'status'")]
    Overview,
    #[command(hide = true, about = "Alias for 'task start'")]
    Start(task::TaskIdArg),
    #[command(
        about = "Create a project and set as default (shortcut for 'plandb project create')"
    )]
    Init {
        #[arg(help = "Project name")]
        name: String,
        #[arg(long, help = "Optional description of the project's goal")]
        description: Option<String>,
    },
    #[command(hide = true, about = "Alias for '--version'")]
    Version,
    #[command(about = "Start MCP server (stdio JSON-RPC for Claude Code, Cursor, Windsurf)")]
    Mcp,
    #[command(about = "Start HTTP server with REST API and SSE event stream")]
    Serve {
        #[arg(long, short, default_value = "8484", help = "Port to listen on")]
        port: u16,
    },
    #[command(
        about = "Generate integration prompt/config for your agent platform.\n\n\
                  Outputs ready-to-paste configuration for:\n\
                  \x20 mcp   — MCP config JSON for Claude Code, Cursor, Windsurf\n\
                  \x20 cli   — System prompt snippet for Codex, Aider, CLI agents\n\
                  \x20 http  — REST API instructions for custom agents"
    )]
    Prompt {
        #[arg(long, value_parser = ["mcp", "cli", "http"], help = "Target platform: mcp, cli, or http")]
        r#for: Option<String>,
        #[arg(long, help = "List available platforms")]
        list: bool,
    },
    #[command(
        about = "Attach context to the project graph — discoveries, decisions, patterns.\n\n\
                  Context persists across sessions and is searchable via BM25.\n\
                  Unlike notes (task-scoped), context is project-wide and queryable.\n\
                  Types: discovery, decision, pattern, blocker, reference"
    )]
    Context {
        #[arg(help = "The context content")]
        content: String,
        #[arg(long, default_value = "discovery", value_parser = ["discovery", "decision", "pattern", "blocker", "reference"],
              help = "Kind of context")]
        r#type: String,
        #[arg(long, help = "Associated task ID (optional — context can be project-wide)")]
        task: Option<String>,
        #[arg(long, help = "Project ID (uses default if not set)")]
        project: Option<String>,
        #[arg(long, help = "Tags for categorization (comma-separated)")]
        tags: Option<String>,
    },
    #[command(
        about = "BM25-ranked search across the entire project graph.\n\n\
                  Searches context entries, task titles, descriptions, and notes.\n\
                  Returns results ranked by relevance — the project's knowledge base."
    )]
    Search {
        #[arg(help = "Search query (BM25 ranked)")]
        query: String,
        #[arg(long, help = "Project ID (uses default if not set)")]
        project: Option<String>,
        #[arg(long, default_value_t = 10, help = "Max results")]
        limit: usize,
    },
    #[command(about = "List context entries attached to the project graph")]
    Contexts {
        #[arg(long, help = "Project ID (uses default if not set)")]
        project: Option<String>,
        #[arg(long, help = "Filter by kind: discovery, decision, pattern, blocker, reference")]
        kind: Option<String>,
        #[arg(long, default_value_t = 50, help = "Max results")]
        limit: usize,
    },
    #[command(about = "Remove a context entry from the project graph")]
    Prune {
        #[arg(help = "Context entry ID to remove")]
        context_id: String,
    },
}

pub fn run(db: &Database, command: Commands, json: bool, compact: bool) -> Result<()> {
    match command {
        Commands::Project(command) => project::run(db, command, json, compact),
        Commands::Task(command) => task::run(db, command, json, compact),
        Commands::WhatIf(command) => task::run_what_if(db, command, json, compact),
        Commands::Artifact(command) => artifact::run(db, command, json),
        Commands::Events(command) => events::run(db, command, json),
        Commands::Ahead { depth, project } => task::ahead_cmd(db, project, depth, json, compact),
        Commands::Use { target, clear } => {
            if clear {
                crate::db::delete_meta(db, "current_project")?;
                crate::db::delete_meta(db, "current_scope")?;
                if json {
                    print_json(&json!({"cleared": true}))?;
                } else {
                    println!("cleared scope");
                }
                return Ok(());
            }

            if let Some(target) = target {
                if target == ".." {
                    // Go up one level
                    if let Some(scope) = crate::db::get_meta(db, "current_scope")? {
                        let task = crate::db::get_task(db, &scope)?;
                        if let Some(parent_id) = task.parent_task_id {
                            crate::db::set_meta(db, "current_scope", &parent_id)?;
                            if json {
                                print_json(&json!({"scope": parent_id}))?;
                            } else {
                                println!("scope: {parent_id}");
                            }
                        } else {
                            crate::db::delete_meta(db, "current_scope")?;
                            if json {
                                print_json(&json!({"scope": null}))?;
                            } else {
                                println!("scope: project root");
                            }
                        }
                    } else if json {
                        print_json(&json!({"scope": null}))?;
                    } else {
                        println!("already at project root");
                    }
                    return Ok(());
                }

                // Try as task ID first (for scoping into composite tasks)
                if target.starts_with("t-") {
                    if let Ok(task) = crate::db::get_task(db, &target) {
                        crate::db::set_meta(db, "current_scope", &task.id)?;
                        crate::db::set_meta(db, "current_project", &task.project_id)?;
                        if json {
                            print_json(&json!({"scope": task.id, "project": task.project_id}))?;
                        } else {
                            println!("scope: {} \"{}\"", task.id, task.title);
                        }
                        return Ok(());
                    }
                }

                // Try as project ID (existing behavior)
                crate::db::get_project(db, &target)?;
                crate::db::set_meta(db, "current_project", &target)?;
                crate::db::delete_meta(db, "current_scope")?;
                if json {
                    print_json(&json!({"current_project": target}))?;
                } else {
                    println!("default project: {target}");
                }
            } else {
                let current_project = crate::db::get_meta(db, "current_project")?;
                let current_scope = crate::db::get_meta(db, "current_scope")?;
                if json {
                    print_json(&json!({"current_project": current_project, "scope": current_scope}))?;
                } else {
                    if let Some(ref p) = current_project {
                        println!("project: {p}");
                    }
                    if let Some(ref s) = current_scope {
                        println!("scope: {s}");
                    }
                    if current_project.is_none() && current_scope.is_none() {
                        println!("no scope set");
                    }
                }
            }
            Ok(())
        }
        Commands::Status {
            project,
            detail,
            full,
            verbose,
        } => project::status_cmd(db, project.as_deref(), detail, full, verbose, json, compact),
        Commands::Go(args) => task::go_cmd(db, &args, json),
        Commands::Done(args) => task::done_cmd(db, args, json, compact),
        Commands::List(args) => task::list_tasks_cmd(db, args, json, compact),
        Commands::Add(args) => task::create_task_cmd(db, args, json, compact),
        Commands::Show(args) => {
            let t = crate::db::fuzzy_find_task(db, &args.task_id, None)?;
            if json || args.json {
                if compact {
                    print_json(&compact_task(&t))?;
                } else {
                    print_json(&t)?;
                }
            } else {
                task::print_task_detail(&t);
            }
            Ok(())
        }
        Commands::Split(args) => task::split_cmd(db, args, json),
        Commands::Complete(args) | Commands::Finish(args) | Commands::Update(args) => {
            task::done_cmd(db, args, json, compact)
        }
        Commands::Tasks(args) | Commands::Ls(args) => task::list_tasks_cmd(db, args, json, compact),
        Commands::CriticalPath { project } => {
            let project_id = resolve_project_id(db, project.as_deref())?;
            if let Some((path, length)) = crate::db::critical_path(db, &project_id)? {
                if json {
                    print_json(&serde_json::json!({"path": path, "length": length}))?;
                } else {
                    println!("Critical path ({length} tasks):");
                    for task_id in path.split(" > ") {
                        if let Ok(task) = crate::db::get_task(db, task_id) {
                            println!(
                                "  {} {} {} [{}]",
                                status_icon(&task.status),
                                task.id,
                                task.title,
                                task.status
                            );
                        }
                    }
                }
            } else if json {
                print_json(&serde_json::json!({"path": null, "length": 0}))?;
            } else {
                println!("no critical path (all tasks done or none exist)");
            }
            Ok(())
        }
        Commands::Bottlenecks { project, limit } => {
            let project_id = resolve_project_id(db, project.as_deref())?;
            let bottlenecks = crate::db::find_bottlenecks(db, &project_id, limit)?;
            if json {
                print_json(
                    &bottlenecks
                        .iter()
                        .map(|b| {
                            serde_json::json!({
                                "task_id": b.task_id,
                                "title": b.title,
                                "status": b.status,
                                "downstream_count": b.downstream_count
                            })
                        })
                        .collect::<Vec<_>>(),
                )?;
            } else if bottlenecks.is_empty() {
                println!("no bottlenecks (nothing blocking downstream work)");
            } else {
                println!("Bottlenecks (tasks blocking the most downstream work):");
                for b in &bottlenecks {
                    println!(
                        "  {} {} — blocks {} tasks [{}]",
                        b.task_id, b.title, b.downstream_count, b.status
                    );
                }
            }
            Ok(())
        }
        Commands::WhatUnlocks { task_id } => {
            let unlocked = crate::db::what_unlocks(db, &task_id)?;
            if json {
                print_json(
                    &unlocked
                        .iter()
                        .map(|u| {
                            serde_json::json!({
                                "task_id": u.task_id,
                                "title": u.title,
                                "status": u.status
                            })
                        })
                        .collect::<Vec<_>>(),
                )?;
            } else if unlocked.is_empty() {
                println!("completing {} unlocks no pending tasks", task_id);
            } else {
                println!("Completing {} unlocks:", task_id);
                for u in &unlocked {
                    println!("  → {} {} [{}]", u.task_id, u.title, u.status);
                }
            }
            Ok(())
        }
        Commands::Watch {
            project,
            interval,
        } => {
            let project_id = resolve_project_id(db, project.as_deref())?;
            loop {
                // Clear screen
                print!("\x1b[2J\x1b[H");
                let _ = project::status_cmd(db, Some(&project_id), true, false, false, json, compact);
                // Show ready count for parallelization hint
                let ready = crate::db::list_tasks(db, crate::db::TaskListFilters {
                    project_id: Some(project_id.clone()),
                    status: Some(crate::models::TaskStatus::Ready),
                    ..Default::default()
                })?;
                if ready.len() > 1 {
                    eprintln!();
                    eprintln!("  {} tasks ready — parallelize!", ready.len());
                }
                // Check if done
                let state = crate::db::project_state(db, &project_id)?;
                if state.done == state.total && state.total > 0 {
                    eprintln!();
                    eprintln!("  all tasks complete!");
                    break;
                }
                std::thread::sleep(std::time::Duration::from_secs(interval));
            }
            Ok(())
        }
        Commands::Overview => project::status_cmd(db, None, false, false, false, json, compact),
        Commands::Start(args) => {
            let task = crate::db::start_task(db, &args.task_id)?;
            if json {
                print_json(&task)?;
            } else {
                println!("started {}", task.id);
            }
            Ok(())
        }
        Commands::Export {
            project,
            name,
            description,
        } => {
            let project_id = resolve_project_id(db, project.as_deref())?;
            let proj = crate::db::get_project(db, &project_id)?;
            let template_name = name.unwrap_or_else(|| proj.name.clone());
            let template = crate::db::export_graph(
                db,
                &project_id,
                &template_name,
                description.as_deref(),
            )?;
            // Always output as YAML regardless of --json flag (it's a file format)
            println!("{}", serde_yaml::to_string(&template)?);
            Ok(())
        }
        Commands::Import { file, project } => {
            let project_id = resolve_project_id(db, project.as_deref())?;
            let content = std::fs::read_to_string(&file)?;
            let template: crate::db::GraphTemplate = serde_yaml::from_str(&content)?;
            let ref_to_id = crate::db::import_graph(db, &project_id, &template)?;
            if json {
                print_json(&serde_json::json!({
                    "imported": ref_to_id.len(),
                    "ref_to_id": ref_to_id,
                    "template_name": template.name,
                }))?;
            } else {
                println!(
                    "imported {} tasks from template \"{}\"",
                    ref_to_id.len(),
                    template.name
                );
                for (ref_id, task_id) in &ref_to_id {
                    println!("  {} -> {}", ref_id, task_id);
                }
            }
            Ok(())
        }
        Commands::Init { name, description } => {
            let project = crate::db::create_project(db, &name, description, None)?;
            crate::db::set_meta(db, "current_project", &project.id)?;
            if json {
                print_json(&project)?;
            } else {
                println!("created {} ({})", project.id, project.name);
                if !compact {
                    eprintln!();
                    eprintln!("next: plandb add \"First task\"");
                    eprintln!("tip:  start with 1-2 tasks. add more as you learn things.");
                }
            }
            Ok(())
        }
        Commands::Version => {
            println!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Commands::Context {
            content,
            r#type,
            task,
            project,
            tags,
        } => {
            let project_id = resolve_project_id(db, project.as_deref())?;
            let kind: crate::db::LearningKind = r#type.parse()?;
            let agent_id = std::env::var("PLANDB_AGENT").ok();
            let tag_list: Vec<String> = tags
                .map(|t| t.split(',').map(|s| s.trim().to_string()).collect())
                .unwrap_or_default();
            let entry = crate::db::add_learning(
                db,
                &project_id,
                task.as_deref(),
                agent_id.as_deref(),
                kind,
                &content,
                &tag_list,
            )?;
            if json {
                print_json(&entry)?;
            } else {
                println!("{} [{}]", entry.id, entry.kind);
                if compact {
                    return Ok(());
                }
                if let Some(ref t) = entry.task_id {
                    println!("  task: {t}");
                }
                if !entry.tags.is_empty() {
                    println!("  tags: {}", entry.tags.join(", "));
                }
            }
            Ok(())
        }
        Commands::Search {
            query,
            project,
            limit,
        } => {
            let project_id = resolve_project_id(db, project.as_deref())?;
            let results = crate::db::recall(db, &project_id, &query, limit)?;
            if json {
                print_json(&results)?;
            } else if results.is_empty() {
                println!("no results for \"{}\"", query);
            } else {
                for r in &results {
                    let source_tag = match r.source.as_str() {
                        "learning" => "ctx",
                        "task" => "task",
                        _ => "?",
                    };
                    let truncated = if r.content.len() > 120 {
                        format!("{}...", &r.content[..120])
                    } else {
                        r.content.clone()
                    };
                    println!("  [{}] {} {} {}", source_tag, r.id, r.kind, truncated);
                }
            }
            Ok(())
        }
        Commands::Contexts {
            project,
            kind,
            limit,
        } => {
            let project_id = resolve_project_id(db, project.as_deref())?;
            let entries =
                crate::db::list_learnings(db, &project_id, kind.as_deref(), limit)?;
            if json {
                print_json(&entries)?;
            } else if entries.is_empty() {
                println!("no context entries yet");
                if !compact {
                    eprintln!("tip: plandb context \"what you discovered\" --type discovery");
                }
            } else {
                for e in &entries {
                    let truncated = if e.content.len() > 100 {
                        format!("{}...", &e.content[..100])
                    } else {
                        e.content.clone()
                    };
                    println!("  {} [{}] {}", e.id, e.kind, truncated);
                }
            }
            Ok(())
        }
        Commands::Prune { context_id } => {
            let deleted = crate::db::delete_learning(db, &context_id)?;
            if json {
                print_json(&serde_json::json!({"deleted": deleted}))?;
            } else if deleted > 0 {
                println!("pruned {context_id}");
            } else {
                println!("context entry {context_id} not found");
            }
            Ok(())
        }
        Commands::Mcp | Commands::Serve { .. } | Commands::Prompt { .. } => {
            unreachable!("handled in main")
        }
    }
}

pub fn resolve_project_id(db: &Database, explicit: Option<&str>) -> Result<String> {
    // 1. Explicit --project flag
    if let Some(project_id) = explicit {
        return Ok(project_id.to_string());
    }
    // 2. Stored current_project from meta table
    if let Some(project_id) = crate::db::get_meta(db, "current_project")? {
        return Ok(project_id);
    }
    // 3. Auto-select if only one project exists
    let projects = crate::db::list_projects(db)?;
    if projects.len() == 1 {
        return Ok(projects[0].id.clone());
    }
    // 4. Multiple projects: list them in error
    if !projects.is_empty() {
        let names: Vec<_> = projects
            .iter()
            .map(|p| format!("  {} ({})", p.id, p.name))
            .collect();
        return Err(anyhow!(
            "Multiple projects found. Use --project <id> or 'plandb use <id>':\n{}",
            names.join("\n")
        ));
    }
    Err(anyhow!(
        "No projects found. Run 'plandb init <name>' to create one."
    ))
}

pub(crate) fn parse_project_status(input: &str) -> std::result::Result<ProjectStatus, String> {
    ProjectStatus::from_str(input)
}

pub(crate) fn parse_task_status(input: &str) -> std::result::Result<TaskStatus, String> {
    TaskStatus::from_str(input)
}

pub(crate) fn parse_task_kind(input: &str) -> std::result::Result<TaskKind, String> {
    TaskKind::from_str(input)
}

pub(crate) fn parse_dependency_kind(input: &str) -> std::result::Result<DependencyKind, String> {
    DependencyKind::from_str(input)
}

pub(crate) fn parse_event_type(input: &str) -> std::result::Result<EventType, String> {
    EventType::from_str(input)
}

pub(crate) fn print_json<T: serde::Serialize>(value: &T) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

pub(crate) fn should_color() -> bool {
    std::env::var_os("NO_COLOR").is_none()
}

pub(crate) fn colorize(text: &str, ansi_code: &str) -> String {
    if should_color() {
        format!("\x1b[{ansi_code}m{text}\x1b[0m")
    } else {
        text.to_string()
    }
}

pub(crate) fn status_icon(status: &TaskStatus) -> &'static str {
    match status {
        TaskStatus::Done | TaskStatus::DonePartial => "✓",
        TaskStatus::Running | TaskStatus::Claimed => "◉",
        TaskStatus::Ready => "○",
        TaskStatus::Pending => "·",
        TaskStatus::Failed => "✗",
        TaskStatus::Cancelled => "⊘",
    }
}

pub(crate) fn color_task_status(status: &TaskStatus) -> String {
    let label = status.to_string();
    match status {
        TaskStatus::Done | TaskStatus::DonePartial => colorize(&label, "32"),
        TaskStatus::Running | TaskStatus::Claimed => colorize(&label, "33"),
        TaskStatus::Ready => colorize(&label, "34"),
        TaskStatus::Pending => colorize(&label, "90"),
        TaskStatus::Failed => colorize(&label, "31"),
        TaskStatus::Cancelled => colorize(&label, "31"),
    }
}

pub(crate) fn compact_task(task: &Task) -> Value {
    json!({
        "id": task.id,
        "title": task.title,
        "status": task.status,
        "kind": task.kind,
        "agent_id": task.agent_id,
        "priority": task.priority,
    })
}

pub(crate) fn minimal_task(task: &Task) -> Value {
    json!({
        "id": task.id,
        "status": task.status,
    })
}

pub(crate) fn print_table(headers: &[&str], rows: &[Vec<String>]) {
    if rows.is_empty() {
        println!("(no rows)");
        return;
    }

    let mut widths = headers.iter().map(|h| h.len()).collect::<Vec<_>>();
    for row in rows {
        for (idx, value) in row.iter().enumerate() {
            widths[idx] = widths[idx].max(value.chars().count());
        }
    }

    let header_line = headers
        .iter()
        .enumerate()
        .map(|(idx, h)| format!("{h:<width$}", width = widths[idx]))
        .collect::<Vec<_>>()
        .join("  ");
    println!("{header_line}");

    let sep_line = widths
        .iter()
        .map(|w| "-".repeat(*w))
        .collect::<Vec<_>>()
        .join("  ");
    println!("{sep_line}");

    for row in rows {
        let line = row
            .iter()
            .enumerate()
            .map(|(idx, value)| format!("{value:<width$}", width = widths[idx]))
            .collect::<Vec<_>>()
            .join("  ");
        println!("{line}");
    }
}

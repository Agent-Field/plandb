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

#[derive(Parser, Debug)]
#[command(
    name = "planq",
    version,
    about = "Task graph primitive for AI agent orchestration.\n\n\
        Manages a dependency-aware task graph in SQLite. Three interfaces: CLI, MCP server, HTTP API.\n\
        Agents decompose work into tasks with dependencies, then execute via a claim-and-complete loop.\n\n\
        WORKFLOW:\n\
        \x20 1. planq project create \"my-project\"                    Create a project\n\
        \x20 2. planq task create --title \"Design API\" --dep t-xxx   Add tasks with dependencies\n\
        \x20 3. planq go --agent agent-1                             Claim next ready task\n\
        \x20 4. planq done <TASK_ID> --next --agent agent-1          Complete + claim next\n\
        \x20 5. planq status                                         Check progress\n\n\
        PLAN ADAPTATION:\n\
        \x20 planq ahead              See upcoming tasks in the lookahead buffer\n\
        \x20 planq what-if cancel     Preview effects of cancelling a task\n\
        \x20 planq task insert        Add a step between existing tasks\n\
        \x20 planq task amend         Annotate a future task with new context\n\
        \x20 planq task pivot         Replace a subtree with new tasks\n\
        \x20 planq task split         Decompose a task mid-execution\n\n\
        MULTI-AGENT:\n\
        \x20 Each agent runs: planq go --agent <NAME> → work → planq done <ID> --next --agent <NAME>\n\
        \x20 The graph ensures no two agents claim the same task. Dependencies are enforced.\n\n\
        CONCEPTS:\n\
        \x20 Task states: pending → ready (deps done) → claimed → running → done/failed\n\
        \x20 Dep types:   feeds_into (default, passes result downstream), blocks, suggests\n\
        \x20 Task kinds:  generic, code, research, review, test, shell\n\
        \x20 IDs:         short 8-char (e.g. t-a1b2c3d4). Fuzzy-matched on typos.\n\n\
        OUTPUT MODES:\n\
        \x20 Default human-readable. --json for structured JSON. -c/--compact for token-efficient output.",
    after_help = "EXAMPLES:\n\
        \x20 planq project create \"auth-system\"                         Create project\n\
        \x20 planq task create --title \"Design schema\" --kind research  Add a task\n\
        \x20 planq task create --title \"Implement\" --dep t-a1b2c3       Add dependent task\n\
        \x20 planq go --agent claude-1                                   Claim + start next ready\n\
        \x20 planq done t-d4e5f6 --result '{\"api\":\"done\"}' --next --agent claude-1\n\
        \x20 planq task insert --after t-a1 --before t-b2 --title \"Add validation\"\n\
        \x20 planq what-if cancel t-a1b2c3                               Preview cancel effects\n\
        \x20 planq status --detail                                       Per-task breakdown\n\
        \x20 planq --json -c status                                      Compact JSON for LLMs\n\n\
        ENVIRONMENT:\n\
        \x20 PLANQ_DB     Path to SQLite database (default: .planq.db)"
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
    pub command: Commands,
}

fn default_db_path() -> String {
    std::env::var("PLANQ_DB").unwrap_or_else(|_| ".planq.db".to_string())
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
        about = "Set or show the default project (avoids --project on every command).\n\n\
                  Once set, all commands that accept --project will use this default.\n\
                  Run without arguments to show current default. Use --clear to unset."
    )]
    Use {
        #[arg(help = "Project ID to set as default")]
        project_id: Option<String>,
        #[arg(long, help = "Clear the default project")]
        clear: bool,
    },
    #[command(
        about = "Show project progress: done/total, ready tasks, running agents.\n\n\
                  Three detail levels:\n\
                  \x20 planq status             One-line summary with counts\n\
                  \x20 planq status --detail     Per-task breakdown with status icons\n\
                  \x20 planq status --full       All tasks + dependency edges"
    )]
    Status {
        #[arg(long, help = "Project ID (uses default if not set)")]
        project: Option<String>,
        #[arg(long, help = "Show per-task breakdown")]
        detail: bool,
        #[arg(long, help = "Show all tasks and dependencies")]
        full: bool,
    },
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
}

pub fn run(db: &Database, cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Project(command) => project::run(db, command, cli.json, cli.compact),
        Commands::Task(command) => task::run(db, command, cli.json, cli.compact),
        Commands::WhatIf(command) => task::run_what_if(db, command, cli.json, cli.compact),
        Commands::Artifact(command) => artifact::run(db, command, cli.json),
        Commands::Events(command) => events::run(db, command, cli.json),
        Commands::Ahead { depth, project } => {
            task::ahead_cmd(db, project, depth, cli.json, cli.compact)
        }
        Commands::Use { project_id, clear } => {
            if clear {
                crate::db::delete_meta(db, "current_project")?;
                if cli.json {
                    print_json(&json!({"cleared": true}))?;
                } else {
                    println!("cleared default project");
                }
                return Ok(());
            }

            if let Some(project_id) = project_id {
                crate::db::get_project(db, &project_id)?;
                crate::db::set_meta(db, "current_project", &project_id)?;
                if cli.json {
                    print_json(&json!({"current_project": project_id}))?;
                } else {
                    println!("default project: {project_id}");
                }
            } else {
                let current = crate::db::get_meta(db, "current_project")?;
                if cli.json {
                    print_json(&json!({"current_project": current}))?;
                } else if let Some(project_id) = current {
                    println!("{project_id}");
                } else {
                    println!("no default set");
                }
            }
            Ok(())
        }
        Commands::Status {
            project,
            detail,
            full,
        } => project::status_cmd(db, project.as_deref(), detail, full, cli.json, cli.compact),
        Commands::Mcp | Commands::Serve { .. } | Commands::Prompt { .. } => {
            unreachable!("handled in main")
        }
    }
}

pub fn resolve_project_id(db: &Database, explicit: Option<&str>) -> Result<String> {
    if let Some(project_id) = explicit {
        return Ok(project_id.to_string());
    }
    if let Some(project_id) = crate::db::get_meta(db, "current_project")? {
        return Ok(project_id);
    }
    Err(anyhow!(
        "No project specified. Use --project or run 'planq use <project_id>'."
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

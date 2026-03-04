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
#[command(name = "planq", version, about = "Planq task DAG CLI")]
pub struct Cli {
    #[arg(long, default_value_t = default_db_path(), global = true)]
    pub db: String,

    #[arg(long, global = true)]
    pub json: bool,

    #[arg(
        long,
        short = 'c',
        global = true,
        help = "Compact output for LLM consumption"
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
    Project(project::ProjectCommand),
    Task(task::TaskCommand),
    WhatIf(task::WhatIfCommand),
    Artifact(artifact::ArtifactCommand),
    Events(events::EventsCommand),
    #[command(about = "Lookahead buffer for running tasks")]
    Ahead {
        #[arg(long, default_value_t = 2)]
        depth: usize,
        #[arg(long)]
        project: Option<String>,
    },
    #[command(about = "Set/show default project")]
    Use {
        project_id: Option<String>,
        #[arg(long)]
        clear: bool,
    },
    #[command(about = "Project status (uses default project)")]
    Status {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, help = "Per-task detail")]
        detail: bool,
        #[arg(long, help = "Full verbose output")]
        full: bool,
    },
    #[command(about = "Start MCP server (stdio JSON-RPC for AI agents)")]
    Mcp,
    #[command(about = "Start HTTP server with REST API and SSE events")]
    Serve {
        #[arg(long, short, default_value = "8484")]
        port: u16,
    },
    #[command(about = "Generate integration prompt for your platform")]
    Prompt {
        #[arg(long, value_parser = ["mcp", "cli", "http"], help = "Platform: mcp, cli, http")]
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

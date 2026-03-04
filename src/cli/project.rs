use crate::cli::{parse_project_status, print_json, print_table, resolve_project_id, status_icon};
use crate::db::{
    get_project, list_dependencies, list_projects, list_tasks, set_meta, Database, TaskListFilters,
};
use crate::models::{ProjectStatus, TaskStatus};
use anyhow::Result;
use clap::{Args, Subcommand};
use serde::Serialize;
use std::collections::{HashMap, HashSet};

#[derive(Args, Debug)]
#[command(about = "Manage projects.\n\n\
              A project is a container for a task graph. Create one first, then add tasks.\n\
              The first project created is automatically set as the default (see 'planq use').")]
pub struct ProjectCommand {
    #[command(subcommand)]
    command: ProjectSubcommand,
}

#[derive(Subcommand, Debug)]
enum ProjectSubcommand {
    #[command(about = "Create a new project and set it as default")]
    Create(CreateProjectArgs),
    #[command(about = "List all projects (optionally filter by status)")]
    List(ListProjectsArgs),
    #[command(about = "Show project status with task counts")]
    Status(ProjectIdArg),
    #[command(about = "Render the task dependency graph as a tree")]
    Dag(ProjectIdArg),
}

#[derive(Args, Debug)]
struct CreateProjectArgs {
    #[arg(help = "Project name")]
    name: String,
    #[arg(long, help = "Optional description of the project's goal")]
    description: Option<String>,
}

#[derive(Args, Debug)]
struct ListProjectsArgs {
    #[arg(long, value_parser = parse_project_status, help = "Filter by status: active, completed, archived")]
    status: Option<ProjectStatus>,
}

#[derive(Args, Debug)]
struct ProjectIdArg {
    #[arg(help = "Project ID (uses default if not set)")]
    project_id: Option<String>,
}

#[derive(Serialize)]
struct ProjectStatusView {
    project: crate::models::Project,
    tasks_total: usize,
    tasks_done: usize,
    tasks_running: usize,
    tasks_failed: usize,
    tasks_ready: usize,
    tasks_pending: usize,
}

pub fn run(db: &Database, command: ProjectCommand, json: bool, compact: bool) -> Result<()> {
    match command.command {
        ProjectSubcommand::Create(args) => {
            let project = crate::db::create_project(db, &args.name, args.description, None)?;
            set_meta(db, "current_project", &project.id)?;
            if json {
                print_json(&project)?;
            } else {
                println!("created project {} ({})", project.id, project.name);
            }
        }
        ProjectSubcommand::List(args) => {
            let mut projects = list_projects(db)?;
            if let Some(status) = args.status {
                projects.retain(|p| p.status == status);
            }
            if json {
                print_json(&projects)?;
            } else {
                let rows = projects
                    .iter()
                    .map(|p| {
                        vec![
                            p.id.clone(),
                            p.name.clone(),
                            p.status.to_string(),
                            p.created_at.format("%Y-%m-%d %H:%M:%S").to_string(),
                            p.description.clone().unwrap_or_default(),
                        ]
                    })
                    .collect::<Vec<_>>();
                print_table(&["ID", "NAME", "STATUS", "CREATED", "DESCRIPTION"], &rows);
            }
        }
        ProjectSubcommand::Status(args) => {
            let project_id = resolve_project_id(db, args.project_id.as_deref())?;
            let project = get_project(db, &project_id)?;
            let tasks = list_tasks(
                db,
                TaskListFilters {
                    project_id: Some(project.id.clone()),
                    ..Default::default()
                },
            )?;
            let view = ProjectStatusView {
                project,
                tasks_total: tasks.len(),
                tasks_done: tasks
                    .iter()
                    .filter(|t| matches!(t.status, TaskStatus::Done | TaskStatus::DonePartial))
                    .count(),
                tasks_running: tasks
                    .iter()
                    .filter(|t| matches!(t.status, TaskStatus::Running | TaskStatus::Claimed))
                    .count(),
                tasks_failed: tasks
                    .iter()
                    .filter(|t| t.status == TaskStatus::Failed)
                    .count(),
                tasks_ready: tasks
                    .iter()
                    .filter(|t| t.status == TaskStatus::Ready)
                    .count(),
                tasks_pending: tasks
                    .iter()
                    .filter(|t| t.status == TaskStatus::Pending)
                    .count(),
            };

            if json {
                if compact {
                    let progress_pct = if view.tasks_total == 0 {
                        0.0
                    } else {
                        (view.tasks_done as f64 / view.tasks_total as f64) * 100.0
                    };
                    print_json(&serde_json::json!({
                        "total": view.tasks_total,
                        "done": view.tasks_done,
                        "ready": view.tasks_ready,
                        "running": view.tasks_running,
                        "pending": view.tasks_pending,
                        "failed": view.tasks_failed,
                        "progress_pct": progress_pct,
                    }))?;
                } else {
                    print_json(&view)?;
                }
            } else {
                println!("project: {} ({})", view.project.id, view.project.name);
                println!("status: {}", view.project.status);
                println!(
                    "tasks: total={} done={} running={} ready={} pending={} failed={}",
                    view.tasks_total,
                    view.tasks_done,
                    view.tasks_running,
                    view.tasks_ready,
                    view.tasks_pending,
                    view.tasks_failed
                );
            }
        }
        ProjectSubcommand::Dag(args) => {
            let project_id = resolve_project_id(db, args.project_id.as_deref())?;
            render_dag(db, &project_id, json)?
        }
    }

    Ok(())
}

pub fn status_cmd(
    db: &Database,
    project: Option<&str>,
    detail: bool,
    full: bool,
    json: bool,
    compact: bool,
) -> Result<()> {
    let project_id = resolve_project_id(db, project)?;
    let project = get_project(db, &project_id)?;
    let tasks = list_tasks(
        db,
        TaskListFilters {
            project_id: Some(project_id),
            ..Default::default()
        },
    )?;

    let total = tasks.len();
    let done = tasks
        .iter()
        .filter(|t| matches!(t.status, TaskStatus::Done | TaskStatus::DonePartial))
        .count();
    let ready: Vec<_> = tasks
        .iter()
        .filter(|t| t.status == TaskStatus::Ready)
        .collect();
    let running: Vec<_> = tasks
        .iter()
        .filter(|t| matches!(t.status, TaskStatus::Running | TaskStatus::Claimed))
        .collect();
    let blocked = tasks
        .iter()
        .filter(|t| t.status == TaskStatus::Pending)
        .count();
    let progress_pct = if total == 0 {
        0
    } else {
        ((done as f64 / total as f64) * 100.0).round() as i32
    };

    if full {
        let mut all_deps = Vec::new();
        for task in &tasks {
            for dep in list_dependencies(db, &task.id)? {
                all_deps.push(dep);
            }
        }
        if json {
            print_json(&serde_json::json!({
                "tasks": tasks,
                "dependencies": all_deps,
                "total": total,
            }))?;
        } else {
            println!("Project overview: {} tasks", total);
            for task in &tasks {
                println!(
                    "  {} {} {} [{}] {}",
                    status_icon(&task.status),
                    task.id,
                    task.title,
                    task.status,
                    task.agent_id.as_deref().unwrap_or("")
                );
            }
        }
        return Ok(());
    }

    if detail {
        if json {
            print_json(&tasks)?;
        } else {
            println!(
                "{} {}: {}/{} done ({}%)",
                project.id, project.name, done, total, progress_pct
            );
            for task in &tasks {
                let mut line =
                    format!("  {} {} {}", status_icon(&task.status), task.id, task.title);
                if let Some(agent) = &task.agent_id {
                    if matches!(task.status, TaskStatus::Running | TaskStatus::Claimed) {
                        line.push_str(&format!(" @{agent}"));
                    }
                }
                println!("{line}");
            }
        }
        return Ok(());
    }

    let ready_ids = ready.iter().map(|t| t.id.clone()).collect::<Vec<_>>();
    let running_tasks = running
        .iter()
        .map(|t| {
            serde_json::json!({
                "id": t.id,
                "agent": t.agent_id,
            })
        })
        .collect::<Vec<_>>();

    if json {
        if compact {
            print_json(&serde_json::json!({
                "id": project.id,
                "name": project.name,
                "total": total,
                "done": done,
                "ready": ready_ids.len(),
                "running": running.len(),
                "blocked": blocked,
                "progress": format!("{}%", progress_pct),
                "ready_ids": ready_ids,
                "running_tasks": running_tasks,
            }))?;
        } else {
            print_json(&serde_json::json!({
                "project": project,
                "tasks": tasks,
            }))?;
        }
    } else {
        let ready_str = if ready_ids.is_empty() {
            "-".to_string()
        } else {
            ready_ids.join(",")
        };
        let running_str = if running.is_empty() {
            "-".to_string()
        } else {
            running
                .iter()
                .map(|t| {
                    format!(
                        "{}@{}",
                        t.id,
                        t.agent_id.clone().unwrap_or_else(|| "-".to_string())
                    )
                })
                .collect::<Vec<_>>()
                .join(",")
        };
        println!(
            "{} {}: {}/{} done ({}%) | ready: {} | running: {} | blocked: {}",
            project.id, project.name, done, total, progress_pct, ready_str, running_str, blocked
        );
    }

    Ok(())
}

fn render_dag(db: &Database, project_id: &str, json: bool) -> Result<()> {
    let tasks = list_tasks(
        db,
        TaskListFilters {
            project_id: Some(project_id.to_string()),
            ..Default::default()
        },
    )?;

    if json {
        #[derive(Serialize)]
        struct DagEdge {
            from: String,
            to: String,
            kind: String,
        }
        #[derive(Serialize)]
        struct DagView {
            tasks: Vec<crate::models::Task>,
            edges: Vec<DagEdge>,
        }

        let mut edges = Vec::new();
        for task in &tasks {
            for dep in list_dependencies(db, &task.id)? {
                if dep.from_task == task.id {
                    edges.push(DagEdge {
                        from: dep.from_task,
                        to: dep.to_task,
                        kind: dep.kind.to_string(),
                    });
                }
            }
        }
        print_json(&DagView { tasks, edges })?;
        return Ok(());
    }

    let by_id = tasks
        .iter()
        .cloned()
        .map(|t| (t.id.clone(), t))
        .collect::<HashMap<_, _>>();

    let mut outgoing: HashMap<String, Vec<(String, String)>> = HashMap::new();
    let mut incoming: HashSet<String> = HashSet::new();
    let mut seen_deps = HashSet::new();
    for task in &tasks {
        for dep in list_dependencies(db, &task.id)? {
            if dep.from_task != task.id {
                continue;
            }
            if !seen_deps.insert(dep.id) {
                continue;
            }
            outgoing
                .entry(dep.from_task.clone())
                .or_default()
                .push((dep.to_task.clone(), dep.kind.to_string()));
            incoming.insert(dep.to_task);
        }
    }

    let mut roots = tasks
        .iter()
        .map(|t| t.id.clone())
        .filter(|id| !incoming.contains(id))
        .collect::<Vec<_>>();
    roots.sort();

    if roots.is_empty() {
        println!("(no tasks)");
        return Ok(());
    }

    let mut printed = HashSet::new();
    for root in roots {
        render_node(&root, &by_id, &outgoing, "", true, &mut printed);
    }
    Ok(())
}

fn render_node(
    task_id: &str,
    by_id: &HashMap<String, crate::models::Task>,
    outgoing: &HashMap<String, Vec<(String, String)>>,
    prefix: &str,
    last: bool,
    printed: &mut HashSet<String>,
) {
    let Some(task) = by_id.get(task_id) else {
        return;
    };

    let status = task.status.to_string();
    let icon = crate::cli::status_icon(&task.status);
    let short_id = if task.id.len() > 12 {
        &task.id[task.id.len() - 8..]
    } else {
        &task.id
    };
    let mut line = if prefix.is_empty() {
        format!("{icon} {status} {short_id} {}", task.title)
    } else {
        format!(
            "{}{} {icon} {status} {short_id} {}",
            prefix,
            if last { "└──" } else { "├──" },
            task.title
        )
    };
    if let Some(agent) = &task.agent_id {
        if matches!(task.status, TaskStatus::Running | TaskStatus::Claimed) {
            line.push_str(&format!(" ({agent})"));
        }
    }
    println!("{line}");

    if !printed.insert(task_id.to_string()) {
        return;
    }

    let mut children = outgoing.get(task_id).cloned().unwrap_or_default();
    children.sort_by(|a, b| a.0.cmp(&b.0));
    for (idx, (to_task, dep_kind)) in children.iter().enumerate() {
        let child_last = idx == children.len() - 1;
        let branch_prefix = if prefix.is_empty() {
            String::new()
        } else if last {
            format!("{prefix}   ")
        } else {
            format!("{prefix}│  ")
        };
        println!(
            "{}{}{}──▶ {}",
            branch_prefix,
            if child_last { "└──" } else { "├──" },
            dep_kind,
            to_task
        );
        let next_prefix = if prefix.is_empty() {
            if child_last {
                "   ".to_string()
            } else {
                "│  ".to_string()
            }
        } else if last {
            format!("{prefix}   ")
        } else {
            format!("{prefix}│  ")
        };
        render_node(&to_task, by_id, outgoing, &next_prefix, child_last, printed);
    }
}

use crate::cli::{parse_dependency_kind, resolve_project_id};
use crate::db::{
    add_dependency, check_file_conflicts, claim_next_task, create_task,
    get_handoff_context, get_task, list_notes, list_task_files, list_tasks, promote_ready_tasks,
    start_task, Database, NewSubtask, PlandbError, SplitPart, TaskListFilters,
};
use crate::models::{
    generate_id, DependencyCondition, DependencyKind, RetryBackoff, Task, TaskKind, TaskStatus,
};
use anyhow::{anyhow, Result};
use chrono::Utc;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;

use super::DecomposeYaml;

/// Resolve agent ID: explicit value wins, then PLANDB_AGENT env var, then "default".
pub(crate) fn resolve_agent(explicit: &str) -> String {
    if explicit != "default" {
        return explicit.to_string();
    }
    std::env::var("PLANDB_AGENT").unwrap_or_else(|_| "default".to_string())
}

/// Parse simple split syntax: comma-separated titles, `>` for chains, or JSON array.
pub(crate) fn parse_simple_split(input: &str) -> Vec<SplitPart> {
    // Check if it's JSON
    if input.trim_start().starts_with('[') {
        if let Ok(parts) = serde_json::from_str::<Vec<SplitPart>>(input) {
            return parts;
        }
    }

    // Check if it uses > for chaining (A > B > C = linear dependency chain)
    if input.contains('>') {
        let titles: Vec<String> = input
            .split('>')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        return titles
            .iter()
            .enumerate()
            .map(|(i, title)| SplitPart {
                title: title.clone(),
                description: None,
                done: None,
                result: None,
                deps_on: if i > 0 {
                    Some(vec![titles[i - 1].clone()])
                } else {
                    None
                },
            })
            .collect();
    }

    // Simple comma-separated titles (all independent)
    input
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .map(|title| SplitPart {
            title,
            description: None,
            done: None,
            result: None,
            deps_on: None,
        })
        .collect()
}

pub(crate) fn parse_dep_arg(dep: &str) -> Result<(String, DependencyKind)> {
    if let Some(idx) = dep.rfind(':') {
        let (from_task, kind_str) = dep.split_at(idx);
        let kind_str = &kind_str[1..];
        if let Ok(kind) = parse_dependency_kind(kind_str) {
            return Ok((from_task.to_string(), kind));
        }
    }
    Ok((dep.to_string(), DependencyKind::FeedsInto))
}

pub fn parse_files_arg(files: &str) -> Vec<String> {
    files
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

pub(crate) fn parse_new_subtasks(
    subtasks: Option<String>,
    file: Option<String>,
) -> Result<Vec<NewSubtask>> {
    if let Some(raw) = subtasks {
        let parsed: Vec<NewSubtask> = serde_json::from_str(&raw)?;
        return Ok(parsed);
    }
    if let Some(path) = file {
        let content = fs::read_to_string(path)?;
        #[derive(Deserialize)]
        struct PivotYaml {
            subtasks: Vec<NewSubtask>,
        }
        let parsed: PivotYaml = serde_yaml::from_str(&content)?;
        return Ok(parsed.subtasks);
    }
    Err(anyhow!("provide either --subtasks or --file"))
}

pub fn print_task_detail(task: &Task) {
    println!("id: {}", task.id);
    println!("project: {}", task.project_id);
    println!("title: {}", task.title);
    println!(
        "status: {} {}",
        crate::cli::status_icon(&task.status),
        task.status
    );
    println!("kind: {}", task.kind);
    println!("priority: {}", task.priority);
    if let Some(agent) = &task.agent_id {
        println!("agent: {agent}");
    }
    if let Some(progress) = task.progress {
        println!("progress: {progress}%");
    }
    if let Some(note) = &task.progress_note {
        println!("note: {note}");
    }
    if let Some(desc) = &task.description {
        println!("description: {desc}");
    }
    if let Some(pre) = &task.pre_condition {
        println!("pre-condition: {pre}");
    }
    if let Some(post) = &task.post_condition {
        println!("post-condition: {post}");
    }
}

pub(crate) fn enrich_transition_error(
    db: &Database,
    task_id: &str,
    action: &str,
    err: anyhow::Error,
) -> anyhow::Error {
    let Some(PlandbError::InvalidTransition(_)) = err.downcast_ref::<PlandbError>() else {
        return err;
    };

    let Ok(task) = get_task(db, task_id) else {
        return anyhow!(
            "cannot {action} task {task_id}: task not found. Try `plandb show {task_id}` to verify the id"
        );
    };

    let suggestion = match (action, &task.status) {
        ("start", TaskStatus::Ready) => {
            format!(
                "`plandb task claim {} --agent <agent>` then `plandb task start {}`",
                task.id, task.id
            )
        }
        ("start", TaskStatus::Pending) => {
            format!(
                "task is blocked by dependencies; inspect with `plandb show {}`",
                task.id
            )
        }
        ("start", TaskStatus::Running) => {
            format!(
                "task is already running; continue work or run `plandb done {}`",
                task.id
            )
        }
        ("complete", TaskStatus::Claimed) => {
            format!(
                "run `plandb task start {}` first, then `plandb done {}`",
                task.id, task.id
            )
        }
        ("complete", TaskStatus::Ready) => {
            format!(
                "run `plandb task claim {} --agent <agent>` and `plandb task start {}` first",
                task.id, task.id
            )
        }
        ("complete", TaskStatus::Pending) => {
            format!(
                "task is blocked by dependencies; inspect with `plandb show {}`",
                task.id
            )
        }
        ("complete", TaskStatus::Done | TaskStatus::DonePartial) => {
            format!(
                "task is already completed; inspect with `plandb show {}`",
                task.id
            )
        }
        ("fail", TaskStatus::Claimed) => {
            format!(
                "run `plandb task start {}` first, then `plandb task fail {} --error <msg>`",
                task.id, task.id
            )
        }
        ("fail", TaskStatus::Ready) => {
            format!(
                "run `plandb task claim {} --agent <agent>` and `plandb task start {}` first",
                task.id, task.id
            )
        }
        _ => format!("inspect task state with `plandb show {}`", task.id),
    };

    anyhow!(
        "cannot {action} task {}: current state is '{}'. Suggestion: {}",
        task.id,
        task.status,
        suggestion
    )
}

pub fn go_payload(
    db: &Database,
    project: Option<&str>,
    agent_id: &str,
) -> Result<serde_json::Value> {
    let project_id = resolve_project_id(db, project)?;
    let claimed = claim_next_task(db, &project_id, agent_id)?;
    let task = match claimed {
        Some(t) => Some(start_task(db, &t.id)?),
        None => None,
    };

    let tasks = list_tasks(
        db,
        TaskListFilters {
            project_id: Some(project_id.clone()),
            ..Default::default()
        },
    )?;
    let total = tasks.len();
    let done = tasks
        .iter()
        .filter(|t| matches!(t.status, TaskStatus::Done | TaskStatus::DonePartial))
        .count();
    let ready = tasks
        .iter()
        .filter(|t| t.status == TaskStatus::Ready)
        .count();
    let running = tasks
        .iter()
        .filter(|t| matches!(t.status, TaskStatus::Running | TaskStatus::Claimed))
        .count();
    let pending = tasks
        .iter()
        .filter(|t| t.status == TaskStatus::Pending)
        .count();
    let progress = if total == 0 {
        "0%".to_string()
    } else {
        format!("{}%", ((done as f64 / total as f64) * 100.0).round() as i32)
    };

    let (task_json, handoff_json, notes_json, conflicts_json) = if let Some(task) = &task {
        let handoff = get_handoff_context(db, &task.id)?;
        let notes = list_notes(db, &task.id)?;
        let task_files = list_task_files(db, &task.id)?;
        let mut conflicts = check_file_conflicts(db, &project_id, Some(&task.id))?;
        if !task_files.is_empty() {
            let path_set = task_files
                .into_iter()
                .collect::<std::collections::HashSet<_>>();
            conflicts.retain(|c| path_set.contains(&c.path));
        }
        (
            serde_json::json!({
                "id": task.id,
                "title": task.title,
                "status": task.status,
                "description": task.description,
                "pre_condition": task.pre_condition,
                "post_condition": task.post_condition,
            }),
            handoff
                .into_iter()
                .map(|h| {
                    serde_json::json!({
                        "from_task": h.from_task_id,
                        "from_title": h.from_title,
                        "result": h.result,
                        "agent_id": h.agent_id,
                    })
                })
                .collect::<Vec<_>>(),
            notes
                .into_iter()
                .map(|n| {
                    serde_json::json!({
                        "content": n.content,
                        "agent_id": n.agent_id,
                        "created_at": n.created_at,
                    })
                })
                .collect::<Vec<_>>(),
            serde_json::to_value(conflicts)?,
        )
    } else {
        (Value::Null, Vec::new(), Vec::new(), serde_json::json!([]))
    };

    Ok(serde_json::json!({
        "task": task_json,
        "handoff": handoff_json,
        "notes": notes_json,
        "file_conflicts": conflicts_json,
        "remaining": {
            "total": total,
            "done": done,
            "ready": ready,
            "running": running,
            "pending": pending,
        },
        "progress": progress,
    }))
}

pub(crate) fn decompose_or_replan(
    db: &Database,
    task_id: &str,
    file: &str,
    cancel_remaining: bool,
) -> Result<HashMap<String, String>> {
    let content = fs::read_to_string(file)?;
    let parsed: DecomposeYaml = serde_yaml::from_str(&content)?;
    if parsed.subtasks.is_empty() {
        return Err(anyhow!("decompose file has no subtasks"));
    }

    {
        let conn = db.lock()?;
        conn.execute(
            "UPDATE tasks SET is_composite = 1, updated_at = datetime('now') WHERE id = ?1",
            rusqlite::params![task_id],
        )?;
        if cancel_remaining {
            conn.execute(
                "UPDATE tasks SET status = 'cancelled', updated_at = datetime('now') WHERE parent_task_id = ?1 AND status NOT IN ('done', 'done_partial', 'running')",
                rusqlite::params![task_id],
            )?;
        }
    }
    let parent = get_task(db, task_id)?;

    let mut title_to_id: HashMap<String, String> = HashMap::new();
    let now = Utc::now().naive_utc();
    for sub in &parsed.subtasks {
        let has_deps = sub.deps_on.as_ref().map(|d| !d.is_empty()).unwrap_or(false);
        let task = Task {
            id: generate_id("task"),
            project_id: parent.project_id.clone(),
            parent_task_id: Some(task_id.to_string()),
            is_composite: false,
            title: sub.title.clone(),
            description: sub.description.clone(),
            status: if has_deps {
                TaskStatus::Pending
            } else {
                TaskStatus::Ready
            },
            kind: sub.kind.clone().unwrap_or(TaskKind::Generic),
            priority: sub.priority.unwrap_or(0),
            agent_id: None,
            claimed_at: None,
            started_at: None,
            completed_at: None,
            result: None,
            error: None,
            progress: None,
            progress_note: None,
            max_retries: 0,
            retry_count: 0,
            retry_backoff: RetryBackoff::Exponential,
            retry_delay_ms: 1000,
            timeout_seconds: None,
            heartbeat_interval: 30,
            last_heartbeat: None,
            requires_approval: false,
            approval_status: None,
            approved_by: None,
            approval_comment: None,
            pre_condition: None,
            post_condition: None,
            metadata: None,
            created_at: now,
            updated_at: now,
        };
        let created = create_task(db, &task, &[])?;
        title_to_id.insert(sub.title.clone(), created.id.clone());
    }

    for sub in &parsed.subtasks {
        if let Some(deps_on) = &sub.deps_on {
            let to_id = title_to_id
                .get(&sub.title)
                .ok_or_else(|| anyhow!("internal error: subtask title not found"))?;
            for dep_title in deps_on {
                let from_id = title_to_id
                    .get(dep_title)
                    .ok_or_else(|| anyhow!("deps_on references unknown subtask: {}", dep_title))?;
                add_dependency(
                    db,
                    from_id,
                    to_id,
                    DependencyKind::FeedsInto,
                    DependencyCondition::All,
                    None,
                )?;
            }
        }
    }
    let _ = promote_ready_tasks(db)?;
    Ok(title_to_id)
}

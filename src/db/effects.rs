use crate::db::Database;
use crate::models::TaskStatus;
use anyhow::Result;
use rusqlite::params;
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, serde::Serialize)]
pub struct MutationEffect {
    pub delayed: Vec<String>,
    pub accelerated: Vec<String>,
    pub ready_now: Vec<String>,
    pub blocked_now: Vec<String>,
    pub critical_path: Vec<String>,
    pub depth: usize,
}

pub fn snapshot_task_statuses(
    db: &Database,
    project_id: &str,
) -> Result<HashMap<String, TaskStatus>> {
    let conn = db.lock()?;
    let mut stmt = conn.prepare("SELECT id, status FROM tasks WHERE project_id = ?1")?;
    let mut rows = stmt.query(params![project_id])?;
    let mut snapshot = HashMap::new();

    while let Some(row) = rows.next()? {
        let task_id: String = row.get(0)?;
        let status: TaskStatus = row.get(1)?;
        snapshot.insert(task_id, status);
    }

    Ok(snapshot)
}

pub fn compute_effects(
    db: &Database,
    project_id: &str,
    before_snapshot: &HashMap<String, TaskStatus>,
    after_snapshot: &HashMap<String, TaskStatus>,
) -> Result<MutationEffect> {
    let mut delayed = Vec::new();
    let mut accelerated = Vec::new();
    let mut ready_now = Vec::new();
    let mut blocked_now = Vec::new();

    for (task_id, after_status) in after_snapshot {
        let before_status = before_snapshot.get(task_id);
        if before_status == Some(after_status) {
            continue;
        }

        if became_ready(before_status, after_status) {
            ready_now.push(task_id.clone());
            accelerated.push(task_id.clone());
        } else if became_blocked(before_status, after_status) {
            blocked_now.push(task_id.clone());
            delayed.push(task_id.clone());
        } else if became_more_available(before_status, after_status) {
            accelerated.push(task_id.clone());
        } else if became_less_available(before_status, after_status) {
            delayed.push(task_id.clone());
        }
    }

    delayed.sort();
    delayed.dedup();
    accelerated.sort();
    accelerated.dedup();
    ready_now.sort();
    ready_now.dedup();
    blocked_now.sort();
    blocked_now.dedup();

    let (critical_path, depth) = compute_critical_path(db, project_id)?;

    Ok(MutationEffect {
        delayed,
        accelerated,
        ready_now,
        blocked_now,
        critical_path,
        depth,
    })
}

fn status_rank(status: &TaskStatus) -> i32 {
    match status {
        TaskStatus::Pending => 0,
        TaskStatus::Ready => 1,
        TaskStatus::Claimed => 2,
        TaskStatus::Running => 3,
        TaskStatus::Done | TaskStatus::DonePartial => 4,
        TaskStatus::Failed | TaskStatus::Cancelled => -1,
    }
}

fn became_ready(before: Option<&TaskStatus>, after: &TaskStatus) -> bool {
    !matches!(before, Some(TaskStatus::Ready)) && matches!(after, TaskStatus::Ready)
}

fn became_blocked(before: Option<&TaskStatus>, after: &TaskStatus) -> bool {
    matches!(
        before,
        Some(TaskStatus::Ready | TaskStatus::Claimed | TaskStatus::Running)
    ) && matches!(after, TaskStatus::Pending)
}

fn became_more_available(before: Option<&TaskStatus>, after: &TaskStatus) -> bool {
    let before_rank = before.map(status_rank).unwrap_or(-1);
    status_rank(after) > before_rank
}

fn became_less_available(before: Option<&TaskStatus>, after: &TaskStatus) -> bool {
    let before_rank = before.map(status_rank).unwrap_or(-1);
    status_rank(after) < before_rank
}

fn compute_critical_path(db: &Database, project_id: &str) -> Result<(Vec<String>, usize)> {
    let conn = db.lock()?;

    let mut task_stmt = conn.prepare("SELECT id FROM tasks WHERE project_id = ?1")?;
    let task_rows = task_stmt.query_map(params![project_id], |row| row.get::<_, String>(0))?;
    let task_ids: Vec<String> = task_rows.collect::<std::result::Result<Vec<_>, _>>()?;

    if task_ids.is_empty() {
        return Ok((Vec::new(), 0));
    }

    let mut adjacency: HashMap<String, Vec<String>> = task_ids
        .iter()
        .map(|task_id| (task_id.clone(), Vec::new()))
        .collect();
    let mut indegree: HashMap<String, usize> = task_ids
        .iter()
        .map(|task_id| (task_id.clone(), 0usize))
        .collect();

    let mut edge_stmt = conn.prepare(
        r#"
        SELECT d.from_task, d.to_task
        FROM dependencies d
        JOIN tasks ft ON ft.id = d.from_task
        JOIN tasks tt ON tt.id = d.to_task
        WHERE ft.project_id = ?1 AND tt.project_id = ?1
        "#,
    )?;

    let edge_rows = edge_stmt.query_map(params![project_id], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;

    for edge in edge_rows {
        let (from_task, to_task) = edge?;
        if let Some(children) = adjacency.get_mut(&from_task) {
            children.push(to_task.clone());
        }
        if let Some(value) = indegree.get_mut(&to_task) {
            *value += 1;
        }
    }

    for children in adjacency.values_mut() {
        children.sort();
    }

    let mut queue: VecDeque<String> = indegree
        .iter()
        .filter_map(|(task_id, degree)| {
            if *degree == 0 {
                Some(task_id.clone())
            } else {
                None
            }
        })
        .collect();
    let queue_sorted = queue.make_contiguous();
    queue_sorted.sort();

    let mut distance: HashMap<String, usize> = task_ids
        .iter()
        .map(|task_id| (task_id.clone(), 1usize))
        .collect();
    let mut parent: HashMap<String, String> = HashMap::new();

    while let Some(current) = queue.pop_front() {
        let current_distance = *distance.get(&current).unwrap_or(&1usize);
        if let Some(children) = adjacency.get(&current) {
            for child in children {
                let candidate = current_distance + 1;
                let current_best = *distance.get(child).unwrap_or(&1usize);
                if candidate > current_best {
                    distance.insert(child.clone(), candidate);
                    parent.insert(child.clone(), current.clone());
                }

                if let Some(degree) = indegree.get_mut(child) {
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(child.clone());
                    }
                }
            }
        }
    }

    let mut best_leaf = task_ids[0].clone();
    let mut best_depth = 1usize;
    for task_id in &task_ids {
        let depth = *distance.get(task_id).unwrap_or(&1usize);
        if depth > best_depth {
            best_depth = depth;
            best_leaf = task_id.clone();
        }
    }

    let mut critical_path = Vec::new();
    let mut cursor = Some(best_leaf);
    while let Some(task_id) = cursor {
        critical_path.push(task_id.clone());
        cursor = parent.get(&task_id).cloned();
    }
    critical_path.reverse();

    Ok((critical_path, best_depth))
}

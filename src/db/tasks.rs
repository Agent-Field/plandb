use crate::db::dependencies::{add_dependency, remove_dependency};
use crate::db::{
    dt_to_sql, json_to_sql, now_utc_naive, parse_dt, parse_json, Database, PlandbError,
};
use crate::models::{generate_id, RetryBackoff, Task, TaskKind, TaskStatus};
use anyhow::Result;
use rusqlite::{params, OptionalExtension};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet, VecDeque};

const INSERT_TASK: &str = r#"
INSERT INTO tasks (
  id, project_id, parent_task_id, is_composite,
  title, description, status, kind, priority,
  agent_id, claimed_at, started_at, completed_at,
  result, error, progress, progress_note,
  max_retries, retry_count, retry_backoff, retry_delay_ms,
  timeout_seconds, heartbeat_interval, last_heartbeat,
  requires_approval, approval_status, approved_by, approval_comment,
  metadata, created_at, updated_at
) VALUES (
  ?1, ?2, ?3, ?4,
  ?5, ?6, ?7, ?8, ?9,
  ?10, ?11, ?12, ?13,
  ?14, ?15, ?16, ?17,
  ?18, ?19, ?20, ?21,
  ?22, ?23, ?24,
  ?25, ?26, ?27, ?28,
  ?29, ?30, ?31
);
"#;

const INSERT_TASK_TAG: &str = "INSERT OR IGNORE INTO task_tags(task_id, tag) VALUES (?1, ?2);";

const SELECT_TASK_BY_ID: &str = r#"
SELECT
id, project_id, parent_task_id, is_composite,
title, description, status, kind, priority,
agent_id, claimed_at, started_at, completed_at,
result, error, progress, progress_note,
max_retries, retry_count, retry_backoff, retry_delay_ms,
timeout_seconds, heartbeat_interval, last_heartbeat,
requires_approval, approval_status, approved_by, approval_comment,
metadata, created_at, updated_at
FROM tasks
WHERE id = ?1;
"#;

const SELECT_TASKS_FILTERED: &str = r#"
SELECT
t.id, t.project_id, t.parent_task_id, t.is_composite,
t.title, t.description, t.status, t.kind, t.priority,
t.agent_id, t.claimed_at, t.started_at, t.completed_at,
t.result, t.error, t.progress, t.progress_note,
t.max_retries, t.retry_count, t.retry_backoff, t.retry_delay_ms,
t.timeout_seconds, t.heartbeat_interval, t.last_heartbeat,
t.requires_approval, t.approval_status, t.approved_by, t.approval_comment,
t.metadata, t.created_at, t.updated_at
FROM tasks t
WHERE (?1 IS NULL OR t.project_id = ?1)
  AND (?2 IS NULL OR t.status = ?2)
  AND (?3 IS NULL OR t.kind = ?3)
  AND (?4 IS NULL OR t.parent_task_id = ?4)
  AND (?5 IS NULL OR t.agent_id = ?5)
  AND (
      ?6 IS NULL OR EXISTS (
        SELECT 1 FROM task_tags tt
        WHERE tt.task_id = t.id
          AND tt.tag IN (SELECT value FROM json_each(?6))
      )
  )
ORDER BY t.priority DESC, t.created_at ASC;
"#;

const CLAIM_TASK: &str = r#"
UPDATE tasks
SET status = 'claimed', agent_id = ?1, claimed_at = ?2, last_heartbeat = ?2, updated_at = ?2
WHERE id = ?3 AND status = 'ready'
RETURNING
id, project_id, parent_task_id, is_composite,
title, description, status, kind, priority,
agent_id, claimed_at, started_at, completed_at,
result, error, progress, progress_note,
max_retries, retry_count, retry_backoff, retry_delay_ms,
timeout_seconds, heartbeat_interval, last_heartbeat,
requires_approval, approval_status, approved_by, approval_comment,
metadata, created_at, updated_at;
"#;

const CLAIM_NEXT_TASK: &str = r#"
UPDATE tasks
SET status = 'claimed', agent_id = ?1, claimed_at = ?3, last_heartbeat = ?3, updated_at = ?3
WHERE id = (
  SELECT id
  FROM tasks
  WHERE project_id = ?2 AND status = 'ready'
  ORDER BY priority DESC, created_at ASC
  LIMIT 1
)
RETURNING
id, project_id, parent_task_id, is_composite,
title, description, status, kind, priority,
agent_id, claimed_at, started_at, completed_at,
result, error, progress, progress_note,
max_retries, retry_count, retry_backoff, retry_delay_ms,
timeout_seconds, heartbeat_interval, last_heartbeat,
requires_approval, approval_status, approved_by, approval_comment,
metadata, created_at, updated_at;
"#;

const START_TASK: &str = r#"
UPDATE tasks
SET status = 'running', started_at = ?2, last_heartbeat = ?2, updated_at = ?2
WHERE id = ?1 AND status = 'claimed';
"#;

const COMPLETE_TASK: &str = r#"
UPDATE tasks
SET status = 'done', result = ?2, error = NULL, completed_at = ?3, updated_at = ?3,
    claimed_at = COALESCE(claimed_at, ?3),
    started_at = COALESCE(started_at, ?3)
WHERE id = ?1 AND status IN ('ready', 'claimed', 'running');
"#;

const FAIL_TASK: &str = r#"
UPDATE tasks
SET status = 'failed', error = ?2, retry_count = retry_count + 1, completed_at = ?3, updated_at = ?3
WHERE id = ?1 AND status = 'running';
"#;

const CANCEL_TASK: &str = r#"
UPDATE tasks
SET status = 'cancelled', updated_at = ?2
WHERE id = ?1 AND status NOT IN ('done', 'done_partial');
"#;

const CANCEL_DOWNSTREAM: &str = r#"
WITH RECURSIVE downstream(task_id) AS (
  SELECT to_task FROM dependencies WHERE from_task = ?1
  UNION ALL
  SELECT d.to_task FROM dependencies d
  JOIN downstream ds ON d.from_task = ds.task_id
)
UPDATE tasks
SET status = 'cancelled', updated_at = ?2
WHERE id IN (SELECT task_id FROM downstream)
  AND status NOT IN ('done', 'done_partial');
"#;

const UPDATE_HEARTBEAT: &str = r#"
UPDATE tasks
SET last_heartbeat = ?2, updated_at = ?2
WHERE id = ?1 AND status IN ('claimed', 'running');
"#;

const UPDATE_PROGRESS: &str = r#"
UPDATE tasks
SET progress = ?2, progress_note = ?3, updated_at = ?4
WHERE id = ?1;
"#;

const APPROVE_TASK: &str = r#"
UPDATE tasks
SET approval_status = ?2, approved_by = ?3, approval_comment = ?4, updated_at = ?5
WHERE id = ?1;
"#;

const PROMOTE_READY: &str = r#"
UPDATE tasks SET status = 'ready', updated_at = ?1
WHERE id IN (SELECT id FROM task_readiness WHERE promotable = 1);
"#;

const PAUSE_TASK: &str = r#"
UPDATE tasks
SET status = 'ready',
    agent_id = NULL,
    progress = ?2,
    progress_note = ?3,
    metadata = ?4,
    updated_at = ?5
WHERE id = ?1
  AND status IN ('running', 'claimed');
"#;

const SELECT_HANDOFF_CONTEXT: &str = r#"
SELECT t.id, t.title, t.result, t.agent_id
FROM dependencies d
JOIN tasks t ON t.id = d.from_task
WHERE d.to_task = ?1
  AND t.status IN ('done', 'done_partial')
  AND t.result IS NOT NULL
ORDER BY t.completed_at ASC, t.created_at ASC;
"#;

const SELECT_TASK_TITLES_LIKE: &str = r#"
SELECT id, title
FROM tasks
WHERE title LIKE ?1
  AND (?2 IS NULL OR project_id = ?2)
ORDER BY created_at DESC
LIMIT 5;
"#;
const SELECT_RECENT_TASK_IDS: &str = r#"
SELECT id, title
FROM tasks
WHERE (?1 IS NULL OR project_id = ?1)
ORDER BY created_at DESC
LIMIT 50;
"#;

fn levenshtein(a: &str, b: &str) -> usize {
    let n = b.chars().count();
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];
    for (i, ca) in a.chars().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.chars().enumerate() {
            curr[j + 1] = if ca == cb {
                prev[j]
            } else {
                1 + prev[j].min(prev[j + 1]).min(curr[j])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

#[derive(Default, Clone, Debug)]
pub struct TaskListFilters {
    pub project_id: Option<String>,
    pub status: Option<TaskStatus>,
    pub kind: Option<TaskKind>,
    pub parent_task_id: Option<String>,
    pub agent_id: Option<String>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct HandoffEntry {
    pub from_task_id: String,
    pub from_title: String,
    pub result: Option<Value>,
    pub agent_id: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ProjectState {
    pub total: usize,
    pub done: usize,
    pub ready: usize,
    pub running: usize,
    pub pending: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LookaheadTask {
    pub id: String,
    pub title: String,
    pub hops: usize,
    pub blocked_by: Vec<String>,
    pub description: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LookaheadResult {
    pub current: Vec<Task>,
    pub upcoming: Vec<LookaheadTask>,
    pub updatable: Vec<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct NewSubtask {
    pub title: String,
    pub description: Option<String>,
    pub kind: Option<TaskKind>,
    pub priority: Option<i32>,
    pub deps_on: Option<Vec<String>>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PivotResult {
    pub kept: Vec<String>,
    pub cancelled: Vec<String>,
    pub created: Vec<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct SplitPart {
    pub title: String,
    pub done: Option<bool>,
    pub result: Option<String>,
    pub deps_on: Option<Vec<String>>,
    pub description: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SplitResult {
    pub parent_task_id: String,
    pub created: Vec<String>,
    pub done: Vec<String>,
    pub title_to_id: HashMap<String, String>,
}

pub(crate) fn row_to_task(row: &rusqlite::Row<'_>) -> rusqlite::Result<Task> {
    let conv = |idx: usize, e: anyhow::Error| {
        rusqlite::Error::FromSqlConversionFailure(
            idx,
            rusqlite::types::Type::Text,
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                e.to_string(),
            )),
        )
    };
    let claimed_at: Option<String> = row.get(10)?;
    let started_at: Option<String> = row.get(11)?;
    let completed_at: Option<String> = row.get(12)?;
    let result: Option<String> = row.get(13)?;
    let last_heartbeat: Option<String> = row.get(23)?;
    let metadata: Option<String> = row.get(28)?;
    Ok(Task {
        id: row.get(0)?,
        project_id: row.get(1)?,
        parent_task_id: row.get(2)?,
        is_composite: row.get(3)?,
        title: row.get(4)?,
        description: row.get(5)?,
        status: row.get(6)?,
        kind: row.get(7)?,
        priority: row.get(8)?,
        agent_id: row.get(9)?,
        claimed_at: claimed_at
            .map(parse_dt)
            .transpose()
            .map_err(|e| conv(10, e))?,
        started_at: started_at
            .map(parse_dt)
            .transpose()
            .map_err(|e| conv(11, e))?,
        completed_at: completed_at
            .map(parse_dt)
            .transpose()
            .map_err(|e| conv(12, e))?,
        result: parse_json(result).map_err(|e| conv(13, e))?,
        error: row.get(14)?,
        progress: row.get(15)?,
        progress_note: row.get(16)?,
        max_retries: row.get(17)?,
        retry_count: row.get(18)?,
        retry_backoff: row.get(19)?,
        retry_delay_ms: row.get(20)?,
        timeout_seconds: row.get(21)?,
        heartbeat_interval: row.get(22)?,
        last_heartbeat: last_heartbeat
            .map(parse_dt)
            .transpose()
            .map_err(|e| conv(23, e))?,
        requires_approval: row.get(24)?,
        approval_status: row.get(25)?,
        approved_by: row.get(26)?,
        approval_comment: row.get(27)?,
        metadata: parse_json(metadata).map_err(|e| conv(28, e))?,
        created_at: parse_dt(row.get::<_, String>(29)?).map_err(|e| conv(29, e))?,
        updated_at: parse_dt(row.get::<_, String>(30)?).map_err(|e| conv(30, e))?,
    })
}

pub fn create_task(db: &Database, task: &Task, tags: &[String]) -> Result<Task> {
    let conn = db.lock()?;
    let mut task_with_defaults = task.clone();
    if task_with_defaults.max_retries < 0 {
        task_with_defaults.max_retries = 0;
    }
    if task_with_defaults.retry_count < 0 {
        task_with_defaults.retry_count = 0;
    }
    if task_with_defaults.retry_delay_ms <= 0 {
        task_with_defaults.retry_delay_ms = 1000;
    }
    if task_with_defaults.heartbeat_interval <= 0 {
        task_with_defaults.heartbeat_interval = 30;
    }
    let task_id = task_with_defaults.id.clone();
    let result = json_to_sql(&task_with_defaults.result)?;
    let metadata = json_to_sql(&task_with_defaults.metadata)?;
    conn.execute(
        INSERT_TASK,
        params![
            &task_with_defaults.id,
            &task_with_defaults.project_id,
            &task_with_defaults.parent_task_id,
            task_with_defaults.is_composite,
            &task_with_defaults.title,
            &task_with_defaults.description,
            &task_with_defaults.status,
            &task_with_defaults.kind,
            task_with_defaults.priority,
            &task_with_defaults.agent_id,
            task_with_defaults.claimed_at.map(dt_to_sql),
            task_with_defaults.started_at.map(dt_to_sql),
            task_with_defaults.completed_at.map(dt_to_sql),
            &result,
            &task_with_defaults.error,
            task_with_defaults.progress,
            &task_with_defaults.progress_note,
            task_with_defaults.max_retries,
            task_with_defaults.retry_count,
            &task_with_defaults.retry_backoff,
            task_with_defaults.retry_delay_ms,
            task_with_defaults.timeout_seconds,
            task_with_defaults.heartbeat_interval,
            task_with_defaults.last_heartbeat.map(dt_to_sql),
            task_with_defaults.requires_approval,
            &task_with_defaults.approval_status,
            &task_with_defaults.approved_by,
            &task_with_defaults.approval_comment,
            &metadata,
            dt_to_sql(task_with_defaults.created_at),
            dt_to_sql(task_with_defaults.updated_at)
        ],
    )?;
    for tag in tags {
        conn.execute(INSERT_TASK_TAG, params![&task_id, tag])?;
    }
    drop(conn);
    let task = get_task(db, &task_id)?;
    let _ = crate::db::insert_event(
        db,
        Some(&task.id),
        Some(&task.project_id),
        task.agent_id.as_deref(),
        crate::models::EventType::TaskCreated,
        Some(serde_json::json!({"title": task.title})),
        crate::db::now_utc_naive(),
    );
    Ok(task)
}

pub fn get_task(db: &Database, task_id: &str) -> Result<Task> {
    let conn = db.lock()?;
    let mut stmt = conn.prepare(SELECT_TASK_BY_ID)?;
    let task = stmt.query_row(params![task_id], row_to_task).optional()?;
    task.ok_or_else(|| PlandbError::NotFound(format!("task {task_id}")).into())
}

pub fn fuzzy_find_task(db: &Database, input: &str, project_id: Option<&str>) -> Result<Task> {
    match get_task(db, input) {
        Ok(task) => return Ok(task),
        Err(err) => {
            if !matches!(
                err.downcast_ref::<PlandbError>(),
                Some(PlandbError::NotFound(_))
            ) {
                return Err(err);
            }
        }
    }

    if !input.starts_with("t-") {
        let matches: Vec<(String, String)> = {
            let conn = db.lock()?;
            let like = format!("%{input}%");
            let mut stmt = conn.prepare(SELECT_TASK_TITLES_LIKE)?;
            let mut rows = stmt.query(params![like, project_id])?;
            let mut matches: Vec<(String, String)> = Vec::new();
            while let Some(row) = rows.next()? {
                matches.push((row.get(0)?, row.get(1)?));
            }
            matches
        };

        if matches.len() == 1 {
            return get_task(db, &matches[0].0);
        }
        if !matches.is_empty() {
            let rendered = matches
                .iter()
                .map(|(id, title)| format!("{id} ({title})"))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(anyhow::anyhow!(
                "Multiple matches for '{input}': {rendered}"
            ));
        }
    }

    if input.starts_with("t-") {
        let conn = db.lock()?;
        let mut stmt = conn.prepare(SELECT_RECENT_TASK_IDS)?;
        let mut rows = stmt.query(params![project_id])?;
        let mut best: Option<(usize, String, String)> = None;
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let title: String = row.get(1)?;
            let dist = levenshtein(input, &id);
            if best.as_ref().map(|(d, _, _)| dist < *d).unwrap_or(true) {
                best = Some((dist, id, title));
            }
        }
        if let Some((dist, id, title)) = best {
            if dist <= 2 {
                return Err(anyhow::anyhow!(
                    "Task '{input}' not found. Did you mean: {id} ({title})?"
                ));
            }
        }
    }

    Err(PlandbError::NotFound(format!("task {input}")).into())
}

pub fn list_tasks(db: &Database, filters: TaskListFilters) -> Result<Vec<Task>> {
    let conn = db.lock()?;
    let mut stmt = conn.prepare(SELECT_TASKS_FILTERED)?;
    let tag_json = if filters.tags.is_empty() {
        None
    } else {
        Some(serde_json::to_string(&filters.tags)?)
    };
    let mut rows = stmt.query(params![
        filters.project_id,
        filters.status.map(|s| s.to_string()),
        filters.kind.map(|k| k.to_string()),
        filters.parent_task_id,
        filters.agent_id,
        tag_json,
    ])?;

    let mut tasks = Vec::new();
    while let Some(row) = rows.next()? {
        tasks.push(row_to_task(row)?);
    }
    Ok(tasks)
}

pub fn claim_task(db: &Database, task_id: &str, agent_id: &str) -> Result<Option<Task>> {
    let task = {
        let conn = db.lock()?;
        let now = dt_to_sql(now_utc_naive());
        let mut stmt = conn.prepare(CLAIM_TASK)?;
        stmt.query_row(params![agent_id, now, task_id], row_to_task)
            .optional()?
    };
    if let Some(ref t) = task {
        let _ = crate::db::insert_event(
            db,
            Some(&t.id),
            Some(&t.project_id),
            Some(agent_id),
            crate::models::EventType::TaskClaimed,
            None,
            crate::db::now_utc_naive(),
        );
    }
    Ok(task)
}

pub fn claim_next_task(db: &Database, project_id: &str, agent_id: &str) -> Result<Option<Task>> {
    let task = {
        let conn = db.lock()?;
        let now = dt_to_sql(now_utc_naive());
        let mut stmt = conn.prepare(CLAIM_NEXT_TASK)?;
        stmt.query_row(params![agent_id, project_id, now], row_to_task)
            .optional()?
    };
    if let Some(ref t) = task {
        let _ = crate::db::insert_event(
            db,
            Some(&t.id),
            Some(&t.project_id),
            Some(agent_id),
            crate::models::EventType::TaskClaimed,
            None,
            crate::db::now_utc_naive(),
        );
    }
    Ok(task)
}

pub fn start_task(db: &Database, task_id: &str) -> Result<Task> {
    let conn = db.lock()?;
    let now = dt_to_sql(now_utc_naive());
    let changed = conn.execute(START_TASK, params![task_id, now])?;
    if changed == 0 {
        return Err(PlandbError::InvalidTransition(format!(
            "task {task_id} must be claimed to start"
        ))
        .into());
    }
    drop(conn);
    let task = get_task(db, task_id)?;
    let _ = crate::db::insert_event(
        db,
        Some(&task.id),
        Some(&task.project_id),
        task.agent_id.as_deref(),
        crate::models::EventType::TaskStarted,
        None,
        crate::db::now_utc_naive(),
    );
    Ok(task)
}

pub fn complete_task(
    db: &Database,
    task_id: &str,
    result: Option<serde_json::Value>,
) -> Result<Task> {
    let conn = db.lock()?;
    let now = dt_to_sql(now_utc_naive());
    let result_json = match result {
        Some(value) => Some(serde_json::to_string(&value)?),
        None => None,
    };
    let changed = conn.execute(COMPLETE_TASK, params![task_id, result_json, now])?;
    if changed == 0 {
        drop(conn);
        let current = get_task(db, task_id);
        let status_msg = match current {
            Ok(t) => format!(
                "task {} is '{}', must be ready/claimed/running to complete",
                task_id, t.status
            ),
            Err(_) => format!("task {task_id} not found"),
        };
        return Err(PlandbError::InvalidTransition(status_msg).into());
    }
    drop(conn);
    let task = get_task(db, task_id)?;
    let _ = crate::db::insert_event(
        db,
        Some(&task.id),
        Some(&task.project_id),
        task.agent_id.as_deref(),
        crate::models::EventType::TaskCompleted,
        task.result
            .as_ref()
            .map(|_| serde_json::json!({"has_result": true})),
        crate::db::now_utc_naive(),
    );
    Ok(task)
}

pub fn fail_task(db: &Database, task_id: &str, error: &str) -> Result<Task> {
    let conn = db.lock()?;
    let now = dt_to_sql(now_utc_naive());
    let changed = conn.execute(FAIL_TASK, params![task_id, error, now])?;
    if changed == 0 {
        return Err(PlandbError::InvalidTransition(format!(
            "task {task_id} must be running to fail"
        ))
        .into());
    }
    drop(conn);
    let task = get_task(db, task_id)?;
    let _ = crate::db::insert_event(
        db,
        Some(&task.id),
        Some(&task.project_id),
        task.agent_id.as_deref(),
        crate::models::EventType::TaskFailed,
        Some(serde_json::json!({"error": error})),
        crate::db::now_utc_naive(),
    );
    Ok(task)
}

pub fn cancel_task(db: &Database, task_id: &str, cascade: bool) -> Result<usize> {
    let conn = db.lock()?;
    let now = dt_to_sql(now_utc_naive());
    let changed = conn.execute(CANCEL_TASK, params![task_id, now])?;
    let mut total = changed;
    if cascade {
        total += conn.execute(CANCEL_DOWNSTREAM, params![task_id, now])?;
    }
    drop(conn);
    let _ = crate::db::insert_event(
        db,
        Some(task_id),
        None,
        None,
        crate::models::EventType::TaskCancelled,
        Some(serde_json::json!({"cascade": cascade, "cancelled_count": total})),
        crate::db::now_utc_naive(),
    );
    Ok(total)
}

pub fn update_heartbeat(db: &Database, task_id: &str) -> Result<usize> {
    let conn = db.lock()?;
    let now = dt_to_sql(now_utc_naive());
    Ok(conn.execute(UPDATE_HEARTBEAT, params![task_id, now])?)
}

pub fn update_progress(
    db: &Database,
    task_id: &str,
    progress: Option<i32>,
    note: Option<String>,
) -> Result<usize> {
    let conn = db.lock()?;
    let now = dt_to_sql(now_utc_naive());
    Ok(conn.execute(UPDATE_PROGRESS, params![task_id, progress, note, now])?)
}

pub fn approve_task(
    db: &Database,
    task_id: &str,
    approval_status: &str,
    approved_by: Option<String>,
    approval_comment: Option<String>,
) -> Result<usize> {
    let conn = db.lock()?;
    let now = dt_to_sql(now_utc_naive());
    Ok(conn.execute(
        APPROVE_TASK,
        params![task_id, approval_status, approved_by, approval_comment, now],
    )?)
}

pub fn update_task(
    db: &Database,
    task_id: &str,
    title: Option<String>,
    description: Option<String>,
    kind: Option<TaskKind>,
    priority: Option<i32>,
    metadata: Option<serde_json::Value>,
) -> Result<Task> {
    let existing = get_task(db, task_id)?;
    match existing.status {
        TaskStatus::Done | TaskStatus::DonePartial | TaskStatus::Cancelled => {
            return Err(PlandbError::InvalidTransition(format!(
                "task {task_id} is {} and cannot be updated",
                existing.status
            ))
            .into());
        }
        _ => {}
    }

    let conn = db.lock()?;
    let now = dt_to_sql(now_utc_naive());

    let mut sets = Vec::new();
    let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

    if let Some(ref t) = title {
        sets.push("title = ?");
        param_values.push(Box::new(t.clone()));
    }
    if let Some(ref d) = description {
        sets.push("description = ?");
        param_values.push(Box::new(d.clone()));
    }
    if let Some(ref k) = kind {
        sets.push("kind = ?");
        param_values.push(Box::new(k.to_string()));
    }
    if let Some(p) = priority {
        sets.push("priority = ?");
        param_values.push(Box::new(p));
    }
    if let Some(ref m) = metadata {
        sets.push("metadata = ?");
        param_values.push(Box::new(serde_json::to_string(m)?));
    }

    if sets.is_empty() {
        return Ok(existing);
    }

    sets.push("updated_at = ?");
    param_values.push(Box::new(now));

    let sql = format!("UPDATE tasks SET {} WHERE id = ?", sets.join(", "));
    param_values.push(Box::new(task_id.to_string()));

    let params: Vec<&dyn rusqlite::types::ToSql> =
        param_values.iter().map(|b| b.as_ref()).collect();
    conn.execute(&sql, rusqlite::params_from_iter(params))?;
    drop(conn);

    get_task(db, task_id)
}

pub fn pause_task(
    db: &Database,
    task_id: &str,
    progress: Option<i32>,
    note: Option<String>,
) -> Result<Task> {
    let current = get_task(db, task_id)?;
    if !matches!(current.status, TaskStatus::Running | TaskStatus::Claimed) {
        return Err(PlandbError::InvalidTransition(format!(
            "task {task_id} must be running or claimed to pause"
        ))
        .into());
    }

    let mut metadata_obj = match current.metadata {
        Some(Value::Object(obj)) => obj,
        _ => Map::new(),
    };
    if let Some(agent) = current.agent_id {
        metadata_obj.insert("previous_agent".to_string(), Value::String(agent));
    }

    let metadata_json = serde_json::to_string(&Value::Object(metadata_obj))?;
    let conn = db.lock()?;
    let now = dt_to_sql(now_utc_naive());
    let changed = conn.execute(
        PAUSE_TASK,
        params![task_id, progress, note, metadata_json, now],
    )?;
    if changed == 0 {
        return Err(PlandbError::InvalidTransition(format!(
            "task {task_id} must be running or claimed to pause"
        ))
        .into());
    }
    drop(conn);
    get_task(db, task_id)
}

pub fn get_handoff_context(db: &Database, task_id: &str) -> Result<Vec<HandoffEntry>> {
    let conn = db.lock()?;
    let mut stmt = conn.prepare(SELECT_HANDOFF_CONTEXT)?;
    let mut rows = stmt.query(params![task_id])?;
    let mut entries = Vec::new();
    while let Some(row) = rows.next()? {
        let result_raw: Option<String> = row.get(2)?;
        entries.push(HandoffEntry {
            from_task_id: row.get(0)?,
            from_title: row.get(1)?,
            result: parse_json(result_raw)?,
            agent_id: row.get(3)?,
        });
    }
    Ok(entries)
}

pub fn batch_create_tasks(db: &Database, tasks: &[Task]) -> Result<usize> {
    let mut conn = db.lock()?;
    let tx = conn.transaction()?;
    let mut inserted = 0usize;
    for task in tasks {
        let result = json_to_sql(&task.result)?;
        let metadata = json_to_sql(&task.metadata)?;
        tx.execute(
            INSERT_TASK,
            params![
                &task.id,
                &task.project_id,
                &task.parent_task_id,
                task.is_composite,
                &task.title,
                &task.description,
                &task.status,
                &task.kind,
                task.priority,
                &task.agent_id,
                task.claimed_at.map(dt_to_sql),
                task.started_at.map(dt_to_sql),
                task.completed_at.map(dt_to_sql),
                &result,
                &task.error,
                task.progress,
                &task.progress_note,
                task.max_retries,
                task.retry_count,
                &task.retry_backoff,
                task.retry_delay_ms,
                task.timeout_seconds,
                task.heartbeat_interval,
                task.last_heartbeat.map(dt_to_sql),
                task.requires_approval,
                &task.approval_status,
                &task.approved_by,
                &task.approval_comment,
                &metadata,
                dt_to_sql(task.created_at),
                dt_to_sql(task.updated_at)
            ],
        )?;
        inserted += 1;
    }
    tx.commit()?;
    Ok(inserted)
}

pub fn promote_ready_tasks(db: &Database) -> Result<usize> {
    let conn = db.lock()?;
    let now = dt_to_sql(now_utc_naive());
    Ok(conn.execute(PROMOTE_READY, params![now])?)
}

pub fn project_state(db: &Database, project_id: &str) -> Result<ProjectState> {
    let tasks = list_tasks(
        db,
        TaskListFilters {
            project_id: Some(project_id.to_string()),
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
    Ok(ProjectState {
        total,
        done,
        ready,
        running,
        pending,
    })
}

pub fn insert_task_between(
    db: &Database,
    project_id: &str,
    after_task: &str,
    before_task: Option<&str>,
    title: &str,
    description: Option<String>,
) -> Result<Task> {
    let after = get_task(db, after_task)?;
    if after.project_id != project_id {
        return Err(
            PlandbError::Conflict("after task belongs to a different project".to_string()).into(),
        );
    }

    if let Some(before_task_id) = before_task {
        let before = get_task(db, before_task_id)?;
        if before.project_id != project_id {
            return Err(PlandbError::Conflict(
                "before task belongs to a different project".to_string(),
            )
            .into());
        }
    }

    let now = now_utc_naive();
    let task = Task {
        id: generate_id("task"),
        project_id: project_id.to_string(),
        parent_task_id: None,
        is_composite: false,
        title: title.to_string(),
        description,
        status: TaskStatus::Pending,
        kind: TaskKind::Generic,
        priority: 0,
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
        metadata: None,
        created_at: now,
        updated_at: now,
    };

    let created = create_task(db, &task, &[])?;

    add_dependency(
        db,
        after_task,
        &created.id,
        crate::models::DependencyKind::FeedsInto,
        crate::models::DependencyCondition::All,
        None,
    )?;

    if let Some(before_task_id) = before_task {
        let _ = remove_dependency(db, after_task, before_task_id)?;
        add_dependency(
            db,
            &created.id,
            before_task_id,
            crate::models::DependencyKind::FeedsInto,
            crate::models::DependencyCondition::All,
            None,
        )?;

        let before = get_task(db, before_task_id)?;
        if before.status == TaskStatus::Ready
            && !matches!(after.status, TaskStatus::Done | TaskStatus::DonePartial)
        {
            let conn = db.lock()?;
            conn.execute(
                "UPDATE tasks SET status = 'pending', updated_at = datetime('now') WHERE id = ?1 AND status = 'ready'",
                params![before_task_id],
            )?;
        }
    }

    let _ = promote_ready_tasks(db)?;
    get_task(db, &created.id)
}

pub fn amend_task_description(db: &Database, task_id: &str, text: &str) -> Result<Task> {
    let task = get_task(db, task_id)?;
    if !matches!(task.status, TaskStatus::Pending | TaskStatus::Ready) {
        return Err(PlandbError::InvalidTransition(format!(
            "task {task_id} must be pending or ready to amend"
        ))
        .into());
    }

    let amended = match task.description {
        Some(description) if !description.is_empty() => format!("{text}\n---\n{description}"),
        _ => text.to_string(),
    };

    let conn = db.lock()?;
    conn.execute(
        "UPDATE tasks SET description = ?2, updated_at = ?3 WHERE id = ?1",
        params![task_id, amended, dt_to_sql(now_utc_naive())],
    )?;
    drop(conn);

    get_task(db, task_id)
}

pub fn get_lookahead(db: &Database, project_id: &str, depth: usize) -> Result<LookaheadResult> {
    let tasks = list_tasks(
        db,
        TaskListFilters {
            project_id: Some(project_id.to_string()),
            ..Default::default()
        },
    )?;

    let task_by_id: HashMap<String, Task> = tasks.into_iter().map(|t| (t.id.clone(), t)).collect();
    let mut downstream: HashMap<String, Vec<String>> = HashMap::new();
    let mut upstream: HashMap<String, Vec<String>> = HashMap::new();

    for task_id in task_by_id.keys() {
        downstream.insert(task_id.clone(), Vec::new());
        upstream.insert(task_id.clone(), Vec::new());
    }

    let conn = db.lock()?;
    let mut stmt = conn.prepare(
        r#"
        SELECT d.from_task, d.to_task
        FROM dependencies d
        JOIN tasks ft ON ft.id = d.from_task
        JOIN tasks tt ON tt.id = d.to_task
        WHERE ft.project_id = ?1 AND tt.project_id = ?1
        "#,
    )?;
    let mut rows = stmt.query(params![project_id])?;
    while let Some(row) = rows.next()? {
        let from_task: String = row.get(0)?;
        let to_task: String = row.get(1)?;
        if let Some(children) = downstream.get_mut(&from_task) {
            children.push(to_task.clone());
        }
        if let Some(parents) = upstream.get_mut(&to_task) {
            parents.push(from_task.clone());
        }
    }
    let mut current: Vec<Task> = task_by_id
        .values()
        .filter(|task| matches!(task.status, TaskStatus::Running | TaskStatus::Claimed))
        .cloned()
        .collect();
    current.sort_by(|a, b| a.id.cmp(&b.id));

    let mut queue: VecDeque<(String, usize)> = current
        .iter()
        .map(|task| (task.id.clone(), 0usize))
        .collect();
    let mut best_hops: HashMap<String, usize> = HashMap::new();

    while let Some((task_id, hops)) = queue.pop_front() {
        if hops >= depth {
            continue;
        }
        if let Some(children) = downstream.get(&task_id) {
            for child in children {
                let next_hops = hops + 1;
                let update = best_hops
                    .get(child)
                    .map(|current_hops| next_hops < *current_hops)
                    .unwrap_or(true);
                if update {
                    best_hops.insert(child.clone(), next_hops);
                    queue.push_back((child.clone(), next_hops));
                }
            }
        }
    }

    let current_ids: HashSet<String> = current.iter().map(|task| task.id.clone()).collect();
    let mut upcoming = Vec::new();
    for (task_id, hops) in &best_hops {
        if *hops == 0 || current_ids.contains(task_id) {
            continue;
        }
        if let Some(task) = task_by_id.get(task_id) {
            let blocked_by = upstream
                .get(task_id)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .filter(|upstream_id| {
                    task_by_id
                        .get(upstream_id)
                        .map(|upstream_task| {
                            !matches!(
                                upstream_task.status,
                                TaskStatus::Done | TaskStatus::DonePartial
                            )
                        })
                        .unwrap_or(false)
                })
                .collect::<Vec<_>>();

            upcoming.push(LookaheadTask {
                id: task.id.clone(),
                title: task.title.clone(),
                hops: *hops,
                blocked_by,
                description: task.description.clone(),
            });
        }
    }

    upcoming.sort_by(|a, b| a.hops.cmp(&b.hops).then_with(|| a.id.cmp(&b.id)));
    let updatable = upcoming
        .iter()
        .map(|task| task.id.clone())
        .collect::<Vec<_>>();

    Ok(LookaheadResult {
        current,
        upcoming,
        updatable,
    })
}

pub fn pivot_subtree(
    db: &Database,
    parent_id: &str,
    keep_done: bool,
    new_subtasks: Vec<NewSubtask>,
) -> Result<PivotResult> {
    let parent = get_task(db, parent_id)?;

    let children = list_tasks(
        db,
        TaskListFilters {
            project_id: Some(parent.project_id.clone()),
            parent_task_id: Some(parent_id.to_string()),
            ..Default::default()
        },
    )?;

    if children
        .iter()
        .any(|child| matches!(child.status, TaskStatus::Running))
    {
        return Err(PlandbError::InvalidTransition(
            "cannot pivot while a child is running; pause or complete it first".to_string(),
        )
        .into());
    }

    let mut kept = Vec::new();
    let mut cancelled = Vec::new();
    let now = dt_to_sql(now_utc_naive());

    {
        let conn = db.lock()?;
        for child in &children {
            if matches!(child.status, TaskStatus::Done | TaskStatus::DonePartial) {
                if keep_done {
                    kept.push(child.id.clone());
                }
                continue;
            }

            if matches!(
                child.status,
                TaskStatus::Pending | TaskStatus::Ready | TaskStatus::Claimed
            ) {
                conn.execute(
                    "UPDATE tasks SET status = 'cancelled', updated_at = ?2 WHERE id = ?1",
                    params![child.id, now],
                )?;
                cancelled.push(child.id.clone());
            }
        }
        conn.execute(
            "UPDATE tasks SET is_composite = 1, updated_at = datetime('now') WHERE id = ?1",
            params![parent_id],
        )?;
    }

    let now = now_utc_naive();
    let mut title_to_id: HashMap<String, String> = HashMap::new();
    let mut created = Vec::new();
    for subtask in &new_subtasks {
        let has_deps = subtask
            .deps_on
            .as_ref()
            .map(|deps| !deps.is_empty())
            .unwrap_or(false);
        let task = Task {
            id: generate_id("task"),
            project_id: parent.project_id.clone(),
            parent_task_id: Some(parent_id.to_string()),
            is_composite: false,
            title: subtask.title.clone(),
            description: subtask.description.clone(),
            status: if has_deps {
                TaskStatus::Pending
            } else {
                TaskStatus::Ready
            },
            kind: subtask.kind.clone().unwrap_or(TaskKind::Generic),
            priority: subtask.priority.unwrap_or(0),
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
            metadata: None,
            created_at: now,
            updated_at: now,
        };
        let created_task = create_task(db, &task, &[])?;
        title_to_id.insert(subtask.title.clone(), created_task.id.clone());
        created.push(created_task.id);
    }

    for subtask in &new_subtasks {
        if let Some(deps_on) = &subtask.deps_on {
            let to_id = title_to_id
                .get(&subtask.title)
                .ok_or_else(|| anyhow::anyhow!("missing subtask id for title {}", subtask.title))?;
            for dep_title in deps_on {
                let from_id = title_to_id.get(dep_title).ok_or_else(|| {
                    anyhow::anyhow!("deps_on references unknown subtask title: {}", dep_title)
                })?;
                add_dependency(
                    db,
                    from_id,
                    to_id,
                    crate::models::DependencyKind::FeedsInto,
                    crate::models::DependencyCondition::All,
                    None,
                )?;
            }
        }
    }

    let _ = promote_ready_tasks(db)?;

    Ok(PivotResult {
        kept,
        cancelled,
        created,
    })
}

pub fn split_task(db: &Database, task_id: &str, parts: Vec<SplitPart>) -> Result<SplitResult> {
    if parts.is_empty() {
        return Err(anyhow::anyhow!("split requires at least one part"));
    }

    let parent = get_task(db, task_id)?;
    {
        let conn = db.lock()?;
        conn.execute(
            "UPDATE tasks SET is_composite = 1, status = 'pending', agent_id = NULL, updated_at = datetime('now') WHERE id = ?1",
            params![task_id],
        )?;
    }

    let mut seen_titles = HashSet::new();
    for part in &parts {
        if !seen_titles.insert(part.title.clone()) {
            return Err(anyhow::anyhow!("duplicate split title: {}", part.title));
        }
    }

    let now = now_utc_naive();
    let mut title_to_id = HashMap::new();
    let mut created = Vec::new();
    let mut done_ids = Vec::new();

    for part in &parts {
        let has_deps = part
            .deps_on
            .as_ref()
            .map(|deps| !deps.is_empty())
            .unwrap_or(false);
        let mut task = Task {
            id: generate_id("task"),
            project_id: parent.project_id.clone(),
            parent_task_id: Some(task_id.to_string()),
            is_composite: false,
            title: part.title.clone(),
            description: part.description.clone(),
            status: if has_deps {
                TaskStatus::Pending
            } else {
                TaskStatus::Ready
            },
            kind: TaskKind::Generic,
            priority: 0,
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
            metadata: None,
            created_at: now,
            updated_at: now,
        };

        if part.done.unwrap_or(false) {
            task.status = TaskStatus::Done;
            task.completed_at = Some(now);
            task.result = part
                .result
                .as_ref()
                .map(|value| serde_json::Value::String(value.clone()));
        }

        let created_task = create_task(db, &task, &[])?;
        if part.done.unwrap_or(false) {
            done_ids.push(created_task.id.clone());
        }
        title_to_id.insert(part.title.clone(), created_task.id.clone());
        created.push(created_task.id);
    }

    for part in &parts {
        if let Some(deps_on) = &part.deps_on {
            let to_id = title_to_id
                .get(&part.title)
                .ok_or_else(|| anyhow::anyhow!("missing split part id for title {}", part.title))?;
            for dep_title in deps_on {
                let from_id = title_to_id.get(dep_title).ok_or_else(|| {
                    anyhow::anyhow!("deps_on references unknown part: {}", dep_title)
                })?;
                add_dependency(
                    db,
                    from_id,
                    to_id,
                    crate::models::DependencyKind::FeedsInto,
                    crate::models::DependencyCondition::All,
                    None,
                )?;
            }
        }
    }

    let _ = promote_ready_tasks(db)?;

    Ok(SplitResult {
        parent_task_id: task_id.to_string(),
        created,
        done: done_ids,
        title_to_id,
    })
}

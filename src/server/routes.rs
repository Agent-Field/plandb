use super::sse::{event_stream_handler, EventStreamQuery};
use crate::db::*;
use crate::models::*;
use anyhow::Result;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::{NaiveDateTime, Utc};
use rusqlite::params;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;

type AppState = Arc<Database>;

pub fn api_routes() -> Router<AppState> {
    Router::new()
        .route("/projects", post(create_project_handler).get(list_projects_handler))
        .route(
            "/projects/:id",
            get(get_project_handler).patch(update_project_status_handler),
        )
        .route("/projects/:id/status", get(project_status_handler))
        .route("/projects/:id/dag", get(project_dag_handler))
        .route(
            "/projects/:project_id/tasks",
            post(create_task_handler).get(list_tasks_handler),
        )
        .route(
            "/projects/:project_id/tasks/batch",
            post(batch_create_tasks_handler),
        )
        .route("/projects/:project_id/events", get(list_events_handler))
        .route("/tasks/:id", get(get_task_handler).patch(update_task_handler))
        .route("/tasks/:id/context", get(get_task_context_handler))
        .route("/go", post(go_handler))
        .route("/tasks/:id/claim", post(claim_task_handler))
        .route("/tasks/:id/start", post(start_task_handler))
        .route("/tasks/:id/heartbeat", post(task_heartbeat_handler))
        .route("/tasks/:id/progress", post(task_progress_handler))
        .route("/tasks/:id/done", post(done_task_handler))
        .route("/tasks/:id/notes", post(add_task_note_handler).get(list_task_notes_handler))
        .route("/tasks/:id/pause", post(pause_task_handler))
        .route("/tasks/:id/fail", post(fail_task_handler))
        .route("/tasks/:id/cancel", post(cancel_task_handler))
        .route("/tasks/:id/approve", post(approve_task_handler))
        .route("/tasks/next", post(next_task_handler))
        .route(
            "/tasks/:task_id/artifacts",
            post(create_artifact_handler).get(list_task_artifacts_handler),
        )
        .route("/tasks/:task_id/upstream-artifacts", get(upstream_artifacts_handler))
        .route("/artifacts/:id", get(get_artifact_handler))
        .route("/events/stream", get(event_stream_handler))
        .route(
            "/tasks/:id/deps",
            post(add_dependency_handler).delete(remove_dependency_handler),
        )
        .route("/projects/:id/overview", get(project_overview_handler))
        .route("/tasks/:id/decompose", post(decompose_task_handler))
        .route("/tasks/:id/replan", post(replan_task_handler))
        .route("/tasks/insert", post(insert_task_handler))
        .route("/tasks/:id/amend", post(amend_task_handler))
        .route("/tasks/:id/pivot", post(pivot_task_handler))
        .route("/tasks/:id/split", post(split_task_handler))
        .route("/ahead", get(ahead_handler))
        .route("/what-if", post(what_if_handler))
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    code: String,
    message: String,
}

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "bad_request",
            message: message.into(),
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            code: "not_found",
            message: message.into(),
        }
    }

    fn conflict(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            code: "conflict",
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "internal_error",
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(ErrorBody {
            error: ErrorDetail {
                code: self.code.to_string(),
                message: self.message,
            },
        });
        (self.status, body).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(value: anyhow::Error) -> Self {
        if let Some(plandb) = value.downcast_ref::<PlandbError>() {
            return match plandb {
                PlandbError::NotFound(msg) => ApiError::not_found(msg.clone()),
                PlandbError::Conflict(msg) => ApiError::conflict(msg.clone()),
                PlandbError::InvalidTransition(msg) => ApiError::conflict(msg.clone()),
            };
        }

        if let Some(sql) = value.downcast_ref::<rusqlite::Error>() {
            if matches!(sql, rusqlite::Error::QueryReturnedNoRows) {
                return ApiError::not_found("resource not found");
            }
        }

        ApiError::internal(value.to_string())
    }
}

#[derive(Debug, Deserialize)]
pub struct CreateProjectRequest {
    name: String,
    description: Option<String>,
    metadata: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateProjectStatusRequest {
    status: String,
}

#[derive(Debug, Serialize)]
struct ProjectStatusResponse {
    project_id: String,
    total_tasks: usize,
    by_status: HashMap<String, usize>,
    progress_percent: f64,
}

#[derive(Debug, Serialize)]
struct DagResponse {
    project_id: String,
    nodes: Vec<Task>,
    edges: Vec<Dependency>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CreateTaskRequest {
    title: String,
    description: Option<String>,
    parent_task_id: Option<String>,
    is_composite: Option<bool>,
    status: Option<String>,
    kind: Option<String>,
    priority: Option<i32>,
    max_retries: Option<i32>,
    retry_backoff: Option<String>,
    retry_delay_ms: Option<i64>,
    timeout_seconds: Option<i64>,
    heartbeat_interval: Option<i32>,
    requires_approval: Option<bool>,
    metadata: Option<Value>,
    tags: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct BatchCreateTasksRequest {
    tasks: Vec<CreateTaskRequest>,
}

#[derive(Debug, Deserialize)]
pub struct ListTasksQuery {
    status: Option<String>,
    kind: Option<String>,
    agent_id: Option<String>,
    tag: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ClaimRequest {
    agent_id: String,
}

#[derive(Debug, Deserialize)]
pub struct ProgressRequest {
    percent: Option<i32>,
    note: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct DoneRequest {
    result: Option<Value>,
    files: Option<Vec<String>>,
    next: Option<bool>,
    agent_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct GoRequest {
    project_id: String,
    agent_id: String,
}

#[derive(Debug, Deserialize)]
pub struct TaskNoteRequest {
    content: String,
    agent_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct PauseRequest {
    progress: Option<i32>,
    note: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct FailRequest {
    error: String,
}

#[derive(Debug, Deserialize)]
pub struct CancelQuery {
    cascade: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct ApproveRequest {
    by: Option<String>,
    comment: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct NextTaskRequest {
    project_id: String,
    agent_id: String,
    claim: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct CreateArtifactRequest {
    name: String,
    content: Option<String>,
    path: Option<String>,
    kind: Option<String>,
    mime_type: Option<String>,
    metadata: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct ListEventsQuery {
    #[serde(rename = "type")]
    event_type: Option<String>,
    since: Option<String>,
    limit: Option<usize>,
}

#[derive(Debug, Serialize)]
struct TaskContextResponse {
    task: Task,
    project: Project,
    upstream_artifacts: Vec<Artifact>,
    downstream: Vec<Task>,
    siblings: Vec<Task>,
}

#[derive(Debug, Serialize)]
struct BatchCreateResponse {
    created: usize,
}

#[derive(Debug, Serialize)]
struct CancelResponse {
    cancelled: usize,
}

#[derive(Debug, Deserialize)]
pub struct AddDependencyRequest {
    from_task: String,
    kind: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RemoveDependencyRequest {
    from_task: String,
}

#[derive(Debug, Deserialize)]
pub struct UpdateTaskRequest {
    title: Option<String>,
    description: Option<String>,
    kind: Option<String>,
    priority: Option<i32>,
    metadata: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct DecomposeSubtaskRequest {
    title: String,
    kind: Option<String>,
    description: Option<String>,
    deps_on: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct DecomposeRequest {
    subtasks: Vec<DecomposeSubtaskRequest>,
}

#[derive(Debug, Deserialize)]
pub struct InsertTaskRequest {
    project: String,
    after_task: String,
    before_task: Option<String>,
    title: String,
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AmendTaskRequest {
    prepend: String,
}

#[derive(Debug, Deserialize)]
pub struct PivotTaskRequest {
    keep_done: Option<bool>,
    subtasks: Vec<NewSubtask>,
}

#[derive(Debug, Deserialize)]
pub struct SplitTaskRequest {
    parts: Vec<SplitPart>,
}

#[derive(Debug, Deserialize)]
pub struct AheadQuery {
    project: String,
    depth: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct WhatIfRequest {
    mutation_type: String,
    task_id: Option<String>,
    after_task: Option<String>,
    before_task: Option<String>,
    title: Option<String>,
    project: Option<String>,
}

#[derive(Debug, Serialize)]
struct OverviewResponse {
    project: Project,
    summary: OverviewSummary,
    ready_task_ids: Vec<String>,
    tasks: Vec<OverviewTask>,
    edges: Vec<OverviewEdge>,
}

#[derive(Debug, Serialize)]
struct OverviewSummary {
    total: usize,
    pending: usize,
    ready: usize,
    claimed: usize,
    running: usize,
    done: usize,
    failed: usize,
    cancelled: usize,
    progress_percent: f64,
}

#[derive(Debug, Serialize)]
struct OverviewTask {
    id: String,
    title: String,
    status: TaskStatus,
    kind: TaskKind,
    priority: i32,
    agent_id: Option<String>,
    parent_task_id: Option<String>,
    is_composite: bool,
}

#[derive(Debug, Serialize)]
struct OverviewEdge {
    from: String,
    to: String,
    kind: String,
}

#[derive(Debug, Serialize)]
struct DecomposeResponse {
    parent_task_id: String,
    subtask_ids: Vec<String>,
}

fn parse_task_status(value: Option<String>) -> Result<Option<TaskStatus>, ApiError> {
    value
        .map(|raw| {
            TaskStatus::from_str(&raw)
                .map_err(|_| ApiError::bad_request(format!("invalid task status: {raw}")))
        })
        .transpose()
}

fn parse_task_kind(value: Option<String>) -> Result<Option<TaskKind>, ApiError> {
    value
        .map(|raw| {
            TaskKind::from_str(&raw)
                .map_err(|_| ApiError::bad_request(format!("invalid task kind: {raw}")))
        })
        .transpose()
}

fn parse_project_status(raw: &str) -> Result<ProjectStatus, ApiError> {
    ProjectStatus::from_str(raw)
        .map_err(|_| ApiError::bad_request(format!("invalid project status: {raw}")))
}

fn parse_retry_backoff(raw: Option<String>) -> Result<RetryBackoff, ApiError> {
    match raw {
        Some(v) => RetryBackoff::from_str(&v)
            .map_err(|_| ApiError::bad_request(format!("invalid retry_backoff: {v}"))),
        None => Ok(RetryBackoff::Exponential),
    }
}

fn parse_event_type(raw: Option<String>) -> Result<Option<EventType>, ApiError> {
    raw.map(|v| {
        EventType::from_str(&v).map_err(|_| ApiError::bad_request(format!("invalid event type: {v}")))
    })
    .transpose()
}

fn parse_since(since: Option<String>) -> Result<Option<NaiveDateTime>, ApiError> {
    let Some(raw) = since else {
        return Ok(None);
    };
    if let Ok(ts) = NaiveDateTime::parse_from_str(&raw, "%Y-%m-%d %H:%M:%S") {
        return Ok(Some(ts));
    }
    let dt = chrono::DateTime::parse_from_rfc3339(&raw)
        .map_err(|_| ApiError::bad_request(format!("invalid since timestamp: {raw}")))?;
    Ok(Some(dt.naive_utc()))
}

fn emit_event(
    db: &Database,
    task_id: Option<&str>,
    project_id: Option<&str>,
    agent_id: Option<&str>,
    event_type: EventType,
    payload: Option<Value>,
) -> Result<()> {
    let _ = insert_event(
        db,
        task_id,
        project_id,
        agent_id,
        event_type,
        payload,
        Utc::now().naive_utc(),
    )?;
    Ok(())
}

fn go_response(db: &Database, project_id: &str, agent_id: &str) -> Result<Value, ApiError> {
    let claimed = claim_next_task(db, project_id, agent_id).map_err(ApiError::from)?;
    let task = match claimed {
        Some(t) => Some(start_task(db, &t.id).map_err(ApiError::from)?),
        None => None,
    };

    let tasks = list_tasks(
        db,
        TaskListFilters {
            project_id: Some(project_id.to_string()),
            ..Default::default()
        },
    )
    .map_err(ApiError::from)?;
    let total = tasks.len();
    let done = tasks
        .iter()
        .filter(|t| matches!(t.status, TaskStatus::Done | TaskStatus::DonePartial))
        .count();
    let ready = tasks.iter().filter(|t| t.status == TaskStatus::Ready).count();
    let running = tasks
        .iter()
        .filter(|t| matches!(t.status, TaskStatus::Running | TaskStatus::Claimed))
        .count();
    let pending = tasks.iter().filter(|t| t.status == TaskStatus::Pending).count();
    let progress = if total == 0 {
        "0%".to_string()
    } else {
        format!("{}%", ((done as f64 / total as f64) * 100.0).round() as i32)
    };

    let mut handoff = Vec::new();
    let mut notes = Vec::new();
    let mut file_conflicts = json!([]);
    let task_json = if let Some(task) = task {
        handoff = get_handoff_context(db, &task.id)
            .map_err(ApiError::from)?
            .into_iter()
            .map(|h| {
                json!({
                    "from_task": h.from_task_id,
                    "from_title": h.from_title,
                    "result": h.result,
                    "agent_id": h.agent_id,
                })
            })
            .collect();
        notes = list_notes(db, &task.id)
            .map_err(ApiError::from)?
            .into_iter()
            .map(|n| {
                json!({
                    "content": n.content,
                    "agent_id": n.agent_id,
                    "created_at": n.created_at,
                })
            })
            .collect();
        file_conflicts = serde_json::to_value(
            check_file_conflicts(db, project_id, Some(&task.id)).map_err(ApiError::from)?,
        )
        .map_err(|e| ApiError::internal(e.to_string()))?;
        json!({
            "id": task.id,
            "title": task.title,
            "status": task.status,
            "description": task.description,
        })
    } else {
        Value::Null
    };

    Ok(json!({
        "task": task_json,
        "handoff": handoff,
        "notes": notes,
        "file_conflicts": file_conflicts,
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

fn ensure_project_exists(db: &Database, project_id: &str) -> Result<Project, ApiError> {
    get_project(db, project_id).map_err(ApiError::from)
}

fn build_task(project_id: &str, req: &CreateTaskRequest) -> Result<(Task, Vec<String>), ApiError> {
    let now = Utc::now().naive_utc();
    let status = parse_task_status(req.status.clone())?.unwrap_or(TaskStatus::Pending);
    let kind = parse_task_kind(req.kind.clone())?.unwrap_or(TaskKind::Generic);
    let retry_backoff = parse_retry_backoff(req.retry_backoff.clone())?;

    let task = Task {
        id: generate_id("task"),
        project_id: project_id.to_string(),
        parent_task_id: req.parent_task_id.clone(),
        is_composite: req.is_composite.unwrap_or(false),
        title: req.title.clone(),
        description: req.description.clone(),
        status,
        kind,
        priority: req.priority.unwrap_or(0),
        agent_id: None,
        claimed_at: None,
        started_at: None,
        completed_at: None,
        result: None,
        error: None,
        progress: None,
        progress_note: None,
        max_retries: req.max_retries.unwrap_or(0),
        retry_count: 0,
        retry_backoff,
        retry_delay_ms: req.retry_delay_ms.unwrap_or(1000),
        timeout_seconds: req.timeout_seconds,
        heartbeat_interval: req.heartbeat_interval.unwrap_or(30),
        last_heartbeat: None,
        requires_approval: req.requires_approval.unwrap_or(false),
        approval_status: None,
        approved_by: None,
        approval_comment: None,
        metadata: req.metadata.clone(),
        created_at: now,
        updated_at: now,
    };

    Ok((task, req.tags.clone().unwrap_or_default()))
}

pub async fn create_project_handler(
    State(db): State<AppState>,
    Json(body): Json<CreateProjectRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let project = create_project(&db, &body.name, body.description, body.metadata).map_err(ApiError::from)?;
    Ok((StatusCode::CREATED, Json(project)))
}

pub async fn list_projects_handler(
    State(db): State<AppState>,
) -> Result<impl IntoResponse, ApiError> {
    let projects = list_projects(&db).map_err(ApiError::from)?;
    Ok(Json(projects))
}

pub async fn get_project_handler(
    State(db): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let project = get_project(&db, &project_id).map_err(ApiError::from)?;
    Ok(Json(project))
}

pub async fn update_project_status_handler(
    State(db): State<AppState>,
    Path(project_id): Path<String>,
    Json(body): Json<UpdateProjectStatusRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let status = parse_project_status(&body.status)?;
    let updated = update_project_status(&db, &project_id, status).map_err(ApiError::from)?;
    Ok(Json(updated))
}

pub async fn project_status_handler(
    State(db): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    ensure_project_exists(&db, &project_id)?;
    let tasks = list_tasks(
        &db,
        TaskListFilters {
            project_id: Some(project_id.clone()),
            ..Default::default()
        },
    )
    .map_err(ApiError::from)?;

    let mut by_status: HashMap<String, usize> = HashMap::new();
    for task in &tasks {
        *by_status.entry(task.status.to_string()).or_insert(0) += 1;
    }
    let total = tasks.len();
    let done_count = by_status.get("done").copied().unwrap_or(0)
        + by_status.get("done_partial").copied().unwrap_or(0);
    let progress_percent = if total == 0 {
        0.0
    } else {
        (done_count as f64 / total as f64) * 100.0
    };

    Ok(Json(ProjectStatusResponse {
        project_id,
        total_tasks: total,
        by_status,
        progress_percent,
    }))
}

pub async fn project_dag_handler(
    State(db): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    ensure_project_exists(&db, &project_id)?;
    let nodes = list_tasks(
        &db,
        TaskListFilters {
            project_id: Some(project_id.clone()),
            ..Default::default()
        },
    )
    .map_err(ApiError::from)?;

    let conn = db.lock().map_err(ApiError::from)?;
    let mut stmt = conn
        .prepare(
            r#"
            SELECT d.id, d.from_task, d.to_task, d.kind, d.condition, d.metadata
            FROM dependencies d
            JOIN tasks t ON t.id = d.to_task
            WHERE t.project_id = ?1
            ORDER BY d.id ASC
            "#,
        )
        .map_err(|e| ApiError::from(anyhow::Error::from(e)))?;
    let rows = stmt
        .query_map(params![project_id], |row| {
            let metadata: Option<String> = row.get(5)?;
            Ok(Dependency {
                id: row.get(0)?,
                from_task: row.get(1)?,
                to_task: row.get(2)?,
                kind: row.get(3)?,
                condition: row.get(4)?,
                metadata: metadata.and_then(|m| serde_json::from_str::<Value>(&m).ok()),
            })
        })
        .map_err(|e| ApiError::from(anyhow::Error::from(e)))?;
    let mut edges = Vec::new();
    for row in rows {
        edges.push(row.map_err(|e| ApiError::from(anyhow::Error::from(e)))?);
    }

    Ok(Json(DagResponse {
        project_id,
        nodes,
        edges,
    }))
}

pub async fn create_task_handler(
    State(db): State<AppState>,
    Path(project_id): Path<String>,
    Json(body): Json<CreateTaskRequest>,
) -> Result<impl IntoResponse, ApiError> {
    ensure_project_exists(&db, &project_id)?;
    let (task, tags) = build_task(&project_id, &body)?;
    let created = create_task(&db, &task, &tags).map_err(ApiError::from)?;
    emit_event(
        &db,
        Some(&created.id),
        Some(&created.project_id),
        None,
        EventType::TaskCreated,
        None,
    )
    .map_err(ApiError::from)?;
    let _ = promote_ready_tasks(&db).map_err(ApiError::from)?;
    Ok((StatusCode::CREATED, Json(created)))
}

pub async fn batch_create_tasks_handler(
    State(db): State<AppState>,
    Path(project_id): Path<String>,
    Json(body): Json<BatchCreateTasksRequest>,
) -> Result<impl IntoResponse, ApiError> {
    ensure_project_exists(&db, &project_id)?;
    let mut tasks = Vec::with_capacity(body.tasks.len());
    for req in &body.tasks {
        let (task, _) = build_task(&project_id, req)?;
        tasks.push(task);
    }

    let created = batch_create_tasks(&db, &tasks).map_err(ApiError::from)?;
    let _ = promote_ready_tasks(&db).map_err(ApiError::from)?;
    Ok((
        StatusCode::CREATED,
        Json(BatchCreateResponse { created }),
    ))
}

pub async fn list_tasks_handler(
    State(db): State<AppState>,
    Path(project_id): Path<String>,
    Query(query): Query<ListTasksQuery>,
) -> Result<impl IntoResponse, ApiError> {
    ensure_project_exists(&db, &project_id)?;
    let tasks = list_tasks(
        &db,
        TaskListFilters {
            project_id: Some(project_id),
            status: parse_task_status(query.status)?,
            kind: parse_task_kind(query.kind)?,
            parent_task_id: None,
            agent_id: query.agent_id,
            tags: query.tag.map(|t| vec![t]).unwrap_or_default(),
        },
    )
    .map_err(ApiError::from)?;
    Ok(Json(tasks))
}

pub async fn get_task_handler(
    State(db): State<AppState>,
    Path(task_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let task = get_task(&db, &task_id).map_err(ApiError::from)?;
    Ok(Json(task))
}

pub async fn get_task_context_handler(
    State(db): State<AppState>,
    Path(task_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let task = get_task(&db, &task_id).map_err(ApiError::from)?;
    let project = get_project(&db, &task.project_id).map_err(ApiError::from)?;
    let upstream_artifacts = get_upstream_artifacts(&db, &task.id).map_err(ApiError::from)?;

    let downstream_ids = get_downstream_tasks(&db, &task.id).map_err(ApiError::from)?;
    let mut downstream = Vec::new();
    for id in downstream_ids {
        if let Ok(t) = get_task(&db, &id) {
            downstream.push(t);
        }
    }

    let siblings = if let Some(parent_id) = task.parent_task_id.clone() {
        list_tasks(
            &db,
            TaskListFilters {
                project_id: Some(task.project_id.clone()),
                parent_task_id: Some(parent_id),
                ..Default::default()
            },
        )
        .map_err(ApiError::from)?
        .into_iter()
        .filter(|t| t.id != task.id)
        .collect()
    } else {
        Vec::new()
    };

    Ok(Json(TaskContextResponse {
        task,
        project,
        upstream_artifacts,
        downstream,
        siblings,
    }))
}

pub async fn claim_task_handler(
    State(db): State<AppState>,
    Path(task_id): Path<String>,
    Json(body): Json<ClaimRequest>,
) -> Result<impl IntoResponse, ApiError> {
    if get_task(&db, &task_id).is_err() {
        return Err(ApiError::not_found(format!("Task {task_id} not found")));
    }
    let Some(task) = claim_task(&db, &task_id, &body.agent_id).map_err(ApiError::from)? else {
        return Err(ApiError::conflict(format!("Task {task_id} is not ready to claim")));
    };
    emit_event(
        &db,
        Some(&task.id),
        Some(&task.project_id),
        Some(&body.agent_id),
        EventType::TaskClaimed,
        None,
    )
    .map_err(ApiError::from)?;
    Ok(Json(task))
}

pub async fn start_task_handler(
    State(db): State<AppState>,
    Path(task_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let task = start_task(&db, &task_id).map_err(ApiError::from)?;
    emit_event(
        &db,
        Some(&task.id),
        Some(&task.project_id),
        task.agent_id.as_deref(),
        EventType::TaskStarted,
        None,
    )
    .map_err(ApiError::from)?;
    Ok(Json(task))
}

pub async fn task_heartbeat_handler(
    State(db): State<AppState>,
    Path(task_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let changed = update_heartbeat(&db, &task_id).map_err(ApiError::from)?;
    if changed == 0 {
        return Err(ApiError::conflict(format!(
            "Task {task_id} is not claimed or running"
        )));
    }
    let task = get_task(&db, &task_id).map_err(ApiError::from)?;
    Ok(Json(task))
}

pub async fn task_progress_handler(
    State(db): State<AppState>,
    Path(task_id): Path<String>,
    Json(body): Json<ProgressRequest>,
) -> Result<impl IntoResponse, ApiError> {
    if let Some(percent) = body.percent {
        if !(0..=100).contains(&percent) {
            return Err(ApiError::bad_request("percent must be between 0 and 100"));
        }
    }
    let changed = update_progress(&db, &task_id, body.percent, body.note).map_err(ApiError::from)?;
    if changed == 0 {
        return Err(ApiError::not_found(format!("Task {task_id} not found")));
    }
    let task = get_task(&db, &task_id).map_err(ApiError::from)?;
    Ok(Json(task))
}

pub async fn done_task_handler(
    State(db): State<AppState>,
    Path(task_id): Path<String>,
    Json(body): Json<DoneRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let task = complete_task(&db, &task_id, body.result).map_err(ApiError::from)?;
    if let Some(files) = body.files {
        let _ = add_task_files(&db, &task.id, &files).map_err(ApiError::from)?;
    }
    let _ = promote_ready_tasks(&db).map_err(ApiError::from)?;
    emit_event(
        &db,
        Some(&task.id),
        Some(&task.project_id),
        task.agent_id.as_deref(),
        EventType::TaskCompleted,
        None,
    )
    .map_err(ApiError::from)?;
    if body.next.unwrap_or(false) {
        let agent_id = body
            .agent_id
            .ok_or_else(|| ApiError::bad_request("agent_id is required when next=true"))?;
        let next = go_response(&db, &task.project_id, &agent_id)?;
        Ok(Json(json!({
            "completed": {"id": task.id, "status": task.status},
            "next": next,
        })))
    } else {
        Ok(Json(serde_json::to_value(task).map_err(|e| ApiError::internal(e.to_string()))?))
    }
}

pub async fn go_handler(
    State(db): State<AppState>,
    Json(body): Json<GoRequest>,
) -> Result<impl IntoResponse, ApiError> {
    ensure_project_exists(&db, &body.project_id)?;
    let payload = go_response(&db, &body.project_id, &body.agent_id)?;
    Ok(Json(payload))
}

pub async fn add_task_note_handler(
    State(db): State<AppState>,
    Path(task_id): Path<String>,
    Json(body): Json<TaskNoteRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let _ = get_task(&db, &task_id).map_err(ApiError::from)?;
    let note = add_note(&db, &task_id, body.agent_id, &body.content).map_err(ApiError::from)?;
    Ok((
        StatusCode::CREATED,
        Json(serde_json::to_value(note).map_err(|e| ApiError::internal(e.to_string()))?),
    ))
}

pub async fn list_task_notes_handler(
    State(db): State<AppState>,
    Path(task_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let _ = get_task(&db, &task_id).map_err(ApiError::from)?;
    let notes = list_notes(&db, &task_id).map_err(ApiError::from)?;
    Ok(Json(
        serde_json::to_value(notes).map_err(|e| ApiError::internal(e.to_string()))?,
    ))
}

pub async fn pause_task_handler(
    State(db): State<AppState>,
    Path(task_id): Path<String>,
    Json(body): Json<PauseRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let task = pause_task(&db, &task_id, body.progress, body.note).map_err(ApiError::from)?;
    Ok(Json(
        serde_json::to_value(task).map_err(|e| ApiError::internal(e.to_string()))?,
    ))
}

pub async fn fail_task_handler(
    State(db): State<AppState>,
    Path(task_id): Path<String>,
    Json(body): Json<FailRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let task = fail_task(&db, &task_id, &body.error).map_err(ApiError::from)?;
    emit_event(
        &db,
        Some(&task.id),
        Some(&task.project_id),
        task.agent_id.as_deref(),
        EventType::TaskFailed,
        Some(json!({"error": body.error})),
    )
    .map_err(ApiError::from)?;
    Ok(Json(task))
}

pub async fn cancel_task_handler(
    State(db): State<AppState>,
    Path(task_id): Path<String>,
    Query(query): Query<CancelQuery>,
) -> Result<impl IntoResponse, ApiError> {
    let task = get_task(&db, &task_id).map_err(ApiError::from)?;
    let cancelled = cancel_task(&db, &task_id, query.cascade.unwrap_or(false)).map_err(ApiError::from)?;
    if cancelled == 0 {
        return Err(ApiError::conflict(format!("Task {task_id} cannot be cancelled")));
    }
    emit_event(
        &db,
        Some(&task.id),
        Some(&task.project_id),
        task.agent_id.as_deref(),
        EventType::TaskCancelled,
        Some(json!({"cascade": query.cascade.unwrap_or(false)})),
    )
    .map_err(ApiError::from)?;
    Ok(Json(CancelResponse { cancelled }))
}

pub async fn approve_task_handler(
    State(db): State<AppState>,
    Path(task_id): Path<String>,
    Json(body): Json<ApproveRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let changed = approve_task(
        &db,
        &task_id,
        "approved",
        body.by.clone(),
        body.comment.clone(),
    )
    .map_err(ApiError::from)?;
    if changed == 0 {
        return Err(ApiError::not_found(format!("Task {task_id} not found")));
    }
    let task = get_task(&db, &task_id).map_err(ApiError::from)?;
    emit_event(
        &db,
        Some(&task.id),
        Some(&task.project_id),
        body.by.as_deref(),
        EventType::ApprovalResolved,
        Some(json!({"comment": body.comment})),
    )
    .map_err(ApiError::from)?;
    Ok(Json(task))
}

pub async fn next_task_handler(
    State(db): State<AppState>,
    Json(body): Json<NextTaskRequest>,
) -> Result<impl IntoResponse, ApiError> {
    ensure_project_exists(&db, &body.project_id)?;
    let claim = body.claim.unwrap_or(true);
    if claim {
        let task = claim_next_task(&db, &body.project_id, &body.agent_id).map_err(ApiError::from)?;
        if let Some(task) = &task {
            emit_event(
                &db,
                Some(&task.id),
                Some(&task.project_id),
                Some(&body.agent_id),
                EventType::TaskClaimed,
                Some(json!({"next": true})),
            )
            .map_err(ApiError::from)?;
        }
        return Ok(Json(task));
    }

    let tasks = list_tasks(
        &db,
        TaskListFilters {
            project_id: Some(body.project_id),
            status: Some(TaskStatus::Ready),
            ..Default::default()
        },
    )
    .map_err(ApiError::from)?;
    Ok(Json(tasks.into_iter().next()))
}

pub async fn create_artifact_handler(
    State(db): State<AppState>,
    Path(task_id): Path<String>,
    Json(body): Json<CreateArtifactRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let task = get_task(&db, &task_id).map_err(ApiError::from)?;
    let content_len = body.content.as_ref().map(|v| v.len() as i64);
    let artifact = Artifact {
        id: generate_id("art"),
        task_id,
        name: body.name,
        kind: body.kind,
        content: body.content,
        path: body.path,
        size_bytes: content_len,
        mime_type: body.mime_type,
        metadata: body.metadata,
        created_at: Utc::now().naive_utc(),
    };
    let created = create_artifact(&db, &artifact).map_err(ApiError::from)?;
    emit_event(
        &db,
        Some(&created.task_id),
        Some(&task.project_id),
        task.agent_id.as_deref(),
        EventType::ArtifactCreated,
        Some(json!({"artifact_id": created.id, "name": created.name})),
    )
    .map_err(ApiError::from)?;
    Ok((StatusCode::CREATED, Json(created)))
}

pub async fn list_task_artifacts_handler(
    State(db): State<AppState>,
    Path(task_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let _ = get_task(&db, &task_id).map_err(ApiError::from)?;
    let artifacts = list_artifacts(&db, &task_id).map_err(ApiError::from)?;
    Ok(Json(artifacts))
}

pub async fn get_artifact_handler(
    State(db): State<AppState>,
    Path(artifact_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let artifact = get_artifact(&db, &artifact_id).map_err(ApiError::from)?;
    Ok(Json(artifact))
}

pub async fn upstream_artifacts_handler(
    State(db): State<AppState>,
    Path(task_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let _ = get_task(&db, &task_id).map_err(ApiError::from)?;
    let artifacts = get_upstream_artifacts(&db, &task_id).map_err(ApiError::from)?;
    Ok(Json(artifacts))
}

pub async fn list_events_handler(
    State(db): State<AppState>,
    Path(project_id): Path<String>,
    Query(query): Query<ListEventsQuery>,
) -> Result<impl IntoResponse, ApiError> {
    ensure_project_exists(&db, &project_id)?;
    let event_type = parse_event_type(query.event_type)?;
    let since = parse_since(query.since)?;
    let mut events = list_events(
        &db,
        EventFilters {
            project_id: Some(project_id),
            event_type,
            since,
            ..Default::default()
        },
    )
    .map_err(ApiError::from)?;

    if let Some(limit) = query.limit {
        events.truncate(limit);
    }
    Ok(Json(events))
}

pub async fn add_dependency_handler(
    State(db): State<AppState>,
    Path(task_id): Path<String>,
    Json(body): Json<AddDependencyRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let _ = get_task(&db, &task_id).map_err(ApiError::from)?;
    let _ = get_task(&db, &body.from_task).map_err(ApiError::from)?;

    let kind_str = body.kind.as_deref().unwrap_or("feeds_into");
    let kind = DependencyKind::from_str(kind_str)
        .map_err(|_| ApiError::bad_request(format!("invalid dependency kind: {kind_str}")))?;

    let dep = add_dependency(
        &db,
        &body.from_task,
        &task_id,
        kind,
        DependencyCondition::All,
        None,
    )
    .map_err(ApiError::from)?;

    let to_task = get_task(&db, &task_id).map_err(ApiError::from)?;
    if to_task.status == TaskStatus::Ready {
        let from_task = get_task(&db, &body.from_task).map_err(ApiError::from)?;
        if from_task.status != TaskStatus::Done && from_task.status != TaskStatus::DonePartial {
            let conn = db.lock().map_err(ApiError::from)?;
            conn.execute(
                "UPDATE tasks SET status = 'pending', updated_at = datetime('now') WHERE id = ?1 AND status = 'ready'",
                params![task_id],
            )
            .map_err(|e| ApiError::from(anyhow::Error::from(e)))?;
        }
    }
    let _ = promote_ready_tasks(&db).map_err(ApiError::from)?;

    emit_event(
        &db,
        Some(&task_id),
        None,
        None,
        EventType::DependencyAdded,
        Some(json!({"dependency_added": {"from": body.from_task, "to": task_id}})),
    )
    .map_err(ApiError::from)?;

    Ok((StatusCode::CREATED, Json(dep)))
}

pub async fn remove_dependency_handler(
    State(db): State<AppState>,
    Path(task_id): Path<String>,
    Json(body): Json<RemoveDependencyRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let removed = remove_dependency(&db, &body.from_task, &task_id).map_err(ApiError::from)?;
    let _ = promote_ready_tasks(&db).map_err(ApiError::from)?;
    Ok(Json(json!({"removed": removed})))
}

pub async fn update_task_handler(
    State(db): State<AppState>,
    Path(task_id): Path<String>,
    Json(body): Json<UpdateTaskRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let kind = match body.kind {
        Some(ref k) => Some(
            TaskKind::from_str(k)
                .map_err(|_| ApiError::bad_request(format!("invalid task kind: {k}")))?,
        ),
        None => None,
    };

    let task = update_task(
        &db,
        &task_id,
        body.title,
        body.description,
        kind,
        body.priority,
        body.metadata,
    )
    .map_err(ApiError::from)?;

    emit_event(
        &db,
        Some(&task.id),
        Some(&task.project_id),
        task.agent_id.as_deref(),
        EventType::TaskCreated,
        Some(json!({"updated": true})),
    )
    .map_err(ApiError::from)?;

    Ok(Json(task))
}

pub async fn project_overview_handler(
    State(db): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let project = ensure_project_exists(&db, &project_id)?;
    let tasks = list_tasks(
        &db,
        TaskListFilters {
            project_id: Some(project_id.clone()),
            ..Default::default()
        },
    )
    .map_err(ApiError::from)?;

    let edges = {
        let conn = db.lock().map_err(ApiError::from)?;
        let mut stmt = conn
            .prepare(
                "SELECT d.from_task, d.to_task, d.kind FROM dependencies d JOIN tasks t ON t.id = d.to_task WHERE t.project_id = ?1 ORDER BY d.id ASC",
            )
            .map_err(|e| ApiError::from(anyhow::Error::from(e)))?;
        let mut rows = stmt
            .query(params![project_id])
            .map_err(|e| ApiError::from(anyhow::Error::from(e)))?;
        let mut edges = Vec::new();
        while let Some(row) = rows
            .next()
            .map_err(|e| ApiError::from(anyhow::Error::from(e)))?
        {
            edges.push(OverviewEdge {
                from: row
                    .get::<_, String>(0)
                    .map_err(|e| ApiError::from(anyhow::Error::from(e)))?,
                to: row
                    .get::<_, String>(1)
                    .map_err(|e| ApiError::from(anyhow::Error::from(e)))?,
                kind: row
                    .get::<_, String>(2)
                    .map_err(|e| ApiError::from(anyhow::Error::from(e)))?,
            });
        }
        edges
    };

    let mut summary = OverviewSummary {
        total: tasks.len(),
        pending: 0,
        ready: 0,
        claimed: 0,
        running: 0,
        done: 0,
        failed: 0,
        cancelled: 0,
        progress_percent: 0.0,
    };
    let mut ready_task_ids = Vec::new();
    let overview_tasks: Vec<OverviewTask> = tasks
        .iter()
        .map(|t| {
            match t.status {
                TaskStatus::Pending => summary.pending += 1,
                TaskStatus::Ready => {
                    summary.ready += 1;
                    ready_task_ids.push(t.id.clone());
                }
                TaskStatus::Claimed => summary.claimed += 1,
                TaskStatus::Running => summary.running += 1,
                TaskStatus::Done | TaskStatus::DonePartial => summary.done += 1,
                TaskStatus::Failed => summary.failed += 1,
                TaskStatus::Cancelled => summary.cancelled += 1,
            }
            OverviewTask {
                id: t.id.clone(),
                title: t.title.clone(),
                status: t.status.clone(),
                kind: t.kind.clone(),
                priority: t.priority,
                agent_id: t.agent_id.clone(),
                parent_task_id: t.parent_task_id.clone(),
                is_composite: t.is_composite,
            }
        })
        .collect();

    summary.progress_percent = if summary.total == 0 {
        0.0
    } else {
        (summary.done as f64 / summary.total as f64) * 100.0
    };

    Ok(Json(OverviewResponse {
        project,
        summary,
        ready_task_ids,
        tasks: overview_tasks,
        edges,
    }))
}

pub async fn decompose_task_handler(
    State(db): State<AppState>,
    Path(task_id): Path<String>,
    Json(body): Json<DecomposeRequest>,
) -> Result<impl IntoResponse, ApiError> {
    if body.subtasks.is_empty() {
        return Err(ApiError::bad_request("subtasks list cannot be empty"));
    }

    let parent = get_task(&db, &task_id).map_err(ApiError::from)?;

    {
        let conn = db.lock().map_err(ApiError::from)?;
        conn.execute(
            "UPDATE tasks SET is_composite = 1, updated_at = datetime('now') WHERE id = ?1",
            params![task_id],
        )
        .map_err(|e| ApiError::from(anyhow::Error::from(e)))?;
    }

    let mut title_to_id: HashMap<String, String> = HashMap::new();
    let mut created_ids = Vec::new();
    let now = Utc::now().naive_utc();

    for sub in &body.subtasks {
        let kind = parse_task_kind(sub.kind.clone())?.unwrap_or(TaskKind::Generic);
        let has_deps = sub.deps_on.as_ref().map(|d| !d.is_empty()).unwrap_or(false);
        let status = if has_deps {
            TaskStatus::Pending
        } else {
            TaskStatus::Ready
        };

        let task = Task {
            id: generate_id("task"),
            project_id: parent.project_id.clone(),
            parent_task_id: Some(task_id.clone()),
            is_composite: false,
            title: sub.title.clone(),
            description: sub.description.clone(),
            status,
            kind,
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
        let created = create_task(&db, &task, &[]).map_err(ApiError::from)?;
        title_to_id.insert(sub.title.clone(), created.id.clone());
        created_ids.push(created.id);
    }

    for sub in &body.subtasks {
        if let Some(deps_on) = &sub.deps_on {
            let to_id = title_to_id
                .get(&sub.title)
                .ok_or_else(|| ApiError::internal("subtask title not found"))?;
            for dep_title in deps_on {
                let from_id = title_to_id.get(dep_title).ok_or_else(|| {
                    ApiError::bad_request(format!(
                        "deps_on references unknown subtask: {dep_title}"
                    ))
                })?;
                add_dependency(
                    &db,
                    from_id,
                    to_id,
                    DependencyKind::FeedsInto,
                    DependencyCondition::All,
                    None,
                )
                .map_err(ApiError::from)?;
            }
        }
    }
    let _ = promote_ready_tasks(&db).map_err(ApiError::from)?;

    emit_event(
        &db,
        Some(&task_id),
        Some(&parent.project_id),
        None,
        EventType::TaskCreated,
        Some(json!({"decomposed_into": created_ids.len()})),
    )
    .map_err(ApiError::from)?;

    Ok((
        StatusCode::CREATED,
        Json(DecomposeResponse {
            parent_task_id: task_id,
            subtask_ids: created_ids,
        }),
    ))
}

pub async fn replan_task_handler(
    State(db): State<AppState>,
    Path(task_id): Path<String>,
    Json(body): Json<DecomposeRequest>,
) -> Result<impl IntoResponse, ApiError> {
    {
        let conn = db.lock().map_err(ApiError::from)?;
        conn.execute(
            "UPDATE tasks SET status = 'cancelled', updated_at = datetime('now') WHERE parent_task_id = ?1 AND status NOT IN ('done', 'done_partial', 'running')",
            params![task_id.clone()],
        )
        .map_err(|e| ApiError::from(anyhow::Error::from(e)))?;
    }
    decompose_task_handler(State(db), Path(task_id), Json(body)).await
}

pub async fn insert_task_handler(
    State(db): State<AppState>,
    Json(body): Json<InsertTaskRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let before_snapshot = snapshot_task_statuses(&db, &body.project).map_err(ApiError::from)?;
    let task = insert_task_between(
        &db,
        &body.project,
        &body.after_task,
        body.before_task.as_deref(),
        &body.title,
        body.description,
    )
    .map_err(ApiError::from)?;
    let after_snapshot = snapshot_task_statuses(&db, &body.project).map_err(ApiError::from)?;
    let effect = compute_effects(&db, &body.project, &before_snapshot, &after_snapshot)
        .map_err(ApiError::from)?;
    Ok(Json(json!({
        "id": task.id,
        "title": task.title,
        "status": task.status,
        "effect": effect,
        "project_state": project_state(&db, &body.project).map_err(ApiError::from)?,
    })))
}

pub async fn amend_task_handler(
    State(db): State<AppState>,
    Path(task_id): Path<String>,
    Json(body): Json<AmendTaskRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let task = amend_task_description(&db, &task_id, &body.prepend).map_err(ApiError::from)?;
    Ok(Json(serde_json::to_value(task).map_err(|e| ApiError::internal(e.to_string()))?))
}

pub async fn pivot_task_handler(
    State(db): State<AppState>,
    Path(task_id): Path<String>,
    Json(body): Json<PivotTaskRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let parent = get_task(&db, &task_id).map_err(ApiError::from)?;
    let before_snapshot =
        snapshot_task_statuses(&db, &parent.project_id).map_err(ApiError::from)?;
    let result = pivot_subtree(
        &db,
        &task_id,
        body.keep_done.unwrap_or(false),
        body.subtasks,
    )
    .map_err(ApiError::from)?;
    let after_snapshot =
        snapshot_task_statuses(&db, &parent.project_id).map_err(ApiError::from)?;
    let effect = compute_effects(&db, &parent.project_id, &before_snapshot, &after_snapshot)
        .map_err(ApiError::from)?;
    Ok(Json(json!({
        "kept": result.kept,
        "cancelled": result.cancelled,
        "created": result.created,
        "effect": effect,
        "project_state": project_state(&db, &parent.project_id).map_err(ApiError::from)?,
    })))
}

pub async fn split_task_handler(
    State(db): State<AppState>,
    Path(task_id): Path<String>,
    Json(body): Json<SplitTaskRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let parent = get_task(&db, &task_id).map_err(ApiError::from)?;
    let before_snapshot =
        snapshot_task_statuses(&db, &parent.project_id).map_err(ApiError::from)?;
    let result = split_task(&db, &task_id, body.parts).map_err(ApiError::from)?;
    let after_snapshot =
        snapshot_task_statuses(&db, &parent.project_id).map_err(ApiError::from)?;
    let effect = compute_effects(&db, &parent.project_id, &before_snapshot, &after_snapshot)
        .map_err(ApiError::from)?;
    Ok(Json(json!({
        "parent_task_id": result.parent_task_id,
        "created": result.created,
        "done": result.done,
        "title_to_id": result.title_to_id,
        "effect": effect,
        "project_state": project_state(&db, &parent.project_id).map_err(ApiError::from)?,
    })))
}

pub async fn ahead_handler(
    State(db): State<AppState>,
    Query(query): Query<AheadQuery>,
) -> Result<impl IntoResponse, ApiError> {
    let lookahead = get_lookahead(&db, &query.project, query.depth.unwrap_or(2)).map_err(ApiError::from)?;
    Ok(Json(serde_json::to_value(lookahead).map_err(|e| ApiError::internal(e.to_string()))?))
}

pub async fn what_if_handler(
    State(db): State<AppState>,
    Json(body): Json<WhatIfRequest>,
) -> Result<impl IntoResponse, ApiError> {
    match body.mutation_type.as_str() {
        "cancel" => {
            let task_id = body
                .task_id
                .ok_or_else(|| ApiError::bad_request("task_id is required for cancel"))?;
            let task = get_task(&db, &task_id).map_err(ApiError::from)?;
            let before_snapshot =
                snapshot_task_statuses(&db, &task.project_id).map_err(ApiError::from)?;
            {
                let mut conn = db.lock().map_err(ApiError::from)?;
                let tx = conn
                    .transaction()
                    .map_err(|e| ApiError::from(anyhow::Error::from(e)))?;
                tx.execute(
                    "UPDATE tasks SET status = 'cancelled', updated_at = datetime('now') WHERE id = ?1 AND status NOT IN ('done', 'done_partial')",
                    params![task_id],
                )
                .map_err(|e| ApiError::from(anyhow::Error::from(e)))?;
                tx.rollback()
                    .map_err(|e| ApiError::from(anyhow::Error::from(e)))?;
            }
            let mut after_snapshot = before_snapshot.clone();
            if let Some(status) = after_snapshot.get_mut(&task.id) {
                if !matches!(status, TaskStatus::Done | TaskStatus::DonePartial) {
                    *status = TaskStatus::Cancelled;
                }
            }
            let effect = compute_effects(&db, &task.project_id, &before_snapshot, &after_snapshot)
                .map_err(ApiError::from)?;
            Ok(Json(json!({
                "action": "cancel",
                "effect": effect,
                "project_state": project_state(&db, &task.project_id).map_err(ApiError::from)?,
            })))
        }
        "insert" => {
            let project_id = body
                .project
                .ok_or_else(|| ApiError::bad_request("project is required for insert"))?;
            let after_task = body
                .after_task
                .ok_or_else(|| ApiError::bad_request("after_task is required for insert"))?;
            let title = body
                .title
                .ok_or_else(|| ApiError::bad_request("title is required for insert"))?;
            let before_snapshot = snapshot_task_statuses(&db, &project_id).map_err(ApiError::from)?;
            {
                let mut conn = db.lock().map_err(ApiError::from)?;
                let tx = conn
                    .transaction()
                    .map_err(|e| ApiError::from(anyhow::Error::from(e)))?;
                tx.execute(
                    "INSERT INTO tasks (id, project_id, title, status, kind) VALUES ('t-whatif-insert', ?1, ?2, 'pending', 'generic')",
                    params![project_id, title],
                )
                .map_err(|e| ApiError::from(anyhow::Error::from(e)))?;
                tx.execute(
                    "INSERT INTO dependencies(from_task, to_task, kind, condition, metadata) VALUES (?1, 't-whatif-insert', 'feeds_into', 'all', NULL)",
                    params![after_task],
                )
                .map_err(|e| ApiError::from(anyhow::Error::from(e)))?;
                if let Some(before_task) = body.before_task {
                    tx.execute(
                        "DELETE FROM dependencies WHERE from_task = ?1 AND to_task = ?2",
                        params![after_task, before_task],
                    )
                    .map_err(|e| ApiError::from(anyhow::Error::from(e)))?;
                    tx.execute(
                        "INSERT INTO dependencies(from_task, to_task, kind, condition, metadata) VALUES ('t-whatif-insert', ?1, 'feeds_into', 'all', NULL)",
                        params![before_task],
                    )
                    .map_err(|e| ApiError::from(anyhow::Error::from(e)))?;
                }
                tx.rollback()
                    .map_err(|e| ApiError::from(anyhow::Error::from(e)))?;
            }
            let mut after_snapshot = before_snapshot.clone();
            after_snapshot.insert("t-whatif-insert".to_string(), TaskStatus::Pending);
            let effect = compute_effects(&db, &project_id, &before_snapshot, &after_snapshot)
                .map_err(ApiError::from)?;
            Ok(Json(json!({
                "action": "insert",
                "effect": effect,
                "project_state": project_state(&db, &project_id).map_err(ApiError::from)?,
            })))
        }
        _ => Err(ApiError::bad_request(
            "mutation_type must be one of: cancel, insert",
        )),
    }
}

pub fn parse_event_stream_query(query: &EventStreamQuery) -> Result<(Option<String>, Option<EventType>), ApiError> {
    let event_type = parse_event_type(query.event_type.clone())?;
    Ok((query.project_id.clone(), event_type))
}

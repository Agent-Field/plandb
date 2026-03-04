use anyhow::{anyhow, Result};
use chrono::Utc;
use serde::Deserialize;
use serde_json::{json, Value};
use std::str::FromStr;

use crate::db::{
    add_dependency, add_note, add_task_files, amend_task_description, check_file_conflicts,
    claim_next_task, claim_task, complete_task, compute_effects, create_artifact, create_project,
    create_task, fail_task, get_artifact, get_downstream_tasks, get_handoff_context, get_lookahead,
    get_project, get_task, get_upstream_artifacts, insert_task_between, list_artifacts,
    list_dependencies, list_notes, list_tasks, pause_task, pivot_subtree, project_state,
    promote_ready_tasks, remove_dependency, run_sweep, snapshot_task_statuses, split_task,
    start_task, update_task, Database, NewSubtask, SplitPart, TaskListFilters,
};
use crate::models::{
    generate_id, Artifact, DependencyCondition, DependencyKind, RetryBackoff, Task, TaskKind,
    TaskStatus,
};

pub type ToolHandlerResult = Result<Value>;

#[derive(Debug, Deserialize)]
struct ToolCallParams {
    name: String,
    #[serde(default)]
    arguments: Value,
}

#[derive(Debug, Deserialize)]
struct ProjectCreateArgs {
    name: String,
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TaskDepArg {
    from: String,
    kind: String,
}

#[derive(Debug, Deserialize)]
struct TaskCreateArgs {
    project_id: String,
    title: String,
    description: Option<String>,
    kind: Option<String>,
    priority: Option<i32>,
    parent_task_id: Option<String>,
    deps: Option<Vec<TaskDepArg>>,
    tags: Option<Vec<String>>,
    max_retries: Option<i32>,
    timeout_seconds: Option<i64>,
    requires_approval: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct BatchTaskInput {
    id: Option<String>,
    title: String,
    kind: Option<String>,
    priority: Option<i32>,
    deps: Option<Vec<TaskDepArg>>,
    tags: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct TaskCreateBatchArgs {
    project_id: String,
    tasks: Vec<BatchTaskInput>,
}

#[derive(Debug, Deserialize)]
struct TaskIdArgs {
    task_id: String,
}

#[derive(Debug, Deserialize)]
struct TaskClaimArgs {
    task_id: String,
    agent_id: String,
}

#[derive(Debug, Deserialize)]
struct TaskDoneArgs {
    task_id: String,
    result: Option<Value>,
    next: Option<bool>,
    agent_id: Option<String>,
    files: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct TaskFailArgs {
    task_id: String,
    error: String,
}

#[derive(Debug, Deserialize)]
struct TaskListArgs {
    project_id: String,
    status: Option<String>,
    kind: Option<String>,
    agent_id: Option<String>,
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct TaskNextArgs {
    project_id: String,
    agent_id: String,
    claim: Option<bool>,
    start: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct TaskGoArgs {
    agent_id: String,
    project_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TaskNoteArgs {
    task_id: String,
    content: String,
    agent_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TaskNotesArgs {
    task_id: String,
}

#[derive(Debug, Deserialize)]
struct TaskPauseArgs {
    task_id: String,
    progress: Option<i32>,
    note: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StatusArgs {
    project_id: String,
    detail_level: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TaskReplanArgs {
    task_id: String,
    subtasks: Vec<DecomposeSubtask>,
}

#[derive(Debug, Deserialize)]
struct WhatIfArgs {
    mutation_type: String,
    task_id: Option<String>,
    after_task: Option<String>,
    before_task: Option<String>,
    title: Option<String>,
    project: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TaskInsertArgs {
    after_task: String,
    before_task: Option<String>,
    title: String,
    description: Option<String>,
    project: String,
}

#[derive(Debug, Deserialize)]
struct AheadArgs {
    project: String,
    depth: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct TaskAmendArgs {
    task_id: String,
    prepend: String,
}

#[derive(Debug, Deserialize)]
struct TaskPivotArgs {
    parent_id: String,
    keep_done: Option<bool>,
    subtasks: Vec<NewSubtask>,
}

#[derive(Debug, Deserialize)]
struct TaskSplitArgs {
    task_id: String,
    parts: Vec<SplitPart>,
}

#[derive(Debug, Deserialize)]
struct ProjectIdArgs {
    project_id: String,
}

#[derive(Debug, Deserialize)]
struct ArtifactWriteArgs {
    task_id: String,
    name: String,
    content: Option<String>,
    path: Option<String>,
    kind: Option<String>,
    mime_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ArtifactReadArgs {
    task_id: String,
    name: String,
}

#[derive(Debug, Deserialize)]
struct DependencyAddArgs {
    from_task: String,
    to_task: String,
    kind: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DependencyRemoveArgs {
    from_task: String,
    to_task: String,
}

#[derive(Debug, Deserialize)]
struct TaskUpdateArgs {
    task_id: String,
    title: Option<String>,
    description: Option<String>,
    kind: Option<String>,
    priority: Option<i32>,
    metadata: Option<Value>,
}

#[derive(Debug, Deserialize, serde::Serialize, Clone)]
struct DecomposeSubtask {
    title: String,
    kind: Option<String>,
    description: Option<String>,
    deps_on: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct TaskDecomposeArgs {
    task_id: String,
    subtasks: Vec<DecomposeSubtask>,
}

pub fn tool_schemas() -> Vec<Value> {
    vec![
        json!({
            "name": "planq_project_create",
            "description": "Create project",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "description": { "type": "string" }
                },
                "required": ["name"]
            }
        }),
        json!({
            "name": "planq_task_create",
            "description": "Create task with optional deps",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_id": { "type": "string" },
                    "title": { "type": "string" },
                    "description": { "type": "string" },
                    "kind": { "type": "string" },
                    "priority": { "type": "integer" },
                    "parent_task_id": { "type": "string" },
                    "deps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "from": { "type": "string" },
                                "kind": { "type": "string" }
                            },
                            "required": ["from", "kind"]
                        }
                    },
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "max_retries": { "type": "integer" },
                    "timeout_seconds": { "type": "integer" },
                    "requires_approval": { "type": "boolean" }
                },
                "required": ["project_id", "title"]
            }
        }),
        json!({
            "name": "planq_task_create_batch",
            "description": "Create task batch",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_id": { "type": "string" },
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string" },
                                "title": { "type": "string" },
                                "kind": { "type": "string" },
                                "priority": { "type": "integer" },
                                "deps": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "from": { "type": "string" },
                                            "kind": { "type": "string" }
                                        },
                                        "required": ["from", "kind"]
                                    }
                                },
                                "tags": {
                                    "type": "array",
                                    "items": { "type": "string" }
                                }
                            },
                            "required": ["title"]
                        }
                    }
                },
                "required": ["project_id", "tasks"]
            }
        }),
        json!({
            "name": "planq_task_get_context",
            "description": "Get task context: project, artifacts, deps",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": { "type": "string" }
                },
                "required": ["task_id"]
            }
        }),
        json!({
            "name": "planq_task_claim",
            "description": "Claim ready task",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": { "type": "string" },
                    "agent_id": { "type": "string" }
                },
                "required": ["task_id", "agent_id"]
            }
        }),
        json!({
            "name": "planq_task_start",
            "description": "Mark task running",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": { "type": "string" }
                },
                "required": ["task_id"]
            }
        }),
        json!({
            "name": "planq_task_done",
            "description": "Mark task done",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": { "type": "string" },
                    "result": { "type": "object" },
                    "next": { "type": "boolean" },
                    "agent_id": { "type": "string" },
                    "files": {
                        "type": "array",
                        "items": { "type": "string" }
                    }
                },
                "required": ["task_id"]
            }
        }),
        json!({
            "name": "planq_go",
            "description": "Get next task, claim+start it, return context",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent_id": { "type": "string" },
                    "project_id": { "type": "string" }
                },
                "required": ["agent_id"]
            }
        }),
        json!({
            "name": "planq_task_fail",
            "description": "Mark task failed",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": { "type": "string" },
                    "error": { "type": "string" }
                },
                "required": ["task_id", "error"]
            }
        }),
        json!({
            "name": "planq_task_list",
            "description": "List tasks",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_id": { "type": "string" },
                    "status": { "type": "string" },
                    "kind": { "type": "string" },
                    "agent_id": { "type": "string" },
                    "limit": { "type": "integer" }
                },
                "required": ["project_id"]
            }
        }),
        json!({
            "name": "planq_task_next",
            "description": "Get next ready task",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_id": { "type": "string" },
                    "agent_id": { "type": "string" },
                    "claim": { "type": "boolean" },
                    "start": { "type": "boolean" }
                },
                "required": ["project_id", "agent_id"]
            }
        }),
        json!({
            "name": "planq_project_status",
            "description": "Get project status",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_id": { "type": "string" }
                },
                "required": ["project_id"]
            }
        }),
        json!({
            "name": "planq_project_dag",
            "description": "Get project DAG",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_id": { "type": "string" }
                },
                "required": ["project_id"]
            }
        }),
        json!({
            "name": "planq_artifact_write",
            "description": "Write artifact",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": { "type": "string" },
                    "name": { "type": "string" },
                    "content": { "type": "string" },
                    "path": { "type": "string" },
                    "kind": { "type": "string" },
                    "mime_type": { "type": "string" }
                },
                "required": ["task_id", "name"]
            }
        }),
        json!({
            "name": "planq_artifact_read",
            "description": "Read artifact",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": { "type": "string" },
                    "name": { "type": "string" }
                },
                "required": ["task_id", "name"]
            }
        }),
        json!({
            "name": "planq_dependency_add",
            "description": "Add dependency",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "from_task": { "type": "string" },
                    "to_task": { "type": "string" },
                    "kind": { "type": "string" }
                },
                "required": ["from_task", "to_task"]
            }
        }),
        json!({
            "name": "planq_dependency_remove",
            "description": "Remove dependency",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "from_task": { "type": "string" },
                    "to_task": { "type": "string" }
                },
                "required": ["from_task", "to_task"]
            }
        }),
        json!({
            "name": "planq_task_update",
            "description": "Update task fields",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": { "type": "string" },
                    "title": { "type": "string" },
                    "description": { "type": "string" },
                    "kind": { "type": "string" },
                    "priority": { "type": "integer" },
                    "metadata": { "type": "object" }
                },
                "required": ["task_id"]
            }
        }),
        json!({
            "name": "planq_project_overview",
            "description": "Project overview: tasks, deps, counts, ready IDs",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_id": { "type": "string" }
                },
                "required": ["project_id"]
            }
        }),
        json!({
            "name": "planq_task_decompose",
            "description": "Decompose task into subtasks",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": { "type": "string" },
                    "subtasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": { "type": "string" },
                                "kind": { "type": "string" },
                                "description": { "type": "string" },
                                "deps_on": {
                                    "type": "array",
                                    "items": { "type": "string" }
                                }
                            },
                            "required": ["title"]
                        }
                    }
                },
                "required": ["task_id", "subtasks"]
            }
        }),
        json!({
            "name": "planq_task_note",
            "description": "Add note to task",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": { "type": "string" },
                    "content": { "type": "string" },
                    "agent_id": { "type": "string" }
                },
                "required": ["task_id", "content"]
            }
        }),
        json!({
            "name": "planq_task_notes",
            "description": "List notes for task",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": { "type": "string" }
                },
                "required": ["task_id"]
            }
        }),
        json!({
            "name": "planq_status",
            "description": "Project status with summary/detail/full levels",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_id": { "type": "string" },
                    "detail_level": { "type": "string" }
                },
                "required": ["project_id"]
            }
        }),
        json!({
            "name": "planq_task_pause",
            "description": "Pause running/claimed task and return it to ready",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": { "type": "string" },
                    "progress": { "type": "integer" },
                    "note": { "type": "string" }
                },
                "required": ["task_id"]
            }
        }),
        json!({
            "name": "planq_task_replan",
            "description": "Cancel remaining subtasks and create replacement subtasks",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": { "type": "string" },
                    "subtasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": { "type": "string" },
                                "kind": { "type": "string" },
                                "description": { "type": "string" },
                                "deps_on": {
                                    "type": "array",
                                    "items": { "type": "string" }
                                }
                            },
                            "required": ["title"]
                        }
                    }
                },
                "required": ["task_id", "subtasks"]
            }
        }),
        json!({
            "name": "planq_what_if",
            "description": "Dry-run mutation and return effect analysis",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "mutation_type": { "type": "string" },
                    "task_id": { "type": "string" },
                    "after_task": { "type": "string" },
                    "before_task": { "type": "string" },
                    "title": { "type": "string" },
                    "project": { "type": "string" }
                },
                "required": ["mutation_type"]
            }
        }),
        json!({
            "name": "planq_task_insert",
            "description": "Insert task between existing tasks",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "after_task": { "type": "string" },
                    "before_task": { "type": "string" },
                    "title": { "type": "string" },
                    "description": { "type": "string" },
                    "project": { "type": "string" }
                },
                "required": ["after_task", "title", "project"]
            }
        }),
        json!({
            "name": "planq_ahead",
            "description": "Get running-task lookahead buffer",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project": { "type": "string" },
                    "depth": { "type": "integer" }
                },
                "required": ["project"]
            }
        }),
        json!({
            "name": "planq_task_amend",
            "description": "Prepend context to a future task",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": { "type": "string" },
                    "prepend": { "type": "string" }
                },
                "required": ["task_id", "prepend"]
            }
        }),
        json!({
            "name": "planq_task_pivot",
            "description": "Replace a parent task subtree",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "parent_id": { "type": "string" },
                    "keep_done": { "type": "boolean" },
                    "subtasks": { "type": "array" }
                },
                "required": ["parent_id", "subtasks"]
            }
        }),
        json!({
            "name": "planq_task_split",
            "description": "Split task into executable parts",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": { "type": "string" },
                    "parts": { "type": "array" }
                },
                "required": ["task_id", "parts"]
            }
        }),
    ]
}

pub fn parse_tool_call(params: Option<Value>) -> Result<(String, Value)> {
    let parsed: ToolCallParams = serde_json::from_value(params.unwrap_or_else(|| json!({})))?;
    Ok((parsed.name, parsed.arguments))
}

pub fn call_tool(db: &Database, tool_name: &str, args: Value) -> ToolHandlerResult {
    match tool_name {
        "planq_project_create" => planq_project_create(db, args),
        "planq_task_create" => planq_task_create(db, args),
        "planq_task_create_batch" => planq_task_create_batch(db, args),
        "planq_task_get_context" => planq_task_get_context(db, args),
        "planq_task_claim" => planq_task_claim(db, args),
        "planq_task_start" => planq_task_start(db, args),
        "planq_task_done" => planq_task_done(db, args),
        "planq_go" => planq_go(db, args),
        "planq_task_fail" => planq_task_fail(db, args),
        "planq_task_list" => planq_task_list(db, args),
        "planq_task_next" => planq_task_next(db, args),
        "planq_project_status" => planq_project_status(db, args),
        "planq_project_dag" => planq_project_dag(db, args),
        "planq_artifact_write" => planq_artifact_write(db, args),
        "planq_artifact_read" => planq_artifact_read(db, args),
        "planq_dependency_add" => planq_dependency_add(db, args),
        "planq_dependency_remove" => planq_dependency_remove(db, args),
        "planq_task_update" => planq_task_update_tool(db, args),
        "planq_project_overview" => planq_project_overview(db, args),
        "planq_task_decompose" => planq_task_decompose(db, args),
        "planq_task_note" => planq_task_note(db, args),
        "planq_task_notes" => planq_task_notes(db, args),
        "planq_status" => planq_status(db, args),
        "planq_task_pause" => planq_task_pause(db, args),
        "planq_task_replan" => planq_task_replan(db, args),
        "planq_what_if" => planq_what_if(db, args),
        "planq_task_insert" => planq_task_insert(db, args),
        "planq_ahead" => planq_ahead(db, args),
        "planq_task_amend" => planq_task_amend(db, args),
        "planq_task_pivot" => planq_task_pivot(db, args),
        "planq_task_split" => planq_task_split(db, args),
        _ => Err(anyhow!("unknown tool: {tool_name}")),
    }
}

pub fn wrap_tool_result(result: Value) -> Value {
    let text = serde_json::to_string(&result).unwrap_or_else(|_| "{}".to_string());
    json!({
        "content": [
            {
                "type": "text",
                "text": text
            }
        ]
    })
}

fn parse_task_kind(value: Option<String>) -> Result<TaskKind> {
    match value {
        Some(raw) => TaskKind::from_str(&raw).map_err(|e| anyhow!(e)),
        None => Ok(TaskKind::Generic),
    }
}

fn parse_task_kind_opt(value: Option<String>) -> Result<Option<TaskKind>> {
    match value {
        Some(raw) => TaskKind::from_str(&raw).map(Some).map_err(|e| anyhow!(e)),
        None => Ok(None),
    }
}

fn parse_dep_kind(value: &str) -> Result<DependencyKind> {
    DependencyKind::from_str(value).map_err(|e| anyhow!(e))
}

fn parse_task_status(value: Option<String>) -> Result<Option<TaskStatus>> {
    match value {
        Some(raw) => TaskStatus::from_str(&raw).map(Some).map_err(|e| anyhow!(e)),
        None => Ok(None),
    }
}

fn compact_task_json(task: &Task) -> Value {
    json!({
        "id": task.id,
        "title": task.title,
        "status": task.status,
        "kind": task.kind,
        "priority": task.priority,
        "agent_id": task.agent_id,
    })
}

fn minimal_task_json(task: &Task) -> Value {
    json!({
        "id": task.id,
        "status": task.status,
    })
}

fn make_task(project_id: &str, title: &str, description: Option<String>) -> Task {
    let now = Utc::now().naive_utc();
    Task {
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
    }
}

fn planq_project_create(db: &Database, args: Value) -> ToolHandlerResult {
    let args: ProjectCreateArgs = serde_json::from_value(args)?;
    let project = create_project(db, &args.name, args.description, None)?;
    Ok(serde_json::to_value(project)?)
}

fn planq_task_create(db: &Database, args: Value) -> ToolHandlerResult {
    let args: TaskCreateArgs = serde_json::from_value(args)?;
    let mut task = make_task(&args.project_id, &args.title, args.description);
    task.kind = parse_task_kind(args.kind)?;
    task.priority = args.priority.unwrap_or(0);
    task.parent_task_id = args.parent_task_id;
    task.max_retries = args.max_retries.unwrap_or(0);
    task.timeout_seconds = args.timeout_seconds;
    task.requires_approval = args.requires_approval.unwrap_or(false);
    let deps = args.deps.unwrap_or_default();
    if deps.is_empty() {
        task.status = TaskStatus::Ready;
    }

    let tags = args.tags.unwrap_or_default();
    let created = create_task(db, &task, &tags)?;

    for dep in deps {
        let kind = parse_dep_kind(&dep.kind)?;
        add_dependency(
            db,
            &dep.from,
            &created.id,
            kind,
            DependencyCondition::All,
            None,
        )?;
    }

    if created.status == TaskStatus::Pending {
        promote_ready_tasks(db)?;
    }

    let refreshed = get_task(db, &created.id)?;
    Ok(json!({
        "id": refreshed.id,
        "title": refreshed.title,
        "status": refreshed.status,
    }))
}

fn planq_task_create_batch(db: &Database, args: Value) -> ToolHandlerResult {
    let args: TaskCreateBatchArgs = serde_json::from_value(args)?;
    let mut created_ids = Vec::with_capacity(args.tasks.len());
    let mut created_with_deps = Vec::with_capacity(args.tasks.len());

    for input in args.tasks {
        let mut task = make_task(&args.project_id, &input.title, None);
        if let Some(explicit_id) = &input.id {
            task.id = explicit_id.clone();
        }
        task.kind = parse_task_kind(input.kind)?;
        task.priority = input.priority.unwrap_or(0);
        let deps = input.deps.unwrap_or_default();
        if deps.is_empty() {
            task.status = TaskStatus::Ready;
        }
        let created = create_task(db, &task, &input.tags.unwrap_or_default())?;
        created_with_deps.push((created.id.clone(), deps));
        created_ids.push(created.id);
    }

    for (to_task_id, deps) in created_with_deps {
        for dep in deps {
            add_dependency(
                db,
                &dep.from,
                &to_task_id,
                parse_dep_kind(&dep.kind)?,
                DependencyCondition::All,
                None,
            )?;
        }
    }

    promote_ready_tasks(db)?;
    Ok(json!({ "task_ids": created_ids }))
}

fn planq_task_get_context(db: &Database, args: Value) -> ToolHandlerResult {
    let args: TaskIdArgs = serde_json::from_value(args)?;
    let task = get_task(db, &args.task_id)?;
    let project = get_project(db, &task.project_id)?;

    let upstream_artifacts_raw = get_upstream_artifacts(db, &task.id)?;
    let mut upstream_artifacts = Vec::with_capacity(upstream_artifacts_raw.len());
    for artifact in upstream_artifacts_raw {
        let from_task = get_task(db, &artifact.task_id)?;
        upstream_artifacts.push(json!({
            "from_task": artifact.task_id,
            "from_title": from_task.title,
            "name": artifact.name,
            "content": artifact.content,
        }));
    }

    let downstream_ids = get_downstream_tasks(db, &task.id)?;
    let mut downstream_tasks = Vec::with_capacity(downstream_ids.len());
    for id in downstream_ids {
        let t = get_task(db, &id)?;
        downstream_tasks.push(json!({
            "id": t.id,
            "title": t.title,
            "status": t.status,
        }));
    }

    let siblings = list_tasks(
        db,
        TaskListFilters {
            project_id: Some(task.project_id.clone()),
            parent_task_id: task.parent_task_id.clone(),
            ..Default::default()
        },
    )?;
    let sibling_tasks: Vec<Value> = siblings
        .into_iter()
        .filter(|t| t.id != task.id)
        .map(|t| {
            json!({
                "id": t.id,
                "title": t.title,
                "status": t.status,
                "agent_id": t.agent_id,
            })
        })
        .collect();

    Ok(json!({
        "task": task,
        "project": project,
        "upstream_artifacts": upstream_artifacts,
        "downstream_tasks": downstream_tasks,
        "sibling_tasks": sibling_tasks,
    }))
}

fn planq_task_claim(db: &Database, args: Value) -> ToolHandlerResult {
    let args: TaskClaimArgs = serde_json::from_value(args)?;
    let task = claim_task(db, &args.task_id, &args.agent_id)?
        .ok_or_else(|| anyhow!("task {} is not ready or already claimed", args.task_id))?;
    Ok(minimal_task_json(&task))
}

fn planq_task_start(db: &Database, args: Value) -> ToolHandlerResult {
    let args: TaskIdArgs = serde_json::from_value(args)?;
    let task = start_task(db, &args.task_id)?;
    Ok(minimal_task_json(&task))
}

fn planq_task_done(db: &Database, args: Value) -> ToolHandlerResult {
    let args: TaskDoneArgs = serde_json::from_value(args)?;
    let task = complete_task(db, &args.task_id, args.result)?;
    if let Some(files) = args.files {
        let _ = add_task_files(db, &task.id, &files)?;
    }
    run_sweep(db)?;
    if args.next.unwrap_or(false) {
        let agent_id = args
            .agent_id
            .ok_or_else(|| anyhow!("agent_id is required when next=true"))?;
        let next = go_response(db, Some(task.project_id.clone()), agent_id)?;
        Ok(json!({
            "completed": minimal_task_json(&task),
            "next": next,
        }))
    } else {
        Ok(minimal_task_json(&task))
    }
}

fn go_response(db: &Database, project_id: Option<String>, agent_id: String) -> ToolHandlerResult {
    let pid = project_id.ok_or_else(|| anyhow!("project_id is required"))?;
    let claimed = claim_next_task(db, &pid, &agent_id)?;
    let task = match claimed {
        Some(t) => Some(start_task(db, &t.id)?),
        None => None,
    };

    let tasks = list_tasks(
        db,
        TaskListFilters {
            project_id: Some(pid.clone()),
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

    let mut handoff = Vec::new();
    let mut notes = Vec::new();
    let mut file_conflicts = json!([]);
    let task_json = if let Some(task) = task {
        handoff = get_handoff_context(db, &task.id)?
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
        notes = list_notes(db, &task.id)?
            .into_iter()
            .map(|n| {
                json!({
                    "content": n.content,
                    "agent_id": n.agent_id,
                    "created_at": n.created_at,
                })
            })
            .collect();
        file_conflicts = serde_json::to_value(check_file_conflicts(db, &pid, Some(&task.id))?)?;
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

fn planq_go(db: &Database, args: Value) -> ToolHandlerResult {
    let args: TaskGoArgs = serde_json::from_value(args)?;
    go_response(db, args.project_id, args.agent_id)
}

fn planq_task_fail(db: &Database, args: Value) -> ToolHandlerResult {
    let args: TaskFailArgs = serde_json::from_value(args)?;
    let task = fail_task(db, &args.task_id, &args.error)?;
    Ok(minimal_task_json(&task))
}

fn planq_task_list(db: &Database, args: Value) -> ToolHandlerResult {
    let args: TaskListArgs = serde_json::from_value(args)?;
    let mut tasks = list_tasks(
        db,
        TaskListFilters {
            project_id: Some(args.project_id),
            status: parse_task_status(args.status)?,
            kind: parse_task_kind_opt(args.kind)?,
            agent_id: args.agent_id,
            ..Default::default()
        },
    )?;

    if let Some(limit) = args.limit {
        tasks.truncate(limit);
    }

    let compact = tasks
        .into_iter()
        .map(|task| {
            json!({
                "id": task.id,
                "title": task.title,
                "status": task.status,
                "kind": task.kind,
                "agent_id": task.agent_id,
            })
        })
        .collect::<Vec<_>>();
    Ok(json!(compact))
}

fn planq_task_next(db: &Database, args: Value) -> ToolHandlerResult {
    let args: TaskNextArgs = serde_json::from_value(args)?;
    let should_start = args.start.unwrap_or(false);
    let should_claim = args.claim.unwrap_or(false) || should_start;

    let next_task = if should_claim {
        let next = claim_next_task(db, &args.project_id, &args.agent_id)?;
        match next {
            Some(task) if should_start => Some(start_task(db, &task.id)?),
            Some(task) => Some(task),
            None => None,
        }
    } else {
        let mut tasks = list_tasks(
            db,
            TaskListFilters {
                project_id: Some(args.project_id.clone()),
                status: Some(TaskStatus::Ready),
                ..Default::default()
            },
        )?;
        if tasks.is_empty() {
            None
        } else {
            Some(tasks.remove(0))
        }
    };

    let ready_tasks = list_tasks(
        db,
        TaskListFilters {
            project_id: Some(args.project_id),
            status: Some(TaskStatus::Ready),
            ..Default::default()
        },
    )?;

    match next_task {
        Some(task) => {
            let mut compact = compact_task_json(&task);
            compact["description"] = json!(task.description);
            Ok(json!({
                "task": compact,
                "remaining_ready_count": ready_tasks.len(),
            }))
        }
        None => Ok(Value::Null),
    }
}

fn planq_project_status(db: &Database, args: Value) -> ToolHandlerResult {
    let args: ProjectIdArgs = serde_json::from_value(args)?;
    let project = get_project(db, &args.project_id)?;
    let tasks = list_tasks(
        db,
        TaskListFilters {
            project_id: Some(args.project_id),
            ..Default::default()
        },
    )?;

    let total_tasks = tasks.len();
    let mut pending = 0usize;
    let mut ready = 0usize;
    let mut running = 0usize;
    let mut done = 0usize;
    let mut failed = 0usize;

    for t in &tasks {
        match t.status {
            TaskStatus::Pending => pending += 1,
            TaskStatus::Ready => ready += 1,
            TaskStatus::Running => running += 1,
            TaskStatus::Done | TaskStatus::DonePartial => done += 1,
            TaskStatus::Failed => failed += 1,
            _ => {}
        }
    }

    let progress_percent = if total_tasks == 0 {
        0.0
    } else {
        (done as f64 / total_tasks as f64) * 100.0
    };

    Ok(json!({
        "project": project,
        "total_tasks": total_tasks,
        "by_status": {
            "pending": pending,
            "ready": ready,
            "running": running,
            "done": done,
            "failed": failed,
        },
        "progress_percent": progress_percent,
    }))
}

fn planq_project_dag(db: &Database, args: Value) -> ToolHandlerResult {
    let args: ProjectIdArgs = serde_json::from_value(args)?;
    let tasks = list_tasks(
        db,
        TaskListFilters {
            project_id: Some(args.project_id.clone()),
            ..Default::default()
        },
    )?;

    let nodes: Vec<Value> = tasks
        .iter()
        .map(|t| {
            json!({
                "id": t.id,
                "title": t.title,
                "status": t.status,
                "agent_id": t.agent_id,
                "kind": t.kind,
            })
        })
        .collect();

    let conn = db.lock()?;
    let mut stmt = conn.prepare(
        "
        SELECT d.from_task, d.to_task, d.kind
        FROM dependencies d
        JOIN tasks t ON t.id = d.to_task
        WHERE t.project_id = ?1
        ORDER BY d.id ASC
        ",
    )?;
    let mut rows = stmt.query([args.project_id])?;
    let mut edges = Vec::new();
    while let Some(row) = rows.next()? {
        edges.push(json!({
            "from": row.get::<_, String>(0)?,
            "to": row.get::<_, String>(1)?,
            "kind": row.get::<_, String>(2)?,
        }));
    }

    Ok(json!({
        "nodes": nodes,
        "edges": edges,
    }))
}

fn planq_artifact_write(db: &Database, args: Value) -> ToolHandlerResult {
    let args: ArtifactWriteArgs = serde_json::from_value(args)?;
    let now = Utc::now().naive_utc();
    let size_bytes = args.content.as_ref().map(|c| c.len() as i64);
    let artifact = Artifact {
        id: generate_id("artifact"),
        task_id: args.task_id,
        name: args.name,
        kind: args.kind,
        content: args.content,
        path: args.path,
        size_bytes,
        mime_type: args.mime_type,
        metadata: None,
        created_at: now,
    };
    let created = create_artifact(db, &artifact)?;
    Ok(serde_json::to_value(created)?)
}

fn planq_artifact_read(db: &Database, args: Value) -> ToolHandlerResult {
    let args: ArtifactReadArgs = serde_json::from_value(args)?;
    let artifacts = list_artifacts(db, &args.task_id)?;
    let artifact = artifacts
        .into_iter()
        .rev()
        .find(|a| a.name == args.name)
        .ok_or_else(|| {
            anyhow!(
                "artifact '{}' not found for task {}",
                args.name,
                args.task_id
            )
        })?;
    let artifact = get_artifact(db, &artifact.id)?;
    Ok(serde_json::to_value(artifact)?)
}

fn planq_dependency_add(db: &Database, args: Value) -> ToolHandlerResult {
    let args: DependencyAddArgs = serde_json::from_value(args)?;
    let kind = match args.kind.as_deref() {
        Some(k) => parse_dep_kind(k)?,
        None => DependencyKind::FeedsInto,
    };
    let dep = add_dependency(
        db,
        &args.from_task,
        &args.to_task,
        kind,
        DependencyCondition::All,
        None,
    )?;

    {
        let to_task = get_task(db, &args.to_task)?;
        if to_task.status == TaskStatus::Ready {
            let from_task = get_task(db, &args.from_task)?;
            if from_task.status != TaskStatus::Done && from_task.status != TaskStatus::DonePartial {
                let conn = db.lock()?;
                conn.execute(
                    "UPDATE tasks SET status = 'pending', updated_at = datetime('now') WHERE id = ?1 AND status = 'ready'",
                    rusqlite::params![args.to_task],
                )?;
            }
        }
    }

    Ok(serde_json::to_value(dep)?)
}

fn planq_dependency_remove(db: &Database, args: Value) -> ToolHandlerResult {
    let args: DependencyRemoveArgs = serde_json::from_value(args)?;
    let removed = remove_dependency(db, &args.from_task, &args.to_task)?;
    promote_ready_tasks(db)?;
    Ok(json!({ "removed": removed }))
}

fn planq_task_update_tool(db: &Database, args: Value) -> ToolHandlerResult {
    let args: TaskUpdateArgs = serde_json::from_value(args)?;
    let kind = match args.kind {
        Some(ref k) => Some(parse_task_kind(Some(k.clone()))?),
        None => None,
    };
    let task = update_task(
        db,
        &args.task_id,
        args.title,
        args.description,
        kind,
        args.priority,
        args.metadata,
    )?;
    Ok(serde_json::to_value(task)?)
}

fn planq_project_overview(db: &Database, args: Value) -> ToolHandlerResult {
    let args: ProjectIdArgs = serde_json::from_value(args)?;
    let project = get_project(db, &args.project_id)?;
    let tasks = list_tasks(
        db,
        TaskListFilters {
            project_id: Some(args.project_id.clone()),
            ..Default::default()
        },
    )?;

    let mut edge_ids = std::collections::HashSet::new();
    let mut edges = Vec::new();
    for task in &tasks {
        for dep in list_dependencies(db, &task.id)? {
            if edge_ids.insert(dep.id) {
                edges.push(json!({
                    "from": dep.from_task,
                    "to": dep.to_task,
                    "kind": dep.kind,
                }));
            }
        }
    }

    let mut pending = 0usize;
    let mut ready = 0usize;
    let mut claimed = 0usize;
    let mut running = 0usize;
    let mut done = 0usize;
    let mut failed = 0usize;
    let mut cancelled = 0usize;
    let mut ready_task_ids: Vec<String> = Vec::new();

    let compact_tasks: Vec<Value> = tasks
        .iter()
        .map(|t| {
            match t.status {
                TaskStatus::Pending => pending += 1,
                TaskStatus::Ready => {
                    ready += 1;
                    ready_task_ids.push(t.id.clone());
                }
                TaskStatus::Claimed => claimed += 1,
                TaskStatus::Running => running += 1,
                TaskStatus::Done | TaskStatus::DonePartial => done += 1,
                TaskStatus::Failed => failed += 1,
                TaskStatus::Cancelled => cancelled += 1,
            }
            let mut item = compact_task_json(t);
            item["parent_task_id"] = json!(t.parent_task_id);
            if t.is_composite {
                item["is_composite"] = json!(true);
            }
            item
        })
        .collect();

    let total = tasks.len();
    let progress_percent = if total == 0 {
        0.0
    } else {
        (done as f64 / total as f64) * 100.0
    };

    Ok(json!({
        "project": project,
        "summary": {
            "total": total,
            "pending": pending,
            "ready": ready,
            "claimed": claimed,
            "running": running,
            "done": done,
            "failed": failed,
            "cancelled": cancelled,
            "progress_percent": progress_percent,
        },
        "ready_task_ids": ready_task_ids,
        "tasks": compact_tasks,
        "edges": edges,
    }))
}

fn planq_task_decompose(db: &Database, args: Value) -> ToolHandlerResult {
    let args: TaskDecomposeArgs = serde_json::from_value(args)?;
    let parent = get_task(db, &args.task_id)?;

    if args.subtasks.is_empty() {
        return Err(anyhow!("subtasks list cannot be empty"));
    }

    let mut seen_titles = std::collections::HashSet::new();
    for sub in &args.subtasks {
        if !seen_titles.insert(sub.title.clone()) {
            return Err(anyhow!("duplicate subtask title: {}", sub.title));
        }
    }

    for sub in &args.subtasks {
        if let Some(deps_on) = &sub.deps_on {
            for dep_title in deps_on {
                if !seen_titles.contains(dep_title) {
                    return Err(anyhow!(
                        "deps_on references unknown subtask title: {}",
                        dep_title
                    ));
                }
            }
        }
    }

    {
        let conn = db.lock()?;
        conn.execute(
            "UPDATE tasks SET is_composite = 1, updated_at = datetime('now') WHERE id = ?1",
            rusqlite::params![args.task_id],
        )?;
    }

    let mut title_to_id: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    let mut created_ids = Vec::new();

    for sub in &args.subtasks {
        let mut task = make_task(&parent.project_id, &sub.title, sub.description.clone());
        task.parent_task_id = Some(args.task_id.clone());
        task.kind = parse_task_kind(sub.kind.clone())?;

        let has_deps = sub.deps_on.as_ref().map(|d| !d.is_empty()).unwrap_or(false);
        if !has_deps {
            task.status = TaskStatus::Ready;
        }

        let created = create_task(db, &task, &[])?;
        title_to_id.insert(sub.title.clone(), created.id.clone());
        created_ids.push(created.id);
    }

    for sub in &args.subtasks {
        if let Some(deps_on) = &sub.deps_on {
            let to_id = title_to_id
                .get(&sub.title)
                .ok_or_else(|| anyhow!("internal error: subtask title not found"))?;
            for dep_title in deps_on {
                let from_id = title_to_id.get(dep_title).ok_or_else(|| {
                    anyhow!("deps_on references unknown subtask title: {}", dep_title)
                })?;
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

    promote_ready_tasks(db)?;

    Ok(json!({
        "parent_task_id": args.task_id,
        "subtask_ids": created_ids,
        "title_to_id": title_to_id,
    }))
}

fn planq_task_note(db: &Database, args: Value) -> ToolHandlerResult {
    let args: TaskNoteArgs = serde_json::from_value(args)?;
    let note = add_note(db, &args.task_id, args.agent_id, &args.content)?;
    Ok(serde_json::to_value(note)?)
}

fn planq_task_notes(db: &Database, args: Value) -> ToolHandlerResult {
    let args: TaskNotesArgs = serde_json::from_value(args)?;
    let notes = list_notes(db, &args.task_id)?;
    Ok(serde_json::to_value(notes)?)
}

fn planq_task_pause(db: &Database, args: Value) -> ToolHandlerResult {
    let args: TaskPauseArgs = serde_json::from_value(args)?;
    let task = pause_task(db, &args.task_id, args.progress, args.note)?;
    Ok(serde_json::to_value(task)?)
}

fn planq_what_if(db: &Database, args: Value) -> ToolHandlerResult {
    let args: WhatIfArgs = serde_json::from_value(args)?;
    match args.mutation_type.as_str() {
        "cancel" => {
            let task_id = args
                .task_id
                .ok_or_else(|| anyhow!("task_id is required for cancel mutation"))?;
            let task = get_task(db, &task_id)?;
            let before_snapshot = snapshot_task_statuses(db, &task.project_id)?;
            {
                let mut conn = db.lock()?;
                let tx = conn.transaction()?;
                tx.execute(
                    "UPDATE tasks SET status = 'cancelled', updated_at = datetime('now') WHERE id = ?1 AND status NOT IN ('done', 'done_partial')",
                    rusqlite::params![task_id],
                )?;
                tx.rollback()?;
            }
            let mut after_snapshot = before_snapshot.clone();
            if let Some(status) = after_snapshot.get_mut(&task.id) {
                if !matches!(status, TaskStatus::Done | TaskStatus::DonePartial) {
                    *status = TaskStatus::Cancelled;
                }
            }
            let effect = compute_effects(db, &task.project_id, &before_snapshot, &after_snapshot)?;
            Ok(json!({
                "action": "cancel",
                "effect": effect,
                "project_state": project_state(db, &task.project_id)?,
            }))
        }
        "insert" => {
            let project_id = args
                .project
                .ok_or_else(|| anyhow!("project is required for insert mutation"))?;
            let after_task = args
                .after_task
                .ok_or_else(|| anyhow!("after_task is required for insert mutation"))?;
            let title = args
                .title
                .ok_or_else(|| anyhow!("title is required for insert mutation"))?;

            let before_snapshot = snapshot_task_statuses(db, &project_id)?;
            {
                let mut conn = db.lock()?;
                let tx = conn.transaction()?;
                tx.execute(
                    "INSERT INTO tasks (id, project_id, title, status, kind) VALUES ('t-whatif-insert', ?1, ?2, 'pending', 'generic')",
                    rusqlite::params![project_id, title],
                )?;
                tx.execute(
                    "INSERT INTO dependencies(from_task, to_task, kind, condition, metadata) VALUES (?1, 't-whatif-insert', 'feeds_into', 'all', NULL)",
                    rusqlite::params![after_task],
                )?;
                if let Some(before_task) = args.before_task {
                    tx.execute(
                        "DELETE FROM dependencies WHERE from_task = ?1 AND to_task = ?2",
                        rusqlite::params![after_task, before_task],
                    )?;
                    tx.execute(
                        "INSERT INTO dependencies(from_task, to_task, kind, condition, metadata) VALUES ('t-whatif-insert', ?1, 'feeds_into', 'all', NULL)",
                        rusqlite::params![before_task],
                    )?;
                }
                tx.rollback()?;
            }
            let mut after_snapshot = before_snapshot.clone();
            after_snapshot.insert("t-whatif-insert".to_string(), TaskStatus::Pending);
            let effect = compute_effects(db, &project_id, &before_snapshot, &after_snapshot)?;
            Ok(json!({
                "action": "insert",
                "effect": effect,
                "project_state": project_state(db, &project_id)?,
            }))
        }
        _ => Err(anyhow!("mutation_type must be one of: cancel, insert")),
    }
}

fn planq_task_insert(db: &Database, args: Value) -> ToolHandlerResult {
    let args: TaskInsertArgs = serde_json::from_value(args)?;
    let before_snapshot = snapshot_task_statuses(db, &args.project)?;
    let task = insert_task_between(
        db,
        &args.project,
        &args.after_task,
        args.before_task.as_deref(),
        &args.title,
        args.description,
    )?;
    let after_snapshot = snapshot_task_statuses(db, &args.project)?;
    let effect = compute_effects(db, &args.project, &before_snapshot, &after_snapshot)?;
    Ok(json!({
        "id": task.id,
        "title": task.title,
        "status": task.status,
        "effect": effect,
        "project_state": project_state(db, &args.project)?,
    }))
}

fn planq_ahead(db: &Database, args: Value) -> ToolHandlerResult {
    let args: AheadArgs = serde_json::from_value(args)?;
    let result = get_lookahead(db, &args.project, args.depth.unwrap_or(2))?;
    Ok(serde_json::to_value(result)?)
}

fn planq_task_amend(db: &Database, args: Value) -> ToolHandlerResult {
    let args: TaskAmendArgs = serde_json::from_value(args)?;
    let task = amend_task_description(db, &args.task_id, &args.prepend)?;
    Ok(serde_json::to_value(task)?)
}

fn planq_task_pivot(db: &Database, args: Value) -> ToolHandlerResult {
    let args: TaskPivotArgs = serde_json::from_value(args)?;
    let parent = get_task(db, &args.parent_id)?;
    let before_snapshot = snapshot_task_statuses(db, &parent.project_id)?;
    let result = pivot_subtree(
        db,
        &args.parent_id,
        args.keep_done.unwrap_or(false),
        args.subtasks,
    )?;
    let after_snapshot = snapshot_task_statuses(db, &parent.project_id)?;
    let effect = compute_effects(db, &parent.project_id, &before_snapshot, &after_snapshot)?;
    Ok(json!({
        "kept": result.kept,
        "cancelled": result.cancelled,
        "created": result.created,
        "effect": effect,
        "project_state": project_state(db, &parent.project_id)?,
    }))
}

fn planq_task_split(db: &Database, args: Value) -> ToolHandlerResult {
    let args: TaskSplitArgs = serde_json::from_value(args)?;
    let parent = get_task(db, &args.task_id)?;
    let before_snapshot = snapshot_task_statuses(db, &parent.project_id)?;
    let result = split_task(db, &args.task_id, args.parts)?;
    let after_snapshot = snapshot_task_statuses(db, &parent.project_id)?;
    let effect = compute_effects(db, &parent.project_id, &before_snapshot, &after_snapshot)?;
    Ok(json!({
        "parent_task_id": result.parent_task_id,
        "created": result.created,
        "done": result.done,
        "title_to_id": result.title_to_id,
        "effect": effect,
        "project_state": project_state(db, &parent.project_id)?,
    }))
}

fn planq_status(db: &Database, args: Value) -> ToolHandlerResult {
    let args: StatusArgs = serde_json::from_value(args)?;
    let detail_level = args.detail_level.unwrap_or_else(|| "summary".to_string());
    match detail_level.as_str() {
        "summary" => planq_project_status(db, json!({"project_id": args.project_id})),
        "detail" => {
            let tasks = list_tasks(
                db,
                TaskListFilters {
                    project_id: Some(args.project_id),
                    ..Default::default()
                },
            )?;
            Ok(serde_json::to_value(tasks)?)
        }
        "full" => planq_project_overview(db, json!({"project_id": args.project_id})),
        _ => Err(anyhow!(
            "detail_level must be one of: summary, detail, full"
        )),
    }
}

fn planq_task_replan(db: &Database, args: Value) -> ToolHandlerResult {
    let args: TaskReplanArgs = serde_json::from_value(args)?;
    let _parent = get_task(db, &args.task_id)?;
    {
        let conn = db.lock()?;
        conn.execute(
            "UPDATE tasks SET status = 'cancelled', updated_at = datetime('now') WHERE parent_task_id = ?1 AND status NOT IN ('done', 'done_partial', 'running')",
            rusqlite::params![args.task_id],
        )?;
    }
    planq_task_decompose(
        db,
        json!({
            "task_id": args.task_id,
            "subtasks": args.subtasks,
        }),
    )
}

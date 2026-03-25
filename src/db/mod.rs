pub mod artifacts;
pub mod dependencies;
pub mod effects;
pub mod events;
pub mod files;
pub mod learnings;
pub mod meta;
pub mod notes;
pub mod projects;
pub mod schema;
pub mod sweeper;
pub mod tasks;
pub mod templates;

use anyhow::{anyhow, Result};
use chrono::{NaiveDateTime, Utc};
use rusqlite::Connection;
use serde_json::Value;
use std::sync::{Arc, Mutex, MutexGuard};
use thiserror::Error;

pub use artifacts::{create_artifact, get_artifact, get_upstream_artifacts, list_artifacts};
pub use dependencies::{
    add_dependency, get_downstream_tasks, get_upstream_tasks, list_dependencies, remove_dependency,
};
pub use effects::{compute_effects, snapshot_task_statuses, MutationEffect};
pub use events::{insert_event, list_events, EventFilters};
pub use files::{add_task_files, check_file_conflicts, list_task_files, FileConflict};
pub use meta::{delete_meta, get_meta, set_meta};
pub use learnings::{
    add_context, delete_context, list_context, search_graph, ContextEntry, SearchResult,
};
pub use notes::{add_note, list_notes};
pub use projects::{
    create_project, fuzzy_find_project, get_project, list_projects, update_project_status,
};
pub use schema::init_db;
pub use sweeper::{run_sweep, SweepResult};
pub use templates::{export_graph, import_graph, GraphTemplate};
pub use tasks::{
    amend_task_description, approve_task, batch_create_tasks, cancel_task, claim_next_task,
    claim_next_task_scoped, claim_task, complete_task, count_children, create_task,
    critical_path, fail_task, find_bottlenecks, fuzzy_find_task, get_handoff_context,
    get_lookahead, get_running_task_for_agent, get_task, insert_task_between, list_subtree,
    list_tasks, pause_task, pivot_subtree, project_state, promote_ready_tasks, split_task,
    start_task, try_complete_composite_parent, update_heartbeat, update_progress, update_task,
    what_unlocks, Bottleneck, HandoffEntry, LookaheadResult, NewSubtask, PivotResult,
    ProjectState, SplitPart, SplitResult, TaskListFilters, UnlockedTask,
};

#[derive(Debug, Error)]
pub enum PlandbError {
    #[error("not found: {0}")]
    NotFound(String),
    #[error("invalid transition: {0}")]
    InvalidTransition(String),
    #[error("conflict: {0}")]
    Conflict(String),
}

#[derive(Clone)]
pub struct Database {
    conn: Arc<Mutex<Connection>>,
}

impl Database {
    pub fn from_connection(conn: Connection) -> Self {
        Self {
            conn: Arc::new(Mutex::new(conn)),
        }
    }

    pub fn lock(&self) -> Result<MutexGuard<'_, Connection>> {
        self.conn
            .lock()
            .map_err(|_| anyhow!("database connection mutex poisoned"))
    }
}

const DATETIME_FMT: &str = "%Y-%m-%d %H:%M:%S";

pub(crate) fn now_utc_naive() -> NaiveDateTime {
    Utc::now().naive_utc()
}

pub(crate) fn dt_to_sql(dt: NaiveDateTime) -> String {
    dt.format(DATETIME_FMT).to_string()
}

pub(crate) fn parse_dt(value: String) -> Result<NaiveDateTime> {
    Ok(NaiveDateTime::parse_from_str(&value, DATETIME_FMT)?)
}

pub(crate) fn json_to_sql(value: &Option<Value>) -> Result<Option<String>> {
    match value {
        Some(v) => Ok(Some(serde_json::to_string(v)?)),
        None => Ok(None),
    }
}

pub(crate) fn parse_json(value: Option<String>) -> Result<Option<Value>> {
    match value {
        Some(v) => Ok(Some(serde_json::from_str(&v)?)),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests;

use chrono::NaiveDateTime;
use rusqlite::types::{FromSql, FromSqlError, FromSqlResult, ToSqlOutput, ValueRef};
use rusqlite::ToSql;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::str::FromStr;

fn parse_err(msg: String) -> FromSqlError {
    FromSqlError::Other(Box::new(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        msg,
    )))
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    Ready,
    Claimed,
    Running,
    Done,
    DonePartial,
    Failed,
    Cancelled,
}

impl Display for TaskStatus {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Pending => "pending",
            Self::Ready => "ready",
            Self::Claimed => "claimed",
            Self::Running => "running",
            Self::Done => "done",
            Self::DonePartial => "done_partial",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        })
    }
}

impl FromStr for TaskStatus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pending" => Ok(Self::Pending),
            "ready" => Ok(Self::Ready),
            "claimed" => Ok(Self::Claimed),
            "running" => Ok(Self::Running),
            "done" => Ok(Self::Done),
            "done_partial" => Ok(Self::DonePartial),
            "failed" => Ok(Self::Failed),
            "cancelled" => Ok(Self::Cancelled),
            _ => Err(format!(
                "invalid task status: {s}. Valid: pending, ready, claimed, running, done, done_partial, failed, cancelled"
            )),
        }
    }
}

impl ToSql for TaskStatus {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::from(self.to_string()))
    }
}

impl FromSql for TaskStatus {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        let text = value.as_str()?;
        Self::from_str(text).map_err(parse_err)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    Generic,
    Code,
    Research,
    Review,
    Test,
    Shell,
}

impl Display for TaskKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Generic => "generic",
            Self::Code => "code",
            Self::Research => "research",
            Self::Review => "review",
            Self::Test => "test",
            Self::Shell => "shell",
        })
    }
}

impl FromStr for TaskKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "generic" => Ok(Self::Generic),
            "code" => Ok(Self::Code),
            "research" => Ok(Self::Research),
            "review" => Ok(Self::Review),
            "test" => Ok(Self::Test),
            "shell" => Ok(Self::Shell),
            _ => Err(format!(
                "invalid task kind: {s}. Valid: generic, code, research, review, test, shell"
            )),
        }
    }
}

impl ToSql for TaskKind {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::from(self.to_string()))
    }
}

impl FromSql for TaskKind {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        let text = value.as_str()?;
        Self::from_str(text).map_err(parse_err)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RetryBackoff {
    Exponential,
    Linear,
    Fixed,
}

impl Display for RetryBackoff {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Exponential => "exponential",
            Self::Linear => "linear",
            Self::Fixed => "fixed",
        })
    }
}

impl FromStr for RetryBackoff {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "exponential" => Ok(Self::Exponential),
            "linear" => Ok(Self::Linear),
            "fixed" => Ok(Self::Fixed),
            _ => Err(format!("invalid retry backoff: {s}")),
        }
    }
}

impl ToSql for RetryBackoff {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::from(self.to_string()))
    }
}

impl FromSql for RetryBackoff {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        let text = value.as_str()?;
        Self::from_str(text).map_err(parse_err)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Task {
    pub id: String,
    pub project_id: String,
    pub parent_task_id: Option<String>,
    pub is_composite: bool,
    pub title: String,
    pub description: Option<String>,
    pub status: TaskStatus,
    pub kind: TaskKind,
    pub priority: i32,
    pub agent_id: Option<String>,
    pub claimed_at: Option<NaiveDateTime>,
    pub started_at: Option<NaiveDateTime>,
    pub completed_at: Option<NaiveDateTime>,
    pub result: Option<serde_json::Value>,
    pub error: Option<String>,
    pub progress: Option<i32>,
    pub progress_note: Option<String>,
    pub max_retries: i32,
    pub retry_count: i32,
    pub retry_backoff: RetryBackoff,
    pub retry_delay_ms: i64,
    pub timeout_seconds: Option<i64>,
    pub heartbeat_interval: i32,
    pub last_heartbeat: Option<NaiveDateTime>,
    pub requires_approval: bool,
    pub approval_status: Option<String>,
    pub approved_by: Option<String>,
    pub approval_comment: Option<String>,
    pub pre_condition: Option<String>,
    pub post_condition: Option<String>,
    pub pre_hook: Option<String>,
    pub post_hook: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub created_at: NaiveDateTime,
    pub updated_at: NaiveDateTime,
}

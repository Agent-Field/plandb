use crate::db::{parse_dt, Database};
use crate::models::generate_id;
use anyhow::Result;
use chrono::NaiveDateTime;
use rusqlite::params;
use serde::Serialize;

/// A learning is a piece of knowledge discovered during project execution.
/// Learnings persist across sessions and are searchable via BM25.
#[derive(Debug, Clone, Serialize)]
pub struct Learning {
    pub id: String,
    pub project_id: String,
    pub task_id: Option<String>,
    pub agent_id: Option<String>,
    pub kind: LearningKind,
    pub content: String,
    pub tags: Vec<String>,
    pub created_at: NaiveDateTime,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LearningKind {
    Discovery,
    Decision,
    Pattern,
    Blocker,
    Reference,
}

impl std::fmt::Display for LearningKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LearningKind::Discovery => write!(f, "discovery"),
            LearningKind::Decision => write!(f, "decision"),
            LearningKind::Pattern => write!(f, "pattern"),
            LearningKind::Blocker => write!(f, "blocker"),
            LearningKind::Reference => write!(f, "reference"),
        }
    }
}

impl std::str::FromStr for LearningKind {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "discovery" => Ok(LearningKind::Discovery),
            "decision" => Ok(LearningKind::Decision),
            "pattern" => Ok(LearningKind::Pattern),
            "blocker" => Ok(LearningKind::Blocker),
            "reference" => Ok(LearningKind::Reference),
            _ => Err(anyhow::anyhow!(
                "unknown learning kind '{}'. Use: discovery, decision, pattern, blocker, reference",
                s
            )),
        }
    }
}

/// A search result from BM25-ranked recall.
#[derive(Debug, Clone, Serialize)]
pub struct RecallResult {
    pub source: String, // "learning" or "task"
    pub id: String,
    pub kind: String,
    pub content: String,
    pub rank: f64,
    pub task_id: Option<String>,
    pub project_id: String,
}

pub fn add_learning(
    db: &Database,
    project_id: &str,
    task_id: Option<&str>,
    agent_id: Option<&str>,
    kind: LearningKind,
    content: &str,
    tags: &[String],
) -> Result<Learning> {
    let conn = db.lock()?;
    let id = generate_id("lrn");
    let kind_str = kind.to_string();

    conn.execute(
        "INSERT INTO learnings (id, project_id, task_id, agent_id, kind, content) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![id, project_id, task_id, agent_id, kind_str, content],
    )?;

    // Insert into FTS index
    conn.execute(
        "INSERT INTO learnings_fts (rowid, content, kind) VALUES (last_insert_rowid(), ?1, ?2)",
        params![content, kind_str],
    )?;

    // Insert tags
    for tag in tags {
        conn.execute(
            "INSERT OR IGNORE INTO learning_tags (learning_id, tag) VALUES (?1, ?2)",
            params![id, tag],
        )?;
    }

    let created_at_raw: String = conn.query_row(
        "SELECT created_at FROM learnings WHERE id = ?1",
        params![id],
        |row| row.get(0),
    )?;
    let created_at = parse_dt(created_at_raw)?;

    Ok(Learning {
        id,
        project_id: project_id.to_string(),
        task_id: task_id.map(|s| s.to_string()),
        agent_id: agent_id.map(|s| s.to_string()),
        kind,
        content: content.to_string(),
        tags: tags.to_vec(),
        created_at,
    })
}

pub fn list_learnings(
    db: &Database,
    project_id: &str,
    kind_filter: Option<&str>,
    limit: usize,
) -> Result<Vec<Learning>> {
    let conn = db.lock()?;
    let mut learnings = if let Some(kind) = kind_filter {
        let mut stmt = conn.prepare(
            "SELECT id, project_id, task_id, agent_id, kind, content, created_at \
             FROM learnings WHERE project_id = ?1 AND kind = ?2 ORDER BY created_at DESC LIMIT ?3",
        )?;
        let mut rows = stmt.query(params![project_id, kind, limit as i64])?;
        let mut result = Vec::new();
        while let Some(row) = rows.next()? {
            result.push(row_to_learning(row)?);
        }
        result
    } else {
        let mut stmt = conn.prepare(
            "SELECT id, project_id, task_id, agent_id, kind, content, created_at \
             FROM learnings WHERE project_id = ?1 ORDER BY created_at DESC LIMIT ?2",
        )?;
        let mut rows = stmt.query(params![project_id, limit as i64])?;
        let mut result = Vec::new();
        while let Some(row) = rows.next()? {
            result.push(row_to_learning(row)?);
        }
        result
    };

    // Attach tags
    for learning in &mut learnings {
        let mut stmt = conn.prepare(
            "SELECT tag FROM learning_tags WHERE learning_id = ?1 ORDER BY tag",
        )?;
        let tags: Vec<String> = stmt
            .query_map(params![learning.id], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        learning.tags = tags;
    }

    Ok(learnings)
}

/// BM25-ranked search across learnings AND task descriptions/titles.
/// Returns ranked results from both sources.
pub fn recall(
    db: &Database,
    project_id: &str,
    query: &str,
    limit: usize,
) -> Result<Vec<RecallResult>> {
    let conn = db.lock()?;
    let mut results = Vec::new();

    // Search learnings via FTS5 with BM25 ranking
    {
        let mut stmt = conn.prepare(
            "SELECT l.id, l.kind, l.content, l.task_id, l.project_id, f.rank \
             FROM learnings_fts f \
             JOIN learnings l ON l.rowid = f.rowid \
             WHERE learnings_fts MATCH ?1 AND l.project_id = ?2 \
             ORDER BY f.rank \
             LIMIT ?3",
        )?;
        let mut rows = stmt.query(params![query, project_id, limit as i64])?;
        while let Some(row) = rows.next()? {
            results.push(RecallResult {
                source: "learning".to_string(),
                id: row.get(0)?,
                kind: row.get(1)?,
                content: row.get(2)?,
                task_id: row.get(3)?,
                project_id: row.get(4)?,
                rank: row.get(5)?,
            });
        }
    }

    // Also search task titles and descriptions via LIKE (simpler but still useful)
    {
        let pattern = format!("%{}%", query);
        let mut stmt = conn.prepare(
            "SELECT id, kind, title, description, project_id \
             FROM tasks \
             WHERE project_id = ?1 AND (title LIKE ?2 OR description LIKE ?2) \
             LIMIT ?3",
        )?;
        let mut rows = stmt.query(params![project_id, pattern, limit as i64])?;
        while let Some(row) = rows.next()? {
            let title: String = row.get(2)?;
            let desc: Option<String> = row.get(3)?;
            let content = if let Some(d) = desc {
                format!("{}: {}", title, d)
            } else {
                title
            };
            results.push(RecallResult {
                source: "task".to_string(),
                id: row.get(0)?,
                kind: row.get(1)?,
                content,
                task_id: Some(row.get::<_, String>(0)?),
                project_id: row.get(4)?,
                rank: -0.5, // Lower priority than FTS matches
            });
        }
    }

    // Sort by rank (BM25 ranks are negative, more negative = better match)
    results.sort_by(|a, b| a.rank.partial_cmp(&b.rank).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);

    Ok(results)
}

pub fn delete_learning(db: &Database, learning_id: &str) -> Result<usize> {
    let conn = db.lock()?;
    // Get rowid for FTS cleanup
    let rowid: Option<i64> = conn
        .query_row(
            "SELECT rowid FROM learnings WHERE id = ?1",
            params![learning_id],
            |row| row.get(0),
        )
        .ok();

    if let Some(rid) = rowid {
        conn.execute(
            "DELETE FROM learnings_fts WHERE rowid = ?1",
            params![rid],
        )?;
    }

    conn.execute(
        "DELETE FROM learning_tags WHERE learning_id = ?1",
        params![learning_id],
    )?;

    let deleted = conn.execute(
        "DELETE FROM learnings WHERE id = ?1",
        params![learning_id],
    )?;

    Ok(deleted)
}

fn row_to_learning(row: &rusqlite::Row<'_>) -> rusqlite::Result<Learning> {
    let created_at_raw: String = row.get(6)?;
    let created_at = parse_dt(created_at_raw).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(
            6,
            rusqlite::types::Type::Text,
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                e.to_string(),
            )),
        )
    })?;
    let kind_str: String = row.get(4)?;
    let kind: LearningKind = kind_str.parse().map_err(|e: anyhow::Error| {
        rusqlite::Error::FromSqlConversionFailure(
            4,
            rusqlite::types::Type::Text,
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                e.to_string(),
            )),
        )
    })?;

    Ok(Learning {
        id: row.get(0)?,
        project_id: row.get(1)?,
        task_id: row.get(2)?,
        agent_id: row.get(3)?,
        kind,
        content: row.get(5)?,
        tags: Vec::new(), // Filled in by caller
        created_at,
    })
}

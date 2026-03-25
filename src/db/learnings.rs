use crate::db::{parse_dt, Database};
use crate::models::generate_id;
use anyhow::Result;
use chrono::NaiveDateTime;
use rusqlite::params;
use serde::Serialize;

/// A context entry attached to the project graph.
/// Context persists across sessions and is searchable via BM25.
/// The `kind` field is a freeform string — agents define their own types.
#[derive(Debug, Clone, Serialize)]
pub struct ContextEntry {
    pub id: String,
    pub project_id: String,
    pub task_id: Option<String>,
    pub agent_id: Option<String>,
    pub kind: String,
    pub content: String,
    pub tags: Vec<String>,
    pub created_at: NaiveDateTime,
}

/// A search result from BM25-ranked recall.
#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub source: String, // "context" or "task"
    pub id: String,
    pub kind: String,
    pub content: String,
    pub rank: f64,
    pub task_id: Option<String>,
    pub project_id: String,
}

pub fn add_context(
    db: &Database,
    project_id: &str,
    task_id: Option<&str>,
    agent_id: Option<&str>,
    kind: &str,
    content: &str,
    tags: &[String],
) -> Result<ContextEntry> {
    let conn = db.lock()?;
    let id = generate_id("ctx");

    conn.execute(
        "INSERT INTO learnings (id, project_id, task_id, agent_id, kind, content) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![id, project_id, task_id, agent_id, kind, content],
    )?;

    // Insert into FTS index
    conn.execute(
        "INSERT INTO learnings_fts (rowid, content, kind) VALUES (last_insert_rowid(), ?1, ?2)",
        params![content, kind],
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

    Ok(ContextEntry {
        id,
        project_id: project_id.to_string(),
        task_id: task_id.map(|s| s.to_string()),
        agent_id: agent_id.map(|s| s.to_string()),
        kind: kind.to_string(),
        content: content.to_string(),
        tags: tags.to_vec(),
        created_at,
    })
}

pub fn list_context(
    db: &Database,
    project_id: &str,
    kind_filter: Option<&str>,
    limit: usize,
) -> Result<Vec<ContextEntry>> {
    let conn = db.lock()?;
    let mut entries = if let Some(kind) = kind_filter {
        let mut stmt = conn.prepare(
            "SELECT id, project_id, task_id, agent_id, kind, content, created_at \
             FROM learnings WHERE project_id = ?1 AND kind = ?2 ORDER BY created_at DESC LIMIT ?3",
        )?;
        let mut rows = stmt.query(params![project_id, kind, limit as i64])?;
        let mut result = Vec::new();
        while let Some(row) = rows.next()? {
            result.push(row_to_entry(row)?);
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
            result.push(row_to_entry(row)?);
        }
        result
    };

    // Attach tags
    for entry in &mut entries {
        let mut stmt = conn.prepare(
            "SELECT tag FROM learning_tags WHERE learning_id = ?1 ORDER BY tag",
        )?;
        let tags: Vec<String> = stmt
            .query_map(params![entry.id], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        entry.tags = tags;
    }

    Ok(entries)
}

/// BM25-ranked search across context entries AND task descriptions/titles.
pub fn search_graph(
    db: &Database,
    project_id: &str,
    query: &str,
    limit: usize,
) -> Result<Vec<SearchResult>> {
    let conn = db.lock()?;
    let mut results = Vec::new();

    // Search context via FTS5 with BM25 ranking
    // Weights: content=1.0, kind=5.0 (searching "decision" finds decision-kind entries)
    {
        let mut stmt = conn.prepare(
            "SELECT l.id, l.kind, l.content, l.task_id, l.project_id, \
                    bm25(learnings_fts, 1.0, 5.0) AS rank \
             FROM learnings_fts f \
             JOIN learnings l ON l.rowid = f.rowid \
             WHERE learnings_fts MATCH ?1 AND l.project_id = ?2 \
             ORDER BY rank \
             LIMIT ?3",
        )?;
        let mut rows = stmt.query(params![query, project_id, limit as i64])?;
        while let Some(row) = rows.next()? {
            results.push(SearchResult {
                source: "context".to_string(),
                id: row.get(0)?,
                kind: row.get(1)?,
                content: row.get(2)?,
                task_id: row.get(3)?,
                project_id: row.get(4)?,
                rank: row.get(5)?,
            });
        }
    }

    // Search task titles and descriptions via FTS5 with BM25 ranking
    // Weights: title=10.0, description=1.0 (title matches rank much higher)
    {
        let mut stmt = conn.prepare(
            "SELECT t.id, t.kind, t.title, t.description, t.project_id, \
                    bm25(tasks_fts, 10.0, 1.0) AS rank \
             FROM tasks_fts f \
             JOIN tasks t ON t.rowid = f.rowid \
             WHERE tasks_fts MATCH ?1 AND t.project_id = ?2 \
             ORDER BY rank \
             LIMIT ?3",
        )?;
        let mut rows = stmt.query(params![query, project_id, limit as i64])?;
        while let Some(row) = rows.next()? {
            let title: String = row.get(2)?;
            let desc: Option<String> = row.get(3)?;
            let content = if let Some(d) = desc {
                format!("{}: {}", title, d)
            } else {
                title
            };
            results.push(SearchResult {
                source: "task".to_string(),
                id: row.get(0)?,
                kind: row.get(1)?,
                content,
                task_id: Some(row.get::<_, String>(0)?),
                project_id: row.get(4)?,
                rank: row.get(5)?,
            });
        }
    }

    // Sort by rank (BM25 ranks are negative, more negative = better match)
    results.sort_by(|a, b| a.rank.partial_cmp(&b.rank).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);

    Ok(results)
}

pub fn delete_context(db: &Database, context_id: &str) -> Result<usize> {
    let conn = db.lock()?;
    let rowid: Option<i64> = conn
        .query_row(
            "SELECT rowid FROM learnings WHERE id = ?1",
            params![context_id],
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
        params![context_id],
    )?;

    let deleted = conn.execute(
        "DELETE FROM learnings WHERE id = ?1",
        params![context_id],
    )?;

    Ok(deleted)
}

fn row_to_entry(row: &rusqlite::Row<'_>) -> rusqlite::Result<ContextEntry> {
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

    Ok(ContextEntry {
        id: row.get(0)?,
        project_id: row.get(1)?,
        task_id: row.get(2)?,
        agent_id: row.get(3)?,
        kind: row.get(4)?,
        content: row.get(5)?,
        tags: Vec::new(), // Filled in by caller
        created_at,
    })
}

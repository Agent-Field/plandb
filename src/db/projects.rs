use crate::db::{
    dt_to_sql, json_to_sql, now_utc_naive, parse_dt, parse_json, Database, PlandbError,
};
use crate::models::{generate_id, Project, ProjectStatus};
use anyhow::Result;
use rusqlite::params;

const SELECT_PROJECTS_NAME_LIKE: &str = r#"
SELECT id, name
FROM projects
WHERE name LIKE ?1
ORDER BY created_at DESC
LIMIT 5;
"#;

const SELECT_RECENT_PROJECT_IDS: &str = r#"
SELECT id, name
FROM projects
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

const INSERT_PROJECT: &str = r#"
INSERT INTO projects (id, name, description, status, metadata, created_at, updated_at)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7);
"#;

const SELECT_PROJECT_BY_ID: &str = r#"
SELECT id, name, description, status, metadata, created_at, updated_at
FROM projects
WHERE id = ?1;
"#;

const SELECT_PROJECTS: &str = r#"
SELECT id, name, description, status, metadata, created_at, updated_at
FROM projects
ORDER BY created_at ASC;
"#;

const UPDATE_PROJECT_STATUS: &str = r#"
UPDATE projects
SET status = ?2, updated_at = ?3
WHERE id = ?1;
"#;

fn row_to_project(row: &rusqlite::Row<'_>) -> Result<Project> {
    let metadata: Option<String> = row.get(4)?;
    Ok(Project {
        id: row.get(0)?,
        name: row.get(1)?,
        description: row.get(2)?,
        status: row.get(3)?,
        metadata: parse_json(metadata)?,
        created_at: parse_dt(row.get::<_, String>(5)?)?,
        updated_at: parse_dt(row.get::<_, String>(6)?)?,
    })
}

pub fn create_project(
    db: &Database,
    name: &str,
    description: Option<String>,
    metadata: Option<serde_json::Value>,
) -> Result<Project> {
    let id = generate_id("proj");
    let now = now_utc_naive();
    let project = Project {
        id,
        name: name.to_owned(),
        description,
        status: ProjectStatus::Active,
        metadata,
        created_at: now,
        updated_at: now,
    };

    let conn = db.lock()?;
    let metadata = json_to_sql(&project.metadata)?;
    conn.execute(
        INSERT_PROJECT,
        params![
            &project.id,
            &project.name,
            &project.description,
            &project.status,
            &metadata,
            dt_to_sql(project.created_at),
            dt_to_sql(project.updated_at)
        ],
    )?;

    Ok(project)
}

pub fn get_project(db: &Database, project_id: &str) -> Result<Project> {
    let conn = db.lock()?;
    let mut stmt = conn.prepare(SELECT_PROJECT_BY_ID)?;
    let project = stmt.query_row(params![project_id], |row| {
        row_to_project(row).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(
                0,
                rusqlite::types::Type::Text,
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    e.to_string(),
                )),
            )
        })
    });
    let project = match project {
        Ok(p) => p,
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            return Err(PlandbError::NotFound(format!("project {project_id}")).into())
        }
        Err(e) => return Err(e.into()),
    };
    Ok(project)
}

pub fn fuzzy_find_project(db: &Database, input: &str) -> Result<Project> {
    match get_project(db, input) {
        Ok(project) => return Ok(project),
        Err(err) => {
            if !matches!(
                err.downcast_ref::<PlandbError>(),
                Some(PlandbError::NotFound(_))
            ) {
                return Err(err);
            }
        }
    }

    if !input.starts_with("p-") {
        let matches: Vec<(String, String)> = {
            let conn = db.lock()?;
            let like = format!("%{input}%");
            let mut stmt = conn.prepare(SELECT_PROJECTS_NAME_LIKE)?;
            let mut rows = stmt.query(params![like])?;
            let mut matches: Vec<(String, String)> = Vec::new();
            while let Some(row) = rows.next()? {
                matches.push((row.get(0)?, row.get(1)?));
            }
            matches
        };

        if matches.len() == 1 {
            return get_project(db, &matches[0].0);
        }
        if !matches.is_empty() {
            let rendered = matches
                .iter()
                .map(|(id, name)| format!("{id} ({name})"))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(anyhow::anyhow!(
                "Multiple matches for '{input}': {rendered}"
            ));
        }
    }

    if input.starts_with("p-") {
        let conn = db.lock()?;
        let mut stmt = conn.prepare(SELECT_RECENT_PROJECT_IDS)?;
        let mut rows = stmt.query([])?;
        let mut best: Option<(usize, String, String)> = None;
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let name: String = row.get(1)?;
            let dist = levenshtein(input, &id);
            if best.as_ref().map(|(d, _, _)| dist < *d).unwrap_or(true) {
                best = Some((dist, id, name));
            }
        }
        if let Some((dist, id, name)) = best {
            if dist <= 2 {
                return Err(anyhow::anyhow!(
                    "Project '{input}' not found. Did you mean: {id} ({name})?"
                ));
            }
        }
    }

    Err(PlandbError::NotFound(format!("project {input}")).into())
}

pub fn list_projects(db: &Database) -> Result<Vec<Project>> {
    let conn = db.lock()?;
    let mut stmt = conn.prepare(SELECT_PROJECTS)?;
    let mut rows = stmt.query([])?;
    let mut result = Vec::new();
    while let Some(row) = rows.next()? {
        result.push(row_to_project(row)?);
    }
    Ok(result)
}

pub fn update_project_status(
    db: &Database,
    project_id: &str,
    status: ProjectStatus,
) -> Result<Project> {
    let conn = db.lock()?;
    let now = now_utc_naive();
    let changed = conn.execute(
        UPDATE_PROJECT_STATUS,
        params![project_id, status, dt_to_sql(now)],
    )?;
    if changed == 0 {
        return Err(PlandbError::NotFound(format!("project {project_id}")).into());
    }
    drop(conn);
    get_project(db, project_id)
}

use crate::db::Database;
use anyhow::Result;
use rusqlite::{params, OptionalExtension};

const SELECT_META: &str = "SELECT value FROM plandb_meta WHERE key = ?1;";
const UPSERT_META: &str =
    "INSERT INTO plandb_meta(key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = excluded.value;";
const DELETE_META: &str = "DELETE FROM plandb_meta WHERE key = ?1;";

pub fn get_meta(db: &Database, key: &str) -> Result<Option<String>> {
    let conn = db.lock()?;
    let value = conn
        .query_row(SELECT_META, params![key], |row| row.get(0))
        .optional()?;
    Ok(value)
}

pub fn set_meta(db: &Database, key: &str, value: &str) -> Result<()> {
    let conn = db.lock()?;
    conn.execute(UPSERT_META, params![key, value])?;
    Ok(())
}

pub fn delete_meta(db: &Database, key: &str) -> Result<()> {
    let conn = db.lock()?;
    conn.execute(DELETE_META, params![key])?;
    Ok(())
}

from __future__ import annotations

import secrets
import sqlite3
from pathlib import Path

BASE62_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def get_connection(database_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(database_path: str | Path) -> None:
    database = Path(database_path)
    database.parent.mkdir(parents=True, exist_ok=True)

    with get_connection(database) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS urls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                long_url TEXT NOT NULL UNIQUE,
                short_code TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def get_url_by_code(database_path: str | Path, short_code: str) -> sqlite3.Row | None:
    with get_connection(database_path) as connection:
        return connection.execute(
            "SELECT * FROM urls WHERE short_code = ?",
            (short_code,),
        ).fetchone()


def get_url_by_long_url(database_path: str | Path, long_url: str) -> sqlite3.Row | None:
    with get_connection(database_path) as connection:
        return connection.execute(
            "SELECT * FROM urls WHERE long_url = ?",
            (long_url,),
        ).fetchone()


def create_short_code(length: int = 6) -> str:
    return "".join(secrets.choice(BASE62_ALPHABET) for _ in range(length))


def create_or_get_short_url(database_path: str | Path, long_url: str) -> sqlite3.Row:
    existing = get_url_by_long_url(database_path, long_url)
    if existing is not None:
        return existing

    with get_connection(database_path) as connection:
        while True:
            short_code = create_short_code()
            try:
                connection.execute(
                    "INSERT INTO urls (long_url, short_code) VALUES (?, ?)",
                    (long_url, short_code),
                )
                return connection.execute(
                    "SELECT * FROM urls WHERE short_code = ?",
                    (short_code,),
                ).fetchone()
            except sqlite3.IntegrityError:
                continue

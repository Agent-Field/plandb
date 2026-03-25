from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


@dataclass(frozen=True)
class UrlRecord:
    id: int
    long_url: str
    short_code: str
    created_at: str


def encode_base62(number: int) -> str:
    if number < 0:
        raise ValueError("number must be non-negative")
    if number == 0:
        return BASE62_ALPHABET[0]

    chars: list[str] = []
    while number:
        number, remainder = divmod(number, 62)
        chars.append(BASE62_ALPHABET[remainder])
    return "".join(reversed(chars))


class SQLiteURLRepository:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS urls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    long_url TEXT NOT NULL UNIQUE,
                    short_code TEXT UNIQUE,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def create_or_get(self, long_url: str) -> tuple[UrlRecord, bool]:
        existing = self.get_by_long_url(long_url)
        if existing:
            return existing, False

        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO urls (long_url) VALUES (?)",
                (long_url,),
            )
            row_id = cursor.lastrowid
            if row_id is None:
                raise RuntimeError("failed to insert URL")

            short_code = encode_base62(row_id)
            connection.execute(
                "UPDATE urls SET short_code = ? WHERE id = ?",
                (short_code, row_id),
            )

        record = self.get_by_short_code(short_code)
        if record is None:
            raise RuntimeError("failed to load inserted URL")
        return record, True

    def get_by_long_url(self, long_url: str) -> UrlRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, long_url, short_code, created_at FROM urls WHERE long_url = ?",
                (long_url,),
            ).fetchone()
        return self._row_to_record(row)

    def get_by_short_code(self, short_code: str) -> UrlRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, long_url, short_code, created_at FROM urls WHERE short_code = ?",
                (short_code,),
            ).fetchone()
        return self._row_to_record(row)

    @staticmethod
    def _row_to_record(row: sqlite3.Row | None) -> UrlRecord | None:
        if row is None:
            return None
        return UrlRecord(
            id=row["id"],
            long_url=row["long_url"],
            short_code=row["short_code"],
            created_at=row["created_at"],
        )

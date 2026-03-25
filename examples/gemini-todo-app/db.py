import sqlite3

DB_NAME = "todo.db"

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                completed BOOLEAN NOT NULL DEFAULT 0
            )
        """)

def add_task(title):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("INSERT INTO tasks (title) VALUES (?)", (title,))

def list_tasks():
    with sqlite3.connect(DB_NAME) as conn:
        return conn.execute("SELECT id, title, completed FROM tasks").fetchall()

def complete_task(task_id):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("UPDATE tasks SET completed = 1 WHERE id = ?", (task_id,))

def delete_task(task_id):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))

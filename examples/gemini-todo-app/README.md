# Example: Gemini CLI builds a todo app using PlanDB

First Gemini CLI validation — proves PlanDB works across different AI agents.
Gemini completed **100% of tasks** (4/4) with proper dependencies and context.

## Prompt

The agent received `plandb prompt --for cli` (120 lines) plus:

```
TASK: Build a Python CLI todo app with SQLite storage. Must support: add, list, complete, delete.
Create at least 4 tasks with dependencies. Use plandb context to record design decisions.
```

## What the Agent Did

### Task Graph (4/4 complete)

```
t-design ──▶ t-db ──▶ t-cli ──▶ t-tests
```

| Task | Status | Description |
|------|--------|-------------|
| Design schema and CLI structure | done | Define SQLite schema and CLI commands |
| Implement SQLite storage layer | done | Create db.py with CRUD operations |
| Implement CLI interface | done | Create todo.py with argparse |
| Verify and test | done | Test all commands |

### Context Entries (3 total)

| Kind | Content |
|------|---------|
| decision | Database schema: tasks table with id (PK), title (TEXT), completed (BOOLEAN) |
| decision | CLI interface: add, list, complete, delete commands using argparse |
| discovery | Tested all commands: add, list, complete, delete. All functional. |

### Files Created

- [`db.py`](db.py) — SQLite storage layer with CRUD operations
- [`todo.py`](todo.py) — CLI interface with argparse

## Cross-Agent Validation

| Agent | Tasks | Dependencies | Context | Custom IDs | Completion |
|-------|-------|-------------|---------|------------|------------|
| Codex | 3-4 | Yes | 3-5 entries | Yes | 33-50% |
| **Gemini** | **4** | **Yes** | **3 entries** | **Yes** | **100%** |

## Reproduce

```bash
cd /tmp/todo-test && git init
gemini -p "$(plandb prompt --for cli)

TASK: Build a Python CLI todo app with SQLite storage." -y --sandbox=none
```

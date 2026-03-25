# Example: Codex builds a Flask URL shortener (v2 — full feature validation)

This example validates the updated PlanDB prompt with all features: dependencies,
structured handoff, context store with freeform kinds, and search.

## Prompt

The agent received `plandb prompt --for cli` output (120 lines) plus:

```
TASK: Build a Python URL shortener with Flask. Create tasks with dependencies.
Record design decisions with plandb context. Use plandb search before starting each task.
After completing tasks, check if plandb go shows relevant context automatically.
```

## What the Agent Did

### Task Graph

```
t-design ──▶ t-scaffold ──▶ t-implement ──▶ t-verify
```

| Task | Kind | Custom ID | Status |
|------|------|-----------|--------|
| Design architecture | research | `t-design` | done |
| Scaffold Flask app | code | `t-scaffold` | done |
| Implement shortener flows | code | `t-implement` | running (2/4 completed) |
| Test and document | test | `t-verify` | pending |

### Structured Handoff

The agent passed structured results via `--result` — downstream tasks receive design decisions as JSON:

```json
{
  "code_strategy": "6-char base62 with collision retry",
  "database": "shortener.db",
  "routes": ["GET /", "POST /shorten", "GET /<code>"],
  "duplicate_policy": "reuse existing code for identical long URL"
}
```

### Context Entries (5 total)

| Kind | Content |
|------|---------|
| decision | Use Flask app package with sqlite3 persistence in shortener.db |
| decision | Generate 6-char Base62 short codes with secrets.choice and collision retry |
| decision | Expose GET /, POST /shorten, GET /<code> — accept only valid HTTP/HTTPS URLs |
| discovery | Scaffolded project with app factory, SQLite bootstrap, HTML form template |
| constraint | Local verification limited until dependencies installed |

### Files Created

```
app.py                  # Entrypoint
app/__init__.py         # Flask app factory
app/storage.py          # SQLite storage layer
app/templates/index.html # Form template
requirements.txt        # Dependencies
```

## Features Validated

| Feature | Agent behavior |
|---------|---------------|
| **Dependencies** | 4-task chain with `feeds_into` |
| **Custom IDs** | `--as design`, `--as scaffold`, etc. |
| **Structured handoff** | `--result '{"routes": [...], ...}'` |
| **Context store** | 5 entries: 3 decisions, 1 discovery, 1 constraint |
| **Freeform kinds** | Agent chose decision/discovery/constraint |
| **Descriptions** | Self-contained with acceptance criteria |

## Reproduce

```bash
cd /tmp/shortener-test && git init
codex exec --full-auto "$(plandb prompt --for cli)

TASK: Build a Python URL shortener with Flask. Create tasks with dependencies.
Record design decisions with plandb context."
```

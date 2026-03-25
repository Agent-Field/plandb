# Example: Codex builds a URL shortener using PlanDB

This example shows what happens when you give [Codex CLI](https://github.com/openai/codex) a task and PlanDB instructions. The agent autonomously decomposes the work, creates proper dependencies, executes each task in order, and records what it learns along the way.

## The Prompt

We gave Codex this prompt (the only human input):

```
You have plandb for task planning AND context tracking.

Task commands:
  plandb init "project"
  plandb add "title" --description "spec" --dep t-upstream --as custom-id
  plandb go → work → plandb done --next
  plandb status --detail

Context commands:
  plandb context "what you discovered" --kind discovery
  plandb search "query"

TASK: Build a URL shortener service in Python (Flask).
Create ALL tasks with dependencies upfront.
Record discoveries and decisions as context entries.
```

## What the Agent Did

### 1. Created the task graph

```
t-research ──▶ t-design ──▶ t-implement ──▶ t-test
```

The agent created 4 tasks with a linear dependency chain — each task depends on the previous one:

| Task | Kind | Description |
|------|------|-------------|
| `t-research` | research | Investigate hash-based vs counter-based short codes |
| `t-design` | code | Define API endpoints and storage approach |
| `t-implement` | code | Build Flask app with SQLite storage |
| `t-test` | test | Write pytest tests |

### 2. Executed tasks and recorded context

While working on each task, the agent recorded what it learned:

| Context | Kind | Content |
|---------|------|---------|
| `c-jacf` | discovery | Hash-based short codes are easy but need collision handling |
| `c-5s66` | decision | Use counter-based strategy with Base62 encoding |
| `c-4hlh` | decision | Use UNIQUE long_url column for idempotent shorten requests |
| `c-m4qk` | pattern | Minimal API: one write endpoint, one redirect, one stats |
| `c-tclp` | discovery | Pytest collection fails without Flask installed |
| `c-s2ry` | discovery | Network-restricted environment prevents pip install |

### 3. Final result

```
url-shortener: 4/4 done (100%)
✓ t-research  Research short code approaches
✓ t-design    Design API and storage
✓ t-implement Implement Flask app
✓ t-test      Write pytest coverage
```

## Files Created by the Agent

- [`app.py`](app.py) — Flask URL shortener with SQLite backend
- [`storage.py`](storage.py) — SQLite storage layer with Base62 encoding
- [`test_app.py`](test_app.py) — pytest test suite
- [`requirements.txt`](requirements.txt) — Python dependencies
- [`README.md`](README.md) — Usage documentation

## Key Observations

1. **Dependencies mattered** — the agent created tasks in dependency order and executed them correctly
2. **Context was natural** — the agent used `plandb context` to document research findings and design decisions without being forced
3. **Auto-linking worked** — context entries were automatically linked to the running task
4. **The agent adapted** — discovered Flask wasn't available in the sandbox and recorded it as context

## Reproduce

```bash
# Install PlanDB
curl -fsSL https://github.com/Agent-Field/plandb/releases/latest/download/plandb-$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m) -o /usr/local/bin/plandb && chmod +x /usr/local/bin/plandb

# Run with Codex
cd /tmp/url-shortener-test
codex exec --full-auto "$(plandb prompt --for cli)

TASK: Build a URL shortener service in Python (Flask).
Create ALL tasks with dependencies upfront.
Record discoveries as context entries."
```

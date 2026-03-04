# plandb Architecture

> SQLite for Agents — a zero-config, embedded task graph primitive for AI agent orchestration.

## Why plandb Exists

Every AI agent framework today solves orchestration by adding infrastructure: message queues, workflow engines, state machines-as-a-service, vector databases. More services to deploy, more endpoints to configure, more things to break at 3am.

plandb takes the opposite approach. The same way SQLite eliminated the need to run a database server for most applications, plandb eliminates the need to run an orchestration server for most agent workloads. It's a single binary that creates a `.plandb.db` file. That file *is* the entire coordination layer — the task graph, the dependency engine, the work queue, the handoff protocol, and the audit log.

No daemon. No config. No network. Just a file.

---

## Core Insight: Tasks Are the Universal Primitive

When an AI agent works, it does exactly three things:

1. **Decides** what to do next
2. **Does** the work
3. **Reports** what happened

Every orchestration framework models this differently — some use state machines, some use message passing, some use function calls. plandb models it as a **directed acyclic graph of tasks with typed dependency edges**, and makes that graph the single source of truth for all coordination.

```mermaid
graph TD
    subgraph "The plandb Model"
        A[Task A<br/>status: done<br/>result: schema.sql] -->|feeds_into| B[Task B<br/>status: running<br/>agent: claude-1]
        A -->|feeds_into| C[Task C<br/>status: ready<br/>waiting for claim]
        B -->|blocks| D[Task D<br/>status: pending<br/>blocked]
        C -->|blocks| D
        D -->|suggests| E[Task E<br/>status: pending<br/>soft dependency]
    end

    style A fill:#2d5a2d,color:#fff
    style B fill:#5a5a2d,color:#fff
    style C fill:#2d2d5a,color:#fff
    style D fill:#3a3a3a,color:#aaa
    style E fill:#3a3a3a,color:#aaa
```

This is not a workflow engine. It's a **coordination primitive** — the smallest possible abstraction that lets multiple agents work together on a shared plan without stepping on each other.

---

## Architecture Overview

```mermaid
graph TB
    subgraph "Agent Interfaces"
        CLI["CLI<br/>(clap)"]
        MCP["MCP Server<br/>(stdio JSON-RPC)"]
        HTTP["HTTP API<br/>(axum REST + SSE)"]
    end

    subgraph "Command Layer"
        PORCELAIN["Porcelain<br/>go · done · add · list · show · status"]
        PLUMBING["Plumbing<br/>claim · start · heartbeat · progress · fail<br/>insert · amend · pivot · split · decompose · replan"]
    end

    subgraph "Core Engine"
        SM["State Machine"]
        DEP["Dependency Engine<br/>(task_readiness VIEW)"]
        HANDOFF["Handoff Protocol<br/>(feeds_into results)"]
        EVENTS["Event Emitter"]
    end

    subgraph "Storage"
        SQLITE["SQLite<br/>(WAL mode)"]
    end

    CLI --> PORCELAIN
    MCP --> PORCELAIN
    HTTP --> PORCELAIN
    CLI --> PLUMBING
    MCP --> PLUMBING
    HTTP --> PLUMBING
    PORCELAIN --> SM
    PLUMBING --> SM
    SM --> DEP
    SM --> HANDOFF
    SM --> EVENTS
    DEP --> SQLITE
    HANDOFF --> SQLITE
    EVENTS --> SQLITE
    SM --> SQLITE
```

### Three Interfaces, One Truth

plandb exposes the same task graph through three interfaces — **CLI**, **MCP** (Model Context Protocol), and **HTTP** — all backed by the same SQLite file. An agent can create tasks via CLI, another can claim them via MCP, and a dashboard can monitor via HTTP SSE. They all see the same state because they all read from the same `.plandb.db`.

This is a deliberate architectural choice. Most orchestration systems treat the API server as the source of truth and the database as an implementation detail. In plandb, the **file is the truth**. The interfaces are just lenses into it. You can copy the file to another machine, open it with `sqlite3`, query it directly, back it up with `cp`. No export/import, no API migration, no schema versioning ceremony.

---

## The Task State Machine

```mermaid
stateDiagram-v2
    [*] --> pending: create
    pending --> ready: promote<br/>(all blocking deps done)
    ready --> claimed: claim<br/>(atomic, one winner)
    claimed --> running: start
    running --> done: complete
    running --> failed: fail
    running --> ready: pause<br/>(release back to queue)
    failed --> ready: retry<br/>(if retries remain)

    ready --> done: lenient done<br/>(auto-transitions)
    claimed --> done: lenient done<br/>(auto-transitions)

    note right of ready: Agent entry point.<br/>go = claim + start in one call.
    note right of done: Result stored as JSON.<br/>Triggers promote_ready_tasks().
```

### Why These States

**`pending` vs `ready`**: This is the core of plandb's dependency engine. A task stays `pending` until *all* its blocking dependencies (`blocks`, `feeds_into`) are in `done` or `done_partial` state. The `task_readiness` SQL VIEW computes this atomically, and `promote_ready_tasks()` does a single bulk UPDATE. This means you never poll for readiness — completion of one task automatically unlocks the next.

**`claimed` vs `running`**: In multi-agent scenarios, claiming a task is a separate concern from starting work on it. The claim is an atomic SQL UPDATE with `WHERE status = 'ready'` — SQLite guarantees only one agent wins. This is optimistic locking without a lock table.

**Lenient transitions**: After real-world testing showed agents wasting 83% of session time fighting the state machine, we added lenient transitions. `done` now accepts tasks in `ready`, `claimed`, or `running` status, auto-filling timestamps. This means a single-agent workflow is `go` → `done` (2 commands), while multi-agent safety is preserved because the claim mechanism still prevents double-assignment.

---

## The Dependency Engine

```mermaid
graph LR
    subgraph "Dependency Types"
        FI["feeds_into<br/>━━━━━━━━━━<br/>Result data flows<br/>downstream via handoff"]
        BL["blocks<br/>━━━━━━━━━━<br/>Ordering only,<br/>no data flow"]
        SU["suggests<br/>━━━━━━━━━━<br/>Soft dependency,<br/>doesn't block promotion"]
    end
```

### How Readiness Is Computed

plandb uses a SQL VIEW — `task_readiness` — to determine which tasks should be promoted from `pending` to `ready`:

```sql
CREATE VIEW task_readiness AS
SELECT
  t.id, t.status,
  COUNT(CASE
    WHEN d.kind IN ('blocks', 'feeds_into')
     AND upstream.status NOT IN ('done', 'done_partial')
    THEN 1
  END) AS unmet_deps,
  CASE
    WHEN t.status = 'pending'
     AND [unmet_deps] = 0
    THEN 1 ELSE 0
  END AS promotable
FROM tasks t
LEFT JOIN dependencies d ON d.to_task = t.id
LEFT JOIN tasks upstream ON upstream.id = d.from_task
GROUP BY t.id;
```

This is evaluated atomically by SQLite. When any task completes, `promote_ready_tasks()` runs:

```sql
UPDATE tasks SET status = 'ready'
WHERE id IN (SELECT id FROM task_readiness WHERE promotable = 1);
```

One query. All newly-unblocked tasks promoted in a single transaction. No event propagation, no message passing, no eventual consistency. The graph structure *is* the scheduling algorithm.

### Why Not a Trigger?

We evaluated using SQLite triggers (auto-promote on every UPDATE to `done`). The VIEW approach was chosen because:

1. **Batch efficiency**: Multiple tasks may complete in rapid succession. The VIEW evaluates all of them at once.
2. **Predictable timing**: Callers control when promotion happens (always after `complete_task`), making the behavior easier to reason about.
3. **No cascading trigger storms**: Deep dependency chains don't create recursive trigger execution.

---

## The Handoff Protocol

This is plandb's answer to "how do agents pass data to each other?"

```mermaid
sequenceDiagram
    participant A as Agent A
    participant P as plandb
    participant B as Agent B

    A->>P: plandb go --agent agent-a
    P-->>A: {task: "Design API", handoff: []}

    Note over A: Does research work...

    A->>P: plandb done t-abc --result '{"schema": "..."}'
    P->>P: promote_ready_tasks()
    Note over P: Task "Implement API" now ready<br/>(feeds_into dependency satisfied)

    B->>P: plandb go --agent agent-b
    P-->>B: {task: "Implement API",<br/>handoff: [{from: "Design API",<br/>result: {"schema": "..."},<br/>agent: "agent-a"}]}

    Note over B: Has Agent A's schema<br/>without knowing Agent A exists
```

When Agent B calls `go`, the `go_payload()` function automatically includes the results from all upstream `feeds_into` dependencies. Agent B receives the schema that Agent A produced, without either agent needing to know about the other. The graph structure creates the data flow.

### Why JSON Results Instead of Files?

Files are paths. Paths are machine-specific. A result like `{"schema": "users(id INT, name TEXT)"}` is portable, inspectable, and small enough to include in the handoff context. For large artifacts (binary files, datasets), plandb has a separate `artifacts` table — but the handoff protocol carries structured results, not file pointers.

---

## Multi-Agent Coordination

```mermaid
graph TD
    subgraph "Agent Coordination Model"
        direction TB
        QUEUE["Ready Queue<br/>━━━━━━━━━━━━━━━<br/>Tasks ordered by priority<br/>Atomic claim prevents<br/>double-assignment"]

        A1["Agent 1<br/>claimed: t-abc"]
        A2["Agent 2<br/>claimed: t-def"]
        A3["Agent 3<br/>waiting for task"]

        QUEUE -->|"atomic claim<br/>UPDATE...WHERE status='ready'"| A1
        QUEUE -->|"atomic claim<br/>next highest priority"| A2
        QUEUE -.->|"no ready tasks"| A3
    end

    subgraph "Safety Mechanisms"
        HB["Heartbeat Monitor<br/>━━━━━━━━━━━━━━━<br/>Detects crashed agents<br/>Reclaims stale tasks"]
        FC["File Conflict Detection<br/>━━━━━━━━━━━━━━━<br/>Tracks which agent<br/>modified which files"]
    end
```

### Atomic Claim — The Core Primitive

The `claim_next_task` operation is a single SQL statement:

```sql
UPDATE tasks
SET status = 'claimed', agent_id = ?1, claimed_at = ?2
WHERE id = (
  SELECT id FROM tasks
  WHERE project_id = ?2 AND status = 'ready'
  ORDER BY priority DESC, created_at ASC
  LIMIT 1
)
RETURNING ...;
```

SQLite serializes writes at the statement level. Two agents executing this simultaneously — even from different threads — are guaranteed to claim different tasks. No advisory locks, no Redis SETNX, no distributed consensus. The database *is* the lock manager.

### Heartbeat and Recovery

Each agent periodically calls `plandb task heartbeat <id>` to prove it's alive. A background sweeper checks for tasks whose `last_heartbeat` exceeds the `heartbeat_interval`:

- If retries remain: reset to `ready` (another agent can claim it)
- If retries exhausted: mark `failed`
- If the task is composite: roll up child task status

This handles the crash scenario — an agent dies mid-task, and the work is automatically returned to the queue.

---

## Plan Adaptation (Mid-Flight Changes)

Real-world agent work is messy. The initial plan is always wrong. plandb provides six primitives for changing the plan while tasks are in flight:

```mermaid
graph TD
    subgraph "Plan Adaptation Primitives"
        INSERT["insert<br/>━━━━━━━━<br/>Add a step between<br/>two existing tasks.<br/>Rewires dependencies."]
        AMEND["amend<br/>━━━━━━━━<br/>Prepend context to a<br/>future task's description.<br/>Non-destructive annotation."]
        PIVOT["pivot<br/>━━━━━━━━<br/>Replace a subtree<br/>with new tasks.<br/>Cancels old, creates new."]
        SPLIT["split<br/>━━━━━━━━<br/>Break one task into<br/>multiple sub-tasks.<br/>Preserves dependencies."]
        DECOMPOSE["decompose<br/>━━━━━━━━<br/>Create subtasks from<br/>a YAML spec.<br/>Parent becomes composite."]
        REPLAN["replan<br/>━━━━━━━━<br/>Cancel pending subtasks,<br/>create new ones from YAML.<br/>Like decompose, but replaces."]
    end
```

### Why These Six?

They map to the six things that go wrong during agent execution:

| What goes wrong | Primitive | Example |
|---|---|---|
| Missed a step | `insert` | "Need to add validation between parse and save" |
| New information surfaced | `amend` | "The API requires OAuth, not API keys" |
| Approach was wrong | `pivot` | "REST won't work, switch to GraphQL" |
| Task is too big | `split` | "Break 'implement auth' into login, signup, forgot-password" |
| Need to plan subtasks | `decompose` | "Here's the YAML spec for all the test cases" |
| Subtasks were wrong | `replan` | "Throw out the test plan, here's the new one" |

### What-If Analysis

Before making destructive changes, agents can preview effects:

```
plandb what-if cancel t-abc
```

This returns which tasks would be cancelled, which would be stranded (no remaining dependencies), and whether any running work would be affected — without actually changing anything. It uses a snapshot-and-simulate approach: snapshot current task statuses, apply the mutation in-memory, diff the results.

---

## The Porcelain/Plumbing Split

Inspired by git's architecture, plandb separates high-level agent-facing commands (porcelain) from low-level precise-control commands (plumbing):

```mermaid
graph TB
    subgraph "Porcelain — what agents use"
        GO["go<br/>claim + start in one call"]
        DONE["done<br/>complete from any active state<br/>auto-transitions timestamps"]
        ADD["add<br/>create a task"]
        LIST["list<br/>show tasks"]
        SHOW["show<br/>task details"]
        STATUS["status<br/>project progress"]
    end

    subgraph "Plumbing — for scripts and advanced use"
        CLAIM["claim"] --> START["start"] --> COMPLETE["complete"]
        HEARTBEAT["heartbeat"]
        PROGRESS["progress"]
        PROMOTE["promote_ready_tasks"]
    end

    subgraph "Hidden Aliases — agent vocabulary mapping"
        FINISH["finish → done"]
        COMP["complete → done"]
        LS["ls → list"]
        TASKS["tasks → list"]
        OVERVIEW["overview → status"]
    end

    GO -.->|"internally calls"| CLAIM
    GO -.->|"internally calls"| START
    DONE -.->|"internally calls"| COMPLETE

    style GO fill:#2d5a2d,color:#fff
    style DONE fill:#2d5a2d,color:#fff
    style ADD fill:#2d5a2d,color:#fff
    style LIST fill:#2d5a2d,color:#fff
    style SHOW fill:#2d5a2d,color:#fff
    style STATUS fill:#2d5a2d,color:#fff
```

### Why This Matters for AI Agents

AI agents don't read documentation. They read `--help` output and try commands. In testing, a Gemini Flash agent tried `plandb list`, `plandb ls`, `plandb tasks`, `plandb show`, `plandb add`, `plandb update`, `plandb start`, `plandb plan`, `plandb track`, and `plandb version` — all before finding `plandb task create`.

plandb now accepts all of these. The hidden aliases and `infer_subcommands` mean that agents can use whatever vocabulary feels natural — `plandb fin`, `plandb comp`, `plandb ta` — and plandb routes to the right handler. The agent doesn't know these are aliases. It just works.

---

## Event System

```mermaid
sequenceDiagram
    participant Agent
    participant Core as Core Operations
    participant Events as Event Emitter
    participant SSE as SSE Endpoint

    Agent->>Core: plandb done t-abc --result '{...}'
    Core->>Core: UPDATE tasks SET status='done'
    Core->>Events: insert_event(TaskCompleted, {has_result: true})
    Core->>Core: promote_ready_tasks()
    Events-->>SSE: Stream to connected clients
    Core-->>Agent: completed t-abc
```

Events are emitted as **side-effects** of core operations, not as the source of truth. This is a deliberate choice against event sourcing:

- **State is always queryable**: `SELECT * FROM tasks WHERE status = 'ready'` — no event replay needed
- **Events are optional**: If event emission fails, the primary operation still succeeds (`let _ =` in Rust)
- **Events are for observability**: Dashboards, logs, SSE streams for monitoring — not for deriving state

The event types map directly to lifecycle transitions: `task_created`, `task_ready`, `task_claimed`, `task_started`, `task_completed`, `task_failed`, `task_cancelled`.

---

## Storage Design

```mermaid
erDiagram
    projects ||--o{ tasks : contains
    tasks ||--o{ tasks : "parent/child"
    tasks ||--o{ dependencies : "from_task"
    tasks ||--o{ dependencies : "to_task"
    tasks ||--o{ task_notes : has
    tasks ||--o{ task_files : tracks
    tasks ||--o{ task_tags : tagged
    tasks ||--o{ artifacts : produces
    tasks ||--o{ events : emits
    
    projects {
        text id PK
        text name
        text status
        json metadata
    }
    
    tasks {
        text id PK
        text project_id FK
        text parent_task_id FK
        text title
        text status
        text kind
        int priority
        text agent_id
        json result
        text error
        datetime claimed_at
        datetime started_at
        datetime completed_at
    }
    
    dependencies {
        int id PK
        text from_task FK
        text to_task FK
        text kind
        text condition
    }
    
    events {
        int id PK
        text task_id FK
        text event_type
        json payload
        datetime timestamp
    }
```

### Why SQLite, Not Postgres

| Concern | SQLite | Postgres |
|---|---|---|
| Setup | `plandb project create "x"` — done | Install, configure, create user, create db, connection string |
| Portability | Copy `.plandb.db` to any machine | pg_dump, transfer, pg_restore |
| Concurrency | Serialized writes (fine for <100 agents) | Full MVCC (needed for 1000+ agents) |
| Deployment | Single binary, no runtime | Requires running service |
| Inspection | `sqlite3 .plandb.db` | psql + connection params |
| Backup | `cp .plandb.db .plandb.db.bak` | pg_dump or continuous archiving |

plandb's target is **1 to ~50 agents working on a shared plan**. At this scale, SQLite's serialized writer model is not a bottleneck — write transactions are sub-millisecond. The readability and portability advantages dominate.

### WAL Mode

plandb enables Write-Ahead Logging (`PRAGMA journal_mode = WAL`), which allows concurrent readers during writes. Multiple agents can read task state while one agent is completing a task. The write serialization only affects concurrent writes, which the atomic claim mechanism already handles.

---

## Design Decisions and Trade-offs

### IDs: Short, Fuzzy-Matched

Task IDs are 8-character prefixes of ULIDs (e.g., `t-a1b2c3d4`). Short enough for agents to type, unique enough to avoid collisions in any realistic project. plandb uses Levenshtein distance matching for typos — `t-a1b2c3d5` will find `t-a1b2c3d4` if it's the closest match.

### JSON Result, Not Typed Schema

Task results are untyped JSON (`result JSON` in the schema). This is intentional — plandb doesn't know what agents produce. A research task might return `{"findings": [...]}`, a code task might return `{"files_modified": [...]}`. Imposing a schema would force agents to serialize their diverse outputs into a common format, adding friction without value. The `feeds_into` handoff protocol passes whatever JSON was stored.

### No Workflow Language

plandb has no DSL, no YAML workflow definitions, no visual graph builder. The task graph is constructed imperatively through commands: `plandb task create`, `plandb task add-dep`. This is because AI agents don't write YAML — they call commands. The graph emerges from the agent's decisions, not from a pre-defined template.

The exception is `decompose` and `replan`, which accept YAML for bulk subtask creation. This is a convenience for the "I have a plan, create all the tasks" use case, not a workflow definition language.

### Agent-Agnostic

plandb has no concept of agent capabilities, routing rules, or agent selection. Any agent can claim any ready task. This is intentional — routing logic belongs in the agent framework, not in the coordination primitive. plandb answers "what needs to be done?" and "who's doing what?", not "who should do what?".

---

## Performance Characteristics

| Operation | Complexity | Typical Latency |
|---|---|---|
| Create task | O(1) | < 1ms |
| Claim next task | O(log n) | < 1ms (index scan) |
| Complete task + promote | O(n) | < 5ms (n = total tasks, VIEW evaluation) |
| List tasks (filtered) | O(n) | < 10ms for 1000 tasks |
| What-if analysis | O(n + e) | < 50ms (n = tasks, e = edges) |
| Lookahead (2 layers) | O(n) | < 10ms |

The `task_readiness` VIEW is the most expensive operation, as it joins `tasks × dependencies × upstream_tasks`. For projects with < 5,000 tasks (which covers virtually all agent workloads), this is sub-50ms. Beyond that, materializing the view with triggers would be the optimization path.

---

## What Makes This Different

**From workflow engines** (Temporal, Airflow, Prefect): plandb is not a service. There's no scheduler process, no worker fleet, no dashboard server. The "scheduler" is a SQL VIEW. The "workers" are agents that call `plandb go`. The "dashboard" is `plandb status`.

**From message queues** (RabbitMQ, SQS, Redis Streams): plandb is not fire-and-forget. Tasks have dependencies, results, and state. A completed task unlocks downstream tasks automatically. You can inspect the entire graph at any point — not just the current queue depth.

**From agent frameworks** (LangGraph, CrewAI, AutoGen): plandb is not an agent framework. It doesn't run agents, doesn't manage prompts, doesn't chain LLM calls. It's the coordination layer that any framework can use. An agent built with LangGraph and an agent built with raw API calls can both coordinate through the same `.plandb.db` file.

**From databases** (SQLite itself, Redis, MongoDB): plandb is not a general-purpose database. It's a domain-specific database for one thing: task coordination. The schema, the state machine, the dependency engine, and the claim mechanism are all built-in. You don't write SQL — you call `plandb go`.

The closest analogy is **SQLite itself**. SQLite didn't compete with Oracle — it created a new category (embedded database) by making the radical simplification that most applications don't need a server. plandb makes the same bet for agent orchestration: most agent workloads don't need a service. They need a file.

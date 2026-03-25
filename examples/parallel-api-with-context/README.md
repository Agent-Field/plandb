# Example: Parallel API Build with Context Store & Lazy Recall

This example demonstrates PlanDB's compound graph features working together:
dependency-driven parallelism, automatic context recall, fan-in handoff,
multi-agent coordination, and mid-flight adaptation.

## The Task Graph

```
t-schema ──┬──▶ t-auth ──┬──▶ t-tests
           │             │
           └──▶ t-crud ──┘
                (parallel)
```

After `t-schema` completes, `t-auth` and `t-crud` become ready simultaneously —
two agents can claim them in parallel. `t-tests` waits for both (fan-in).

## Setup

```bash
plandb init "microservice-api"

# Root task: schema design (research)
plandb add "Design data model" --as schema --kind research \
  --description "Define SQLite schema for users, sessions, and API keys. Consider rate limiting storage."

# Two parallel implementation tasks, both depend on schema
plandb add "Implement auth endpoints" --as auth --kind code --dep t-schema \
  --description "Build /register, /login, /logout endpoints. Use bcrypt for passwords."

plandb add "Implement CRUD endpoints" --as crud --kind code --dep t-schema \
  --description "Build /users CRUD. Paginate list endpoint. Validate all inputs."

# Fan-in: tests depend on BOTH auth and crud
plandb add "Write integration tests" --as tests --kind test --dep t-auth --dep t-crud \
  --description "pytest tests covering auth flow, CRUD operations, edge cases, and error responses."

# Seed project knowledge (simulates discoveries from a previous session)
plandb context "SQLite WAL mode is required for concurrent readers during API requests" --kind discovery
plandb context "Use argon2id for password hashing — bcrypt has 72-byte limit that truncates long passwords" --kind decision
plandb context "Rate limiting: use token bucket algorithm, store counters in same SQLite DB" --kind pattern
plandb context "All API responses must use envelope format: {ok: bool, data: ..., error: ...}" --kind constraint
```

## Execution: Lazy Recall in Action

### 1. Claim schema task — context auto-surfaced

```
$ plandb go
→ t-schema "Design data model" [0/4 · 0 ready · 3 blocked]

downstream: t-auth "Implement auth endpoints" (receives YOUR result)
downstream: t-crud "Implement CRUD endpoints" (receives YOUR result)

relevant context:
  [constraint] All API responses must use envelope format: {ok: bool, data: ..., error: ...}
  [discovery] SQLite WAL mode is required for concurrent readers during API requests
  [pattern] Rate limiting: use token bucket algorithm, store counters in same SQLite DB
```

The agent didn't search — PlanDB extracted key terms from the task ("schema", "SQLite",
"rate limiting") and matched them against the context store via BM25.

### 2. Complete schema, claim auth — different context surfaces

```
$ plandb done --result '{"tables": ["users", "sessions", "api_keys", "rate_limits"]}'
$ plandb go
→ t-auth "Implement auth endpoints" [1/4 · 1 ready · 1 blocked]

upstream:
  t-schema → (structured)

downstream: t-tests "Write integration tests" (receives YOUR result)

relevant context:
  [decision] Use argon2id for password hashing — bcrypt has 72-byte limit that truncates long passwords
```

Now the auth task sees the password hashing decision — the one context entry most
relevant to authentication work. The schema result is also handed off as upstream data.

### 3. Parallel execution — two agents, two tasks

```
$ PLANDB_AGENT=worker-2 plandb go
→ t-crud "Implement CRUD endpoints" [1/4 · 0 ready · 1 blocked]

$ plandb status --detail
p-dj20 microservice-api: 1/4 done (25%)
✓ t-schema Design data model
  ├─◉ t-auth Implement auth endpoints @default
  │ └─· t-tests Write integration tests
  └─◉ t-crud Implement CRUD endpoints @worker-2
    └─· t-tests Write integration tests
```

Two agents running in parallel. Atomic claiming prevents double-assignment.

### 4. Fan-in — tests receive both handoffs

```
$ plandb done --result '{"endpoints": ["/register", "/login", "/logout"]}'
$ PLANDB_AGENT=worker-2 plandb done --result '{"endpoints": ["/users"]}'
✓ t-crud done [3/4 · 1 ready · 0 blocked]
unlocked:
  → t-tests "Write integration tests"  (now ready)

$ plandb go
→ t-tests "Write integration tests" [3/4 · 0 ready · 0 blocked]

upstream:
  t-auth → (structured)
  t-crud → (structured)
```

The test task receives structured results from both upstream tasks — it knows
which endpoints exist without the agent having to discover them.

### 5. Mid-flight adaptation — insert a missed step

While writing tests, the agent discovers auth middleware is missing:

```
$ plandb context "Auth middleware missing — endpoints unprotected without it" --kind blocker
$ plandb task insert --after t-auth --before t-tests --title "Add auth middleware" \
    --description "Protect CRUD endpoints with JWT verification middleware"

$ plandb status --detail
p-dj20 microservice-api: 3/5 done (60%)
✓ t-schema Design data model
  ├─✓ t-auth Implement auth endpoints
  │ └─○ t-xyq1 Add auth middleware
  │   └─◉ t-tests Write integration tests @default
  └─✓ t-crud Implement CRUD endpoints
    └─◉ t-tests Write integration tests @default

$ plandb search "auth"
  [ctx] c-uf1b blocker Auth middleware missing — endpoints unprotected without it
  [task] t-auth code Implement auth endpoints: Build /register, /login, /logout endpoints.
  [task] t-xyq1 generic Add auth middleware: Protect CRUD endpoints with JWT verification middleware
```

The graph rewired itself: `t-xyq1` was inserted between `t-auth` and `t-tests`,
and `plandb search` finds both the blocker context and the new task.

## Key Features Demonstrated

| Feature | How it appeared |
|---------|----------------|
| **Dependency graph** | 4 tasks with fan-out (schema → auth + crud) and fan-in (auth + crud → tests) |
| **Lazy recall** | `plandb go` auto-surfaces relevant context — no explicit search needed |
| **Context store** | 4 context entries with different kinds: discovery, decision, pattern, constraint |
| **Multi-agent** | Two agents (`default` and `worker-2`) claiming parallel tasks atomically |
| **Handoff** | Structured results flow downstream — tests know which endpoints exist |
| **Mid-flight adaptation** | `task insert` adds auth middleware step, rewires dependencies automatically |
| **BM25 search** | `plandb search "auth"` returns ranked results across context and tasks |

## Reproduce

```bash
curl -fsSL https://github.com/Agent-Field/plandb/releases/latest/download/plandb-$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m) -o /usr/local/bin/plandb && chmod +x /usr/local/bin/plandb
# Then run the setup commands above
```

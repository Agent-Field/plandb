#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# PlanDB installer — downloads pre-built binary + configures agent frameworks
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/Agent-Field/plandb/main/install.sh | bash
#   curl -fsSL ... | bash -s -- --all          # silent: install + configure all detected
#   curl -fsSL ... | bash -s -- --binary-only  # just the binary, no framework config
# ─────────────────────────────────────────────────────────────────────

REPO="Agent-Field/plandb"
BINARY="plandb"
INSTALL_DIR="${PLANDB_INSTALL_DIR:-$HOME/.local/bin}"
PLANDB_VERSION="${PLANDB_VERSION:-latest}"

# ── Colors ──────────────────────────────────────────────────────────

if [[ -t 1 ]]; then
  BOLD="\033[1m"
  DIM="\033[2m"
  RESET="\033[0m"
  GREEN="\033[32m"
  CYAN="\033[36m"
  YELLOW="\033[33m"
  RED="\033[31m"
  WHITE="\033[37m"
  CHECK="${GREEN}✓${RESET}"
  CROSS="${RED}✗${RESET}"
  ARROW="${CYAN}→${RESET}"
else
  BOLD="" DIM="" RESET="" GREEN="" CYAN="" YELLOW="" RED="" WHITE=""
  CHECK="[ok]" CROSS="[!!]" ARROW="->"
fi

# ── Utility ─────────────────────────────────────────────────────────

info()  { echo -e "  ${ARROW} $1"; }
ok()    { echo -e "  ${CHECK} $1"; }
warn()  { echo -e "  ${YELLOW}!${RESET} $1"; }
fail()  { echo -e "  ${CROSS} $1"; exit 1; }
header() { echo -e "\n${BOLD}$1${RESET}"; }

# ── Detect platform ─────────────────────────────────────────────────

detect_platform() {
  local os arch
  os="$(uname -s)"
  arch="$(uname -m)"

  case "$os" in
    Linux*)  OS="linux" ;;
    Darwin*) OS="darwin" ;;
    *)       fail "Unsupported OS: $os — pre-built binaries available for Linux and macOS" ;;
  esac

  case "$arch" in
    x86_64|amd64)  ARCH="x86_64" ;;
    arm64|aarch64) ARCH="aarch64" ;;
    *)             fail "Unsupported architecture: $arch — pre-built binaries available for x86_64 and aarch64" ;;
  esac
}

# ── Install binary ──────────────────────────────────────────────────

install_binary() {
  header "Installing PlanDB"

  detect_platform
  info "Platform: ${OS}/${ARCH}"

  # Resolve release URL
  local release_url
  if [[ "$PLANDB_VERSION" == "latest" ]]; then
    release_url="https://api.github.com/repos/${REPO}/releases/latest"
  else
    release_url="https://api.github.com/repos/${REPO}/releases/tags/${PLANDB_VERSION}"
  fi

  local asset_name="plandb-${OS}-${ARCH}"

  # Build auth header if token is available
  local gh_token="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
  if [[ -z "$gh_token" ]] && command -v gh &>/dev/null; then
    gh_token=$(gh auth token 2>/dev/null || true)
  fi

  # Download binary — try plain curl first (works for public repos),
  # fall back to gh CLI (handles private repos with auth)
  local downloaded=false
  local tmpfile
  tmpfile=$(mktemp)

  # Try curl first (fast, no auth needed for public repos)
  local curl_auth=()
  if [[ -n "$gh_token" ]]; then
    curl_auth=(-H "Authorization: token ${gh_token}")
  fi

  local download_url=""
  download_url=$(curl -fsSL "${curl_auth[@]}" "$release_url" 2>/dev/null \
    | grep -o "\"browser_download_url\": *\"[^\"]*${asset_name}[^\"]*\"" \
    | head -1 \
    | sed 's/.*"browser_download_url": *"\(.*\)"/\1/' || true)

  if [[ -n "$download_url" ]]; then
    info "Downloading ${asset_name}..."
    curl -fsSL "${curl_auth[@]}" "$download_url" -o "$tmpfile"
    downloaded=true
  fi

  # Fall back to gh CLI for private repos
  if [[ "$downloaded" != "true" ]] && command -v gh &>/dev/null && gh auth status &>/dev/null; then
    info "Downloading ${asset_name} via gh CLI..."
    local release_tag="latest"
    if [[ "$PLANDB_VERSION" != "latest" ]]; then
      release_tag="$PLANDB_VERSION"
    fi
    local tmpdir
    tmpdir=$(mktemp -d)
    if gh release download "$release_tag" \
        --repo "${REPO}" \
        --pattern "${asset_name}" \
        --dir "$tmpdir" \
        --clobber 2>/dev/null; then
      mv "$tmpdir/${asset_name}" "$tmpfile"
      downloaded=true
    fi
    rm -rf "$tmpdir"
  fi

  if [[ "$downloaded" != "true" ]]; then
    rm -f "$tmpfile"

    # Download failed — check if plandb is already installed
    local existing=""
    if command -v plandb &>/dev/null; then
      existing=$(plandb --version 2>/dev/null || true)
    elif [[ -x "${INSTALL_DIR}/${BINARY}" ]]; then
      existing=$("${INSTALL_DIR}/${BINARY}" --version 2>/dev/null || true)
    fi

    if [[ -n "$existing" ]]; then
      ok "plandb already installed (${existing}) — using existing binary"
      info "To upgrade, run again when a new release is published"
    else
      warn "No pre-built binary found and plandb is not installed."
      info "Build from source:"
      echo -e "    ${DIM}git clone https://github.com/${REPO}.git && cd plandb${RESET}"
      echo -e "    ${DIM}cargo build --release${RESET}"
      echo -e "    ${DIM}cp target/release/plandb ${INSTALL_DIR}/plandb${RESET}"
      echo ""
      info "Continuing with framework configuration..."
    fi
  else
    chmod +x "$tmpfile"

    # Verify checksum if available
    local checksums_url=""
    checksums_url=$(curl -fsSL "${curl_auth[@]:-}" "$release_url" 2>/dev/null \
      | grep -o "\"browser_download_url\": *\"[^\"]*checksums[^\"]*\"" \
      | head -1 \
      | sed 's/.*"browser_download_url": *"\(.*\)"/\1/' || true)

    if [[ -n "$checksums_url" ]]; then
      local expected_hash
      expected_hash=$(curl -fsSL "${curl_auth[@]:-}" "$checksums_url" 2>/dev/null \
        | grep "${asset_name}$" | awk '{print $1}' || true)

      if [[ -n "$expected_hash" ]]; then
        local actual_hash=""
        if command -v sha256sum &>/dev/null; then
          actual_hash=$(sha256sum "$tmpfile" | awk '{print $1}')
        elif command -v shasum &>/dev/null; then
          actual_hash=$(shasum -a 256 "$tmpfile" | awk '{print $1}')
        fi

        if [[ -n "$actual_hash" && "$actual_hash" != "$expected_hash" ]]; then
          rm -f "$tmpfile"
          fail "Checksum mismatch — expected ${expected_hash}, got ${actual_hash}"
        fi
        [[ -n "$actual_hash" ]] && ok "Checksum verified"
      fi
    fi

    # Install
    mkdir -p "$INSTALL_DIR"
    mv "$tmpfile" "${INSTALL_DIR}/${BINARY}"
    chmod +x "${INSTALL_DIR}/${BINARY}"

    ok "Installed to ${INSTALL_DIR}/${BINARY}"
  fi

  # Verify (if binary exists anywhere)
  if command -v plandb &>/dev/null; then
    ok "plandb $(plandb --version 2>/dev/null || echo 'installed')"
  elif [[ -x "${INSTALL_DIR}/${BINARY}" ]]; then
    ok "plandb $("${INSTALL_DIR}/${BINARY}" --version 2>/dev/null || echo 'installed')"
  fi

  # Check PATH
  if ! echo "$PATH" | tr ':' '\n' | grep -qx "$INSTALL_DIR"; then
    warn "${INSTALL_DIR} is not in PATH"
    info "Add to your shell profile:"
    echo -e "    ${DIM}export PATH=\"${INSTALL_DIR}:\$PATH\"${RESET}"
  fi
}

# ── Agent instructions content ──────────────────────────────────────

plandb_instructions() {
  cat << 'PLANDB_BLOCK'
# PlanDB — Task Planning for Agents

<plandb-philosophy>
You have `plandb` installed. It is your task graph — use it to decompose every non-trivial task into a dependency graph, track progress, and coordinate parallel work.

**BEFORE you start any task, ask yourself: can this be done in a single action with no dependencies?**
- If YES (one action, no parallelism possible, no follow-up steps) — just do it.
- If NO — **STOP. Plan in PlanDB first.** This includes: multiple steps, parallel work, research + execution, anything with dependencies between parts.
- If there are **2+ independent actions** — that's parallelism. Use PlanDB even if each action is simple.

This is non-negotiable. Do NOT justify skipping PlanDB because individual steps are "simple." If there are multiple steps or any opportunity to parallelize, the graph must exist before work begins. PlanDB is for any work you do — not just a specific kind of task.

PlanDB is a compound graph with two orthogonal structures:
- **Containment** (place graph): tasks contain subtasks recursively, to any depth
- **Dependencies** (link graph): edges between tasks at ANY level, crossing containment boundaries

This is more general than a flat list or a simple DAG. A subtask at depth 3 can depend on a task in a completely different branch.
</plandb-philosophy>

<plandb-core-loop>
## Core Loop

Every task follows this cycle. No exceptions.

```bash
plandb init "project-name"                            # 1. Create project (once)
plandb add "title" --description "spec" --dep t-xxx   # 2. Build the task graph
plandb go                                             # 3. Claim next ready task
# ... do the work ...
plandb done --next                                    # 4. Complete + claim next
plandb status --detail                                # 5. Reassess after each task
```

**Description is mandatory.** Every task's `--description` must be a self-contained work order: what to do, what to produce, acceptance criteria. The title is a label — the description is the spec.
</plandb-core-loop>

<plandb-decomposition>
## Decomposition

Break tasks down aggressively — not just at the project level, but within each task. The more granular your graph, the more parallelism you unlock and the faster you recover from failures.

```bash
# Split into independent subtasks (creates parallelism)
plandb split --into "A, B, C"

# Split with dependency chain (sequential)
plandb split --into "A > B > C"

# Scope into a composite task to add deeper subtasks
plandb use t-xxx
plandb add "sub-subtask" --description "..."
plandb use ..                                         # Zoom back out
```

**The decomposition rule:** if a task would take more than ~30 seconds to complete, split it further. Keep splitting until each leaf task is a single focused action. Every split is a new opportunity for parallelism — the more leaves your graph has, the more work can happen simultaneously. Composite tasks auto-complete when all children finish.
</plandb-decomposition>

<plandb-parallelism>
## Parallel Execution

**When `plandb list --status ready` returns multiple tasks, run them concurrently.** This is where PlanDB creates the most value — it tells you exactly which tasks are independent and safe to parallelize.

**Use sub-agents for parallelism.** If a task would take more than ~30 seconds to complete, and there are other ready tasks, dispatch sub-agents to work on them simultaneously. Don't serialize work that the graph says is independent.

```bash
plandb list --status ready                            # See what can run NOW
plandb what-unlocks t-xxx                             # What opens up when this completes
plandb ahead --depth 3                                # Preview next 3 layers of work
```

Sub-agent workflow:
1. Run `plandb list --status ready` to find independent tasks
2. Spawn a sub-agent per ready task — each claims with `plandb go --agent <name>`
3. Each completes with `plandb done --next --agent <name>`
4. Atomic claiming prevents conflicts — two agents cannot claim the same task
5. Collect results and continue with the next wave of ready tasks

Set `PLANDB_AGENT=<name>` to avoid passing `--agent` on every command.
</plandb-parallelism>

<plandb-adaptation>
## Mid-Flight Adaptation

Plans are hypotheses. Adapt as you learn — don't abandon PlanDB when reality diverges from the plan.

| Situation | Command |
|-----------|---------|
| Missed a step | `plandb task insert --after t-a --before t-b --title "..."` |
| Task too large | `plandb split --into "A, B, C"` |
| New info for a future task | `plandb task amend t-xxx --prepend "NOTE: ..."` |
| Need to replace a subtree | `plandb task pivot t-xxx --file new-tasks.yaml` |
| Unsure about cancelling | `plandb what-if cancel t-xxx` |
</plandb-adaptation>

<plandb-introspection>
## Introspection

Use these to decide what to work on and where effort is wasted.

```bash
plandb status --detail                                # Per-task breakdown with status
plandb status --full                                  # Compound graph: containment + deps
plandb critical-path                                  # Longest chain — prioritize this
plandb bottlenecks                                    # Tasks blocking the most downstream work
plandb watch                                          # Live-updating dashboard
```
</plandb-introspection>

<plandb-knowledge>
## Knowledge Store

Record discoveries as you work. Context persists across sessions and auto-surfaces when relevant.

```bash
plandb context "what you learned" --kind discovery    # Record project knowledge
plandb search "query"                                 # BM25 search across context + tasks
plandb contexts                                       # List all context entries
```

`--kind` is freeform: `discovery`, `decision`, `pattern`, `blocker`, `reference`, `constraint`, `insight` — use whatever fits. Context auto-links to the running task and auto-surfaces on `plandb go` (lazy recall).
</plandb-knowledge>

<plandb-strategy>
## Planning Strategy

Your goal is maximum quality at maximum speed. PlanDB enables this — but only if you plan deliberately.

**Think before you add tasks.** Before building the graph, reason about the problem structure:
- What are the independent dimensions? Those become parallel branches.
- What has to happen in sequence? Those get `--dep` chains.
- What will you only learn by doing? Keep those tasks small so pivoting is cheap.

**Maximize the width of your graph.** The more tasks that are `ready` simultaneously, the more work runs in parallel. Structure dependencies to unlock breadth early:
- Front-load research and design tasks — they unblock everything downstream.
- Split work by independent outcome, not by sequential phase. "Fetch ad performance" and "Check audience status" can parallelize. "Gather all data" then "analyze everything" cannot.
- Use `plandb ahead --depth 3` to check if your graph has enough width.

**Use the critical path to prioritize.** `plandb critical-path` shows the longest chain to completion — that's your bottleneck. If you have a choice between working on something on the critical path or off it, always choose the critical path. `plandb bottlenecks` shows which tasks block the most downstream work.

**Record context aggressively.** Every discovery, constraint, or decision you make is knowledge that a future task (or a future agent) may need. Use `plandb context` immediately — don't wait until later. It costs almost nothing and compounds over time.
</plandb-strategy>

<plandb-rules>
## Rules

1. **Use PlanDB for every non-trivial task.** Multiple steps, parallel actions, or dependencies between parts — plan it.
2. **Description is the spec.** `--description` must contain everything needed to execute: context, approach, files, acceptance criteria.
3. **Maximize graph width.** Structure tasks to unlock as many parallel branches as possible.
4. **Parallelize ready tasks.** When `plandb list --status ready` returns multiple tasks, run them concurrently — with sub-agents if available.
5. **Prioritize the critical path.** `plandb critical-path` determines completion time. Focus there first.
6. **Adapt, don't abandon.** When reality changes, use `insert`, `split`, `amend`, `pivot`. Don't stop using PlanDB.
7. **Record what you learn.** Use `plandb context` for discoveries, blockers, patterns. It auto-surfaces later.
8. **Dependencies flow data.** `feeds_into` = data flows, `blocks` = ordering only, `suggests` = soft/optional.
9. **Kinds are freeform.** Common: `code`, `research`, `review`, `test`, `shell`, `generic`.
10. **Status is automatic.** `pending` → `ready` (deps done) → `claimed` → `running` → `done`/`failed`. Don't manage state manually.
</plandb-rules>
PLANDB_BLOCK
}

# ── Claude Code skill content ──────────────────────────────────────

SKILL_DIR="${HOME}/.claude/skills/plandb"
SKILL_MARKER="<!-- plandb-skill -->"

plandb_skill_content() {
  cat << 'SKILL_BLOCK'
---
name: plandb
description: "Enforce PlanDB task graph usage for all non-trivial work. Triggers on multi-step tasks, parallel work, feature implementation, refactoring, or any request with dependencies. Ensures proper init, decomposition, parallelism, context recording, and critical-path prioritization."
allowed-tools: [Bash, Agent, Read, Write, Edit, Grep, Glob]
version: 1.0.0
---

# PlanDB — Mandatory Task Planning Workflow

<purpose>
This skill enforces disciplined use of the PlanDB task graph for every non-trivial task.
It is a **rigid** skill — follow the phases exactly. Do not skip steps or rationalize shortcuts.
</purpose>

## Command Gotchas (READ FIRST — common mistakes)

<critical-command-rules>

**These are the most common PlanDB mistakes. Memorize them.**

| WRONG | WHY IT FAILS | CORRECT |
|-------|-------------|---------|
| `plandb go t-xxx` | `go` takes NO positional args. It auto-picks the next ready task. | `plandb task claim t-xxx` (to claim a specific task) |
| `plandb task complete t-xxx` | No `complete` subcommand exists. | `plandb done` or `plandb task done` |
| `plandb task finish t-xxx` | No `finish` subcommand exists. | `plandb done` or `plandb task done` |
| `plandb go && plandb done` | You must do the actual work between `go` and `done`. | `plandb go` → do work → `plandb done` |
| `plandb done t-xxx` | `done` completes the *currently running* task. No task ID arg. | `plandb task claim t-xxx && plandb task start t-xxx` then `plandb done` |
| `plandb add "title"` (no description) | Description is mandatory — title is just a label. | `plandb add "title" --description "full spec"` |
| `plandb status t-xxx` | `status` is project-level. To see one task: | `plandb show t-xxx` or `plandb task get t-xxx` |

**Key command semantics:**
- **`plandb go`** = claim + start the next ready task (no args, auto-selects)
- **`plandb task claim <id>`** = claim a *specific* task by ID
- **`plandb done`** = complete the currently running task (no task ID arg)
- **`plandb done --next`** = complete current + auto-claim next ready task
- **`plandb show <id>`** = view details of a specific task
- **`plandb list --status ready`** = see all claimable tasks

</critical-command-rules>

---

## Decision Gate (MUST run first)

Before ANY work, answer this question:

> **Can this task be completed in a single action with no dependencies and no parallelism opportunity?**

- **YES** (truly single action) → Do it directly. PlanDB not needed.
- **NO** (multiple steps, parallel branches, dependencies, research + execution) → **STOP. Enter PlanDB workflow below.**
- **2+ independent actions** (even if each is simple) → That's parallelism. **Use PlanDB.**

```
EXAMPLES — USE PLANDB:
  "Add auth to the API"              → research + schema + handler + tests + review
  "Fix these 3 bugs"                 → 3 independent fixes = parallel branches
  "Refactor the storage layer"       → plan + multiple files + tests
  "Build a dashboard page"           → API + components + styling + wiring

EXAMPLES — SKIP PLANDB:
  "Rename this variable"             → single find-and-replace
  "What does this function do?"      → read + explain
  "Add a comment to line 42"         → single edit
```

---

## Phase 0: Project Bootstrap

**Detect existing state before creating anything.**

```bash
# Step 1: Check for existing PlanDB database in the working directory
ls .plandb.db 2>/dev/null

# Step 2: If database exists, check active projects
plandb project list --compact 2>/dev/null

# Step 3: Check current scope
plandb status 2>/dev/null
```

### Decision Tree

```dot
digraph bootstrap {
  "Check .plandb.db exists" -> "DB found" [label="yes"];
  "Check .plandb.db exists" -> "No DB" [label="no"];

  "DB found" -> "plandb project list";
  "plandb project list" -> "Active project matches task?" [label="projects exist"];
  "plandb project list" -> "plandb init 'task-name'" [label="no projects"];

  "Active project matches task?" -> "plandb use <project>" [label="yes, reuse"];
  "Active project matches task?" -> "plandb init 'new-task'" [label="no, new project"];

  "No DB" -> "plandb init 'task-name'" [label="creates .plandb.db"];

  "plandb use <project>" -> "Phase 1: Decomposition";
  "plandb init 'task-name'" -> "Phase 1: Decomposition";
  "plandb init 'new-task'" -> "Phase 1: Decomposition";
}
```

**Rules:**
- If a `.plandb.db` exists and has a project matching the current task scope, **reuse it** with `plandb use <project>`.
- If a `.plandb.db` exists but the current task is unrelated, create a **new project** in the same database with `plandb init "new-task-name"`.
- If no `.plandb.db` exists, `plandb init "task-name"` creates both the database and the project.
- Project names should be short, descriptive kebab-case: `auth-system`, `dashboard-redesign`, `fix-memory-leak`.
- **NEVER** use `PLANDB_DB` to point at a global/shared database unless the user explicitly requests it. Default to per-directory `.plandb.db`.

### Shared Database (only if user requests)

If the user says "use a common plandb" or "shared database":

```bash
export PLANDB_DB=~/.plandb/shared.db
plandb init "task-name"
```

---

## Phase 1: Decomposition

**Think before adding tasks.** Reason about the problem structure:

1. **What are the independent dimensions?** → Those become parallel branches (no `--dep`).
2. **What must happen in sequence?** → Those get `--dep` chains.
3. **What will you only learn by doing?** → Keep those tasks small so pivoting is cheap.
4. **What's the critical path?** → Front-load research/design tasks that unblock everything.

### Building the Graph

```bash
# Add root-level tasks with full descriptions
plandb add "Design API schema" \
  --kind research \
  --description "Review existing models in src/models/. Design REST endpoints for user CRUD. Output: endpoint list with request/response shapes. Acceptance: covers all user stories from the spec." \
  --as schema

plandb add "Implement handlers" \
  --kind code \
  --dep t-schema \
  --description "Create handler functions for each endpoint from the schema design. Follow existing patterns in src/handlers/. Each handler must have input validation. Acceptance: all endpoints return correct status codes." \
  --as handlers

plandb add "Write tests" \
  --kind test \
  --dep t-schema \
  --description "Write unit + integration tests for all endpoints. Use table-driven tests. Target 80%+ coverage. Acceptance: all tests pass, coverage verified." \
  --as tests

# Independent tasks (no --dep) = parallel branches
plandb add "Update README" \
  --kind generic \
  --description "Add API documentation section. Include endpoint table, auth requirements, example requests." \
  --as docs
```

### Decomposition Rules

| Guideline | Why |
|-----------|-----|
| **Every task MUST have `--description`** | Description is the spec. Title is just a label. |
| **If a task takes >30 seconds, split it** | Smaller = more parallelism, cheaper pivots |
| **Split by outcome, not by phase** | "Auth handler" + "User handler" parallelizes. "Write all handlers" then "test all" does not. |
| **Use `--kind`** | `code`, `research`, `review`, `test`, `shell`, `generic` — helps prioritization |
| **Use `--pre` and `--post` for quality gates** | `--pre "schema doc exists"` / `--post "all tests pass"` |
| **Use `--as` for readable IDs** | `--as auth` creates `t-auth` — easier than `t-k3m9` |

### Splitting Existing Tasks

When a task turns out to be too large mid-execution:

```bash
# Parallel split (independent subtasks)
plandb split --into "Validate input, Transform data, Write response"

# Sequential split (dependency chain)
plandb split --into "Parse request > Validate > Execute > Respond"

# Deep nesting — scope into a composite, add subtasks
plandb use t-handlers
plandb add "Auth middleware" --kind code --description "..."
plandb add "Rate limiter" --kind code --description "..."
plandb use ..
```

**Composite tasks auto-complete when all children finish.** Never manually mark a composite task as done.

---

## Phase 2: Validate the Graph

**Before starting ANY work, verify the graph is well-formed.**

```bash
# Check graph structure
plandb status --full

# Verify critical path makes sense
plandb critical-path

# Check for bottlenecks
plandb bottlenecks

# Preview first 3 waves of work
plandb ahead --depth 3
```

### Quality Checks

- [ ] **Width check**: Are there 2+ tasks with `ready` status? If the graph is purely sequential, reconsider — can anything parallelize?
- [ ] **Description check**: Does every leaf task have a self-contained `--description`? Could a sub-agent execute it without additional context?
- [ ] **Dependency check**: Are dependencies minimal? Don't add deps "just in case" — only where data flows or ordering is genuinely required.
- [ ] **Granularity check**: Is every leaf task completable in <30 seconds? If not, plan to split during execution.

---

## Phase 3: Execute

### The Core Loop

```bash
# 1. Claim next ready task
plandb go

# 2. Do the work (the task description is your spec)

# 3. Record any discoveries immediately
plandb context "learned that X uses Y pattern" --kind discovery

# 4. Complete + claim next
plandb done --next

# 5. Reassess after each task
plandb status --detail
```

### Parallel Execution (CRITICAL)

**When `plandb list --status ready` returns multiple tasks, run them concurrently.**

```bash
# Check what's ready
plandb list --status ready

# If multiple tasks ready → spawn sub-agents
# Each agent claims atomically — no conflicts possible
```

**Sub-agent dispatch pattern:**

1. Run `plandb list --status ready` — get list of independent tasks.
2. For each ready task, spawn a sub-agent with:
   - The task description as the prompt
   - `PLANDB_AGENT=<agent-name>` set in environment
   - Instructions to run `plandb go --agent <name>` then `plandb done --next --agent <name>`
3. Collect results from all sub-agents.
4. Run `plandb status --detail` to see what unlocked.
5. Repeat with next wave of ready tasks.

**Never serialize work that the graph says is independent.**

### Critical Path Priority

When choosing what to work on:

```bash
plandb critical-path    # Shows the longest chain — this determines total completion time
plandb bottlenecks      # Shows tasks blocking the most downstream work
```

**Rule: Always prefer critical-path tasks over non-critical ones.** Off-critical work can wait; critical-path delays extend the total timeline.

---

## Phase 4: Adapt Mid-Flight

**Plans are hypotheses. When reality changes, adapt — never abandon PlanDB.**

| Situation | Action |
|-----------|--------|
| Discovered a missed step | `plandb task insert --after t-a --before t-b --title "..." --description "..."` |
| Current task is too large | `plandb split --into "A, B, C"` (from within the running task) |
| Learned something a future task needs | `plandb task amend t-xxx --prepend "NOTE: discovered that ..."` |
| Need to replace an entire approach | `plandb task pivot t-xxx --file new-tasks.yaml` |
| Considering cancelling a branch | `plandb what-if cancel t-xxx` (preview first!) |
| New dependency discovered | `plandb task add-dep --from t-src --to t-dst` |

**Never skip adaptation.** If you find yourself working outside the graph, STOP and update the graph first. The graph is your source of truth.

---

## Phase 5: Record Knowledge

**Record context aggressively throughout execution.**

```bash
# Discoveries — technical findings
plandb context "API requires OAuth2 bearer tokens, not API keys" --kind discovery

# Decisions — choices made and why
plandb context "chose SQLite over Postgres for local dev: simpler setup, sufficient scale" --kind decision

# Patterns — reusable approaches found
plandb context "all handlers in this codebase follow the middleware(validate(handle())) pattern" --kind pattern

# Blockers — things that are stuck
plandb context "CI pipeline timeout at 10min, need to split test suite" --kind blocker

# Constraints — boundaries that limit solutions
plandb context "must maintain backward compat with v2 API clients" --kind constraint
```

**Context auto-links to the running task and auto-surfaces on `plandb go`.** Record immediately — don't wait until later. It costs nothing and compounds over time.

---

## Phase 6: Completion

When all tasks are done:

```bash
# Final status check
plandb status --detail

# Verify nothing left
plandb list --status ready
plandb list --status running

# Review what was learned
plandb contexts
```

**A project is complete when `plandb status` shows 100% and no tasks are `ready` or `running`.**

---

## Anti-Patterns (NEVER do these)

| Anti-Pattern | Why It's Wrong | Do This Instead |
|-------------|----------------|-----------------|
| Skip PlanDB because "it's simple" | Multi-step is multi-step, regardless of individual step complexity | Use the Decision Gate honestly |
| Add tasks without `--description` | Title is a label, not a spec. Sub-agents can't execute labels. | Write self-contained descriptions |
| Create a purely sequential graph | No parallelism = no speed gain from PlanDB | Find independent dimensions and split |
| Work outside the graph | Untracked work can't be coordinated or resumed | Update the graph, then work |
| Abandon PlanDB when plans change | The whole point is adaptation | Use `insert`, `amend`, `pivot`, `split` |
| Manually manage task status | PlanDB handles state transitions automatically | Use `go`, `done`, `fail` — never set status directly |
| Serialize ready tasks | Wastes the parallelism PlanDB provides | Dispatch sub-agents for concurrent ready tasks |
| Forget to record context | Knowledge is lost between sessions | Use `plandb context` immediately on any discovery |
| Create tasks with no deps when deps exist | Graph won't enforce ordering | Think about data flow and add proper `--dep` edges |
| `plandb go t-xxx` | `go` takes NO args — it auto-selects | Use `plandb task claim t-xxx` to claim a specific task |
| `plandb task complete` | No such subcommand | Use `plandb done` or `plandb task done` |
| `plandb go && done && go && done` chaining | Must do actual work between go and done | `go` → work → `done --next` → work → `done --next` |
| Passing task ID to `plandb done` | `done` completes the currently running task | Just `plandb done` — it knows which task is running |

---

## Quick Reference

```
BOOTSTRAP:
  plandb init "name"                          Create project + DB
  plandb project list                         List existing projects
  plandb use <project>                        Switch to existing project

BUILD GRAPH:
  plandb add "title" --description "spec"     Add task (ALWAYS use --description)
  plandb add "t" --dep t-xxx --as my-id       Add with dependency + custom ID
  plandb split --into "A, B, C"               Parallel split
  plandb split --into "A > B > C"             Sequential split
  plandb use t-xxx / plandb use ..            Navigate hierarchy

EXECUTE:
  plandb go                                   Claim next ready task (NO args!)
  plandb task claim t-xxx                     Claim a SPECIFIC task by ID
  plandb done                                 Complete current task (NO task ID arg!)
  plandb done --next                          Complete + claim next
  plandb done --result '{"key":"val"}'        Complete with structured output
  plandb show t-xxx                           View details of a specific task

INTROSPECT:
  plandb status --detail                      Per-task breakdown
  plandb status --full                        Full compound graph
  plandb critical-path                        Longest chain (prioritize this)
  plandb bottlenecks                          Most-blocking tasks
  plandb list --status ready                  What can run NOW
  plandb ahead --depth 3                      Preview next 3 waves

ADAPT:
  plandb task insert --after t-a --before t-b Insert step
  plandb task amend t-xxx --prepend "NOTE:"   Annotate future task
  plandb task pivot t-xxx --file tasks.yaml   Replace subtree
  plandb what-if cancel t-xxx                 Preview cancel effects

KNOWLEDGE:
  plandb context "learned X" --kind discovery Record knowledge
  plandb search "query"                       Search everything
  plandb contexts                             List all context entries

MULTI-AGENT:
  plandb go --agent worker-1                  Agent-specific claiming
  plandb done --next --agent worker-1         Agent-specific completion
  PLANDB_AGENT=worker-1                       Set default agent ID
```
SKILL_BLOCK
}

plandb_skillkit_json() {
  local now
  now=$(date -u +"%Y-%m-%dT%H:%M:%S.000Z")
  cat <<EOF
{
  "name": "plandb",
  "description": "Enforce PlanDB task graph usage for all non-trivial work. Ensures proper init, decomposition, parallelism, context recording, and critical-path prioritization.",
  "source": "local",
  "sourceType": "local",
  "subpath": "plandb",
  "installedAt": "${now}",
  "enabled": true
}
EOF
}

install_claude_skill() {
  mkdir -p "$SKILL_DIR"
  plandb_skill_content > "${SKILL_DIR}/SKILL.md"
  plandb_skillkit_json > "${SKILL_DIR}/.skillkit.json"
  ok "Claude Code skill: installed (${SKILL_DIR}/)"
}

# ── Framework configuration ─────────────────────────────────────────

FRAMEWORKS=(
  "claude|Claude Code|${HOME}/.claude/rules/plandb.md|dir"
  "cursor|Cursor|CURSOR_SETTINGS|settings"
  "codex|Codex (OpenAI)|${HOME}/.codex/AGENTS.override.md|file"
  "gemini|Gemini CLI|${HOME}/.gemini/GEMINI.md|file"
  "opencode|OpenCode|${HOME}/.config/opencode/AGENTS.md|file"
  "windsurf|Windsurf|${HOME}/.codeium/windsurf/memories/global_rules.md|file"
  "aider|Aider|${HOME}/.aider.conventions.md|file"
)

MARKER="<!-- plandb-agent-instructions -->"

framework_id()   { echo "$1" | cut -d'|' -f1; }
framework_name() { echo "$1" | cut -d'|' -f2; }
framework_path() { echo "$1" | cut -d'|' -f3; }
framework_method() { echo "$1" | cut -d'|' -f4; }

is_framework_installed() {
  local id="$1"
  case "$id" in
    claude)   [[ -d "$HOME/.claude" ]] ;;
    cursor)   command -v cursor &>/dev/null || [[ -d "$HOME/.cursor" ]] || [[ -d "$HOME/Library/Application Support/Cursor" ]] ;;
    codex)    command -v codex &>/dev/null || [[ -d "$HOME/.codex" ]] ;;
    gemini)   command -v gemini &>/dev/null || [[ -d "$HOME/.gemini" ]] ;;
    opencode) command -v opencode &>/dev/null || [[ -d "$HOME/.config/opencode" ]] ;;
    windsurf) [[ -d "$HOME/.codeium" ]] || [[ -d "$HOME/Library/Application Support/Windsurf" ]] ;;
    aider)    command -v aider &>/dev/null ;;
    *)        return 1 ;;
  esac
}

already_configured() {
  local path="$1"
  [[ -f "$path" ]] && grep -q "$MARKER" "$path" 2>/dev/null
}

configure_framework() {
  local entry="$1"
  local id=$(framework_id "$entry")
  local name=$(framework_name "$entry")
  local path=$(framework_path "$entry")
  local method=$(framework_method "$entry")

  if [[ "$method" == "settings" ]]; then
    warn "${name}: global rules are in Settings UI, not a file"
    info "Paste the plandb instructions into: Cursor → Settings → General → Rules for AI"
    info "Or add to a project: .cursor/rules/plandb.mdc"
    return 0
  fi

  # Create parent directories
  mkdir -p "$(dirname "$path")"

  # If previously configured, strip old block first (for file-append frameworks)
  if [[ "$method" != "dir" ]] && already_configured "$path"; then
    local tmp="${path}.plandb-tmp"
    sed "/${MARKER}/,/${MARKER}/d" "$path" > "$tmp"
    mv "$tmp" "$path"
  fi

  # For "file" method: append to existing file (or create)
  # For "dir" method: create/overwrite file with plandb instructions
  if [[ "$method" == "dir" ]]; then
    {
      echo "$MARKER"
      plandb_instructions
      echo "$MARKER"
    } > "$path"
  else
    # Append to existing file
    {
      echo ""
      echo "$MARKER"
      plandb_instructions
      echo "$MARKER"
    } >> "$path"
  fi

  if [[ "$id" == "aider" ]]; then
    local aider_conf="$HOME/.aider.conf.yml"
    if [[ -f "$aider_conf" ]]; then
      if ! grep -q "aider.conventions.md" "$aider_conf" 2>/dev/null; then
        echo "read: ${HOME}/.aider.conventions.md" >> "$aider_conf"
      fi
    else
      echo "read: ${HOME}/.aider.conventions.md" > "$aider_conf"
    fi
  fi

  # Install Claude Code skill alongside the rules file
  if [[ "$id" == "claude" ]]; then
    install_claude_skill
  fi

  ok "${name}: configured (${path})"
}

# ── Interactive framework selector ──────────────────────────────────

select_frameworks() {
  header "Configure Agent Frameworks"
  echo ""
  echo -e "  PlanDB works with any AI coding agent. Select which frameworks"
  echo -e "  to configure with planning instructions."
  echo ""

  local detected=()
  for entry in "${FRAMEWORKS[@]}"; do
    local id=$(framework_id "$entry")
    local name=$(framework_name "$entry")
    if is_framework_installed "$id"; then
      detected+=("$id")
      echo -e "  ${GREEN}●${RESET} ${BOLD}${name}${RESET} ${DIM}(detected)${RESET}"
    else
      echo -e "  ${DIM}○ ${name}${RESET}"
    fi
  done

  echo ""

  if [[ ${#detected[@]} -eq 0 ]]; then
    warn "No frameworks detected. You can still configure them manually."
    echo ""
  fi

  echo -e "  ${BOLD}Options:${RESET}"
  echo -e "    ${WHITE}a${RESET}  Configure all detected frameworks"
  echo -e "    ${WHITE}A${RESET}  Configure ALL frameworks (even undetected)"
  echo -e "    ${WHITE}n${RESET}  Skip framework configuration"
  echo -e "    ${WHITE}1-${#FRAMEWORKS[@]}${RESET}  Toggle individual frameworks"
  echo ""

  local i=1
  for entry in "${FRAMEWORKS[@]}"; do
    local name=$(framework_name "$entry")
    local path=$(framework_path "$entry")
    local marker=""
    if already_configured "$path" 2>/dev/null; then
      marker=" ${DIM}(will update)${RESET}"
    fi
    echo -e "    ${DIM}${i}.${RESET} ${name}${marker}"
    i=$((i + 1))
  done

  echo ""
  echo -ne "  ${BOLD}Choice${RESET} [a]: "
  if [[ -t 0 ]]; then
    read -r choice
  else
    read -r choice < /dev/tty || choice=""
  fi
  choice="${choice:-a}"

  SELECTED_FRAMEWORKS=()

  case "$choice" in
    a)
      for entry in "${FRAMEWORKS[@]}"; do
        local id=$(framework_id "$entry")
        for d in "${detected[@]}"; do
          if [[ "$d" == "$id" ]]; then
            SELECTED_FRAMEWORKS+=("$entry")
          fi
        done
      done
      ;;
    A)
      SELECTED_FRAMEWORKS=("${FRAMEWORKS[@]}")
      ;;
    n|N)
      info "Skipping framework configuration"
      return 0
      ;;
    *)
      IFS=',' read -ra nums <<< "$choice"
      for num in "${nums[@]}"; do
        num=$(echo "$num" | tr -d ' ')
        if [[ "$num" =~ ^[0-9]+$ ]] && (( num >= 1 && num <= ${#FRAMEWORKS[@]} )); then
          SELECTED_FRAMEWORKS+=("${FRAMEWORKS[$((num - 1))]}")
        fi
      done
      ;;
  esac

  if [[ ${#SELECTED_FRAMEWORKS[@]} -eq 0 ]]; then
    info "No frameworks selected"
    return 0
  fi

  echo ""
  for entry in "${SELECTED_FRAMEWORKS[@]}"; do
    configure_framework "$entry"
  done
}

# ── Banner ──────────────────────────────────────────────────────────

banner() {
  echo ""
  echo -e "${BOLD}  PlanDB${RESET} ${DIM}— the issue tracker for AI agents${RESET}"
  echo -e "  ${DIM}Task graphs with dependencies, decomposition, and knowledge tracking${RESET}"
  echo ""
}

# ── Main ────────────────────────────────────────────────────────────

main() {
  banner

  local mode="interactive"

  for arg in "$@"; do
    case "$arg" in
      --all)         mode="all" ;;
      --binary-only) mode="binary-only" ;;
      --help|-h)
        echo "Usage: install.sh [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --all          Install binary + configure all detected frameworks"
        echo "  --binary-only  Install binary only, skip framework configuration"
        echo "  --help         Show this help"
        echo ""
        echo "Environment:"
        echo "  PLANDB_INSTALL_DIR  Installation directory (default: ~/.local/bin)"
        echo "  PLANDB_VERSION      Version to install (default: latest)"
        echo "  GITHUB_TOKEN        GitHub token for authenticated downloads"
        exit 0
        ;;
    esac
  done

  # Step 1: Install binary
  install_binary

  # Step 2: Configure frameworks
  case "$mode" in
    all)
      header "Configure Agent Frameworks"
      echo ""
      for entry in "${FRAMEWORKS[@]}"; do
        configure_framework "$entry"
      done
      ;;
    binary-only)
      info "Skipping framework configuration (--binary-only)"
      ;;
    interactive)
      select_frameworks
      ;;
  esac

  # Done
  header "Done"
  echo ""
  echo -e "  Get started:"
  echo -e "    ${DIM}\$ plandb init \"my-project\"${RESET}"
  echo -e "    ${DIM}\$ plandb add \"First task\" --description \"What to build\"${RESET}"
  echo -e "    ${DIM}\$ plandb go${RESET}"
  echo ""
  echo -e "  ${DIM}Docs: https://github.com/${REPO}${RESET}"
  echo ""
}

main "$@"

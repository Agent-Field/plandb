#!/usr/bin/env bash
set -euo pipefail

PLANDB="./target/debug/plandb"
DB="/tmp/plandb-functional-test.db"
PASS=0
FAIL=0
ERRORS=""

cleanup() {
  rm -f "$DB" "$DB-wal" "$DB-shm" /tmp/plandb-batch-test.yaml /tmp/plandb-replan-test.yaml /tmp/plandb-template-test.yaml
}

assert_eq() {
  local label="$1" expected="$2" actual="$3"
  if [ "$expected" = "$actual" ]; then
    PASS=$((PASS + 1))
    printf "  \033[32m✓\033[0m %s\n" "$label"
  else
    FAIL=$((FAIL + 1))
    printf "  \033[31m✗\033[0m %s (expected='%s' actual='%s')\n" "$label" "$expected" "$actual"
    ERRORS="$ERRORS\n  ✗ $label (expected='$expected' actual='$actual')"
  fi
}

assert_contains() {
  local label="$1" needle="$2" haystack="$3"
  if echo "$haystack" | grep -q "$needle"; then
    PASS=$((PASS + 1))
    printf "  \033[32m✓\033[0m %s\n" "$label"
  else
    FAIL=$((FAIL + 1))
    printf "  \033[31m✗\033[0m %s (expected to contain '%s')\n" "$label" "$needle"
    ERRORS="$ERRORS\n  ✗ $label (expected to contain '$needle')"
  fi
}

assert_not_empty() {
  local label="$1" value="$2"
  if [ -n "$value" ]; then
    PASS=$((PASS + 1))
    printf "  \033[32m✓\033[0m %s\n" "$label"
  else
    FAIL=$((FAIL + 1))
    printf "  \033[31m✗\033[0m %s (was empty)\n" "$label"
    ERRORS="$ERRORS\n  ✗ $label (was empty)"
  fi
}

assert_regex() {
  local label="$1" pattern="$2" value="$3"
  if [[ "$value" =~ $pattern ]]; then
    PASS=$((PASS + 1))
    printf "  \033[32m✓\033[0m %s\n" "$label"
  else
    FAIL=$((FAIL + 1))
    printf "  \033[31m✗\033[0m %s (value='%s' pattern='%s')\n" "$label" "$value" "$pattern"
    ERRORS="$ERRORS\n  ✗ $label (value='$value' pattern='$pattern')"
  fi
}

jq_field() {
  echo "$1" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('$2','') if d.get('$2') is not None else '')" 2>/dev/null
}

jq_len() {
  echo "$1" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null
}

jq_field_at() {
  echo "$1" | python3 -c "import sys,json; arr=json.load(sys.stdin); print(arr[$2].get('$3','') if arr[$2].get('$3') is not None else '')" 2>/dev/null
}

cleanup

echo ""
echo "=========================================="
echo "  PLANDB FUNCTIONAL TEST SUITE"
echo "=========================================="
echo ""

# ─────────────────────────────────────────────
echo "1. PROJECT CRUD"
echo "─────────────────────────────────────────"

# Create project
OUT=$($PLANDB --db "$DB" --json project create "test-project" --description "A test project")
PROJ_ID=$(jq_field "$OUT" "id")
assert_not_empty "project create returns id" "$PROJ_ID"
assert_regex "project id uses short format" '^p-[a-z0-9]{4}$' "$PROJ_ID"
assert_eq "project name" "test-project" "$(jq_field "$OUT" "name")"
assert_eq "project status" "active" "$(jq_field "$OUT" "status")"
assert_eq "project description" "A test project" "$(jq_field "$OUT" "description")"

# Create second project
OUT2=$($PLANDB --db "$DB" --json project create "second-project")
PROJ2_ID=$(jq_field "$OUT2" "id")
assert_not_empty "second project create returns id" "$PROJ2_ID"

# List projects
LIST=$($PLANDB --db "$DB" --json project list)
LIST_LEN=$(jq_len "$LIST")
assert_eq "project list count" "2" "$LIST_LEN"

# Project status (no tasks yet)
STATUS=$($PLANDB --db "$DB" project status "$PROJ_ID")
assert_contains "project status shows 0 tasks" "total=0" "$STATUS"

# Human-readable project list (non-json)
HR_LIST=$($PLANDB --db "$DB" project list)
assert_contains "human-readable list shows project name" "test-project" "$HR_LIST"

echo ""

# ─────────────────────────────────────────────
echo "2. TASK CREATION & BASIC LIFECYCLE"
echo "─────────────────────────────────────────"

# Create task (no deps → should auto-promote to ready)
T_A=$($PLANDB --db "$DB" --json task create --project "$PROJ_ID" "Task Alpha" --kind code --priority 10 --description "First task")
T_A_ID=$(jq_field "$T_A" "id")
assert_not_empty "task A create returns id" "$T_A_ID"
assert_regex "task id uses short format" '^t-[a-z0-9]{4}$' "$T_A_ID"
assert_eq "task A kind" "code" "$(jq_field "$T_A" "kind")"
assert_eq "task A priority" "10" "$(jq_field "$T_A" "priority")"

# Verify auto-promotion to ready
T_A_GET=$($PLANDB --db "$DB" --json task get "$T_A_ID")
assert_eq "task A auto-promoted to ready" "ready" "$(jq_field "$T_A_GET" "status")"

# Create task with dep (should stay pending)
T_B=$($PLANDB --db "$DB" --json task create --project "$PROJ_ID" "Task Beta" --dep "$T_A_ID")
T_B_ID=$(jq_field "$T_B" "id")
T_B_GET=$($PLANDB --db "$DB" --json task get "$T_B_ID")
assert_eq "task B stays pending (has unmet dep)" "pending" "$(jq_field "$T_B_GET" "status")"

# Task list with filters
READY_TASKS=$($PLANDB --db "$DB" --json task list --project "$PROJ_ID" --status ready)
READY_LEN=$(jq_len "$READY_TASKS")
assert_eq "only 1 ready task (A)" "1" "$READY_LEN"

PENDING_TASKS=$($PLANDB --db "$DB" --json task list --project "$PROJ_ID" --status pending)
PENDING_LEN=$(jq_len "$PENDING_TASKS")
assert_eq "only 1 pending task (B)" "1" "$PENDING_LEN"

# Human-readable task list
HR_TASKS=$($PLANDB --db "$DB" task list --project "$PROJ_ID")
assert_contains "human-readable task list shows Task Alpha" "Task Alpha" "$HR_TASKS"

# Task get human readable
HR_GET=$($PLANDB --db "$DB" task get "$T_A_ID")
assert_contains "human-readable task get shows title" "Task Alpha" "$HR_GET"
assert_contains "human-readable task get shows kind" "code" "$HR_GET"

echo ""

# ─────────────────────────────────────────────
echo "3. CLAIM → START → HEARTBEAT → PROGRESS → DONE"
echo "─────────────────────────────────────────"

# Next task (peek, no claim)
NEXT=$($PLANDB --db "$DB" task next --project "$PROJ_ID" --agent agent-x)
assert_contains "next (peek) shows ready task" "Task Alpha" "$NEXT"

# Next task with claim
CLAIMED=$($PLANDB --db "$DB" --json task next --project "$PROJ_ID" --agent agent-1 --claim)
assert_eq "claimed task is A" "$T_A_ID" "$(jq_field "$CLAIMED" "id")"
assert_eq "claimed status" "claimed" "$(jq_field "$CLAIMED" "status")"
assert_eq "claimed by agent-1" "agent-1" "$(jq_field "$CLAIMED" "agent_id")"

# Double claim should fail (no ready tasks left)
DOUBLE=$($PLANDB --db "$DB" task next --project "$PROJ_ID" --agent agent-2 --claim)
assert_contains "double claim gets no task" "no ready task" "$DOUBLE"

# Claim specific task that's already claimed
CLAIM_AGAIN=$($PLANDB --db "$DB" task claim "$T_A_ID" --agent agent-2)
assert_contains "claim already-claimed task fails" "not claimable" "$CLAIM_AGAIN"

# Start
STARTED=$($PLANDB --db "$DB" --json task start "$T_A_ID")
assert_eq "started status" "running" "$(jq_field "$STARTED" "status")"

# Heartbeat
HB=$($PLANDB --db "$DB" --json task heartbeat "$T_A_ID")
assert_contains "heartbeat succeeds" "1" "$(jq_field "$HB" "updated")"

# Progress
PROG=$($PLANDB --db "$DB" --json task progress "$T_A_ID" --percent 50 --note "halfway done")
assert_contains "progress update succeeds" "1" "$(jq_field "$PROG" "updated")"

# Verify progress on task
T_A_MID=$($PLANDB --db "$DB" --json task get "$T_A_ID")
assert_eq "progress at 50" "50" "$(jq_field "$T_A_MID" "progress")"
assert_eq "progress note" "halfway done" "$(jq_field "$T_A_MID" "progress_note")"

# Done
DONE=$($PLANDB --db "$DB" --json task done "$T_A_ID" --result '{"output": "success"}')
assert_eq "done status" "done" "$(jq_field "$DONE" "status")"
assert_not_empty "done has completed_at" "$(jq_field "$DONE" "completed_at")"

echo ""

# ─────────────────────────────────────────────
echo "4. DEPENDENCY AUTO-PROMOTION"
echo "─────────────────────────────────────────"

# Task B should now be ready
T_B_AFTER=$($PLANDB --db "$DB" --json task get "$T_B_ID")
assert_eq "task B promoted to ready after A done" "ready" "$(jq_field "$T_B_AFTER" "status")"

# Complete B for clean state
$PLANDB --db "$DB" --json task next --project "$PROJ_ID" --agent agent-1 --claim >/dev/null
$PLANDB --db "$DB" --json task start "$T_B_ID" >/dev/null
$PLANDB --db "$DB" --json task done "$T_B_ID" >/dev/null

# Project status should show all done
STATUS_AFTER=$($PLANDB --db "$DB" project status "$PROJ_ID")
assert_contains "project status shows done=2" "done=2" "$STATUS_AFTER"

echo ""

# ─────────────────────────────────────────────
echo "5. FAIL & RETRY"
echo "─────────────────────────────────────────"

# Create task with retries
T_R=$($PLANDB --db "$DB" --json task create --project "$PROJ_ID" "Retry Task" --max-retries 2)
T_R_ID=$(jq_field "$T_R" "id")
T_R_GET=$($PLANDB --db "$DB" --json task get "$T_R_ID")
assert_eq "retry task ready" "ready" "$(jq_field "$T_R_GET" "status")"

# Claim, start, fail
$PLANDB --db "$DB" --json task claim "$T_R_ID" --agent agent-1 >/dev/null
$PLANDB --db "$DB" --json task start "$T_R_ID" >/dev/null
FAILED=$($PLANDB --db "$DB" --json task fail "$T_R_ID" --error "network timeout")
assert_eq "task failed" "failed" "$(jq_field "$FAILED" "status")"
assert_eq "retry count is 1" "1" "$(jq_field "$FAILED" "retry_count")"
assert_eq "error message" "network timeout" "$(jq_field "$FAILED" "error")"

echo ""

# ─────────────────────────────────────────────
echo "6. CANCEL WITH CASCADE"
echo "─────────────────────────────────────────"

# Create parent → child chain
T_P=$($PLANDB --db "$DB" --json task create --project "$PROJ_ID" "Parent Task")
T_P_ID=$(jq_field "$T_P" "id")
T_C1=$($PLANDB --db "$DB" --json task create --project "$PROJ_ID" "Child 1" --dep "$T_P_ID:feeds_into")
T_C1_ID=$(jq_field "$T_C1" "id")
T_C2=$($PLANDB --db "$DB" --json task create --project "$PROJ_ID" "Child 2" --dep "$T_P_ID:feeds_into")
T_C2_ID=$(jq_field "$T_C2" "id")

# Cancel parent with cascade
CANCEL=$($PLANDB --db "$DB" --json task cancel "$T_P_ID" --cascade)
CANCEL_COUNT=$(jq_field "$CANCEL" "cancelled")
assert_eq "cascade cancelled 3 tasks" "3" "$CANCEL_COUNT"

# Verify all cancelled
T_P_CHECK=$($PLANDB --db "$DB" --json task get "$T_P_ID")
T_C1_CHECK=$($PLANDB --db "$DB" --json task get "$T_C1_ID")
T_C2_CHECK=$($PLANDB --db "$DB" --json task get "$T_C2_ID")
assert_eq "parent cancelled" "cancelled" "$(jq_field "$T_P_CHECK" "status")"
assert_eq "child 1 cancelled" "cancelled" "$(jq_field "$T_C1_CHECK" "status")"
assert_eq "child 2 cancelled" "cancelled" "$(jq_field "$T_C2_CHECK" "status")"

# Cancel without cascade (just the one task)
T_SOLO=$($PLANDB --db "$DB" --json task create --project "$PROJ_ID" "Solo Cancel")
T_SOLO_ID=$(jq_field "$T_SOLO" "id")
CANCEL_SOLO=$($PLANDB --db "$DB" --json task cancel "$T_SOLO_ID")
assert_eq "solo cancel cancels 1" "1" "$(jq_field "$CANCEL_SOLO" "cancelled")"

echo ""

# ─────────────────────────────────────────────
echo "7. BATCH YAML IMPORT"
echo "─────────────────────────────────────────"

cat > /tmp/plandb-batch-test.yaml << 'YAML'
tasks:
  - id: batch-design
    title: "Design the API"
    kind: code
    tags:
      - backend
      - api
  - id: batch-impl
    title: "Implement the API"
    kind: code
    deps:
      - from: batch-design
        kind: feeds_into
  - id: batch-test
    title: "Write tests"
    kind: test
    deps:
      - from: batch-impl
        kind: feeds_into
YAML

BATCH=$($PLANDB --db "$DB" --json task create-batch --project "$PROJ_ID" --file /tmp/plandb-batch-test.yaml)
INSERTED=$(jq_field "$BATCH" "inserted")
assert_eq "batch inserted 3 tasks" "3" "$INSERTED"

# Verify DAG structure: design=ready, impl=pending, test=pending
DESIGN=$($PLANDB --db "$DB" --json task get "batch-design")
IMPL=$($PLANDB --db "$DB" --json task get "batch-impl")
TEST=$($PLANDB --db "$DB" --json task get "batch-test")
assert_eq "batch design is ready" "ready" "$(jq_field "$DESIGN" "status")"
assert_eq "batch impl is pending" "pending" "$(jq_field "$IMPL" "status")"
assert_eq "batch test is pending" "pending" "$(jq_field "$TEST" "status")"

# Complete design → impl should promote
$PLANDB --db "$DB" --json task claim "batch-design" --agent agent-1 >/dev/null
$PLANDB --db "$DB" --json task start "batch-design" >/dev/null
$PLANDB --db "$DB" --json task done "batch-design" >/dev/null

IMPL_AFTER=$($PLANDB --db "$DB" --json task get "batch-impl")
TEST_AFTER=$($PLANDB --db "$DB" --json task get "batch-test")
assert_eq "batch impl promoted to ready" "ready" "$(jq_field "$IMPL_AFTER" "status")"
assert_eq "batch test still pending" "pending" "$(jq_field "$TEST_AFTER" "status")"

# Complete impl → test should promote
$PLANDB --db "$DB" --json task claim "batch-impl" --agent agent-2 >/dev/null
$PLANDB --db "$DB" --json task start "batch-impl" >/dev/null
$PLANDB --db "$DB" --json task done "batch-impl" >/dev/null

TEST_FINAL=$($PLANDB --db "$DB" --json task get "batch-test")
assert_eq "batch test promoted to ready" "ready" "$(jq_field "$TEST_FINAL" "status")"

echo ""

# ─────────────────────────────────────────────
echo "8. ARTIFACTS"
echo "─────────────────────────────────────────"

# Write artifact on a completed task
ART=$($PLANDB --db "$DB" --json artifact write --task "$T_A_ID" --name "output.json" --content '{"result": 42}' --kind output --mime "application/json")
ART_ID=$(jq_field "$ART" "id")
assert_not_empty "artifact write returns id" "$ART_ID"
assert_regex "artifact id uses short format" '^a-[a-z0-9]{4}$' "$ART_ID"
assert_eq "artifact name" "output.json" "$(jq_field "$ART" "name")"
assert_eq "artifact kind" "output" "$(jq_field "$ART" "kind")"

# Write second artifact
ART2=$($PLANDB --db "$DB" --json artifact write --task "$T_A_ID" --name "log.txt" --content "some log" --kind log)
ART2_ID=$(jq_field "$ART2" "id")

# List artifacts for task
ART_LIST=$($PLANDB --db "$DB" --json artifact list --task "$T_A_ID")
ART_LIST_LEN=$(jq_len "$ART_LIST")
assert_eq "2 artifacts on task A" "2" "$ART_LIST_LEN"

# Read artifact by name
ART_READ=$($PLANDB --db "$DB" --json artifact read --task "$T_A_ID" --name "output.json")
assert_eq "artifact read name" "output.json" "$(jq_field "$ART_READ" "name")"
assert_contains "artifact read content" "42" "$(jq_field "$ART_READ" "content")"

echo ""

# ─────────────────────────────────────────────
echo "9. EVENTS"
echo "─────────────────────────────────────────"

EVENTS=$($PLANDB --db "$DB" --json events list --project "$PROJ_ID")
EVENTS_LEN=$(jq_len "$EVENTS")
# CLI operations now emit events for all core operations (create, claim, start, done, fail, cancel)
assert_eq "events list returns array" "0" "$(echo "$EVENTS" | python3 -c "import sys,json;d=json.load(sys.stdin);print('0' if isinstance(d,list) else '1')")"
assert_regex "events list has entries" '^[1-9][0-9]*$' "$EVENTS_LEN"

# Human-readable events should show event entries
HR_EVENTS=$($PLANDB --db "$DB" events list --project "$PROJ_ID")
assert_contains "events list shows event type" "task_" "$HR_EVENTS"

echo ""

# ─────────────────────────────────────────────
echo "10. DAG VISUALIZATION"
echo "─────────────────────────────────────────"

DAG=$($PLANDB --db "$DB" project dag "$PROJ_ID")
assert_contains "DAG shows Task Alpha title" "Task Alpha" "$DAG"
assert_contains "DAG shows feeds_into edges" "feeds_into" "$DAG"

# JSON dag
DAG_JSON=$($PLANDB --db "$DB" --json project dag "$PROJ_ID")
assert_contains "JSON DAG has tasks" "tasks" "$DAG_JSON"
assert_contains "JSON DAG has edges" "edges" "$DAG_JSON"

echo ""

# ─────────────────────────────────────────────
echo "11. PARENT-CHILD HIERARCHY"
echo "─────────────────────────────────────────"

T_PARENT=$($PLANDB --db "$DB" --json task create --project "$PROJ_ID" "Epic Task")
T_PARENT_ID=$(jq_field "$T_PARENT" "id")

T_SUB1=$($PLANDB --db "$DB" --json task create --project "$PROJ_ID" "Subtask 1" --parent "$T_PARENT_ID")
T_SUB1_ID=$(jq_field "$T_SUB1" "id")
assert_eq "subtask parent_task_id set" "$T_PARENT_ID" "$(jq_field "$T_SUB1" "parent_task_id")"

T_SUB2=$($PLANDB --db "$DB" --json task create --project "$PROJ_ID" "Subtask 2" --parent "$T_PARENT_ID")
T_SUB2_ID=$(jq_field "$T_SUB2" "id")

echo ""

# ─────────────────────────────────────────────
echo "12. TASK KIND FILTERING"
echo "─────────────────────────────────────────"

# Create tasks of different kinds
$PLANDB --db "$DB" --json task create --project "$PROJ2_ID" "Code task" --kind code >/dev/null
$PLANDB --db "$DB" --json task create --project "$PROJ2_ID" "Test task" --kind test >/dev/null
$PLANDB --db "$DB" --json task create --project "$PROJ2_ID" "Review task" --kind review >/dev/null

CODE_TASKS=$($PLANDB --db "$DB" --json task list --project "$PROJ2_ID" --kind code)
CODE_LEN=$(jq_len "$CODE_TASKS")
assert_eq "filter by kind=code returns 1" "1" "$CODE_LEN"

TEST_TASKS=$($PLANDB --db "$DB" --json task list --project "$PROJ2_ID" --kind test)
TEST_LEN=$(jq_len "$TEST_TASKS")
assert_eq "filter by kind=test returns 1" "1" "$TEST_LEN"

ALL_TASKS=$($PLANDB --db "$DB" --json task list --project "$PROJ2_ID")
ALL_LEN=$(jq_len "$ALL_TASKS")
assert_eq "all tasks in proj2 = 3" "3" "$ALL_LEN"

echo ""

# ─────────────────────────────────────────────
echo "13. APPROVAL WORKFLOW"
echo "─────────────────────────────────────────"

T_APR=$($PLANDB --db "$DB" --json task create --project "$PROJ_ID" "Needs Approval" --requires-approval)
T_APR_ID=$(jq_field "$T_APR" "id")
T_APR_GET=$($PLANDB --db "$DB" --json task get "$T_APR_ID")
assert_eq "requires_approval is true" "True" "$(echo "$T_APR_GET" | python3 -c "import sys,json;print(json.load(sys.stdin)['requires_approval'])")"

# Approve it
APR_RESULT=$($PLANDB --db "$DB" --json task approve "$T_APR_ID" --by "reviewer-1" --comment "looks good")
assert_contains "approve returns updated" "1" "$(jq_field "$APR_RESULT" "updated")"

# Verify approval fields
T_APR_AFTER=$($PLANDB --db "$DB" --json task get "$T_APR_ID")
assert_eq "approval_status set" "approved" "$(jq_field "$T_APR_AFTER" "approval_status")"
assert_eq "approved_by set" "reviewer-1" "$(jq_field "$T_APR_AFTER" "approved_by")"

echo ""

# ─────────────────────────────────────────────
echo "14. EDGE CASES"
echo "─────────────────────────────────────────"

# Invalid progress (out of range)
BAD_PROG=$($PLANDB --db "$DB" task progress "$T_APR_ID" --percent 150 2>&1 || true)
assert_contains "invalid progress rejected" "must be between" "$BAD_PROG"

# Get non-existent task
BAD_GET=$($PLANDB --db "$DB" --json task get "t-zzzzzz" 2>&1 || true)
assert_contains "non-existent task errors" "error" "$BAD_GET"

# Double complete (already done task)
BAD_DONE=$($PLANDB --db "$DB" task done "$T_A_ID" 2>&1 || true)
# Should error since task A is already done
assert_contains "double-done errors" "error" "$BAD_DONE"

echo ""

# ─────────────────────────────────────────────
echo "15. STICKY PROJECT + STATUS"
echo "─────────────────────────────────────────"

USE_SHOW=$($PLANDB --db "$DB" use)
assert_contains "use shows current project" "$PROJ2_ID" "$USE_SHOW"

$PLANDB --db "$DB" use "$PROJ_ID" >/dev/null
USE_SHOW2=$($PLANDB --db "$DB" use)
assert_contains "use sets explicit default" "$PROJ_ID" "$USE_SHOW2"

STATUS_LINE=$($PLANDB --db "$DB" status)
assert_contains "status summary includes project id" "$PROJ_ID" "$STATUS_LINE"
assert_contains "status summary includes done ratio" "done" "$STATUS_LINE"

STATUS_DETAIL=$($PLANDB --db "$DB" status --detail)
assert_contains "status detail includes task rows" "Task Alpha" "$STATUS_DETAIL"

echo ""

# ─────────────────────────────────────────────
echo "16. GO + NOTES + DONE FILES"
echo "─────────────────────────────────────────"

GO_PROJECT=$($PLANDB --db "$DB" --json project create "go-project")
GO_PROJECT_ID=$(jq_field "$GO_PROJECT" "id")

GO_TASK=$($PLANDB --db "$DB" --json task create --project "$GO_PROJECT_ID" "Go Candidate")
GO_TASK_ID=$(jq_field "$GO_TASK" "id")

GO_OUT=$($PLANDB --db "$DB" --json task go --agent test-agent --project "$GO_PROJECT_ID")
GO_ID=$(echo "$GO_OUT" | python3 -c "import sys,json; d=json.load(sys.stdin); t=d.get('task'); print('' if t is None else t.get('id',''))")
assert_eq "task go starts expected task" "$GO_TASK_ID" "$GO_ID"
assert_contains "task go reports running" "running" "$GO_OUT"

NOTE_OUT=$($PLANDB --db "$DB" --json task note "$GO_TASK_ID" "test note" --agent test-agent)
assert_eq "task note content saved" "test note" "$(jq_field "$NOTE_OUT" "content")"

NOTES_OUT=$($PLANDB --db "$DB" --json task notes "$GO_TASK_ID")
NOTES_LEN=$(jq_len "$NOTES_OUT")
assert_eq "task notes returns one note" "1" "$NOTES_LEN"

DONE_NEXT=$($PLANDB --db "$DB" --json task done "$GO_TASK_ID" --result "done" --files src/main.rs,src/lib.rs)
assert_eq "done marks task complete" "done" "$(jq_field "$DONE_NEXT" "status")"

echo ""

# ─────────────────────────────────────────────
echo "17. PAUSE + REPLAN"
echo "─────────────────────────────────────────"

PAUSE_TASK=$($PLANDB --db "$DB" --json task create --project "$PROJ_ID" "Pause Task")
PAUSE_TASK_ID=$(jq_field "$PAUSE_TASK" "id")
$PLANDB --db "$DB" --json task claim "$PAUSE_TASK_ID" --agent pauser >/dev/null
$PLANDB --db "$DB" --json task start "$PAUSE_TASK_ID" >/dev/null
PAUSE_OUT=$($PLANDB --db "$DB" --json task pause "$PAUSE_TASK_ID" --progress 60 --note "handoff")
assert_eq "pause returns task to ready" "ready" "$(jq_field "$PAUSE_OUT" "status")"
assert_contains "pause metadata keeps previous agent" "pauser" "$PAUSE_OUT"

REPLAN_PARENT=$($PLANDB --db "$DB" --json task create --project "$PROJ_ID" "Replan Parent")
REPLAN_PARENT_ID=$(jq_field "$REPLAN_PARENT" "id")
cat > /tmp/plandb-replan-test.yaml << 'YAML'
subtasks:
  - title: "New Subtask A"
  - title: "New Subtask B"
    deps_on:
      - "New Subtask A"
YAML

REPLAN_OUT=$($PLANDB --db "$DB" --json task replan "$REPLAN_PARENT_ID" --file /tmp/plandb-replan-test.yaml)
assert_eq "replan created two subtasks" "2" "$(jq_field "$REPLAN_OUT" "subtasks_created")"

echo ""

# ─────────────────────────────────────────────
echo "18. VERSION & HELP"
echo "─────────────────────────────────────────"

VERSION=$($PLANDB --version)
assert_contains "version shows plandb" "plandb" "$VERSION"

HELP=$($PLANDB --help)
assert_contains "help shows project" "project" "$HELP"
assert_contains "help shows task" "task" "$HELP"
assert_contains "help shows mcp" "mcp" "$HELP"
assert_contains "help shows serve" "serve" "$HELP"

echo ""

# ─────────────────────────────────────────────
echo "19. JIT ADAPTIVE PLANNING PRIMITIVES"
echo "─────────────────────────────────────────"

JIT_PROJECT=$($PLANDB --db "$DB" --json project create "jit-project")
JIT_PROJECT_ID=$(jq_field "$JIT_PROJECT" "id")

JIT_A=$($PLANDB --db "$DB" --json task create --project "$JIT_PROJECT_ID" "JIT A")
JIT_A_ID=$(jq_field "$JIT_A" "id")
JIT_B=$($PLANDB --db "$DB" --json task create --project "$JIT_PROJECT_ID" "JIT B" --dep "$JIT_A_ID")
JIT_B_ID=$(jq_field "$JIT_B" "id")
JIT_C=$($PLANDB --db "$DB" --json task create --project "$JIT_PROJECT_ID" "JIT C" --dep "$JIT_B_ID")
JIT_C_ID=$(jq_field "$JIT_C" "id")

WHATIF_CANCEL=$($PLANDB --db "$DB" --json what-if cancel "$JIT_B_ID" 2>&1 || true)
assert_contains "what-if cancel includes effect field" '"effect"' "$WHATIF_CANCEL"
assert_contains "what-if cancel includes project_state" '"project_state"' "$WHATIF_CANCEL"

WHATIF_INSERT=$($PLANDB --db "$DB" --json what-if insert --after "$JIT_A_ID" --before "$JIT_B_ID" --title "JIT Mid" 2>&1 || true)
assert_contains "what-if insert includes effect field" '"effect"' "$WHATIF_INSERT"
assert_contains "what-if insert includes project_state" '"project_state"' "$WHATIF_INSERT"

INSERT_OUT=$($PLANDB --db "$DB" --json task insert --project "$JIT_PROJECT_ID" --after "$JIT_A_ID" --before "$JIT_B_ID" --title "Inserted Between" --description "bridge step" 2>&1 || true)
INSERT_ID=$(jq_field "$INSERT_OUT" "id" || true)
assert_not_empty "task insert returns created id" "$INSERT_ID"
assert_contains "task insert response has effect" '"effect"' "$INSERT_OUT"
assert_contains "task insert response has project_state" '"project_state"' "$INSERT_OUT"

$PLANDB --db "$DB" --json task claim "$JIT_A_ID" --agent jit-agent >/dev/null
$PLANDB --db "$DB" --json task start "$JIT_A_ID" >/dev/null
LOOKAHEAD=$($PLANDB --db "$DB" --json ahead --project "$JIT_PROJECT_ID" --depth 2 2>&1 || true)
assert_contains "ahead includes current" '"current"' "$LOOKAHEAD"
assert_contains "ahead includes upcoming" '"upcoming"' "$LOOKAHEAD"
assert_contains "ahead includes blocked_by" '"blocked_by"' "$LOOKAHEAD"

AMEND_OUT=$($PLANDB --db "$DB" --json task amend "$JIT_B_ID" --prepend "NOTE: learned context" 2>&1 || true)
assert_contains "task amend response has updated task" '"id"' "$AMEND_OUT"
JIT_B_AFTER_AMEND=$($PLANDB --db "$DB" --json task get "$JIT_B_ID")
assert_contains "task amend prepends text" "NOTE: learned context" "$(jq_field "$JIT_B_AFTER_AMEND" "description")"

PIVOT_PARENT=$($PLANDB --db "$DB" --json task create --project "$JIT_PROJECT_ID" "Pivot Parent")
PIVOT_PARENT_ID=$(jq_field "$PIVOT_PARENT" "id")
PIVOT_DONE=$($PLANDB --db "$DB" --json task create --project "$JIT_PROJECT_ID" "Pivot Done Child" --parent "$PIVOT_PARENT_ID")
PIVOT_DONE_ID=$(jq_field "$PIVOT_DONE" "id")
$PLANDB --db "$DB" --json task claim "$PIVOT_DONE_ID" --agent jit-agent >/dev/null
$PLANDB --db "$DB" --json task start "$PIVOT_DONE_ID" >/dev/null
$PLANDB --db "$DB" --json task done "$PIVOT_DONE_ID" >/dev/null
PIVOT_PENDING=$($PLANDB --db "$DB" --json task create --project "$JIT_PROJECT_ID" "Pivot Pending Child" --parent "$PIVOT_PARENT_ID")
PIVOT_PENDING_ID=$(jq_field "$PIVOT_PENDING" "id")
PIVOT_OUT=$($PLANDB --db "$DB" --json task pivot "$PIVOT_PARENT_ID" --keep-done --subtasks '[{"title":"Pivot New A"},{"title":"Pivot New B","deps_on":["Pivot New A"]}]' 2>&1 || true)
assert_contains "task pivot response has kept list" '"kept"' "$PIVOT_OUT"
assert_contains "task pivot response has cancelled list" '"cancelled"' "$PIVOT_OUT"
assert_contains "task pivot response has created list" '"created"' "$PIVOT_OUT"
assert_contains "task pivot response has effect" '"effect"' "$PIVOT_OUT"

SPLIT_PARENT=$($PLANDB --db "$DB" --json task create --project "$JIT_PROJECT_ID" "Split Parent")
SPLIT_PARENT_ID=$(jq_field "$SPLIT_PARENT" "id")
SPLIT_OUT=$($PLANDB --db "$DB" --json task split "$SPLIT_PARENT_ID" --into '[{"title":"Split A","done":true,"result":"already done"},{"title":"Split B","deps_on":["Split A"]}]' 2>&1 || true)
assert_contains "task split response has created list" '"created"' "$SPLIT_OUT"
assert_contains "task split response has done list" '"done"' "$SPLIT_OUT"
assert_contains "task split response has effect" '"effect"' "$SPLIT_OUT"

echo ""

# ─────────────────────────────────────────────
echo "20. PRE/POST CONDITIONS"
echo "─────────────────────────────────────────"

PRE_POST_TASK=$($PLANDB --db "$DB" --json task create --project "$PROJ_ID" "Conditioned Task" --pre "upstream must have schema" --post "all endpoints return JSON" --description "test pre/post")
PRE_POST_ID=$(jq_field "$PRE_POST_TASK" "id")
assert_not_empty "pre/post task created" "$PRE_POST_ID"
assert_eq "pre_condition set" "upstream must have schema" "$(jq_field "$PRE_POST_TASK" "pre_condition")"
assert_eq "post_condition set" "all endpoints return JSON" "$(jq_field "$PRE_POST_TASK" "post_condition")"

# Verify get returns conditions
PRE_POST_GET=$($PLANDB --db "$DB" --json task get "$PRE_POST_ID")
assert_eq "get returns pre_condition" "upstream must have schema" "$(jq_field "$PRE_POST_GET" "pre_condition")"
assert_eq "get returns post_condition" "all endpoints return JSON" "$(jq_field "$PRE_POST_GET" "post_condition")"

# Task without conditions should have null
PLAIN_TASK=$($PLANDB --db "$DB" --json task create --project "$PROJ_ID" "No Conditions")
assert_eq "no pre_condition is empty" "" "$(jq_field "$PLAIN_TASK" "pre_condition")"

echo ""

# ─────────────────────────────────────────────
echo "21. CRITICAL PATH & BOTTLENECKS"
echo "─────────────────────────────────────────"

# Use JIT project which has a chain: JIT A → Inserted Between → JIT B → JIT C
CP_OUT=$($PLANDB --db "$DB" --json critical-path --project "$JIT_PROJECT_ID" 2>&1 || true)
# May return path or null depending on task states — just verify it runs without error
assert_contains "critical-path runs" "path" "$CP_OUT"

# Bottlenecks
BN_OUT=$($PLANDB --db "$DB" --json bottlenecks --project "$JIT_PROJECT_ID" --limit 3 2>&1 || true)
# Should return an array with task_id fields
assert_contains "bottlenecks runs" "task_id" "$BN_OUT"

# What-unlocks on a specific task
WU_OUT=$($PLANDB --db "$DB" --json what-unlocks "$JIT_A_ID" 2>&1 || true)
assert_contains "what-unlocks runs" "task_id" "$WU_OUT"

echo ""

# ─────────────────────────────────────────────
echo "22. EXPORT/IMPORT TEMPLATES"
echo "─────────────────────────────────────────"

# Create a small project for export test
TMPL_PROJ=$($PLANDB --db "$DB" --json project create "template-test")
TMPL_PROJ_ID=$(jq_field "$TMPL_PROJ" "id")
$PLANDB --db "$DB" --json task create --project "$TMPL_PROJ_ID" "Step A" --as step-a >/dev/null
$PLANDB --db "$DB" --json task create --project "$TMPL_PROJ_ID" "Step B" --dep t-step-a >/dev/null

# Export
EXPORT_OUT=$($PLANDB --db "$DB" export --project "$TMPL_PROJ_ID")
assert_contains "export contains template name" "template-test" "$EXPORT_OUT"
assert_contains "export contains Step A" "Step A" "$EXPORT_OUT"
assert_contains "export contains Step B" "Step B" "$EXPORT_OUT"
assert_contains "export contains dependencies" "feeds_into" "$EXPORT_OUT"

# Write to file and import into a new project
TMPL_FILE="/tmp/plandb-template-test.yaml"
echo "$EXPORT_OUT" > "$TMPL_FILE"

IMPORT_PROJ=$($PLANDB --db "$DB" --json project create "import-target")
IMPORT_PROJ_ID=$(jq_field "$IMPORT_PROJ" "id")
IMPORT_OUT=$($PLANDB --db "$DB" --json import "$TMPL_FILE" --project "$IMPORT_PROJ_ID")
IMPORT_COUNT=$(jq_field "$IMPORT_OUT" "imported")
assert_eq "imported 2 tasks" "2" "$IMPORT_COUNT"

# Verify imported tasks exist
IMPORT_TASKS=$($PLANDB --db "$DB" --json task list --project "$IMPORT_PROJ_ID")
IMPORT_TASK_COUNT=$(jq_len "$IMPORT_TASKS")
assert_eq "imported project has 2 tasks" "2" "$IMPORT_TASK_COUNT"

rm -f "$TMPL_FILE"

echo ""

# ─────────────────────────────────────────────
# SUMMARY
echo "=========================================="
echo "  RESULTS"
echo "=========================================="
echo ""
printf "  \033[32m%d passed\033[0m, \033[31m%d failed\033[0m\n" "$PASS" "$FAIL"
echo ""

if [ "$FAIL" -gt 0 ]; then
  echo "  FAILURES:"
  printf "$ERRORS\n"
  echo ""
fi

cleanup

if [ "$FAIL" -gt 0 ]; then
  exit 1
fi

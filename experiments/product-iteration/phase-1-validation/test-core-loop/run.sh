#!/usr/bin/env bash
# Test: Can an AI agent (Codex) use PlanDB's core loop effectively?
# This test gives Codex a simple multi-step task and PlanDB instructions,
# then captures how it interacts with PlanDB.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
DB="$DIR/.plandb.db"
LOG="$DIR/agent-log.txt"
RESULT="$DIR/result.json"

# Clean previous runs
rm -f "$DB" "$DB-wal" "$DB-shm" "$LOG" "$RESULT"

echo "=== PlanDB Core Loop Test ==="
echo "Testing: Can an AI agent use init → add → go → done --next?"
echo ""

# The task: Create a simple Python fibonacci module with tests
TASK_PROMPT='You have plandb installed for task planning. Use it to decompose work and track progress.

IMPORTANT: Use plandb for ALL task management. Do NOT use any other task system.

Database path: '"$DB"'

Core loop:    plandb --db '"$DB"' go → work → plandb --db '"$DB"' done --next
Add tasks:    plandb --db '"$DB"' add "title" --description "detailed spec" --dep t-xxx
Status:       plandb --db '"$DB"' status --detail

TASK: Create a Python project in this directory with:
1. A fibonacci module (fibonacci.py) with iterative and recursive implementations
2. Unit tests (test_fibonacci.py) using pytest
3. A README.md explaining usage

Start by running: plandb --db '"$DB"' init "fibonacci-project"
Then decompose the work into tasks with dependencies.
Execute each task using the plandb go/done loop.
After each done, check status and continue until all tasks complete.

CRITICAL: Every plandb command must include --db '"$DB"'
Do NOT create files outside this directory: '"$DIR"'
Work ONLY in '"$DIR"''

echo "Launching Codex agent..."
echo "Prompt:" > "$LOG"
echo "$TASK_PROMPT" >> "$LOG"
echo "" >> "$LOG"
echo "=== Agent Output ===" >> "$LOG"

# Launch codex with the task
codex exec \
  --full-auto \
  --skip-git-repo-check \
  -C "$DIR" \
  "$TASK_PROMPT" \
  2>&1 | tee -a "$LOG"

echo ""
echo "=== Post-Run Analysis ==="

# Capture final PlanDB state
echo ""
echo "--- PlanDB Status ---"
plandb --db "$DB" --json status --detail 2>/dev/null || echo "No project found"

echo ""
echo "--- Task List ---"
plandb --db "$DB" --json list 2>/dev/null || echo "No tasks"

echo ""
echo "--- Files Created ---"
find "$DIR" -name "*.py" -o -name "*.md" | grep -v __pycache__ | sort

# Generate result summary
python3 -c "
import json, os, subprocess

db = '$DB'
result = {
    'test': 'core-loop',
    'agent': 'codex',
    'success': False,
    'findings': []
}

# Check if project was created
try:
    status = subprocess.run(['plandb', '--db', db, '--json', 'status'], capture_output=True, text=True)
    if status.returncode == 0:
        result['plandb_status'] = json.loads(status.stdout)
        result['findings'].append('Project created successfully')
    else:
        result['findings'].append('FAIL: Project not created')
except Exception as e:
    result['findings'].append(f'FAIL: Could not check status: {e}')

# Check if tasks were created
try:
    tasks = subprocess.run(['plandb', '--db', db, '--json', 'list'], capture_output=True, text=True)
    if tasks.returncode == 0:
        task_list = json.loads(tasks.stdout)
        result['task_count'] = len(task_list)
        done_count = sum(1 for t in task_list if t.get('status') == 'done')
        result['done_count'] = done_count
        result['findings'].append(f'Tasks created: {len(task_list)}, completed: {done_count}')
    else:
        result['findings'].append('FAIL: Could not list tasks')
except Exception as e:
    result['findings'].append(f'FAIL: {e}')

# Check deliverables
for f in ['fibonacci.py', 'test_fibonacci.py', 'README.md']:
    path = os.path.join('$DIR', f)
    if os.path.exists(path):
        result['findings'].append(f'Deliverable created: {f}')
    else:
        result['findings'].append(f'MISSING: {f}')

# Overall success
result['success'] = result.get('task_count', 0) >= 3 and result.get('done_count', 0) >= 3

print(json.dumps(result, indent=2))
" > "$RESULT"

echo ""
echo "--- Result ---"
cat "$RESULT"

# PlanDB Product Iteration — Findings

## Phase 1: Existing Feature Validation

### Test 1: Core Loop (Codex)

**Date**: 2026-03-25
**Agent**: Codex (codex-cli 0.116.0)
**Status**: PASS (with observations)

**What happened:**
- Codex initialized project, created 3 tasks, executed go/done loop — all 3 tasks completed
- Created fibonacci.py, test_fibonacci.py, README.md — all correct
- Used 40,945 tokens total
- Event log: task_created x3 → task_claimed/started/completed x3 (sequential)

**Observations:**
1. **No dependencies created** — all 3 tasks were independent (no `--dep` flags used)
2. **No decomposition** — didn't use `split` or subtasks
3. **Sequential execution** — didn't consider parallelization
4. **Good task descriptions** — descriptive enough for self-contained execution
5. **Custom IDs not used** — auto-generated IDs only

**Root cause analysis:**
The prompt didn't create sufficient pressure to use dependencies. For a simple 3-task project, flat independent tasks are actually rational — Codex optimized for simplicity. Dependencies become essential with more complex work.

**Action items:**
- [ ] Test with a more complex task that *requires* dependencies
- [ ] Evaluate if prompt should enforce `--dep` usage or if it's correctly optional

### Test 2: Decomposition (Gemini)

**Date**: 2026-03-25
**Agent**: Gemini CLI (0.32.1)
**Status**: IN PROGRESS (12 tasks created, 2/12 done)

**What happened:**
- Gemini created a well-structured graph with 12 tasks
- Used custom IDs (`--as design-layout`, `--as base-template`, etc.)
- Created proper dependencies (design → implement chains)
- Used `split` for decomposition (Hero Section, Feature Section)
- Multiple tasks ready for parallel execution

**Observations:**
1. **Excellent graph structure** — dependencies model real workflow constraints
2. **Used split correctly** — decomposed complex tasks into subtasks
3. **Good naming** — custom IDs are human-readable
4. **Parallel-ready** — multiple independent design tasks ready simultaneously

### Test 3: Edge Cases (Manual)

**Date**: 2026-03-25
**Status**: PASS (with 1 issue)

**Findings:**
1. PASS: Init without description works
2. PASS: Add task without flags works
3. **ISSUE**: Non-existent dep gives raw SQL error: "FOREIGN KEY constraint failed" instead of friendly message
4. PASS: Go correctly claims first ready task
5. PASS: Done correctly completes and reports unlocked tasks
6. PASS: Done --next reports "no ready task" when nothing available
7. PASS: Split without running task gives clear error
8. PASS: Status shows correct progress

**Action items:**
- [ ] Fix: Friendly error for non-existent dependency (translate FK error)
- [ ] Consider: `plandb add --dep t-nonexistent` should say "task t-nonexistent not found"

### Test 4: Feature Completeness (Manual)

**Date**: 2026-03-25
**Status**: All features working

Validated: init, add, go, done, split, status --detail, status --full, critical-path, bottlenecks, what-unlocks, task insert, task amend, export. All working correctly.

### Test 5: Prompt Quality Analysis

**Current `plandb prompt --for cli` stats:**
- 199 lines, 9,585 characters (~2,400 tokens)
- Comprehensive but possibly too long for token-constrained agents

**Prompt issues identified:**
1. The prompt doesn't create urgency around using dependencies — agents can ignore them
2. No examples of dependency chains in the quick-start section
3. Missing: explicit instruction to `plandb status --detail` after task creation to verify graph
4. The `--db` flag pattern is cumbersome — agents must remember to include it on every command
5. Split syntax (`--into "A, B, C"` vs `--into "A > B > C"`) not prominent enough

**Proposed prompt improvements:**
- Lead with a concrete 4-task example showing dependencies
- Emphasize `--dep` in the first "add" example
- Add "verify your graph" step after task creation
- Consider making PLANDB_DB env var more prominent to avoid --db repetition

---

## Key Meta-Findings

### 1. Agents optimize for simplicity
When the task doesn't clearly require dependencies, agents skip them. The prompt must demonstrate WHY dependencies matter (ordering, parallelization signals) rather than just HOW to use them.

### 2. Different agents, different behaviors
- **Codex**: Efficient, minimal — uses the simplest PlanDB features that get the job done
- **Gemini**: More thorough — creates richer task graphs with custom IDs and dependencies
- Both correctly used the go/done loop

### 3. The `--db` flag is friction
Every command needing `--db` is error-prone and token-wasteful. The PLANDB_DB env var should be the primary recommendation, not a footnote.

### 4. Event log is valuable for debugging
The events list shows exactly how the agent interacted with PlanDB — essential for diagnosis.

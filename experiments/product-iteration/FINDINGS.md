# PlanDB Product Iteration — Findings

## Phase 1: Existing Feature Validation

### Test 1: Core Loop (Codex)
**Agent**: Codex 0.116.0 | **Status**: PASS | **Tokens**: 40,945

Codex successfully used plandb go/done loop, completing 3/3 tasks. However, it created no dependencies between tasks — all were independent. Root cause: the prompt didn't demonstrate dependencies as essential.

### Test 2: Decomposition (Gemini)
**Agent**: Gemini CLI 0.32.1 | **Status**: PARTIAL PASS (2/12 complete)

Gemini created an excellent 12-task graph with custom IDs, proper dependencies, and used split for decomposition. Only completed 2 tasks before stopping (likely context/timeout).

### Test 3: Edge Cases (Manual)
**Status**: PASS with 1 bug found

Bug: `plandb add --dep t-nonexistent` created the task first, then failed on FK constraint, leaving an orphaned task. **Fixed**: now validates deps before creating task with friendly error message.

### Test 4: Feature Completeness (Manual)
All features validated: init, add, go, done, split, status, critical-path, bottlenecks, what-unlocks, insert, amend, export/import.

## Phase 2: Prompt Optimization

### A/B Test: Original vs Improved Prompt
| Metric | Original (199 lines) | Improved (110 lines) |
|--------|---------------------|---------------------|
| Token cost | ~2,400 tokens | ~1,200 tokens |
| Dependencies used | No | Yes (full graph) |
| Custom IDs | No | Yes |
| Task completion | 3/3 | 4/4 |
| Dependency ordering | N/A | Correct |

**Key insight**: Example-first prompts produce better agent behavior than documentation-heavy prompts. Leading with a concrete dependency example made Codex create proper dependency graphs.

## Phase 3: New Features

### Context Store with BM25 Search

**Agent validation (Codex)**:
- Created 6 context entries while building a URL shortener
- Types used: discovery (3), decision (2), pattern (1)
- Auto-linking to running task worked correctly
- Used context to record research findings, design decisions, environment constraints

**Feature summary**:
- `plandb context` — freeform typed context entries on the project graph
- `plandb search` — BM25 search via SQLite FTS5 (zero extra deps)
- Auto-link to current running task (no --task needed)
- Freeform --kind (not a fixed enum — agent decides the taxonomy)

## Meta-Findings

1. **The prompt IS the product** — agents' behavior is shaped entirely by the prompt quality
2. **Test with real agents, not humans** — agent behavior differs from human expectations
3. **Smaller prompts > larger prompts** — 51% reduction produced better results
4. **Freeform > fixed taxonomies** — agents are better judges of what types they need
5. **Auto-detection > explicit flags** — auto-linking to running task reduces friction

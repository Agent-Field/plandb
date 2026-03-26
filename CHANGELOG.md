# Changelog

All notable changes to PlanDB are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: [SemVer](https://semver.org/).

---

## [0.2.0] — 2026-03-26

The "issue tracker for AI agents" release. Four platform features, production CI/CD, and a complete repositioning.

### Added
- **Context store with BM25 search** — `plandb context "..." --kind discovery` records project knowledge. `plandb search "query"` does BM25-ranked search across context entries and task descriptions. Freeform `--kind` labels. Available on CLI, MCP, and HTTP.
- **Lazy recall** — `plandb go` auto-surfaces relevant context entries for the claimed task. Agents get knowledge without explicitly searching.
- **Task lifecycle hooks** — `--pre-hook` and `--post-hook` shell commands on tasks. Fire at state transitions with env vars (`PLANDB_TASK_ID`, `PLANDB_TASK_TITLE`, etc.). Advisory — failures warn, never block.
- **Contextual action hints** — every CLI response teaches the agent what to do next. Kind-aware: research tasks hint "record discoveries", code tasks hint "split if complex".
- **Agent-scale diagram** (`assets/agent-scale.png`) — panoramic visualization of decomposition, parallelization, and mid-flight adaptation.
- **CI workflow** (`ci.yml`) — runs on every push/PR: build, test, functional tests, clippy, fmt.
- **Cargo caching** in CI for faster builds.

### Changed
- **CLI prompt redesigned** — 51% smaller (199 → 110 lines, ~2400 → ~1200 tokens). Example-first instead of documentation-heavy. Agents now create proper dependency graphs (validated with Codex and Gemini).
- **Release workflow** — releases now trigger on version tags (`v*`) only, not every push to main. Release names use "PlanDB" (was "Plandb").
- **README repositioned** — "The issue tracker your AI agents are missing. Think Linear or Jira — but for your agents." Mapping table, thought leadership section, showcase moved up.
- **All prompts synced** — CLI, MCP, HTTP, and `--help` all reference context store, lazy recall, and hooks.
- **Branch protection** enabled on main: require CI pass, no force push, no branch deletion.

### Fixed
- `plandb add --dep t-nonexistent` no longer creates orphaned tasks. Validates all dependency targets upfront with fuzzy-match suggestions.
- `batch_create_tasks` now correctly passes `pre_hook`/`post_hook` parameters (was passing 33 params, needed 35).
- Context `--kind` default corrected to "discovery" (was "note").
- Help text references fixed from `--type` to `--kind`.
- Removed references to `plandb resume` and `plandb templates` from prompts (commands don't exist yet).

### Removed
- 10 stale branches cleaned up.
- Template references removed from all agent-facing prompts (feature not yet shipped).

---

## [0.1.x] — 2026-03-04 to 2026-03-25

Foundation releases. Core task graph, compound graph (containment + dependencies), recursive decomposition, mid-flight adaptation primitives, multi-agent coordination, quality gates, CLI/MCP/HTTP interfaces, cross-platform binary builds.

See [GitHub releases](https://github.com/Agent-Field/plandb/releases) for per-patch details.

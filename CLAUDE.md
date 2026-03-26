# PlanDB — Project Instructions

## What This Is

PlanDB is "the issue tracker your AI agents are missing" — a task graph database (CLI + MCP + HTTP) that agents use to decompose work, parallelize, adapt mid-flight, and persist knowledge. Single Rust binary, SQLite-backed.

## Development

```bash
cargo build            # build
cargo test             # unit tests (8 tests)
bash tests/functional_test.sh  # functional tests (7 scenarios)
cargo clippy           # lints
cargo fmt --check      # formatting
```

Rust 1.75+ required. No external dependencies — SQLite is bundled via rusqlite.

## Branch Rules

- `main` is protected: CI must pass, no force push, no branch deletion
- All changes go through PRs (admin can bypass for hotfixes)
- Branch naming: `feat/`, `fix/`, `docs/`, `refactor/`, `ci/`
- Delete branches after merge

## Commit Conventions

```
feat: new user-facing capability
fix: bug fix
refactor: code change that doesn't add features or fix bugs
docs: documentation only
ci: CI/CD changes
test: adding or fixing tests
chore: maintenance (deps, config, cleanup)
```

These prefixes drive automatic PR labeling via release-drafter.

## Release Flow

Releases are intentional, not automatic. Every push to main runs CI but does NOT create a release.

### How it works

```
PR merges to main
  → ci.yml runs (build, test, functional tests)
  → release-drafter updates a DRAFT release on GitHub (accumulates changes)

Ready to ship:
  → git tag v0.X.Y -m "description"
  → git push origin v0.X.Y
  → release.yml triggers:
      1. Runs full test suite
      2. Builds cross-platform binaries (linux x86/arm, macOS x86/arm, windows)
      3. Generates checksums
      4. Publishes GitHub release with auto-generated notes from the draft
```

### Version bumping

- **Patch** (v0.2.0 → v0.2.1): bug fixes, docs, refactors
- **Minor** (v0.2.x → v0.3.0): new features
- **Major** (v0.x → v1.0): breaking changes

The release-drafter auto-suggests the next version based on PR labels (`feat:` → minor, `fix:` → patch).

### Checklist before tagging a release

1. All PRs for this release are merged
2. CI is green on main
3. CHANGELOG.md updated with the new version section
4. `git tag vX.Y.Z -m "PlanDB vX.Y.Z — summary"` + `git push origin vX.Y.Z`

## CI/CD Workflows

| Workflow | Trigger | What it does |
|----------|---------|-------------|
| `ci.yml` | Push to main, PRs | Build, unit tests, functional tests, clippy, fmt |
| `release.yml` | Push `v*` tag | Cross-platform build + GitHub release |
| `release-drafter.yml` | Push to main, PRs | Auto-labels PRs, maintains draft release notes |

## Project Structure

```
src/
  cli/           # CLI commands (task/, mod.rs)
  db/            # Database layer (tasks.rs, schema.rs, learnings.rs, templates.rs)
  mcp/           # MCP server (tools.rs)
  server/        # HTTP API (routes.rs)
  models/        # Data models (task.rs)
  main.rs        # Entry point + prompt generation
tests/
  functional_test.sh  # End-to-end CLI tests
examples/        # Agent prompt examples
experiments/     # Real agent session outputs
assets/          # Images for README
docs/            # Architecture docs
```

## Three Interfaces — Keep in Sync

PlanDB has three interfaces that must stay consistent:

1. **CLI** (`src/cli/`) — `plandb <command>`
2. **MCP** (`src/mcp/tools.rs`) — `plandb mcp` for Claude Code / Cursor / Windsurf
3. **HTTP** (`src/server/routes.rs`) — `plandb serve`

When adding a feature, implement it in all three. When updating prompts (`src/main.rs`), update all three prompt functions: `print_prompt_cli`, `print_prompt_mcp`, `print_prompt_http`.

## Prompt Is Product

The agent-facing prompts in `src/main.rs` are the primary product surface. Changes to prompts should be:
- Tested with real agents (Codex, Gemini, Claude Code)
- Kept compact — every token costs the agent context window
- Example-first, not documentation-heavy
- Only reference features that actually exist in the current build

## Key Design Decisions

- **Agent-first**: the primary user is an AI agent, not a human
- **Zero infra**: single binary, SQLite, no Docker/cloud/config
- **Freeform over enums**: `--kind` is a freeform string, not a fixed enum
- **Lazy over explicit**: auto-link context to running task, auto-recall on `plandb go`
- **Advisory hooks**: `--pre-hook`/`--post-hook` warn on failure but never block
- **Compound graph**: containment (hierarchy) + dependencies (flow) are orthogonal

# Contributing to PlanDB

Thanks for your interest in contributing. Here's how to get started.

## Development Setup

```bash
git clone https://github.com/Agent-Field/plandb.git
cd plandb
cargo build
cargo test
bash tests/functional_test.sh
```

Requires Rust 1.75+. No other dependencies — SQLite is bundled.

## Making Changes

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Run `cargo test` and `bash tests/functional_test.sh`
4. Open a PR against `main`

CI must pass before merge. Branch protection requires the `test` status check.

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new command
fix: handle edge case in batch import
refactor: simplify context store lookup
docs: update README
ci: add clippy check
test: add functional test for hooks
```

## What to Work On

- Issues labeled `good first issue` are a great starting point
- Check the [CHANGELOG](CHANGELOG.md) for context on recent changes
- Run `plandb --help` to understand the full command surface

## Code Style

- `cargo fmt` for formatting
- `cargo clippy` for lints
- Keep functions under 50 lines where possible
- Prefer explicit error handling over `.unwrap()`

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

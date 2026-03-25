# Example: Codex builds a Fibonacci module using PlanDB

A minimal example showing PlanDB's core loop: `init → add → go → done --next`.

## Prompt

```
You have plandb for task planning.
plandb init "fibonacci" → plandb add "title" --description "spec" → plandb go → plandb done --next

TASK: Create a Python fibonacci module with iterative and recursive implementations, tests, and README.
```

## What Happened

The agent created 3 tasks and executed them sequentially:

```
fibonacci: 3/3 done (100%)
✓ t-2x8t  Implement fibonacci module
✓ t-gqng  Document usage
✓ t-hvq4  Write pytest coverage
```

**Tokens used**: 40,945

## Files Created

- [`fibonacci.py`](fibonacci.py) — Iterative and recursive Fibonacci with negative input validation
- [`test_fibonacci.py`](test_fibonacci.py) — pytest parametrized tests
- [`README.md`](README.md) — Usage documentation

## Reproduce

```bash
cd /tmp/fibonacci-test
codex exec --full-auto "$(plandb prompt --for cli)
TASK: Create a Python fibonacci module with iterative and recursive implementations, tests, and README."
```

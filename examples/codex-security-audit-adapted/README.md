# Example: Codex runs a security audit with mid-flight adaptation

This is the definitive example of **guided autonomy** — an agent follows a template
pattern while dynamically adapting when it discovers gaps.

## Key Moment: Mid-Flight Task Insertion

During the summary task, the agent realized it had missed reviewing hardcoded secrets.
Instead of ignoring the gap, it used `plandb task insert` to add a new review step:

```
result: {"inserted_task": "t-20dc", "summary": {"critical": 1, "high": 2, "medium": 1}}
```

The graph evolved from 5 tasks to 6 mid-execution — the agent self-corrected.

## Task Graph (6 tasks, all done)

```
t-9vef "Review entrypoints" ──┬──▶ t-q8sn "Review /exec"      ──┐
                               ├──▶ t-l354 "Review /file"      ──┤
                               ├──▶ t-qmev "Review /search"    ──┼──▶ t-km7n "Summarize"
                               └──▶ t-20dc "Review secrets" ────┘
                                     (INSERTED mid-flight!)
```

## Findings (structured results)

| Task | Severity | Finding |
|------|----------|---------|
| Review /exec | Critical | Unauthenticated RCE via `shell=True` |
| Review /file | High | Arbitrary file read via unvalidated path param |
| Review /search | High | SQL injection via f-string query |
| Review secrets | Medium | Hardcoded secret `SECRET` in source |

## Context Entries (6 total)

```
c-ttdw [finding] Critical RCE in app.py:10-12 — subprocess.check_output with shell=True
c-emyo [finding] High-severity arbitrary file read in app.py:16-18
c-qfoh [finding] High-severity SQL injection in app.py:22-24
c-xvrb [finding] Medium hardcoded secret in app.py:6
c-yqvy [discovery] Route inventory: 3 unauthenticated routes
```

## What This Proves

1. **Template patterns work** — agent naturally creates fan-out review structure
2. **Mid-flight adaptation works** — agent inserts missed steps during execution
3. **Structured handoff works** — JSON results flow between tasks
4. **Context store works** — findings documented with severity and location
5. **Self-correction works** — agent notices gaps and fills them autonomously

## Reproduce

```bash
cd /tmp/audit-test && git init
# Create app.py with vulnerabilities (see app.py in this directory)
codex exec --full-auto --skip-git-repo-check "$(plandb prompt --for cli)
You MUST:
1. plandb init 'security-audit'
2. plandb import templates/security-audit.yaml
3. Work through ALL tasks. Record findings with plandb context --kind finding
4. If you discover gaps, use plandb task insert"
```

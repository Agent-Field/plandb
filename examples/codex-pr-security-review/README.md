# Example: Codex security reviews a Flask app using PlanDB

Codex was given a Flask app with intentional vulnerabilities and asked to review it.
The agent found **all 5 security issues** and documented them as typed context entries.

## What the Agent Found

The agent created 2 review tasks and recorded 5 findings:

```
c-nf62 [finding] SQL injection — f-string interpolation in login query (app.py:12)
c-lyvg [finding] Plaintext password comparison — no hashing (app.py:12)
c-k2u3 [finding] Hardcoded token — same token for every user (app.py:15)
c-ovve [finding] No auth on /users — unauthenticated access to all user data (app.py:18-23)
c-iwhp [finding] Data exposure — SELECT * returns raw rows including passwords (app.py:22-23)
```

## Key Observation: Guided Autonomy

The prompt suggested importing the PR review template, but the agent chose a simpler
2-task structure that was better suited for reviewing a single file. This is guided
autonomy — the agent adapts the approach while using PlanDB's core features
(context store with typed entries, task decomposition, structured results).

## Reproduce

```bash
cd /tmp/security-review-test && git init
# Create app.py with vulnerabilities (see examples/codex-pr-security-review/app.py)
codex exec --full-auto --skip-git-repo-check "$(plandb prompt --for cli)
TASK: Review app.py for security vulnerabilities. Record all findings with plandb context."
```

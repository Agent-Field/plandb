# Example: Template Transfer — Security audit on a different codebase

Proves templates are reusable: the comprehensive security audit template (from Flask app #1)
was imported and used to audit a completely different e-commerce Flask API.

## Template Compounding

| Generation | Tasks | Findings | Context | Source |
|-----------|-------|----------|---------|--------|
| Original (hand-seeded) | 7 | 0 | 5 patterns | templates/security-audit.yaml |
| 1st evolution (Flask app #1) | 10 | 20 | 27 entries | templates/security-audit-comprehensive.yaml |
| **2nd evolution (e-commerce)** | **14** | **39** | **50 entries** | **templates/security-audit-transfer-evolved.yaml** |

Each generation is MORE valuable — carries real findings from previous audits as "clues."

## Key Observations

1. **Template transferred successfully** — the 10-task template from a different app worked on the e-commerce API
2. **Agent adapted further** — added 4 new tasks (business logic, deduplication, severity ranking, remediation)
3. **Multi-agent parallelism** — two reviewers ran simultaneously (@default and @logic-reviewer)
4. **39 findings** on a 75-line app — more thorough than the original 20-finding audit
5. **50 context entries** — the richest template yet, carrying 2 generations of findings

## E-commerce App Vulnerabilities Found

The target `server.py` had: SQL injection (product search, orders), XSS (reviews),
weak JWT (HS256 + hardcoded secret), IDOR (order listing), XXE (XML import),
negative quantity fraud, unauthenticated admin endpoints, CORS misconfiguration.

## Reproduce

```bash
cd /tmp/transfer-test && git init
# Copy server.py and templates/ from this example
codex exec --full-auto --skip-git-repo-check "$(plandb prompt --for cli)
plandb init 'audit' && plandb import templates/security-audit-comprehensive.yaml
Work through ALL tasks. Record findings with plandb context --kind finding."
```

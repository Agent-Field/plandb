# Example: Comprehensive Security Audit (120-line Flask App, 20 Findings)

The most thorough PlanDB test — Codex audited a realistic Flask application with
11 routes and found **20 security vulnerabilities** across 7 OWASP categories.

## Results

| Severity | Count | Examples |
|----------|-------|---------|
| Critical | 6 | RCE via pickle.loads, yaml.load, shell=True; SQLi in auth |
| High | 12 | XSS/SSTI, IDOR, broken access control, path traversal, CSRF |
| Medium | 2 | CVE in Flask dependency, predictable API key generation |

## Task Graph (10 tasks, adapted from 7-task template)

```
t-lg1g "Enumerate attack surface"
  ├──▶ t-s1ir "Input validation audit"        ──┐
  ├──▶ t-p2vr "Auth and access control"       ──┤
  ├──▶ t-v3oy "Secrets and configuration"     ──┤
  ├──▶ t-dhdd "Dependency audit"              ──┼──▶ t-yg0b "Compile" ──▶ t-j9kb "Report"
  ├──▶ t-m8st "Serialization/config" (ADDED)  ──┤
  ├──▶ t-3jfp "File handling/exec" (ADDED)    ──┤
  └──▶ t-ws9f "Session/CSRF" (ADDED)          ──┘
```

The agent imported the 7-task security audit template, then **added 3 domain-specific
tasks** (serialization, file handling, session management) based on what it discovered
during the attack surface enumeration. This is guided autonomy at work.

## Template Learning

The exported template (27 context entries) carries ALL findings as institutional knowledge:

```bash
plandb import templates/security-audit-comprehensive.yaml
plandb go  # → lazy recall surfaces findings from THIS audit
```

## OWASP Coverage

| Category | Findings |
|----------|----------|
| A01: Broken Access Control | 4 (admin users, IDOR, CSRF, file write) |
| A02: Cryptographic Failures | 3 (MD5 passwords, weak token, predictable keys) |
| A03: Injection | 6 (SQLi x4, XSS/SSTI, command injection) |
| A05: Security Misconfiguration | 5 (debug mode, hardcoded secret, path traversal, config mutation, fallback key) |
| A06: Vulnerable Components | 1 (Flask CVE) |
| A07: Auth Failures | 1 (forgeable non-expiring token) |
| A08: Data Integrity | 2 (pickle RCE, yaml RCE) |

## Reproduce

```bash
cd /tmp/audit-test && git init
# Copy app.py from this directory
cp templates/ .  # copy templates directory
codex exec --full-auto --skip-git-repo-check "$(plandb prompt --for cli)
plandb init 'audit' && plandb import templates/security-audit.yaml
Work through ALL tasks. Record findings with plandb context --kind finding."
```

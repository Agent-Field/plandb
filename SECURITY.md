# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in PlanDB, please report it responsibly.

**Email:** security@agentfield.ai

Please include:
- Description of the vulnerability
- Steps to reproduce
- Impact assessment

We will acknowledge reports within 48 hours and aim to release a fix within 7 days for critical issues.

## Scope

PlanDB runs as a local binary with a SQLite database. The primary attack surfaces are:

- **HTTP server mode** (`plandb serve`) — exposed network endpoints
- **MCP server mode** (`plandb mcp`) — tool call handling
- **Shell hooks** (`--pre-hook`, `--post-hook`) — arbitrary command execution by design
- **YAML import** (`plandb import`) — deserialization of untrusted input
- **Template URLs** — fetching and parsing remote YAML files

## Supported Versions

Only the latest release receives security updates.

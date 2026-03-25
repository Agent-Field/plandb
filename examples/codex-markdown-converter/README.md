# Example: Clean E2E — Just the prompt, nothing else

The simplest possible test: paste `plandb prompt --for cli` + a task. No special
instructions, no template import, no forcing. Just the prompt.

**Result**: Agent created 3 tasks with dependencies, 4 context entries, completed 100%.

## Prompt (verbatim)

```
<output of plandb prompt --for cli>

TASK: Build a Python markdown-to-HTML converter with support for headers, bold, italic, and links.
```

## What Happened

```
t-design ──▶ t-implement ──▶ t-test    (3/3 done, 100%)
```

Context recorded:
- [decision] Small package layout: markdown_converter/converter.py
- [decision] Inline parsing via recursive scanner, not regex-only
- [discovery] Implementation complete: convert_markdown() renders headers, bold, italic, links
- [discovery] Verification: unittest suite covers all features

## Why This Matters

This proves the prompt alone is sufficient. No human intervention, no template import,
no special instructions. The agent auto-discovers PlanDB features from the 120-line prompt.

## Reproduce

```bash
cd /tmp/test && git init
codex exec --full-auto --skip-git-repo-check "$(plandb prompt --for cli)
TASK: Build a Python markdown-to-HTML converter."
```

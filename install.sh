#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# PlanDB installer — downloads pre-built binary + configures agent frameworks
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/Agent-Field/plandb/main/install.sh | bash
#   curl -fsSL ... | bash -s -- --all          # silent: install + configure all detected
#   curl -fsSL ... | bash -s -- --binary-only  # just the binary, no framework config
# ─────────────────────────────────────────────────────────────────────

REPO="Agent-Field/plandb"
BINARY="plandb"
INSTALL_DIR="${PLANDB_INSTALL_DIR:-$HOME/.local/bin}"
PLANDB_VERSION="${PLANDB_VERSION:-latest}"

# ── Colors ──────────────────────────────────────────────────────────

if [[ -t 1 ]]; then
  BOLD="\033[1m"
  DIM="\033[2m"
  RESET="\033[0m"
  GREEN="\033[32m"
  CYAN="\033[36m"
  YELLOW="\033[33m"
  RED="\033[31m"
  WHITE="\033[37m"
  CHECK="${GREEN}✓${RESET}"
  CROSS="${RED}✗${RESET}"
  ARROW="${CYAN}→${RESET}"
else
  BOLD="" DIM="" RESET="" GREEN="" CYAN="" YELLOW="" RED="" WHITE=""
  CHECK="[ok]" CROSS="[!!]" ARROW="->"
fi

# ── Utility ─────────────────────────────────────────────────────────

info()  { echo -e "  ${ARROW} $1"; }
ok()    { echo -e "  ${CHECK} $1"; }
warn()  { echo -e "  ${YELLOW}!${RESET} $1"; }
fail()  { echo -e "  ${CROSS} $1"; exit 1; }
header() { echo -e "\n${BOLD}$1${RESET}"; }

# ── Detect platform ─────────────────────────────────────────────────

detect_platform() {
  local os arch
  os="$(uname -s)"
  arch="$(uname -m)"

  case "$os" in
    Linux*)  OS="linux" ;;
    Darwin*) OS="darwin" ;;
    *)       fail "Unsupported OS: $os — pre-built binaries available for Linux and macOS" ;;
  esac

  case "$arch" in
    x86_64|amd64)  ARCH="x86_64" ;;
    arm64|aarch64) ARCH="aarch64" ;;
    *)             fail "Unsupported architecture: $arch — pre-built binaries available for x86_64 and aarch64" ;;
  esac
}

# ── Install binary ──────────────────────────────────────────────────

install_binary() {
  header "Installing PlanDB"

  detect_platform
  info "Platform: ${OS}/${ARCH}"

  # Resolve release URL
  local release_url
  if [[ "$PLANDB_VERSION" == "latest" ]]; then
    release_url="https://api.github.com/repos/${REPO}/releases/latest"
  else
    release_url="https://api.github.com/repos/${REPO}/releases/tags/${PLANDB_VERSION}"
  fi

  local asset_name="plandb-${OS}-${ARCH}"

  # Build auth header if token is available
  local gh_token="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
  if [[ -z "$gh_token" ]] && command -v gh &>/dev/null; then
    gh_token=$(gh auth token 2>/dev/null || true)
  fi

  # Download binary — try plain curl first (works for public repos),
  # fall back to gh CLI (handles private repos with auth)
  local downloaded=false
  local tmpfile
  tmpfile=$(mktemp)

  # Try curl first (fast, no auth needed for public repos)
  local curl_auth=()
  if [[ -n "$gh_token" ]]; then
    curl_auth=(-H "Authorization: token ${gh_token}")
  fi

  local download_url=""
  download_url=$(curl -fsSL "${curl_auth[@]}" "$release_url" 2>/dev/null \
    | grep -o "\"browser_download_url\": *\"[^\"]*${asset_name}[^\"]*\"" \
    | head -1 \
    | sed 's/.*"browser_download_url": *"\(.*\)"/\1/' || true)

  if [[ -n "$download_url" ]]; then
    info "Downloading ${asset_name}..."
    curl -fsSL "${curl_auth[@]}" "$download_url" -o "$tmpfile"
    downloaded=true
  fi

  # Fall back to gh CLI for private repos
  if [[ "$downloaded" != "true" ]] && command -v gh &>/dev/null && gh auth status &>/dev/null; then
    info "Downloading ${asset_name} via gh CLI..."
    local release_tag="latest"
    if [[ "$PLANDB_VERSION" != "latest" ]]; then
      release_tag="$PLANDB_VERSION"
    fi
    local tmpdir
    tmpdir=$(mktemp -d)
    if gh release download "$release_tag" \
        --repo "${REPO}" \
        --pattern "${asset_name}" \
        --dir "$tmpdir" \
        --clobber 2>/dev/null; then
      mv "$tmpdir/${asset_name}" "$tmpfile"
      downloaded=true
    fi
    rm -rf "$tmpdir"
  fi

  if [[ "$downloaded" != "true" ]]; then
    rm -f "$tmpfile"

    # Download failed — check if plandb is already installed
    local existing=""
    if command -v plandb &>/dev/null; then
      existing=$(plandb --version 2>/dev/null || true)
    elif [[ -x "${INSTALL_DIR}/${BINARY}" ]]; then
      existing=$("${INSTALL_DIR}/${BINARY}" --version 2>/dev/null || true)
    fi

    if [[ -n "$existing" ]]; then
      ok "plandb already installed (${existing}) — using existing binary"
      info "To upgrade, run again when a new release is published"
    else
      warn "No pre-built binary found and plandb is not installed."
      info "Build from source:"
      echo -e "    ${DIM}git clone https://github.com/${REPO}.git && cd plandb${RESET}"
      echo -e "    ${DIM}cargo build --release${RESET}"
      echo -e "    ${DIM}cp target/release/plandb ${INSTALL_DIR}/plandb${RESET}"
      echo ""
      info "Continuing with framework configuration..."
    fi
  else
    chmod +x "$tmpfile"

    # Verify checksum if available
    local checksums_url=""
    checksums_url=$(curl -fsSL "${curl_auth[@]:-}" "$release_url" 2>/dev/null \
      | grep -o "\"browser_download_url\": *\"[^\"]*checksums[^\"]*\"" \
      | head -1 \
      | sed 's/.*"browser_download_url": *"\(.*\)"/\1/' || true)

    if [[ -n "$checksums_url" ]]; then
      local expected_hash
      expected_hash=$(curl -fsSL "${curl_auth[@]:-}" "$checksums_url" 2>/dev/null \
        | grep "${asset_name}$" | awk '{print $1}' || true)

      if [[ -n "$expected_hash" ]]; then
        local actual_hash=""
        if command -v sha256sum &>/dev/null; then
          actual_hash=$(sha256sum "$tmpfile" | awk '{print $1}')
        elif command -v shasum &>/dev/null; then
          actual_hash=$(shasum -a 256 "$tmpfile" | awk '{print $1}')
        fi

        if [[ -n "$actual_hash" && "$actual_hash" != "$expected_hash" ]]; then
          rm -f "$tmpfile"
          fail "Checksum mismatch — expected ${expected_hash}, got ${actual_hash}"
        fi
        [[ -n "$actual_hash" ]] && ok "Checksum verified"
      fi
    fi

    # Install
    mkdir -p "$INSTALL_DIR"
    mv "$tmpfile" "${INSTALL_DIR}/${BINARY}"
    chmod +x "${INSTALL_DIR}/${BINARY}"

    ok "Installed to ${INSTALL_DIR}/${BINARY}"
  fi

  # Verify (if binary exists anywhere)
  if command -v plandb &>/dev/null; then
    ok "plandb $(plandb --version 2>/dev/null || echo 'installed')"
  elif [[ -x "${INSTALL_DIR}/${BINARY}" ]]; then
    ok "plandb $("${INSTALL_DIR}/${BINARY}" --version 2>/dev/null || echo 'installed')"
  fi

  # Check PATH
  if ! echo "$PATH" | tr ':' '\n' | grep -qx "$INSTALL_DIR"; then
    warn "${INSTALL_DIR} is not in PATH"
    info "Add to your shell profile:"
    echo -e "    ${DIM}export PATH=\"${INSTALL_DIR}:\$PATH\"${RESET}"
  fi
}

# ── Agent instructions content ──────────────────────────────────────

plandb_instructions() {
  cat << 'PLANDB_BLOCK'
# PlanDB — Task Planning for Agents

<plandb-philosophy>
You have `plandb` installed. It is your task graph — use it to decompose every non-trivial task into a dependency graph, track progress, and coordinate parallel work.

**You MUST use PlanDB for any task with more than 2 steps.** Do not rely on mental tracking, inline checklists, or ad-hoc notes. PlanDB is the single source of truth for what needs to be done, what's blocked, what's ready, and what's complete.

PlanDB is a compound graph with two orthogonal structures:
- **Containment** (place graph): tasks contain subtasks recursively, to any depth
- **Dependencies** (link graph): edges between tasks at ANY level, crossing containment boundaries

This is more general than a flat list or a simple DAG. A subtask at depth 3 can depend on a task in a completely different branch.
</plandb-philosophy>

<plandb-core-loop>
## Core Loop

Every task follows this cycle. No exceptions.

```bash
plandb init "project-name"                            # 1. Create project (once)
plandb add "title" --description "spec" --dep t-xxx   # 2. Build the task graph
plandb go                                             # 3. Claim next ready task
# ... do the work ...
plandb done --next                                    # 4. Complete + claim next
plandb status --detail                                # 5. Reassess after each task
```

**Description is mandatory.** Every task's `--description` must be a self-contained work order: what to do, which files to touch, acceptance criteria. The title is a label — the description is the spec.
</plandb-core-loop>

<plandb-decomposition>
## Decomposition

Break tasks down aggressively. Small tasks complete faster and unlock parallelism.

```bash
# Split into independent subtasks (creates parallelism)
plandb split --into "A, B, C"

# Split with dependency chain (sequential)
plandb split --into "A > B > C"

# Scope into a composite task to add deeper subtasks
plandb use t-xxx
plandb add "sub-subtask" --description "..."
plandb use ..                                         # Zoom back out
```

Composite tasks auto-complete when all children finish. Split further at any depth.
</plandb-decomposition>

<plandb-parallelism>
## Parallel Execution

**When `plandb list --status ready` returns multiple tasks, run them concurrently.** This is where PlanDB creates the most value — it tells you exactly which tasks are independent and safe to parallelize.

```bash
plandb list --status ready                            # See what can run NOW
plandb what-unlocks t-xxx                             # What opens up when this completes
plandb ahead --depth 3                                # Preview next 3 layers of work
```

If you have access to sub-agents, spawn one per ready task:
1. Run `plandb list --status ready` to find independent tasks
2. Each agent claims its task: `plandb go --agent <agent-name>`
3. Each agent completes: `plandb done --next --agent <agent-name>`
4. Atomic claiming prevents conflicts — two agents cannot claim the same task

Set `PLANDB_AGENT=<name>` to avoid passing `--agent` on every command.
</plandb-parallelism>

<plandb-adaptation>
## Mid-Flight Adaptation

Plans are hypotheses. Adapt as you learn — don't abandon PlanDB when reality diverges from the plan.

| Situation | Command |
|-----------|---------|
| Missed a step | `plandb task insert --after t-a --before t-b --title "..."` |
| Task too large | `plandb split --into "A, B, C"` |
| New info for a future task | `plandb task amend t-xxx --prepend "NOTE: ..."` |
| Need to replace a subtree | `plandb task pivot t-xxx --file new-tasks.yaml` |
| Unsure about cancelling | `plandb what-if cancel t-xxx` |
</plandb-adaptation>

<plandb-introspection>
## Introspection

Use these to decide what to work on and where effort is wasted.

```bash
plandb status --detail                                # Per-task breakdown with status
plandb status --full                                  # Compound graph: containment + deps
plandb critical-path                                  # Longest chain — prioritize this
plandb bottlenecks                                    # Tasks blocking the most downstream work
plandb watch                                          # Live-updating dashboard
```
</plandb-introspection>

<plandb-knowledge>
## Knowledge Store

Record discoveries as you work. Context persists across sessions and auto-surfaces when relevant.

```bash
plandb context "what you learned" --kind discovery    # Record project knowledge
plandb search "query"                                 # BM25 search across context + tasks
plandb contexts                                       # List all context entries
```

`--kind` is freeform: `discovery`, `decision`, `pattern`, `blocker`, `reference`, `constraint`, `insight` — use whatever fits. Context auto-links to the running task and auto-surfaces on `plandb go` (lazy recall).
</plandb-knowledge>

<plandb-rules>
## Rules

1. **Use PlanDB for every non-trivial task.** If it has more than 2 steps, decompose it into a graph.
2. **Description is the spec.** `--description` must contain everything needed to execute: context, approach, files, acceptance criteria.
3. **Decompose aggressively.** Smaller tasks = more parallelism = faster completion.
4. **Parallelize ready tasks.** When `plandb list --status ready` returns multiple tasks, run them concurrently — with sub-agents if available.
5. **Adapt, don't abandon.** When reality changes, use `insert`, `split`, `amend`, `pivot`. Don't stop using PlanDB.
6. **Record what you learn.** Use `plandb context` for discoveries, blockers, patterns. It auto-surfaces later.
7. **Check the critical path.** `plandb critical-path` shows what actually determines completion time. Focus there.
8. **Dependencies flow data.** `feeds_into` = data flows, `blocks` = ordering only, `suggests` = soft/optional.
9. **Kinds are freeform.** Common: `code`, `research`, `review`, `test`, `shell`, `generic`.
10. **Status is automatic.** `pending` → `ready` (deps done) → `claimed` → `running` → `done`/`failed`. Don't manage state manually.
</plandb-rules>
PLANDB_BLOCK
}

# ── Framework configuration ─────────────────────────────────────────

FRAMEWORKS=(
  "claude|Claude Code|${HOME}/.claude/rules/plandb.md|dir"
  "cursor|Cursor|CURSOR_SETTINGS|settings"
  "codex|Codex (OpenAI)|${HOME}/.codex/AGENTS.override.md|file"
  "gemini|Gemini CLI|${HOME}/.gemini/GEMINI.md|file"
  "opencode|OpenCode|${HOME}/.config/opencode/AGENTS.md|file"
  "windsurf|Windsurf|${HOME}/.codeium/windsurf/memories/global_rules.md|file"
  "aider|Aider|${HOME}/.aider.conventions.md|file"
)

MARKER="<!-- plandb-agent-instructions -->"

framework_id()   { echo "$1" | cut -d'|' -f1; }
framework_name() { echo "$1" | cut -d'|' -f2; }
framework_path() { echo "$1" | cut -d'|' -f3; }
framework_method() { echo "$1" | cut -d'|' -f4; }

is_framework_installed() {
  local id="$1"
  case "$id" in
    claude)   [[ -d "$HOME/.claude" ]] ;;
    cursor)   command -v cursor &>/dev/null || [[ -d "$HOME/.cursor" ]] || [[ -d "$HOME/Library/Application Support/Cursor" ]] ;;
    codex)    command -v codex &>/dev/null || [[ -d "$HOME/.codex" ]] ;;
    gemini)   command -v gemini &>/dev/null || [[ -d "$HOME/.gemini" ]] ;;
    opencode) command -v opencode &>/dev/null || [[ -d "$HOME/.config/opencode" ]] ;;
    windsurf) [[ -d "$HOME/.codeium" ]] || [[ -d "$HOME/Library/Application Support/Windsurf" ]] ;;
    aider)    command -v aider &>/dev/null ;;
    *)        return 1 ;;
  esac
}

already_configured() {
  local path="$1"
  [[ -f "$path" ]] && grep -q "$MARKER" "$path" 2>/dev/null
}

configure_framework() {
  local entry="$1"
  local id=$(framework_id "$entry")
  local name=$(framework_name "$entry")
  local path=$(framework_path "$entry")
  local method=$(framework_method "$entry")

  if [[ "$method" == "settings" ]]; then
    warn "${name}: global rules are in Settings UI, not a file"
    info "Paste the plandb instructions into: Cursor → Settings → General → Rules for AI"
    info "Or add to a project: .cursor/rules/plandb.mdc"
    return 0
  fi

  # Create parent directories
  mkdir -p "$(dirname "$path")"

  # If previously configured, strip old block first (for file-append frameworks)
  if [[ "$method" != "dir" ]] && already_configured "$path"; then
    local tmp="${path}.plandb-tmp"
    sed "/${MARKER}/,/${MARKER}/d" "$path" > "$tmp"
    mv "$tmp" "$path"
  fi

  # For "file" method: append to existing file (or create)
  # For "dir" method: create/overwrite file with plandb instructions
  if [[ "$method" == "dir" ]]; then
    {
      echo "$MARKER"
      plandb_instructions
      echo "$MARKER"
    } > "$path"
  else
    # Append to existing file
    {
      echo ""
      echo "$MARKER"
      plandb_instructions
      echo "$MARKER"
    } >> "$path"
  fi

  if [[ "$id" == "aider" ]]; then
    local aider_conf="$HOME/.aider.conf.yml"
    if [[ -f "$aider_conf" ]]; then
      if ! grep -q "aider.conventions.md" "$aider_conf" 2>/dev/null; then
        echo "read: ${HOME}/.aider.conventions.md" >> "$aider_conf"
      fi
    else
      echo "read: ${HOME}/.aider.conventions.md" > "$aider_conf"
    fi
  fi

  ok "${name}: configured (${path})"
}

# ── Interactive framework selector ──────────────────────────────────

select_frameworks() {
  header "Configure Agent Frameworks"
  echo ""
  echo -e "  PlanDB works with any AI coding agent. Select which frameworks"
  echo -e "  to configure with planning instructions."
  echo ""

  local detected=()
  for entry in "${FRAMEWORKS[@]}"; do
    local id=$(framework_id "$entry")
    local name=$(framework_name "$entry")
    if is_framework_installed "$id"; then
      detected+=("$id")
      echo -e "  ${GREEN}●${RESET} ${BOLD}${name}${RESET} ${DIM}(detected)${RESET}"
    else
      echo -e "  ${DIM}○ ${name}${RESET}"
    fi
  done

  echo ""

  if [[ ${#detected[@]} -eq 0 ]]; then
    warn "No frameworks detected. You can still configure them manually."
    echo ""
  fi

  echo -e "  ${BOLD}Options:${RESET}"
  echo -e "    ${WHITE}a${RESET}  Configure all detected frameworks"
  echo -e "    ${WHITE}A${RESET}  Configure ALL frameworks (even undetected)"
  echo -e "    ${WHITE}n${RESET}  Skip framework configuration"
  echo -e "    ${WHITE}1-${#FRAMEWORKS[@]}${RESET}  Toggle individual frameworks"
  echo ""

  local i=1
  for entry in "${FRAMEWORKS[@]}"; do
    local name=$(framework_name "$entry")
    local path=$(framework_path "$entry")
    local marker=""
    if already_configured "$path" 2>/dev/null; then
      marker=" ${DIM}(already configured)${RESET}"
    fi
    echo -e "    ${DIM}${i}.${RESET} ${name}${marker}"
    i=$((i + 1))
  done

  echo ""
  echo -ne "  ${BOLD}Choice${RESET} [a]: "
  if [[ -t 0 ]]; then
    read -r choice
  else
    read -r choice < /dev/tty || choice=""
  fi
  choice="${choice:-a}"

  SELECTED_FRAMEWORKS=()

  case "$choice" in
    a)
      for entry in "${FRAMEWORKS[@]}"; do
        local id=$(framework_id "$entry")
        for d in "${detected[@]}"; do
          if [[ "$d" == "$id" ]]; then
            SELECTED_FRAMEWORKS+=("$entry")
          fi
        done
      done
      ;;
    A)
      SELECTED_FRAMEWORKS=("${FRAMEWORKS[@]}")
      ;;
    n|N)
      info "Skipping framework configuration"
      return 0
      ;;
    *)
      IFS=',' read -ra nums <<< "$choice"
      for num in "${nums[@]}"; do
        num=$(echo "$num" | tr -d ' ')
        if [[ "$num" =~ ^[0-9]+$ ]] && (( num >= 1 && num <= ${#FRAMEWORKS[@]} )); then
          SELECTED_FRAMEWORKS+=("${FRAMEWORKS[$((num - 1))]}")
        fi
      done
      ;;
  esac

  if [[ ${#SELECTED_FRAMEWORKS[@]} -eq 0 ]]; then
    info "No frameworks selected"
    return 0
  fi

  echo ""
  for entry in "${SELECTED_FRAMEWORKS[@]}"; do
    configure_framework "$entry"
  done
}

# ── Banner ──────────────────────────────────────────────────────────

banner() {
  echo ""
  echo -e "${BOLD}  PlanDB${RESET} ${DIM}— the issue tracker for AI agents${RESET}"
  echo -e "  ${DIM}Task graphs with dependencies, decomposition, and knowledge tracking${RESET}"
  echo ""
}

# ── Main ────────────────────────────────────────────────────────────

main() {
  banner

  local mode="interactive"

  for arg in "$@"; do
    case "$arg" in
      --all)         mode="all" ;;
      --binary-only) mode="binary-only" ;;
      --help|-h)
        echo "Usage: install.sh [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --all          Install binary + configure all detected frameworks"
        echo "  --binary-only  Install binary only, skip framework configuration"
        echo "  --help         Show this help"
        echo ""
        echo "Environment:"
        echo "  PLANDB_INSTALL_DIR  Installation directory (default: ~/.local/bin)"
        echo "  PLANDB_VERSION      Version to install (default: latest)"
        echo "  GITHUB_TOKEN        GitHub token for authenticated downloads"
        exit 0
        ;;
    esac
  done

  # Step 1: Install binary
  install_binary

  # Step 2: Configure frameworks
  case "$mode" in
    all)
      header "Configure Agent Frameworks"
      echo ""
      for entry in "${FRAMEWORKS[@]}"; do
        configure_framework "$entry"
      done
      ;;
    binary-only)
      info "Skipping framework configuration (--binary-only)"
      ;;
    interactive)
      select_frameworks
      ;;
  esac

  # Done
  header "Done"
  echo ""
  echo -e "  Get started:"
  echo -e "    ${DIM}\$ plandb init \"my-project\"${RESET}"
  echo -e "    ${DIM}\$ plandb add \"First task\" --description \"What to build\"${RESET}"
  echo -e "    ${DIM}\$ plandb go${RESET}"
  echo ""
  echo -e "  ${DIM}Docs: https://github.com/${REPO}${RESET}"
  echo ""
}

main "$@"

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
INSTALL_DIR="${PLANDB_INSTALL_DIR:-/usr/local/bin}"
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

  # Download binary
  local downloaded=false
  local tmpfile
  tmpfile=$(mktemp)

  if command -v gh &>/dev/null && gh auth status &>/dev/null; then
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
  fi

  if [[ "$downloaded" != "true" ]]; then
    rm -f "$tmpfile"
    fail "No pre-built binary found for ${OS}/${ARCH}.

  Check available releases at:
    https://github.com/${REPO}/releases

  Or build from source:
    git clone https://github.com/${REPO}.git && cd plandb
    cargo build --release
    cp target/release/plandb ${INSTALL_DIR}/plandb"
  fi

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

  # Install — use sudo only if needed
  if [[ -d "$INSTALL_DIR" ]] && [[ -w "$INSTALL_DIR" ]]; then
    mv "$tmpfile" "${INSTALL_DIR}/${BINARY}"
  else
    info "Need permissions for ${INSTALL_DIR} — using sudo"
    sudo mkdir -p "$INSTALL_DIR"
    sudo mv "$tmpfile" "${INSTALL_DIR}/${BINARY}"
    sudo chmod +x "${INSTALL_DIR}/${BINARY}"
  fi

  ok "Installed to ${INSTALL_DIR}/${BINARY}"

  # Verify
  if "${INSTALL_DIR}/${BINARY}" --version &>/dev/null; then
    ok "$(${INSTALL_DIR}/${BINARY} --version 2>/dev/null || echo 'plandb installed')"
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

You have `plandb` installed. Use it to decompose work into a dependency graph and track progress.

## When to Use

- Starting any non-trivial task (>2 steps)
- Multi-file features, refactors, or bug investigations
- Work that benefits from dependency ordering or parallelization

## Core Loop

```bash
plandb init "project-name"                          # Create project
plandb add "title" --description "spec" --dep t-xxx # Add task with deps
plandb go                                           # Claim next ready task
# ... do the work ...
plandb done --next                                  # Complete + claim next
plandb status --detail                              # Reassess the plan
```

## Key Commands

```bash
# Planning
plandb add "title" --description "detailed spec" --dep t-upstream [--as id] [--kind code]
plandb split --into "A, B, C"                  # Independent subtasks (parallel)
plandb split --into "A > B > C"                # Dependency chain (sequential)
plandb task insert --after t-a --before t-b    # Insert missed step
plandb task amend t-xxx --prepend "NOTE: ..."  # Annotate future task

# Introspection
plandb status --detail                         # Dependency tree view
plandb critical-path                           # Longest chain — prioritize this
plandb bottlenecks                             # What blocks the most work
plandb list --status ready                     # Tasks safe to parallelize NOW
plandb what-if cancel t-xxx                    # Preview before destructive action

# Knowledge
plandb context "what you discovered" --kind discovery
plandb search "query"                          # BM25 across context + tasks
```

## Rules

- `--description` must be a self-contained spec (what, files, acceptance criteria)
- Create tasks in dependency order (`--dep` upstream must exist)
- `--kind`: generic, code, research, review, test, shell
- When `plandb list --status ready` shows multiple → parallelize them
- Plans are hypotheses — split, insert, and adapt as you learn
- Record discoveries with `plandb context` — auto-surfaces on `plandb go`
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

  if already_configured "$path"; then
    ok "${name}: already configured (${path})"
    return 0
  fi

  mkdir -p "$(dirname "$path")"

  if [[ "$method" == "dir" ]]; then
    {
      echo "$MARKER"
      plandb_instructions
      echo "$MARKER"
    } > "$path"
  else
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
        echo "  PLANDB_INSTALL_DIR  Installation directory (default: /usr/local/bin)"
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

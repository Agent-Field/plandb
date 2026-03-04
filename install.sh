#!/bin/sh
set -e

REPO="Agent-Field/plandb"
BINARY="plandb"

# Detect OS
OS=$(uname -s)
case "$OS" in
  Linux)  OS="linux" ;;
  Darwin) OS="darwin" ;;
  *)      echo "Error: unsupported OS '$OS'"; exit 1 ;;
esac

# Detect architecture
ARCH=$(uname -m)
case "$ARCH" in
  x86_64|amd64)    ARCH="x86_64" ;;
  aarch64|arm64)   ARCH="aarch64" ;;
  *)               echo "Error: unsupported architecture '$ARCH'"; exit 1 ;;
esac

ASSET="plandb-${OS}-${ARCH}"

# Get latest release tag
LATEST=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/')

if [ -z "$LATEST" ]; then
  echo "Error: could not determine latest release"
  exit 1
fi

URL="https://github.com/${REPO}/releases/download/${LATEST}/${ASSET}"
CHECKSUMS_URL="https://github.com/${REPO}/releases/download/${LATEST}/checksums.txt"
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"

echo "Installing plandb ${LATEST} (${OS}/${ARCH})..."

# Download binary to temp first
TMPFILE=$(mktemp)
curl -fsSL "$URL" -o "$TMPFILE"
chmod +x "$TMPFILE"

# Verify checksum if available
EXPECTED_HASH=$(curl -fsSL "$CHECKSUMS_URL" 2>/dev/null | grep "${ASSET}$" | awk '{print $1}')
if [ -n "$EXPECTED_HASH" ]; then
  if command -v sha256sum >/dev/null 2>&1; then
    ACTUAL_HASH=$(sha256sum "$TMPFILE" | awk '{print $1}')
  elif command -v shasum >/dev/null 2>&1; then
    ACTUAL_HASH=$(shasum -a 256 "$TMPFILE" | awk '{print $1}')
  else
    echo "Warning: no sha256sum or shasum found, skipping checksum verification"
    ACTUAL_HASH=""
  fi

  if [ -n "$ACTUAL_HASH" ] && [ "$ACTUAL_HASH" != "$EXPECTED_HASH" ]; then
    echo "Error: checksum mismatch"
    echo "  expected: ${EXPECTED_HASH}"
    echo "  got:      ${ACTUAL_HASH}"
    rm -f "$TMPFILE"
    exit 1
  fi
  [ -n "$ACTUAL_HASH" ] && echo "Checksum verified."
else
  echo "Warning: checksums not available for this release, skipping verification"
fi

# Install — use sudo only if needed
if [ -d "$INSTALL_DIR" ] && [ -w "$INSTALL_DIR" ]; then
  mv "$TMPFILE" "${INSTALL_DIR}/${BINARY}"
else
  echo "Need permissions for ${INSTALL_DIR} — using sudo"
  sudo mkdir -p "$INSTALL_DIR"
  sudo mv "$TMPFILE" "${INSTALL_DIR}/${BINARY}"
  sudo chmod +x "${INSTALL_DIR}/${BINARY}"
fi

echo "plandb ${LATEST} installed to ${INSTALL_DIR}/${BINARY}"
"${INSTALL_DIR}/${BINARY}" --version 2>/dev/null || true

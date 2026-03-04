#!/bin/sh
set -e

REPO="Agent-Field/planq"
BINARY="planq"

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

ASSET="planq-${OS}-${ARCH}"

# Get latest release tag
LATEST=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/')

if [ -z "$LATEST" ]; then
  echo "Error: could not determine latest release"
  exit 1
fi

URL="https://github.com/${REPO}/releases/download/${LATEST}/${ASSET}"
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"

echo "Installing planq ${LATEST} (${OS}/${ARCH})..."

# Create install dir if needed
if [ ! -d "$INSTALL_DIR" ]; then
  mkdir -p "$INSTALL_DIR" 2>/dev/null || sudo mkdir -p "$INSTALL_DIR"
fi

# Download and install
if [ -w "$INSTALL_DIR" ]; then
  curl -fsSL "$URL" -o "${INSTALL_DIR}/${BINARY}"
  chmod +x "${INSTALL_DIR}/${BINARY}"
else
  sudo curl -fsSL "$URL" -o "${INSTALL_DIR}/${BINARY}"
  sudo chmod +x "${INSTALL_DIR}/${BINARY}"
fi

echo "planq ${LATEST} installed to ${INSTALL_DIR}/${BINARY}"

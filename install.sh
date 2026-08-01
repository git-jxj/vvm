#!/usr/bin/env bash
# Install vvm — build from source, install to ~/.local/bin,
# and configure shell integration.
#
# Usage:
#   ./install.sh              # build release, install to ~/.local/bin
#   ./install.sh --to ~/bin   # custom install directory

set -euo pipefail

INSTALL_DIR="${HOME}/.local/bin"

while [ $# -gt 0 ]; do
    case "$1" in
        --to)   INSTALL_DIR="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: install.sh [--to DIR]"
            echo "  --to DIR   Install directory (default: ~/.local/bin)"
            exit 0 ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Check for Rust toolchain
if ! command -v cargo >/dev/null 2>&1; then
    echo "Error: cargo not found. Install Rust first:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

echo "Building vvm (release)..."
cargo build --release --manifest-path "${ROOT}/Cargo.toml"

mkdir -p "$INSTALL_DIR"
# Use `install` (unlink + create) instead of `cp` (in-place overwrite) so we
# don't get ETXTBSY when the existing binary is being exec'd elsewhere — common
# on shared filesystems (Lustre/NFS) where a vvm running on a compute node
# blocks the login-node copy.
install -m 755 "${ROOT}/target/release/vvm" "${INSTALL_DIR}/vvm"

echo "Installed vvm to ${INSTALL_DIR}/vvm"

# Make vvm available in the current script
case ":${PATH}:" in
    *":${INSTALL_DIR}:"*) ;;
    *)  export PATH="${INSTALL_DIR}:${PATH}" ;;
esac

# Detect shell and profile file
SHELL_NAME="$(basename "${SHELL:-bash}")"
case "$SHELL_NAME" in
    zsh)  PROFILE="${HOME}/.zshrc";  INIT_CMD='eval "$(vvm init zsh)"' ;;
    fish) PROFILE="${HOME}/.config/fish/config.fish"; INIT_CMD='vvm init fish | source' ;;
    *)    PROFILE="${HOME}/.bashrc"; INIT_CMD='eval "$(vvm init bash)"' ;;
esac

# Add shell integration if not already present
if [ -f "$PROFILE" ] && grep -qF 'vvm init' "$PROFILE" 2>/dev/null; then
    echo "Shell integration already configured in ${PROFILE}"
else
    echo "" >> "$PROFILE"
    echo "# vvm (vLLM Version Manager)" >> "$PROFILE"

    # Ensure PATH includes install dir in profile too
    case ":${PATH}:" in
        *":${INSTALL_DIR}:"*) ;;
        *)
            echo "export PATH=\"${INSTALL_DIR}:\$PATH\"" >> "$PROFILE"
            ;;
    esac

    echo "${INIT_CMD}" >> "$PROFILE"
    echo "Added vvm init to ${PROFILE}"
fi

echo ""
echo "Done! Run this to activate in your current shell:"
echo "  source ${PROFILE}"
echo ""
echo "Or start a new terminal. Then try:"
echo "  vvm install 0.18.0"
echo "  vvm use 0.18.0"

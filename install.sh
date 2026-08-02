#!/usr/bin/env bash
# Install vvm — download a prebuilt binary (or build from source when run
# inside a checkout), install to ~/.local/bin, and configure shell integration.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/vllm-project/vvm/main/install.sh | bash
#   curl -fsSL .../install.sh | bash -s -- --to ~/bin
#   ./install.sh                # inside a checkout: build from source
#   ./install.sh --prebuilt     # inside a checkout: still download the release binary
#
# Options:
#   --to DIR        Install directory (default: ~/.local/bin)
#   --prebuilt      Force downloading the release binary
#   --from-source   Force building with cargo (requires a checkout or git)

set -euo pipefail

REPO="vllm-project/vvm"
INSTALL_DIR="${HOME}/.local/bin"
MODE="auto"

while [ $# -gt 0 ]; do
    case "$1" in
        --to)          INSTALL_DIR="$2"; shift 2 ;;
        --prebuilt)    MODE="prebuilt"; shift ;;
        --from-source) MODE="source"; shift ;;
        --help|-h)
            echo "Usage: install.sh [--to DIR] [--prebuilt|--from-source]"
            echo "  --to DIR        Install directory (default: ~/.local/bin)"
            echo "  --prebuilt      Force downloading the release binary"
            echo "  --from-source   Force building with cargo"
            exit 0 ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done

err() { echo "Error: $*" >&2; exit 1; }

# Temp paths cleaned up on exit. Globals, not locals: the EXIT trap fires
# after any function's locals are gone (and set -u would trip on them).
TMP_FILE=""
TMP_ROOT=""
trap 'rm -f "${TMP_FILE:-}"; rm -rf "${TMP_ROOT:-}"' EXIT

# Where is this script? Empty when piped from curl.
SCRIPT_PATH="${BASH_SOURCE[0]:-}"
ROOT=""
if [ -n "$SCRIPT_PATH" ] && [ -f "$SCRIPT_PATH" ]; then
    ROOT="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
    # Only treat it as a source checkout if it really is vvm's own repo —
    # not some other directory the script happened to land in.
    if ! grep -q '^name = "vvm"' "${ROOT}/Cargo.toml" 2>/dev/null; then
        ROOT=""
    fi
fi

if [ "$MODE" = "auto" ]; then
    if [ -n "$ROOT" ]; then
        MODE="source"      # developer running ./install.sh in a clone
    else
        MODE="prebuilt"    # piped from curl
    fi
fi

download() {
    # download URL DEST — curl preferred, wget fallback
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$1" -o "$2"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$2" "$1"
    else
        err "need curl or wget to download vvm"
    fi
}

install_prebuilt() {
    local os arch asset url
    os="$(uname -s)"
    arch="$(uname -m)"
    case "$arch" in
        arm64) arch="aarch64" ;;
    esac

    if [ "$os" != "Linux" ] || { [ "$arch" != "x86_64" ] && [ "$arch" != "aarch64" ]; }; then
        echo "No prebuilt binary for ${os}/${arch} (releases ship Linux x86_64/aarch64 only)."
        if command -v cargo >/dev/null 2>&1; then
            echo "Falling back to building from source..."
            install_from_source
            return
        fi
        err "install Rust (https://rustup.rs) and re-run, or use: cargo install --git https://github.com/${REPO}"
    fi

    asset="vvm-${arch}-linux-musl"
    url="https://github.com/${REPO}/releases/latest/download/${asset}"
    TMP_FILE="$(mktemp "${TMPDIR:-/tmp}/vvm.XXXXXX")"

    echo "Downloading ${asset} (latest release)..."
    download "$url" "$TMP_FILE"

    mkdir -p "$INSTALL_DIR"
    # `install` (unlink + create) instead of `cp` (in-place overwrite) so we
    # don't get ETXTBSY when the existing binary is being exec'd elsewhere —
    # common on shared filesystems (Lustre/NFS) where a vvm running on a
    # compute node blocks the login-node copy.
    install -m 755 "$TMP_FILE" "${INSTALL_DIR}/vvm"
}

install_from_source() {
    command -v cargo >/dev/null 2>&1 || err "cargo not found. Install Rust first:
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"

    mkdir -p "$INSTALL_DIR"
    if [ -n "$ROOT" ]; then
        echo "Building vvm (release)..."
        cargo build --release --manifest-path "${ROOT}/Cargo.toml"
        install -m 755 "${ROOT}/target/release/vvm" "${INSTALL_DIR}/vvm"
    else
        echo "Building vvm from git (release)..."
        TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/vvm-cargo.XXXXXX")"
        cargo install --git "https://github.com/${REPO}" --root "$TMP_ROOT" --force
        install -m 755 "${TMP_ROOT}/bin/vvm" "${INSTALL_DIR}/vvm"
    fi
}

case "$MODE" in
    prebuilt) install_prebuilt ;;
    source)   install_from_source ;;
esac

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
    mkdir -p "$(dirname "$PROFILE")"
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

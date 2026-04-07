#!/usr/bin/env bash
# Set up the gw-kb Python virtualenv (uv-managed) for PyO3 embedding.
#
# Run once after cloning:
#   bash crates/gw-kb/setup.sh
#
# Then export the printed environment variables before `cargo build`.

set -euo pipefail

CRATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$CRATE_DIR/python"
VENV_DIR="$PY_DIR/.venv"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is not installed. Install from https://github.com/astral-sh/uv" >&2
    exit 1
fi

cd "$PY_DIR"

# uv sync creates .venv (if missing) and installs deps from pyproject.toml
uv sync

cat <<EOF

gw-kb venv ready at: $VENV_DIR

Add to your shell profile (or run before cargo build):

    export PYO3_PYTHON="$VENV_DIR/bin/python"
    export GW_KB_PYTHON_PATH="$PY_DIR"

EOF

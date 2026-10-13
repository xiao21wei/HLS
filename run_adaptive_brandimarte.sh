#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Only Python's standard library is used; no Conda activation or GPU is needed.
if [[ -n "${PYTHON:-}" ]]; then
    python_command="${PYTHON}"
else
    python_command=""
    for candidate in python3 "${HOME}/miniconda3/bin/python" "${HOME}/miniconda3/envs/hls-rl/bin/python"; do
        if command -v "${candidate}" >/dev/null 2>&1 &&
            "${candidate}" -c 'import sys; sys.exit(sys.version_info < (3, 8))' 2>/dev/null; then
            python_command="${candidate}"
            break
        fi
    done
fi
if [[ -z "${python_command}" ]]; then
    echo "Python >= 3.8 is required. Set PYTHON=/path/to/python." >&2
    exit 2
fi
exec "${python_command}" -u "${SCRIPT_DIR}/run_adaptive_brandimarte.py" "$@"

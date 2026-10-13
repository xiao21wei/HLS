#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_COMMAND="${PYTHON:-${HOME}/miniconda3/bin/python}"
if ! command -v "${PYTHON_COMMAND}" >/dev/null 2>&1; then
    PYTHON_COMMAND=python3
fi
"${PYTHON_COMMAND}" -c 'import sys; sys.exit(sys.version_info < (3, 8))' || {
    echo 'Python >= 3.8 is required; set PYTHON=/path/to/python.' >&2
    exit 2
}
exec "${PYTHON_COMMAND}" -u "${SCRIPT_DIR}/run_smartlab_hls.py" "$@"

#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EXECUTABLE="${EXECUTABLE:-${SCRIPT_DIR}/build/optimized/production_scheduling}"
export COPT_TIME="${COPT_TIME:-0}"
export TOTAL_TIME="${TOTAL_TIME:-600}"
export RUNS="${RUNS:-10}"
export MAX_PARALLEL="${MAX_PARALLEL:-14}"
export CPU_USAGE_LIMIT="${CPU_USAGE_LIMIT:-0.95}"
export CPU_RESERVE="${CPU_RESERVE:-0}"
if [[ -z "${LOG_DIR:-}" ]]; then
    export LOG_DIR="${SCRIPT_DIR}/run_logs/$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_optimized"
fi
if [[ -z "${CANDIDATE_HLS_ENV:-}" ]]; then
    export CANDIDATE_HLS_ENV='{"HLS_GREEDY_FLEXIBLE_REPAIR":"1","HLS_GREEDY_REPAIR_RELAYS":"1","HLS_GREEDY_RELEASE_COUNT":"32","HLS_GREEDY_REPAIR_SECONDS":"8","HLS_TABU_CONTINUOUS_TRAJECTORY":"1","HLS_ENABLE_RESTARTS":"1","HLS_RESTART_STAGNATION_CALLS":"16","HLS_GA_WORKLOAD_REDISTRIBUTION":"1","HLS_DISPATCH_INITIAL":"1"}'
fi
python_command="${PYTHON:-${HOME}/miniconda3/bin/python}"
if ! command -v "${python_command}" >/dev/null 2>&1; then python_command=python3; fi
"${python_command}" -c 'import sys; sys.exit(sys.version_info < (3, 8))' || {
    echo 'Python >= 3.8 is required; set PYTHON=/path/to/python.' >&2; exit 2;
}
exec "${python_command}" -u "${SCRIPT_DIR}/run_optimization_candidate.py" "$@"

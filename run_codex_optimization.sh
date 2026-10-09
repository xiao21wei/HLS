#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_FILE="${TASK_FILE:-${SCRIPT_DIR}/task.md}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/codex_runs}"
RUN_TIMEZONE="${RUN_TIMEZONE:-Asia/Shanghai}"
WATCHDOG_LIMIT="${WATCHDOG_LIMIT:-430m}"
CODEX_BIN="${CODEX_BIN:-codex}"
CODEX_SANDBOX="${CODEX_SANDBOX:-workspace-write}"
CODEX_BYPASS_SANDBOX="${CODEX_BYPASS_SANDBOX:-1}"
CODEX_MODEL="${CODEX_MODEL:-}"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/rl_greedy/models/greedy_policy_brandimarte.txt}"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Required command not found: $1" >&2
        exit 1
    fi
}

require_command "${CODEX_BIN}"
require_command timeout
require_command tee
require_command flock

if ! timeout --signal=INT --kill-after=10m "${WATCHDOG_LIMIT}" true \
    >/dev/null 2>&1; then
    echo "Invalid WATCHDOG_LIMIT for GNU timeout: ${WATCHDOG_LIMIT}" >&2
    echo "Use one duration with a single suffix, for example 430m or 25800s." >&2
    exit 1
fi

if [[ ! -f "${TASK_FILE}" ]]; then
    echo "Task file not found: ${TASK_FILE}" >&2
    exit 1
fi
if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "RL model not found: ${MODEL_PATH}" >&2
    exit 1
fi

export COPT_HOME="${COPT_HOME:-${HOME}/copt80}"
export COPT_LICENSE_DIR="${COPT_LICENSE_DIR:-${COPT_HOME}}"
export LD_LIBRARY_PATH="${COPT_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [[ ! -f "${COPT_HOME}/lib/libcopt_cpp.so" ]]; then
    echo "COPT library not found under ${COPT_HOME}/lib." >&2
    exit 1
fi
if pgrep -f "${SCRIPT_DIR}/build/[p]roduction_scheduling" >/dev/null 2>&1; then
    echo "An HLS production_scheduling process is already running." >&2
    exit 1
fi

mkdir -p "${RUN_ROOT}"
exec 9>"${RUN_ROOT}/.optimization.lock"
if ! flock -n 9; then
    echo "Another Codex optimization launcher is already running." >&2
    exit 1
fi

RUN_ID="$(TZ="${RUN_TIMEZONE}" date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
CURRENT_LINK="${RUN_ROOT}/current"
mkdir -p "${RUN_DIR}"
if [[ -e "${CURRENT_LINK}" && ! -L "${CURRENT_LINK}" ]]; then
    echo "Refusing to replace non-symlink path: ${CURRENT_LINK}" >&2
    exit 1
fi
ln -sfn "${RUN_DIR}" "${CURRENT_LINK}"

export HLS_OPT_RUN_DIR="${RUN_DIR}"
export HLS_RL_MODEL_PATH="${MODEL_PATH}"

case "${CODEX_BYPASS_SANDBOX}" in
    1|true|TRUE|yes|YES)
        CODEX_EXECUTION_MODE="bypass-sandbox"
        ;;
    0|false|FALSE|no|NO)
        CODEX_EXECUTION_MODE="${CODEX_SANDBOX}"
        ;;
    *)
        echo "CODEX_BYPASS_SANDBOX must be 0 or 1." >&2
        exit 1
        ;;
esac

{
    echo "# HLS Codex optimization status"
    echo
    echo "- phase: launcher-started"
    echo "- run_id: ${RUN_ID}"
    echo "- started_at: $(TZ="${RUN_TIMEZONE}" date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "- task_file: ${TASK_FILE}"
    echo "- watchdog_limit: ${WATCHDOG_LIMIT}"
    echo "- codex_execution_mode: ${CODEX_EXECUTION_MODE}"
    echo "- codex_version: $(${CODEX_BIN} --version)"
    echo "- waiting_for_agent_update: true"
} >"${RUN_DIR}/status.md"

CODEX_ARGS=(--cd "${SCRIPT_DIR}")
if [[ "${CODEX_EXECUTION_MODE}" == "bypass-sandbox" ]]; then
    CODEX_ARGS=(
        --dangerously-bypass-approvals-and-sandbox
        "${CODEX_ARGS[@]}"
    )
else
    CODEX_ARGS=(
        --ask-for-approval never
        --sandbox "${CODEX_SANDBOX}"
        "${CODEX_ARGS[@]}"
    )
fi
if [[ -n "${CODEX_MODEL}" ]]; then
    CODEX_ARGS+=(--model "${CODEX_MODEL}")
fi
CODEX_ARGS+=(
    exec
    --output-last-message "${RUN_DIR}/final-response.md"
)

echo "Run directory: ${RUN_DIR}"
echo "Live log: ${RUN_DIR}/codex-live.log"
echo "Status: ${RUN_DIR}/status.md"
echo "Codex execution mode: ${CODEX_EXECUTION_MODE}"
if [[ "${CODEX_EXECUTION_MODE}" == "bypass-sandbox" ]]; then
    echo "Warning: Codex is unsandboxed because bwrap is unavailable on this host."
fi
echo "The launcher remains in the foreground until Codex exits."

set +e
timeout --signal=INT --kill-after=10m "${WATCHDOG_LIMIT}" \
    "${CODEX_BIN}" "${CODEX_ARGS[@]}" - \
    <"${TASK_FILE}" 2>&1 | tee "${RUN_DIR}/codex-live.log"
CODEX_STATUS="${PIPESTATUS[0]}"
set -e

{
    echo
    echo "launcher_finished_at: $(TZ="${RUN_TIMEZONE}" date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "launcher_exit_code: ${CODEX_STATUS}"
} >>"${RUN_DIR}/launcher-status.txt"

echo "Codex exit code: ${CODEX_STATUS}"
echo "Final response: ${RUN_DIR}/final-response.md"
exit "${CODEX_STATUS}"

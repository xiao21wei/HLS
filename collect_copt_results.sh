#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/run_logs}"
RESULT_FILE="${RESULT_FILE:-${SCRIPT_DIR}/copt.csv}"
RUNS="${RUNS:-10}"
INSTANCE_LIST="${INSTANCES:-Mk01 Mk02 Mk03 Mk04 Mk05 Mk06 Mk07 Mk08 Mk09 Mk10}"

case "${RUNS}" in
    ''|*[!0-9]*) echo "RUNS must be a positive integer." >&2; exit 1 ;;
esac
if (( RUNS == 0 )); then
    echo "RUNS must be positive." >&2
    exit 1
fi

if [[ ! -d "${LOG_DIR}" ]]; then
    echo "Log directory not found: ${LOG_DIR}" >&2
    exit 1
fi

RESULT_DIR="$(dirname -- "${RESULT_FILE}")"
mkdir -p "${RESULT_DIR}"
TEMP_RESULT="$(mktemp "${RESULT_FILE}.tmp.XXXXXX")"

cleanup() {
    if [[ -n "${TEMP_RESULT}" ]]; then
        rm -f "${TEMP_RESULT}"
    fi
}
trap cleanup EXIT

extract_copt_makespan() {
    local log_file="$1"

    if grep -a -q '^COPT initial solution unavailable:' "${log_file}"; then
        printf 'NA'
        return 0
    fi

    local makespan
    makespan="$(awk -F': *' '/^COPT initial makespan:/ { print $2; exit }' "${log_file}")"
    if [[ -z "${makespan}" ]]; then
        # Backward compatibility with logs created before the explicit label.
        makespan="$(awk -F': *' '/^Total time:/ { print $2; exit }' "${log_file}")"
    fi
    if [[ ! "${makespan}" =~ ^[0-9]+$ ]]; then
        echo "Cannot extract COPT initial makespan from ${log_file}." >&2
        return 1
    fi
    printf '%s' "${makespan}"
}

{
    printf 'instance'
    for ((run_number = 1; run_number <= RUNS; run_number++)); do
        printf ',run%d' "${run_number}"
    done
    printf '\n'

    for instance in ${INSTANCE_LIST}; do
        printf '%s' "${instance}"
        for ((run_number = 1; run_number <= RUNS; run_number++)); do
            log_file="${LOG_DIR}/${instance}_run$(printf '%02d' "${run_number}").txt"
            if [[ ! -f "${log_file}" ]]; then
                echo "Missing log file: ${log_file}" >&2
                exit 1
            fi
            printf ',%s' "$(extract_copt_makespan "${log_file}")"
        done
        printf '\n'
    done
} >"${TEMP_RESULT}"

mv -f "${TEMP_RESULT}" "${RESULT_FILE}"
TEMP_RESULT=""
echo "Wrote ${RESULT_FILE}"

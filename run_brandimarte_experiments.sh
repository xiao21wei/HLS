#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

EXECUTABLE="${EXECUTABLE:-${SCRIPT_DIR}/build/production_scheduling}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/datasets/Brandimarte_Data/Text}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/run_logs/${RUN_TIMESTAMP}}"
RESULT_FILE="${RESULT_FILE:-${LOG_DIR}/result.csv}"
RUNS="${RUNS:-1}"
START_RUN="${START_RUN:-1}"
COPT_TIME="${COPT_TIME:-300}"
TOTAL_TIME="${TOTAL_TIME:-3600}"
SEARCH_MODE="${SEARCH_MODE:-0.5}"
RANDOM_STRATEGY="${RANDOM_STRATEGY:-0.5}"
MAX_REPEAT="${MAX_REPEAT:-100}"
MAX_ITER="${MAX_ITER:-30}"
POPULATION_SIZE="${POPULATION_SIZE:-20}"
TABU_LENGTH="${TABU_LENGTH:-25}"
SEED_BASE="${SEED_BASE:-100000}"
PARALLEL_JOBS="${PARALLEL_JOBS:-10}"
ALL_INSTANCE_LIST="Mk01 Mk02 Mk03 Mk04 Mk05 Mk06 Mk07 Mk08 Mk09 Mk10"
INSTANCE_LIST="${INSTANCES:-${ALL_INSTANCE_LIST}}"
AGGREGATE_INSTANCE_LIST="${AGGREGATE_INSTANCES:-${ALL_INSTANCE_LIST}}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

case "${RUNS}" in
    ''|*[!0-9]*) echo "RUNS must be a positive integer." >&2; exit 1 ;;
esac
if (( RUNS == 0 )); then
    echo "RUNS must be positive." >&2
    exit 1
fi

case "${START_RUN}" in
    ''|*[!0-9]*) echo "START_RUN must be a positive integer." >&2; exit 1 ;;
esac
if (( START_RUN == 0 || START_RUN > RUNS )); then
    echo "START_RUN must be between 1 and RUNS." >&2
    exit 1
fi

case "${PARALLEL_JOBS}" in
    ''|*[!0-9]*) echo "PARALLEL_JOBS must be a positive integer." >&2; exit 1 ;;
esac
if (( PARALLEL_JOBS == 0 )); then
    echo "PARALLEL_JOBS must be positive." >&2
    exit 1
fi

if [[ ! -x "${EXECUTABLE}" ]]; then
    echo "Executable not found or not executable: ${EXECUTABLE}" >&2
    exit 1
fi

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "Dataset directory not found: ${DATA_DIR}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname -- "${RESULT_FILE}")"

echo "Run timestamp: ${RUN_TIMESTAMP}"
echo "Log directory: ${LOG_DIR}"
echo "Result file: ${RESULT_FILE}"

if [[ ! -f "${RESULT_FILE}" ]]; then
    {
        printf 'instance'
        for ((run_number = 1; run_number <= RUNS; run_number++)); do
            printf ',run%d' "${run_number}"
        done
        printf '\n'
        for instance in ${AGGREGATE_INSTANCE_LIST}; do
            printf '%s' "${instance}"
            for ((run_number = 1; run_number <= RUNS; run_number++)); do
                printf ',PENDING'
            done
            printf '\n'
        done
    } >"${RESULT_FILE}"
fi

export COPT_HOME="${COPT_HOME:-${HOME}/copt80}"
export COPT_LICENSE_DIR="${COPT_LICENSE_DIR:-${COPT_HOME}}"
export LD_LIBRARY_PATH="${COPT_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

TEMP_RESULT=""
declare -a RUNNING_PIDS=()
declare -a RUNNING_LABELS=()
declare -a RUNNING_LOGS=()

cleanup() {
    local pid
    for pid in "${RUNNING_PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
    if [[ -n "${TEMP_RESULT}" ]]; then
        rm -f "${TEMP_RESULT}"
    fi
}
trap cleanup EXIT

wait_for_batch() {
    local failed=0
    local index

    for index in "${!RUNNING_PIDS[@]}"; do
        if wait "${RUNNING_PIDS[$index]}"; then
            echo "[done] ${RUNNING_LABELS[$index]}"
        else
            echo "[failed] ${RUNNING_LABELS[$index]}; see ${RUNNING_LOGS[$index]}" >&2
            failed=1
        fi
    done

    RUNNING_PIDS=()
    RUNNING_LABELS=()
    RUNNING_LOGS=()

    if (( failed != 0 )); then
        exit 1
    fi
}

run_one() {
    local instance="$1"
    local run_number="$2"
    local seed="$3"
    local dataset="${DATA_DIR}/${instance}.fjs"
    local log_file="${LOG_DIR}/${instance}_run$(printf '%02d' "${run_number}").txt"

    "${EXECUTABLE}" \
        "${dataset}" \
        "${SEARCH_MODE}" \
        "${RANDOM_STRATEGY}" \
        "${MAX_REPEAT}" \
        "${MAX_ITER}" \
        "${POPULATION_SIZE}" \
        "${TABU_LENGTH}" \
        "${COPT_TIME}" \
        "${TOTAL_TIME}" \
        "${seed}" \
        >"${log_file}" 2>&1
}

instance_index=0
for instance in ${INSTANCE_LIST}; do
    dataset="${DATA_DIR}/${instance}.fjs"
    if [[ ! -f "${dataset}" ]]; then
        echo "Dataset not found: ${dataset}" >&2
        exit 1
    fi

    for ((run_number = START_RUN; run_number <= RUNS; run_number++)); do
        seed=$((SEED_BASE + instance_index * 1000 + run_number))
        label="${instance} run ${run_number}/${RUNS}"
        log_file="${LOG_DIR}/${instance}_run$(printf '%02d' "${run_number}").txt"
        if [[ "${SKIP_COMPLETED}" == "1" ]] && [[ -f "${log_file}" ]] && grep -q '^Final makespan:' "${log_file}"; then
            echo "[skip] ${label}"
            continue
        fi
        run_one "${instance}" "${run_number}" "${seed}" &
        RUNNING_PIDS+=("$!")
        RUNNING_LABELS+=("${label}")
        RUNNING_LOGS+=("${log_file}")

        if (( ${#RUNNING_PIDS[@]} >= PARALLEL_JOBS )); then
            wait_for_batch
        fi
    done
    ((instance_index += 1))
done

wait_for_batch

TEMP_RESULT="$(mktemp "${RESULT_FILE}.tmp.XXXXXX")"

{
    printf 'instance'
    for ((run_number = 1; run_number <= RUNS; run_number++)); do
        printf ',run%d' "${run_number}"
    done
    printf '\n'

    for instance in ${AGGREGATE_INSTANCE_LIST}; do
        printf '%s' "${instance}"
        for ((run_number = 1; run_number <= RUNS; run_number++)); do
            log_file="${LOG_DIR}/${instance}_run$(printf '%02d' "${run_number}").txt"
            makespan="$(awk -F': *' '/^Final makespan:/ { value = $2 } END { if (value == "") exit 1; print value }' "${log_file}")"
            if [[ ! "${makespan}" =~ ^[0-9]+$ ]]; then
                echo "Invalid final makespan in ${log_file}: ${makespan}" >&2
                exit 1
            fi
            printf ',%s' "${makespan}"
        done
        printf '\n'
    done
} >"${TEMP_RESULT}"

mv -f "${TEMP_RESULT}" "${RESULT_FILE}"
TEMP_RESULT=""
echo "Wrote ${RESULT_FILE}"

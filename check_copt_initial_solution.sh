#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

EXECUTABLE="${EXECUTABLE:-${SCRIPT_DIR}/build/production_scheduling}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/datasets/Brandimarte_Data/Text}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/copt_progress_checks}"
PROGRESS_RESULT_FILE="${PROGRESS_RESULT_FILE:-${OUTPUT_DIR}/copt_progress.csv}"
FINAL_RESULT_FILE="${FINAL_RESULT_FILE:-${OUTPUT_DIR}/copt_final_results.csv}"
INSTANCES="${INSTANCES:-Mk01 Mk02 Mk03 Mk04 Mk05 Mk06 Mk07 Mk08 Mk09 Mk10}"
COPT_TIME="${COPT_TIME:-3600}"
TOTAL_TIME="${TOTAL_TIME:-3600}"
CHECKPOINT_SECONDS="${CHECKPOINT_SECONDS:-300}"
PARALLEL_JOBS="${PARALLEL_JOBS:-5}"
WAIT_GRACE_SECONDS="${WAIT_GRACE_SECONDS:-120}"
POLL_INTERVAL="${POLL_INTERVAL:-1}"

SEARCH_MODE="${SEARCH_MODE:-0.5}"
RANDOM_STRATEGY="${RANDOM_STRATEGY:-0.5}"
MAX_REPEAT="${MAX_REPEAT:-1}"
MAX_ITER="${MAX_ITER:-1}"
POPULATION_SIZE="${POPULATION_SIZE:-2}"
TABU_LENGTH="${TABU_LENGTH:-2}"
SEED_BASE="${SEED_BASE:-100000}"

TEMP_PROGRESS_RESULT=""
TEMP_FINAL_RESULT=""
declare -a RUNNING_PIDS=()
declare -a RUNNING_LABELS=()

cleanup() {
    local pid
    for pid in "${RUNNING_PIDS[@]:-}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
            wait "${pid}" 2>/dev/null || true
        fi
    done
    if [[ -n "${TEMP_PROGRESS_RESULT}" ]]; then
        rm -f "${TEMP_PROGRESS_RESULT}"
    fi
    if [[ -n "${TEMP_FINAL_RESULT}" ]]; then
        rm -f "${TEMP_FINAL_RESULT}"
    fi
}

trap cleanup EXIT
trap 'exit 130' INT TERM

if [[ ! -x "${EXECUTABLE}" ]]; then
    echo "Executable not found or not executable: ${EXECUTABLE}" >&2
    exit 1
fi

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "Dataset directory not found: ${DATA_DIR}" >&2
    exit 1
fi

for value_name in COPT_TIME TOTAL_TIME CHECKPOINT_SECONDS PARALLEL_JOBS WAIT_GRACE_SECONDS; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer: ${value}" >&2
        exit 1
    fi
done

if (( COPT_TIME > TOTAL_TIME )); then
    echo "COPT_TIME (${COPT_TIME}) cannot exceed TOTAL_TIME (${TOTAL_TIME})." >&2
    exit 1
fi

if (( CHECKPOINT_SECONDS > COPT_TIME )); then
    echo "CHECKPOINT_SECONDS (${CHECKPOINT_SECONDS}) cannot exceed COPT_TIME (${COPT_TIME})." >&2
    exit 1
fi

if ! [[ "${POLL_INTERVAL}" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
    ! awk -v value="${POLL_INTERVAL}" 'BEGIN { exit !(value > 0) }'; then
    echo "POLL_INTERVAL must be a positive number: ${POLL_INTERVAL}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/rows"

export COPT_HOME="${COPT_HOME:-${HOME}/copt80}"
export COPT_LICENSE_DIR="${COPT_LICENSE_DIR:-${COPT_HOME}}"
export LD_LIBRARY_PATH="${COPT_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

latest_at_or_before() {
    local progress_file="$1"
    local checkpoint="$2"

    if [[ ! -s "${progress_file}" ]]; then
        printf 'NA,NA\n'
        return 0
    fi

    awk -F, -v checkpoint="${checkpoint}" '
        NR > 1 &&
        $1 ~ /^[0-9]+([.][0-9]+)?$/ &&
        $2 ~ /^-?[0-9]+([.][0-9]+)?$/ &&
        ($1 + 0) <= (checkpoint + 0) {
            observed_elapsed = $1
            makespan = $2
        }
        END {
            if (makespan == "") {
                print "NA,NA"
            } else {
                print observed_elapsed "," makespan
            }
        }
    ' "${progress_file}"
}

run_instance() (
    set -euo pipefail

    local instance="$1"
    local instance_index="$2"
    local log_file="${OUTPUT_DIR}/logs/${instance}.log"
    local progress_file="${OUTPUT_DIR}/logs/${instance}_incumbents.csv"
    local rows_file="${OUTPUT_DIR}/rows/${instance}.csv"
    local final_file="${OUTPUT_DIR}/rows/${instance}_final.csv"
    local seed=$((SEED_BASE + instance_index * 1000))
    local pid=""
    local start_time
    local elapsed
    local status="timeout"
    local final_makespan="NA"
    local copt_elapsed="NA"
    local checkpoint
    local sample
    local observed_elapsed
    local sample_makespan
    local sample_status

    cleanup_instance() {
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
            wait "${pid}" 2>/dev/null || true
        fi
    }
    trap cleanup_instance EXIT INT TERM

    printf 'elapsed_seconds,makespan\n' >"${progress_file}"
    printf 'instance,checkpoint_seconds,makespan,observed_elapsed_seconds,status\n' >"${rows_file}"
    : >"${final_file}"
    : >"${log_file}"

    HLS_COPT_PROGRESS_FILE="${progress_file}" \
        "${EXECUTABLE}" \
        "${DATA_DIR}/${instance}.fjs" \
        "${SEARCH_MODE}" \
        "${RANDOM_STRATEGY}" \
        "${MAX_REPEAT}" \
        "${MAX_ITER}" \
        "${POPULATION_SIZE}" \
        "${TABU_LENGTH}" \
        "${COPT_TIME}" \
        "${TOTAL_TIME}" \
        "${seed}" \
        >"${log_file}" 2>&1 &
    pid=$!
    start_time="$(date +%s)"

    while kill -0 "${pid}" 2>/dev/null; do
        if grep -a -q '^COPT initial makespan:' "${log_file}" || \
            grep -a -q '^COPT initial solution unavailable:' "${log_file}"; then
            break
        fi
        elapsed=$(( $(date +%s) - start_time ))
        if (( elapsed >= COPT_TIME + WAIT_GRACE_SECONDS )); then
            break
        fi
        sleep "${POLL_INTERVAL}"
    done

    if grep -a -q '^COPT initial makespan:' "${log_file}"; then
        status="copt_solution"
        final_makespan="$(awk -F': *' '/^COPT initial makespan:/ { value = $2 } END { print value }' "${log_file}")"
    elif grep -a -q '^COPT initial solution unavailable:' "${log_file}"; then
        status="no_copt_solution"
    elif ! kill -0 "${pid}" 2>/dev/null; then
        status="error"
    fi

    if kill -0 "${pid}" 2>/dev/null; then
        kill "${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
    pid=""

    copt_elapsed="$(awk -F, 'NR > 1 { value = $1 } END { print value }' "${progress_file}")"
    if [[ -z "${copt_elapsed}" ]]; then
        copt_elapsed="NA"
    fi

    for ((checkpoint = CHECKPOINT_SECONDS; checkpoint <= COPT_TIME; checkpoint += CHECKPOINT_SECONDS)); do
        sample="$(latest_at_or_before "${progress_file}" "${checkpoint}")"
        observed_elapsed="${sample%%,*}"
        sample_makespan="${sample##*,}"

        if [[ "${sample_makespan}" == "NA" &&
              "${status}" == "copt_solution" &&
              "${copt_elapsed}" != "NA" ]] &&
            awk -v elapsed="${copt_elapsed}" -v checkpoint="${checkpoint}" \
                'BEGIN { exit !(elapsed <= checkpoint) }'; then
            observed_elapsed="${copt_elapsed}"
            sample_makespan="${final_makespan}"
        fi

        sample_status="${status}"
        if [[ "${sample_makespan}" == "NA" && "${status}" == "copt_solution" ]]; then
            sample_status="no_incumbent_at_checkpoint"
        fi
        printf '%s,%s,%s,%s,%s\n' \
            "${instance}" "${checkpoint}" "${sample_makespan}" \
            "${observed_elapsed}" "${sample_status}" >>"${rows_file}"
    done

    printf '%s,%s,%s,%s,%s,%s\n' \
        "${instance}" "${final_makespan}" "${status}" "${copt_elapsed}" \
        "${log_file}" "${progress_file}" >"${final_file}"
    printf '%-5s status=%-18s final_makespan=%s copt_elapsed=%ss\n' \
        "${instance}" "${status}" "${final_makespan}" "${copt_elapsed}"
)

wait_for_batch() {
    local failed=0
    local index

    for index in "${!RUNNING_PIDS[@]}"; do
        if wait "${RUNNING_PIDS[$index]}"; then
            echo "[done] ${RUNNING_LABELS[$index]}"
        else
            echo "[failed] ${RUNNING_LABELS[$index]}" >&2
            failed=1
        fi
    done

    RUNNING_PIDS=()
    RUNNING_LABELS=()

    if (( failed != 0 )); then
        exit 1
    fi
}

instance_index=0
for instance in ${INSTANCES}; do
    if [[ ! -f "${DATA_DIR}/${instance}.fjs" ]]; then
        echo "Dataset not found: ${DATA_DIR}/${instance}.fjs" >&2
        exit 1
    fi

    run_instance "${instance}" "${instance_index}" &
    RUNNING_PIDS+=("$!")
    RUNNING_LABELS+=("${instance}")
    ((instance_index += 1))

    if (( ${#RUNNING_PIDS[@]} >= PARALLEL_JOBS )); then
        wait_for_batch
    fi
done
wait_for_batch

PROGRESS_RESULT_DIR="$(dirname -- "${PROGRESS_RESULT_FILE}")"
FINAL_RESULT_DIR="$(dirname -- "${FINAL_RESULT_FILE}")"
mkdir -p "${PROGRESS_RESULT_DIR}" "${FINAL_RESULT_DIR}"
TEMP_PROGRESS_RESULT="$(mktemp "${PROGRESS_RESULT_FILE}.tmp.XXXXXX")"
TEMP_FINAL_RESULT="$(mktemp "${FINAL_RESULT_FILE}.tmp.XXXXXX")"

printf 'instance,checkpoint_seconds,makespan,observed_elapsed_seconds,status\n' >"${TEMP_PROGRESS_RESULT}"
printf 'instance,final_makespan,status,copt_elapsed_seconds,log_file,progress_file\n' >"${TEMP_FINAL_RESULT}"

for instance in ${INSTANCES}; do
    awk 'NR > 1' "${OUTPUT_DIR}/rows/${instance}.csv" >>"${TEMP_PROGRESS_RESULT}"
    cat "${OUTPUT_DIR}/rows/${instance}_final.csv" >>"${TEMP_FINAL_RESULT}"
done

mv -f "${TEMP_PROGRESS_RESULT}" "${PROGRESS_RESULT_FILE}"
TEMP_PROGRESS_RESULT=""
mv -f "${TEMP_FINAL_RESULT}" "${FINAL_RESULT_FILE}"
TEMP_FINAL_RESULT=""

echo
echo "Final COPT results:"
awk -F, 'NR > 1 { printf "%-5s %s (%s)\n", $1, $2, $3 }' "${FINAL_RESULT_FILE}"
echo "Wrote ${PROGRESS_RESULT_FILE}"
echo "Wrote ${FINAL_RESULT_FILE}"

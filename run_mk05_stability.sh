#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXECUTABLE="${EXECUTABLE:-${SCRIPT_DIR}/build/production_scheduling}"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/rl_greedy/models/greedy_policy_brandimarte.txt}"
DATA_FILE="${DATA_FILE:-${SCRIPT_DIR}/datasets/Brandimarte_Data/Text/Mk05.fjs}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/run_logs/${RUN_TIMESTAMP}/Mk05_stability}"
SEEDS_FILE="${SEEDS_FILE:-${LOG_DIR}/seeds.txt}"
RUNS="${RUNS:-10}"
PARALLEL_JOBS="${PARALLEL_JOBS:-5}"

SEARCH_MODE="${SEARCH_MODE:-0.60}"
RANDOM_STRATEGY="${RANDOM_STRATEGY:-0.50}"
MAX_REPEAT="${MAX_REPEAT:-1000}"
MAX_ITER="${MAX_ITER:-45}"
POPULATION_SIZE="${POPULATION_SIZE:-32}"
TABU_LENGTH="${TABU_LENGTH:-30}"
COPT_TIME="${COPT_TIME:-30}"
TOTAL_TIME="${TOTAL_TIME:-300}"
GUIDED_TOP_K="${GUIDED_TOP_K:-96}"
GREEDY_RL_TOP_K="${GREEDY_RL_TOP_K:-256}"
TABU_RL_TOP_K="${TABU_RL_TOP_K:-1024}"
GREEDY_RL_RULE_FRACTION="${GREEDY_RL_RULE_FRACTION:-0.30}"
GREEDY_RL_EXPLORATION="${GREEDY_RL_EXPLORATION:-0.20}"
TABU_RL_RULE_FRACTION="${TABU_RL_RULE_FRACTION:-0.30}"
TABU_RL_EXPLORATION="${TABU_RL_EXPLORATION:-0.25}"
TABU_RL_MIN_PER_NEIGHBORHOOD="${TABU_RL_MIN_PER_NEIGHBORHOOD:-24}"
TABU_RL_FALLBACK_ON_NON_IMPROVEMENT="${TABU_RL_FALLBACK_ON_NON_IMPROVEMENT:-0}"
TARGET_MAKESPAN="${TARGET_MAKESPAN:-173}"
GA_ACTIVE_DECODE="${GA_ACTIVE_DECODE:-1}"
GA_LOCAL_RESTART="${GA_LOCAL_RESTART:-0}"
ENABLE_RESTARTS="${ENABLE_RESTARTS:-0}"
RESTART_STAGNATION_CALLS="${RESTART_STAGNATION_CALLS:-40}"
GREEDY_COPT_REPAIR="${GREEDY_COPT_REPAIR:-0}"

for value_name in RUNS PARALLEL_JOBS; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer." >&2
        exit 2
    fi
done
if (( PARALLEL_JOBS > 5 )); then
    echo "PARALLEL_JOBS must be at most 5." >&2
    exit 2
fi
if [[ ! -x "${EXECUTABLE}" ]]; then
    echo "Executable not found: ${EXECUTABLE}" >&2
    exit 2
fi
if [[ ! -f "${MODEL_PATH}" || ! -f "${DATA_FILE}" ]]; then
    echo "Model or dataset file not found." >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"
if [[ ! -f "${SEEDS_FILE}" ]]; then
    od -An -N"$((4 * RUNS))" -tu4 /dev/urandom |
        tr -s '[:space:]' '\n' | sed '/^$/d' >"${SEEDS_FILE}"
fi
mapfile -t seeds <"${SEEDS_FILE}"
if (( ${#seeds[@]} != RUNS )); then
    echo "Expected ${RUNS} seeds in ${SEEDS_FILE}." >&2
    exit 2
fi
for seed in "${seeds[@]}"; do
    if [[ ! "${seed}" =~ ^[0-9]+$ ]] || (( seed > 4294967295 )); then
        echo "Invalid unsigned 32-bit seed: ${seed}" >&2
        exit 2
    fi
done

export COPT_HOME="${COPT_HOME:-${HOME}/copt80}"
export COPT_LICENSE_DIR="${COPT_LICENSE_DIR:-${COPT_HOME}}"
export LD_LIBRARY_PATH="${COPT_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

declare -a pids=()
declare -a labels=()

cleanup() {
    local pid
    for pid in "${pids[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

run_one() {
    local run_index="$1"
    local seed="$2"
    local label log_file started finished exit_code
    label="$(printf 'run%02d' "$((run_index + 1))")"
    log_file="${LOG_DIR}/${label}.txt"
    started="$(date +%s)"
    set +e
    env \
        HLS_TARGET_MAKESPAN=0 \
        HLS_ENABLE_GUIDED_SEARCH=force \
        HLS_GA_ACTIVE_DECODE="${GA_ACTIVE_DECODE}" \
        HLS_GA_LOCAL_RESTART="${GA_LOCAL_RESTART}" \
        HLS_ENABLE_RESTARTS="${ENABLE_RESTARTS}" \
        HLS_RESTART_STAGNATION_CALLS="${RESTART_STAGNATION_CALLS}" \
        HLS_GREEDY_COPT_REPAIR="${GREEDY_COPT_REPAIR}" \
        HLS_GREEDY_SCORE_BONUS=0 \
        HLS_GENETIC_SCORE_BONUS=0 \
        HLS_TABU_SCORE_BONUS=0 \
        HLS_GUIDED_SCORE_BONUS=0 \
        HLS_GUIDED_TOP_K="${GUIDED_TOP_K}" \
        HLS_GREEDY_RL=1 \
        HLS_GREEDY_RL_MODEL="${MODEL_PATH}" \
        HLS_GREEDY_RL_TOP_K="${GREEDY_RL_TOP_K}" \
        HLS_GREEDY_RL_DYNAMIC_TOP_K=0 \
        HLS_GREEDY_RL_RULE_FRACTION="${GREEDY_RL_RULE_FRACTION}" \
        HLS_GREEDY_RL_EXPLORATION="${GREEDY_RL_EXPLORATION}" \
        HLS_GREEDY_RL_FALLBACK_ON_MISS=1 \
        HLS_GREEDY_RL_VERIFY_WITH_BASELINE=0 \
        HLS_TABU_RL=1 \
        HLS_TABU_RL_MODEL="${MODEL_PATH}" \
        HLS_TABU_RL_TOP_K="${TABU_RL_TOP_K}" \
        HLS_TABU_RL_DYNAMIC_TOP_K=0 \
        HLS_TABU_RL_RULE_FRACTION="${TABU_RL_RULE_FRACTION}" \
        HLS_TABU_RL_EXPLORATION="${TABU_RL_EXPLORATION}" \
        HLS_TABU_RL_MIN_PER_NEIGHBORHOOD="${TABU_RL_MIN_PER_NEIGHBORHOOD}" \
        HLS_TABU_RL_FALLBACK_ON_MISS=1 \
        HLS_TABU_RL_FALLBACK_ON_NON_IMPROVEMENT="${TABU_RL_FALLBACK_ON_NON_IMPROVEMENT}" \
        HLS_TABU_RL_VERIFY_FULL_NEIGHBORHOOD=0 \
        "${EXECUTABLE}" "${DATA_FILE}" \
        "${SEARCH_MODE}" "${RANDOM_STRATEGY}" \
        "${MAX_REPEAT}" "${MAX_ITER}" "${POPULATION_SIZE}" \
        "${TABU_LENGTH}" "${COPT_TIME}" "${TOTAL_TIME}" "${seed}" \
        >"${log_file}" 2>&1
    exit_code=$?
    set -e
    finished="$(date +%s)"
    printf 'wall_seconds=%d\nexit_code=%d\n' \
        "$((finished - started))" "${exit_code}" >>"${log_file}"
    return "${exit_code}"
}

wait_batch() {
    local failed=0 index
    for index in "${!pids[@]}"; do
        if wait "${pids[$index]}"; then
            echo "[done] ${labels[$index]}"
        else
            echo "[failed] ${labels[$index]}" >&2
            failed=1
        fi
    done
    pids=()
    labels=()
    (( failed == 0 )) || exit 1
}

if [[ "${COLLECT_ONLY:-0}" != "1" ]]; then
    for index in "${!seeds[@]}"; do
        run_one "${index}" "${seeds[$index]}" &
        pids+=("$!")
        labels+=("$(printf 'run%02d' "$((index + 1))")")
        if (( ${#pids[@]} >= PARALLEL_JOBS )); then
            wait_batch
        fi
    done
    if (( ${#pids[@]} > 0 )); then
        wait_batch
    fi
fi

result_file="${LOG_DIR}/result.csv"
printf 'run,seed,makespan,wall_seconds,status\n' >"${result_file}"
for index in "${!seeds[@]}"; do
    label="$(printf 'run%02d' "$((index + 1))")"
    log_file="${LOG_DIR}/${label}.txt"
    makespan="$(awk '/^Final makespan:/ {value=$3} END {print value}' "${log_file}")"
    wall_seconds="$(awk -F= '/^wall_seconds=/ {print $2}' "${log_file}")"
    if [[ ! "${makespan}" =~ ^[0-9]+$ ]]; then
        echo "Missing final makespan in ${log_file}." >&2
        exit 1
    fi
    status="$(awk -v value="${makespan}" -v target="${TARGET_MAKESPAN}" \
        'BEGIN {print value <= target ? "SUCCESS" : "MISS"}')"
    printf '%s,%s,%s,%s,%s\n' \
        "${label}" "${seeds[$index]}" "${makespan}" \
        "${wall_seconds}" "${status}" >>"${result_file}"
done

awk -F, -v target="${TARGET_MAKESPAN}" '
    NR > 1 {
        sum += $3; count += 1; success += ($3 <= target);
        if (count == 1 || $3 < minimum) minimum = $3;
        if (count == 1 || $3 > maximum) maximum = $3;
    }
    END {
        printf "runs=%d\nmean=%.2f\nminimum=%d\nmaximum=%d\nsuccesses=%d\nsuccess_rate=%.1f%%\n",
            count, sum / count, minimum, maximum, success, 100 * success / count;
    }
' "${result_file}" >"${LOG_DIR}/summary.txt"

{
    printf 'EXECUTABLE=%q\n' "${EXECUTABLE}"
    printf 'EXECUTABLE_SHA256=%q\n' "$(sha256sum "${EXECUTABLE}" | awk '{print $1}')"
    printf 'MODEL_SHA256=%q\n' "$(sha256sum "${MODEL_PATH}" | awk '{print $1}')"
    printf 'MODEL_PATH=%q\nDATA_FILE=%q\nSEEDS_FILE=%q\n' \
        "${MODEL_PATH}" "${DATA_FILE}" "${SEEDS_FILE}"
    printf 'MAX_REPEAT=%q\nTARGET_MAKESPAN=%q\nHLS_GA_ACTIVE_DECODE=%q\n' \
        "${MAX_REPEAT}" "${TARGET_MAKESPAN}" "${GA_ACTIVE_DECODE}"
    printf 'HLS_GA_LOCAL_RESTART=%q\n' "${GA_LOCAL_RESTART}"
    printf 'HLS_ENABLE_RESTARTS=%q\n' "${ENABLE_RESTARTS}"
    printf 'HLS_RESTART_STAGNATION_CALLS=%q\n' "${RESTART_STAGNATION_CALLS}"
    printf 'HLS_GREEDY_COPT_REPAIR=%q\n' "${GREEDY_COPT_REPAIR}"
    printf 'RUNS=%q\nPARALLEL_JOBS=%q\n' "${RUNS}" "${PARALLEL_JOBS}"
    printf 'SEARCH_MODE=%q\nRANDOM_STRATEGY=%q\n' "${SEARCH_MODE}" "${RANDOM_STRATEGY}"
    printf 'MAX_ITER=%q\nPOPULATION_SIZE=%q\nTABU_LENGTH=%q\n' \
        "${MAX_ITER}" "${POPULATION_SIZE}" "${TABU_LENGTH}"
    printf 'COPT_TIME=%q\nTOTAL_TIME=%q\n' "${COPT_TIME}" "${TOTAL_TIME}"
    printf 'GUIDED_TOP_K=%q\nGREEDY_RL_TOP_K=%q\nTABU_RL_TOP_K=%q\n' \
        "${GUIDED_TOP_K}" "${GREEDY_RL_TOP_K}" "${TABU_RL_TOP_K}"
    printf 'GREEDY_RL_RULE_FRACTION=%q\nGREEDY_RL_EXPLORATION=%q\n' \
        "${GREEDY_RL_RULE_FRACTION}" "${GREEDY_RL_EXPLORATION}"
    printf 'TABU_RL_RULE_FRACTION=%q\nTABU_RL_EXPLORATION=%q\n' \
        "${TABU_RL_RULE_FRACTION}" "${TABU_RL_EXPLORATION}"
    printf 'TABU_RL_MIN_PER_NEIGHBORHOOD=%q\nTABU_RL_FALLBACK_ON_NON_IMPROVEMENT=%q\n' \
        "${TABU_RL_MIN_PER_NEIGHBORHOOD}" "${TABU_RL_FALLBACK_ON_NON_IMPROVEMENT}"
} >"${LOG_DIR}/config.env"

trap - EXIT INT TERM
echo "Result file: ${result_file}"
echo "Summary: ${LOG_DIR}/summary.txt"

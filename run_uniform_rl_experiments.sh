#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXECUTABLE="${EXECUTABLE:-${SCRIPT_DIR}/build/production_scheduling}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/datasets/Brandimarte_Data/Text}"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/rl_greedy/models/greedy_policy_brandimarte.txt}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/run_logs/${RUN_TIMESTAMP}}"
PARALLEL_JOBS="${PARALLEL_JOBS:-5}"
SEEDS_FILE="${SEEDS_FILE:-${LOG_DIR}/seeds.txt}"
INSTANCE_LIST="${INSTANCES:-Mk01 Mk02 Mk03 Mk04 Mk05 Mk06 Mk07 Mk08 Mk09 Mk10}"

SEARCH_MODE="${SEARCH_MODE:-0.60}"
RANDOM_STRATEGY="${RANDOM_STRATEGY:-0.50}"
MAX_REPEAT="${MAX_REPEAT:-1000}"
MAX_ITER="${MAX_ITER:-45}"
POPULATION_SIZE="${POPULATION_SIZE:-32}"
TABU_LENGTH="${TABU_LENGTH:-30}"
COPT_TIME="${COPT_TIME:-300}"
TOTAL_TIME="${TOTAL_TIME:-3600}"
GREEDY_SCORE_BONUS="${GREEDY_SCORE_BONUS:-0}"
GENETIC_SCORE_BONUS="${GENETIC_SCORE_BONUS:-0}"
TABU_SCORE_BONUS="${TABU_SCORE_BONUS:-0}"
GUIDED_SCORE_BONUS="${GUIDED_SCORE_BONUS:-0}"
GUIDED_TOP_K="${GUIDED_TOP_K:-96}"
GREEDY_RL_TOP_K="${GREEDY_RL_TOP_K:-256}"
TABU_RL_TOP_K="${TABU_RL_TOP_K:-1024}"
GREEDY_RL_RULE_FRACTION="${GREEDY_RL_RULE_FRACTION:-0.30}"
GREEDY_RL_EXPLORATION="${GREEDY_RL_EXPLORATION:-0.20}"
TABU_RL_RULE_FRACTION="${TABU_RL_RULE_FRACTION:-0.30}"
TABU_RL_EXPLORATION="${TABU_RL_EXPLORATION:-0.25}"
TABU_RL_MIN_PER_NEIGHBORHOOD="${TABU_RL_MIN_PER_NEIGHBORHOOD:-24}"
TABU_RL_FALLBACK_ON_NON_IMPROVEMENT="${TABU_RL_FALLBACK_ON_NON_IMPROVEMENT:-0}"

read -r -a instances <<<"${INSTANCE_LIST}"
if (( ${#instances[@]} == 0 )); then
    echo "INSTANCES must contain at least one instance." >&2
    exit 2
fi
declare -A requested_instances=()
for instance in "${instances[@]}"; do
    case "${instance}" in
        Mk01|Mk02|Mk03|Mk04|Mk05|Mk06|Mk07|Mk08|Mk09|Mk10) ;;
        *) echo "Unsupported instance: ${instance}" >&2; exit 2 ;;
    esac
    if [[ -n "${requested_instances[${instance}]:-}" ]]; then
        echo "Duplicate instance: ${instance}" >&2
        exit 2
    fi
    requested_instances["${instance}"]=1
done

case "${PARALLEL_JOBS}" in
    ''|*[!0-9]*) echo "PARALLEL_JOBS must be a positive integer." >&2; exit 2 ;;
esac
if (( PARALLEL_JOBS < 1 || PARALLEL_JOBS > 5 )); then
    echo "PARALLEL_JOBS must be between 1 and 5." >&2
    exit 2
fi
if [[ ! -x "${EXECUTABLE}" ]]; then
    echo "Executable not found: ${EXECUTABLE}" >&2
    exit 2
fi
if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "RL model not found: ${MODEL_PATH}" >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"
if [[ ! -f "${SEEDS_FILE}" ]]; then
    od -An -N"$((4 * ${#instances[@]}))" -tu4 /dev/urandom | tr -s '[:space:]' '\n' |
        sed '/^$/d' >"${SEEDS_FILE}"
fi
mapfile -t seeds <"${SEEDS_FILE}"
if (( ${#seeds[@]} != ${#instances[@]} )); then
    echo "Expected exactly ${#instances[@]} seeds in ${SEEDS_FILE}." >&2
    exit 2
fi
for seed in "${seeds[@]}"; do
    if [[ ! "${seed}" =~ ^[0-9]+$ ]] ||
       (( seed < 0 || seed > 4294967295 )); then
        echo "Invalid unsigned 32-bit seed: ${seed}" >&2
        exit 2
    fi
done

export COPT_HOME="${COPT_HOME:-${HOME}/copt80}"
export COPT_LICENSE_DIR="${COPT_LICENSE_DIR:-${COPT_HOME}}"
export LD_LIBRARY_PATH="${COPT_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

declare -a running_pids=()
declare -a running_instances=()

cleanup() {
    local pid
    for pid in "${running_pids[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

run_one() {
    local instance="$1"
    local seed="$2"
    local log_file="${LOG_DIR}/${instance}.txt"
    local started finished exit_code
    started="$(date +%s)"
    set +e
    env \
        HLS_TARGET_MAKESPAN=0 \
        HLS_ENABLE_GUIDED_SEARCH=force \
        HLS_GREEDY_SCORE_BONUS="${GREEDY_SCORE_BONUS}" \
        HLS_GENETIC_SCORE_BONUS="${GENETIC_SCORE_BONUS}" \
        HLS_TABU_SCORE_BONUS="${TABU_SCORE_BONUS}" \
        HLS_GUIDED_SCORE_BONUS="${GUIDED_SCORE_BONUS}" \
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
        "${EXECUTABLE}" \
        "${DATA_DIR}/${instance}.fjs" \
        "${SEARCH_MODE}" "${RANDOM_STRATEGY}" \
        "${MAX_REPEAT}" "${MAX_ITER}" \
        "${POPULATION_SIZE}" "${TABU_LENGTH}" \
        "${COPT_TIME}" "${TOTAL_TIME}" "${seed}" \
        >"${log_file}" 2>&1
    exit_code=$?
    set -e
    finished="$(date +%s)"
    printf 'wall_seconds=%d\nexit_code=%d\n' \
        "$((finished - started))" "${exit_code}" >>"${log_file}"
    return "${exit_code}"
}

wait_batch() {
    local failed=0
    local index
    for index in "${!running_pids[@]}"; do
        if wait "${running_pids[$index]}"; then
            echo "[done] ${running_instances[$index]}"
        else
            echo "[failed] ${running_instances[$index]}" >&2
            failed=1
        fi
    done
    running_pids=()
    running_instances=()
    if (( failed != 0 )); then
        exit 1
    fi
}

for index in "${!instances[@]}"; do
    run_one "${instances[$index]}" "${seeds[$index]}" &
    running_pids+=("$!")
    running_instances+=("${instances[$index]}")
    if (( ${#running_pids[@]} >= PARALLEL_JOBS )); then
        wait_batch
    fi
done
if (( ${#running_pids[@]} > 0 )); then
    wait_batch
fi

declare -A l2s_mean=(
    [Mk01]=40.9 [Mk02]=27.6 [Mk03]=204.0 [Mk04]=63.9 [Mk05]=173.1
    [Mk06]=72.1 [Mk07]=142.4 [Mk08]=523.0 [Mk09]=316.5 [Mk10]=215.8
)

result_file="${LOG_DIR}/result.csv"
comparison_file="${LOG_DIR}/comparison_l2s.csv"
printf 'instance,run1\n' >"${result_file}"
printf 'instance,hls,l2s_mean,hls_minus_l2s,status,seed\n' >"${comparison_file}"
for index in "${!instances[@]}"; do
    instance="${instances[$index]}"
    log_file="${LOG_DIR}/${instance}.txt"
    makespan="$(awk '/^Final makespan:/ {value=$3} END {print value}' "${log_file}")"
    if [[ ! "${makespan}" =~ ^[0-9]+$ ]]; then
        echo "Missing final makespan in ${log_file}." >&2
        exit 1
    fi
    difference="$(awk -v h="${makespan}" -v l="${l2s_mean[$instance]}" \
        'BEGIN {printf "%.1f", h-l}')"
    status="$(awk -v d="${difference}" \
        'BEGIN {print d < 0 ? "BETTER" : (d == 0 ? "TIE" : "WORSE")}')"
    printf '%s,%s\n' "${instance}" "${makespan}" >>"${result_file}"
    printf '%s,%s,%s,%s,%s,%s\n' \
        "${instance}" "${makespan}" "${l2s_mean[$instance]}" \
        "${difference}" "${status}" "${seeds[$index]}" \
        >>"${comparison_file}"
done

model_sha256="$(sha256sum "${MODEL_PATH}" | awk '{print $1}')"
executable_sha256="$(sha256sum "${EXECUTABLE}" | awk '{print $1}')"
{
    printf 'EXECUTABLE=%q\n' "${EXECUTABLE}"
    printf 'EXECUTABLE_SHA256=%q\n' "${executable_sha256}"
    printf 'DATA_DIR=%q\n' "${DATA_DIR}"
    printf 'MODEL_PATH=%q\n' "${MODEL_PATH}"
    printf 'MODEL_SHA256=%q\n' "${model_sha256}"
    printf 'INSTANCES=%q\n' "${instances[*]}"
    printf 'PARALLEL_JOBS=%q\n' "${PARALLEL_JOBS}"
    printf 'SEARCH_MODE=%q\n' "${SEARCH_MODE}"
    printf 'RANDOM_STRATEGY=%q\n' "${RANDOM_STRATEGY}"
    printf 'MAX_REPEAT=%q\n' "${MAX_REPEAT}"
    printf 'MAX_ITER=%q\n' "${MAX_ITER}"
    printf 'POPULATION_SIZE=%q\n' "${POPULATION_SIZE}"
    printf 'TABU_LENGTH=%q\n' "${TABU_LENGTH}"
    printf 'COPT_TIME=%q\n' "${COPT_TIME}"
    printf 'TOTAL_TIME=%q\n' "${TOTAL_TIME}"
    printf 'GREEDY_SCORE_BONUS=%q\n' "${GREEDY_SCORE_BONUS}"
    printf 'GENETIC_SCORE_BONUS=%q\n' "${GENETIC_SCORE_BONUS}"
    printf 'TABU_SCORE_BONUS=%q\n' "${TABU_SCORE_BONUS}"
    printf 'GUIDED_SCORE_BONUS=%q\n' "${GUIDED_SCORE_BONUS}"
    printf 'GUIDED_TOP_K=%q\n' "${GUIDED_TOP_K}"
    printf 'GREEDY_RL_TOP_K=%q\n' "${GREEDY_RL_TOP_K}"
    printf 'TABU_RL_TOP_K=%q\n' "${TABU_RL_TOP_K}"
    printf 'GREEDY_RL_RULE_FRACTION=%q\n' "${GREEDY_RL_RULE_FRACTION}"
    printf 'GREEDY_RL_EXPLORATION=%q\n' "${GREEDY_RL_EXPLORATION}"
    printf 'TABU_RL_RULE_FRACTION=%q\n' "${TABU_RL_RULE_FRACTION}"
    printf 'TABU_RL_EXPLORATION=%q\n' "${TABU_RL_EXPLORATION}"
    printf 'TABU_RL_MIN_PER_NEIGHBORHOOD=%q\n' \
        "${TABU_RL_MIN_PER_NEIGHBORHOOD}"
    printf 'TABU_RL_FALLBACK_ON_NON_IMPROVEMENT=%q\n' \
        "${TABU_RL_FALLBACK_ON_NON_IMPROVEMENT}"
    printf 'HLS_TARGET_MAKESPAN=0\n'
    printf 'HLS_ENABLE_GUIDED_SEARCH=force\n'
    printf 'HLS_GREEDY_RL=1\n'
    printf 'HLS_GREEDY_RL_DYNAMIC_TOP_K=0\n'
    printf 'HLS_GREEDY_RL_FALLBACK_ON_MISS=1\n'
    printf 'HLS_GREEDY_RL_VERIFY_WITH_BASELINE=0\n'
    printf 'HLS_TABU_RL=1\n'
    printf 'HLS_TABU_RL_DYNAMIC_TOP_K=0\n'
    printf 'HLS_TABU_RL_FALLBACK_ON_MISS=1\n'
    printf 'HLS_TABU_RL_FALLBACK_ON_NON_IMPROVEMENT=%q\n' \
        "${TABU_RL_FALLBACK_ON_NON_IMPROVEMENT}"
    printf 'HLS_TABU_RL_VERIFY_FULL_NEIGHBORHOOD=0\n'
    printf 'COMMAND_TEMPLATE=%q\n' \
        "${EXECUTABLE} ${DATA_DIR}/%INSTANCE%.fjs ${SEARCH_MODE} ${RANDOM_STRATEGY} ${MAX_REPEAT} ${MAX_ITER} ${POPULATION_SIZE} ${TABU_LENGTH} ${COPT_TIME} ${TOTAL_TIME} %SEED%"
    for index in "${!instances[@]}"; do
        printf 'SEED_%s=%q\n' "${instances[$index]}" "${seeds[$index]}"
    done
} >"${LOG_DIR}/best-config.env"

trap - EXIT INT TERM
echo "Result file: ${result_file}"
echo "L2S comparison: ${comparison_file}"

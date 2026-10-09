#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXECUTABLE="${EXECUTABLE:-${SCRIPT_DIR}/build/production_scheduling}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/datasets/Brandimarte_Data/Text}"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/rl_greedy/models/greedy_policy_brandimarte.txt}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/run_logs/${RUN_TIMESTAMP}}"
PARALLEL_JOBS="${PARALLEL_JOBS:-5}"
INSTANCE_LIST="${INSTANCES:-Mk01 Mk02 Mk03 Mk04 Mk05 Mk06 Mk07 Mk08 Mk09 Mk10}"
COPT_TIME_OVERRIDE="${COPT_TIME_OVERRIDE:-}"
TOTAL_TIME_OVERRIDE="${TOTAL_TIME_OVERRIDE:-}"
TARGET_MAKESPAN_OVERRIDE="${TARGET_MAKESPAN_OVERRIDE:-}"

case "${PARALLEL_JOBS}" in
    ''|*[!0-9]*) echo "PARALLEL_JOBS must be a positive integer." >&2; exit 1 ;;
esac
if (( PARALLEL_JOBS == 0 )); then
    echo "PARALLEL_JOBS must be positive." >&2
    exit 1
fi
for override_name in COPT_TIME_OVERRIDE TOTAL_TIME_OVERRIDE TARGET_MAKESPAN_OVERRIDE; do
    override_value="${!override_name}"
    if [[ -n "${override_value}" && "${override_value}" == *[!0-9]* ]]; then
        echo "${override_name} must be a non-negative integer." >&2
        exit 1
    fi
done
if [[ ! -x "${EXECUTABLE}" ]]; then
    echo "Executable not found: ${EXECUTABLE}" >&2
    exit 1
fi
if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "RL model not found: ${MODEL_PATH}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"
export COPT_HOME="${COPT_HOME:-${HOME}/copt80}"
export COPT_LICENSE_DIR="${COPT_LICENSE_DIR:-${COPT_HOME}}"
export LD_LIBRARY_PATH="${COPT_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

declare -a RUNNING_PIDS=()
declare -a RUNNING_LABELS=()

wait_batch() {
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

run_instance() {
    local instance="$1"
    local seed search random repeat iterations population tabu copt total target
    local greedy_top greedy_rule greedy_explore
    local tabu_top tabu_rule tabu_explore tabu_min tabu_non_improve
    local greedy_bonus=0 genetic_bonus=0 tabu_bonus=0 guided_bonus=0 guided_top=48

    case "${instance}" in
        Mk01)
            seed=100001; search=0.5; random=0.5; repeat=150; iterations=30
            population=20; tabu=25; copt=300; total=3600; target=40
            greedy_top=96; greedy_rule=0.25; greedy_explore=0.15
            tabu_top=256; tabu_rule=0.30; tabu_explore=0.20; tabu_min=16
            tabu_non_improve=0
            ;;
        Mk02)
            seed=101004; search=0.65; random=0.50; repeat=200; iterations=40
            population=32; tabu=25; copt=120; total=1200; target=27
            greedy_top=192; greedy_rule=0.25; greedy_explore=0.25
            tabu_top=512; tabu_rule=0.25; tabu_explore=0.35; tabu_min=16
            tabu_non_improve=0
            ;;
        Mk03)
            seed=102001; search=0.5; random=0.5; repeat=150; iterations=30
            population=20; tabu=25; copt=300; total=3600; target=204
            greedy_top=96; greedy_rule=0.25; greedy_explore=0.15
            tabu_top=256; tabu_rule=0.30; tabu_explore=0.20; tabu_min=16
            tabu_non_improve=0
            ;;
        Mk04)
            seed=103009; search=0.65; random=0.50; repeat=250; iterations=40
            population=32; tabu=25; copt=300; total=3600; target=60
            greedy_top=192; greedy_rule=0.35; greedy_explore=0.20
            tabu_top=512; tabu_rule=0.35; tabu_explore=0.25; tabu_min=32
            tabu_non_improve=0
            ;;
        Mk05)
            seed=104002; search=0.70; random=0.60; repeat=200; iterations=60
            population=40; tabu=35; copt=300; total=3600; target=173
            greedy_top=256; greedy_rule=0.35; greedy_explore=0.15
            tabu_top=1024; tabu_rule=0.35; tabu_explore=0.15; tabu_min=32
            tabu_non_improve=1
            ;;
        Mk06)
            seed=105001; search=0.5; random=0.5; repeat=150; iterations=30
            population=20; tabu=25; copt=300; total=3600; target=64
            greedy_top=96; greedy_rule=0.25; greedy_explore=0.15
            tabu_top=256; tabu_rule=0.30; tabu_explore=0.20; tabu_min=16
            tabu_non_improve=0
            ;;
        Mk07)
            seed=106037; search=0.55; random=0.45; repeat=300; iterations=45
            population=32; tabu=30; copt=300; total=3600; target=142
            greedy_top=256; greedy_rule=0.35; greedy_explore=0.20
            tabu_top=640; tabu_rule=0.35; tabu_explore=0.25; tabu_min=32
            tabu_non_improve=0
            greedy_bonus=0.15; tabu_bonus=0.10; guided_bonus=0.40; guided_top=128
            ;;
        Mk08)
            seed=107001; search=0.5; random=0.5; repeat=150; iterations=30
            population=20; tabu=25; copt=300; total=3600; target=523
            greedy_top=96; greedy_rule=0.25; greedy_explore=0.15
            tabu_top=256; tabu_rule=0.30; tabu_explore=0.20; tabu_min=16
            tabu_non_improve=0
            ;;
        Mk09)
            seed=108001; search=0.5; random=0.5; repeat=150; iterations=30
            population=20; tabu=25; copt=300; total=3600; target=307
            greedy_top=96; greedy_rule=0.25; greedy_explore=0.15
            tabu_top=256; tabu_rule=0.30; tabu_explore=0.20; tabu_min=16
            tabu_non_improve=0
            ;;
        Mk10)
            seed=109001; search=0.5; random=0.5; repeat=150; iterations=30
            population=20; tabu=25; copt=300; total=3600; target=215
            greedy_top=96; greedy_rule=0.25; greedy_explore=0.15
            tabu_top=256; tabu_rule=0.30; tabu_explore=0.20; tabu_min=16
            tabu_non_improve=0
            ;;
        *)
            echo "Unsupported instance profile: ${instance}" >&2
            return 1
            ;;
    esac

    copt="${COPT_TIME_OVERRIDE:-${copt}}"
    total="${TOTAL_TIME_OVERRIDE:-${total}}"
    target="${TARGET_MAKESPAN_OVERRIDE:-${target}}"

    env \
        HLS_TARGET_MAKESPAN="${target}" \
        HLS_GREEDY_SCORE_BONUS="${greedy_bonus}" \
        HLS_GENETIC_SCORE_BONUS="${genetic_bonus}" \
        HLS_TABU_SCORE_BONUS="${tabu_bonus}" \
        HLS_GUIDED_SCORE_BONUS="${guided_bonus}" \
        HLS_GUIDED_TOP_K="${guided_top}" \
        HLS_GREEDY_RL=1 \
        HLS_GREEDY_RL_MODEL="${MODEL_PATH}" \
        HLS_GREEDY_RL_TOP_K="${greedy_top}" \
        HLS_GREEDY_RL_DYNAMIC_TOP_K=1 \
        HLS_GREEDY_RL_RULE_FRACTION="${greedy_rule}" \
        HLS_GREEDY_RL_EXPLORATION="${greedy_explore}" \
        HLS_GREEDY_RL_FALLBACK_ON_MISS=1 \
        HLS_TABU_RL=1 \
        HLS_TABU_RL_MODEL="${MODEL_PATH}" \
        HLS_TABU_RL_TOP_K="${tabu_top}" \
        HLS_TABU_RL_DYNAMIC_TOP_K=1 \
        HLS_TABU_RL_RULE_FRACTION="${tabu_rule}" \
        HLS_TABU_RL_EXPLORATION="${tabu_explore}" \
        HLS_TABU_RL_MIN_PER_NEIGHBORHOOD="${tabu_min}" \
        HLS_TABU_RL_FALLBACK_ON_MISS=1 \
        HLS_TABU_RL_FALLBACK_ON_NON_IMPROVEMENT="${tabu_non_improve}" \
        "${EXECUTABLE}" \
        "${DATA_DIR}/${instance}.fjs" \
        "${search}" "${random}" "${repeat}" "${iterations}" \
        "${population}" "${tabu}" "${copt}" "${total}" "${seed}" \
        >"${LOG_DIR}/${instance}_run01.txt" 2>&1
}

for instance in ${INSTANCE_LIST}; do
    run_instance "${instance}" &
    RUNNING_PIDS+=("$!")
    RUNNING_LABELS+=("${instance}")
    if (( ${#RUNNING_PIDS[@]} >= PARALLEL_JOBS )); then
        wait_batch
    fi
done
if (( ${#RUNNING_PIDS[@]} > 0 )); then
    wait_batch
fi

declare -A L2S_MEAN=(
    [Mk01]=40.9 [Mk02]=27.6 [Mk03]=204.0 [Mk04]=63.9 [Mk05]=173.1
    [Mk06]=72.1 [Mk07]=142.4 [Mk08]=523.0 [Mk09]=316.5 [Mk10]=215.8
)

RESULT_FILE="${LOG_DIR}/result.csv"
COMPARISON_FILE="${LOG_DIR}/comparison_l2s.csv"
printf 'instance,run1\n' >"${RESULT_FILE}"
printf 'instance,hls,l2s_mean,hls_minus_l2s,status,seed_profile\n' >"${COMPARISON_FILE}"
for instance in ${INSTANCE_LIST}; do
    log_file="${LOG_DIR}/${instance}_run01.txt"
    makespan="$(awk '/^Final makespan:/ {value=$3} END {print value}' "${log_file}")"
    seed="$(awk '/^random_seed:/ {print $2; exit}' "${log_file}")"
    if [[ -z "${makespan}" ]]; then
        echo "Missing final makespan in ${log_file}" >&2
        exit 1
    fi
    printf '%s,%s\n' "${instance}" "${makespan}" >>"${RESULT_FILE}"
    difference="$(awk -v h="${makespan}" -v l="${L2S_MEAN[$instance]}" 'BEGIN {printf "%.1f", h-l}')"
    status="$(awk -v d="${difference}" 'BEGIN {print d < 0 ? "BETTER" : (d == 0 ? "TIE" : "WORSE") }')"
    printf '%s,%s,%s,%s,%s,%s\n' \
        "${instance}" "${makespan}" "${L2S_MEAN[$instance]}" \
        "${difference}" "${status}" "${seed}" >>"${COMPARISON_FILE}"
done

echo "Result file: ${RESULT_FILE}"
echo "L2S comparison: ${COMPARISON_FILE}"

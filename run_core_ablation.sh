#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CMAKE_COMMAND="${CMAKE:-${HOME}/.local/bin/cmake}"
PYTHON_COMMAND="${PYTHON:-${HOME}/miniconda3/bin/python}"
BUILD_JOBS="${BUILD_JOBS:-8}"
COPT_ROOT="${COPT_HOME:-${HOME}/copt80}"

if [[ ! -x "$CMAKE_COMMAND" ]]; then
    CMAKE_COMMAND="$(command -v cmake || true)"
fi
if [[ ! -x "$PYTHON_COMMAND" ]]; then
    PYTHON_COMMAND="$(command -v python3 || true)"
fi
if [[ -z "$PYTHON_COMMAND" || ! -x "$PYTHON_COMMAND" ]]; then
    echo "Error: Python was not found; set PYTHON=/path/to/python." >&2
    exit 2
fi

export COPT_HOME="$COPT_ROOT"
export COPT_LICENSE_DIR="${COPT_LICENSE_DIR:-$COPT_ROOT}"
export LD_LIBRARY_PATH="$COPT_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
    if [[ ! -x "$ROOT_DIR/build/ablation/production_scheduling" ||
          ! -x "$ROOT_DIR/build/ablation/test_adaptive_method_ablation" ]]; then
        echo "Error: SKIP_BUILD=1 but ablation binaries are missing." >&2
        exit 2
    fi
elif [[ -n "$CMAKE_COMMAND" ]] && "$CMAKE_COMMAND" --version >/dev/null 2>&1; then
    "$CMAKE_COMMAND" -S "$ROOT_DIR" -B "$ROOT_DIR/build/ablation" \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
    "$CMAKE_COMMAND" --build "$ROOT_DIR/build/ablation" \
        --target production_scheduling test_adaptive_method_ablation \
        -j "$BUILD_JOBS"
else
    CXX_COMMAND="${CXX:-$(command -v g++ || true)}"
    if [[ -z "$CXX_COMMAND" || ! -x "$CXX_COMMAND" ]]; then
        echo "Error: neither a working CMake nor g++ was found." >&2
        exit 2
    fi
    mkdir -p "$ROOT_DIR/build/ablation"
    TEMP_TEST_BINARY="$ROOT_DIR/build/ablation/.test_adaptive_method_ablation.tmp.$$"
    TEMP_SOLVER_BINARY="$ROOT_DIR/build/ablation/.production_scheduling.tmp.$$"
    trap 'rm -f -- "$TEMP_TEST_BINARY" "$TEMP_SOLVER_BINARY"' EXIT
    "$CXX_COMMAND" -std=c++20 -O2 -DNDEBUG -I"$ROOT_DIR" \
        "$ROOT_DIR/tests/test_adaptive_method_ablation.cpp" \
        "$ROOT_DIR/AdaptiveSearch.cpp" \
        -o "$TEMP_TEST_BINARY"
    "$CXX_COMMAND" -std=c++20 -O2 -DNDEBUG \
        -I"$ROOT_DIR" -I"$COPT_ROOT/include" \
        -I"$COPT_ROOT/include/coptcpp_inc" \
        "$ROOT_DIR/main.cpp" "$ROOT_DIR/AdaptiveSearch.cpp" \
        "$ROOT_DIR/RouteRepairProfile.cpp" "$ROOT_DIR/Order.cpp" \
        "$ROOT_DIR/Job.cpp" "$ROOT_DIR/Machine.cpp" \
        "$ROOT_DIR/Schedule.cpp" "$ROOT_DIR/DataProc.cpp" \
        "$ROOT_DIR/GreedySearch.cpp" "$ROOT_DIR/GreedyRlPolicy.cpp" \
        "$ROOT_DIR/GuidedSearch.cpp" "$ROOT_DIR/RandomSearch.cpp" \
        "$ROOT_DIR/TabuSearch.cpp" "$ROOT_DIR/TabuRlPolicy.cpp" \
        "$ROOT_DIR/Experiment.cpp" "$ROOT_DIR/CoptInitialSolution.cpp" \
        -L"$COPT_ROOT/lib" -Wl,-rpath,"$COPT_ROOT/lib" \
        -lcopt_cpp -pthread -static-libgcc -static-libstdc++ \
        -o "$TEMP_SOLVER_BINARY"
    mv -f -- "$TEMP_TEST_BINARY" \
        "$ROOT_DIR/build/ablation/test_adaptive_method_ablation"
    mv -f -- "$TEMP_SOLVER_BINARY" \
        "$ROOT_DIR/build/ablation/production_scheduling"
    trap - EXIT
fi
"$ROOT_DIR/build/ablation/test_adaptive_method_ablation"

exec "$PYTHON_COMMAND" -u "$ROOT_DIR/run_core_ablation.py" \
    --executable "$ROOT_DIR/build/ablation/production_scheduling" "$@"

#!/usr/bin/env bash
# Staged experiment grid for DSM Benchmark B v3.
#
# Usage:
#   bash scripts/run_v3_grid.sh <stage> [init_eta_out] [init_temperature]
#
# Stages (run in order; only move on when the previous one passes):
#   capacity : fixed 8/16/32/64, metacognition OFF  -> is capacity a bottleneck?
#   metacog  : fixed 16, metacognition ON vs OFF     -> does doubt help learning?
#   growth   : grow / full / nolearn / basic from 16 -> does development help?
#
# Take init_eta_out / init_temperature from the rule-test VERDICT line.
# Summarise with:
#   python benchmarks/dsm_benchmark_b_v3.py --mode summarize --results-dir results_v3

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="$ROOT/benchmarks/dsm_benchmark_b_v3.py"

STAGE="${1:?stage: capacity | metacog | growth}"
ETA="${2:-1.0}"
TEMP="${3:-0.3}"

SEEDS="${SEEDS:-0 1 2}"
GENERATIONS="${GENERATIONS:-100}"
POPULATION="${POPULATION:-64}"
LIVES="${LIVES:-32}"

COMMON=(--generations "$GENERATIONS" --population "$POPULATION" --lives "$LIVES"
        --init-eta-out "$ETA" --init-temperature "$TEMP")

run() {
  echo ">>> $*"
  python "$BENCH" "${COMMON[@]}" "$@"
}

for seed in $SEEDS; do
  case "$STAGE" in
    capacity)
      for n in 8 16 32 64; do
        run --structure fixed --hidden "$n" --no-metacognition --seed "$seed" \
            --results "results_v3/capacity_fixed-${n}_seed${seed}.json"
      done
      ;;
    metacog)
      run --structure fixed --hidden 16 --no-metacognition --seed "$seed" \
          --results "results_v3/metacog_off_seed${seed}.json"
      run --structure fixed --hidden 16 --seed "$seed" \
          --results "results_v3/metacog_on_seed${seed}.json"
      ;;
    growth)
      for mode in grow full nolearn basic; do
        run --structure "$mode" --hidden 16 --seed "$seed" \
            --results "results_v3/growth_${mode}-16_seed${seed}.json"
      done
      ;;
    *)
      echo "unknown stage: $STAGE" >&2
      exit 1
      ;;
  esac
done

#!/usr/bin/env bash
# Metacognition grid for DSM Benchmark B v4: does evolution make the
# self-model matter when errors are ambiguous (reward noise, reversals)?
#
# Usage:
#   bash scripts/run_v4_meta_grid.sh
#   SEEDS="0 1" GENERATIONS=60 bash scripts/run_v4_meta_grid.sh
# Then:
#   python benchmarks/dsm_benchmark_b_v4.py --mode summarize --results-dir results_v4_meta
#
# Success = full-16 clearly beats basic-16 under noise+reversal, with the gap
# larger after evolution than in the hand-built comparison.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="$ROOT/benchmarks/dsm_benchmark_b_v4.py"

SEEDS="${SEEDS:-0 1 2}"
GENERATIONS="${GENERATIONS:-100}"
POPULATION="${POPULATION:-64}"
LIVES="${LIVES:-32}"
CONFIGS="${CONFIGS:-fixed-128 full-16 nolearn-16 basic-16}"
EVAL_LIVES="${EVAL_LIVES:-1000}"

TASK=(--steps 480 --rehearsal-fraction 0.5 --fault-probability 0
      --init-repair-slot 20 --init-repair-wrong 2 --init-gate-slot 6
      --eval-lives "$EVAL_LIVES")
EVOLVE=(--generations "$GENERATIONS" --population "$POPULATION" --lives "$LIVES"
        --results-dir results_v4_meta)

CONDITIONS=(
  ""
  "--reward-noise 0.15"
  "--reversal"
  "--reward-noise 0.15 --reversal"
)

echo ">>> hand-built comparison (no evolution) for each condition"
for cond in "${CONDITIONS[@]}"; do
  # shellcheck disable=SC2086
  python "$BENCH" --mode hand --seeds 0,1,2 --results-dir "" \
    "${TASK[@]}" $cond --configs "fixed-16,fixed-128,full-16,nolearn-16,basic-16"
done

for seed in $SEEDS; do
  for cond in "${CONDITIONS[@]}"; do
    for spec in $CONFIGS; do
      echo ">>> $spec seed $seed [$cond]"
      # shellcheck disable=SC2086
      python "$BENCH" --structure "${spec%-*}" --units "${spec##*-}" \
        --seed "$seed" "${TASK[@]}" "${EVOLVE[@]}" $cond
    done
  done
done

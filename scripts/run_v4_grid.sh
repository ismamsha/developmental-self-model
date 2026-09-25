#!/usr/bin/env bash
# Evolution grid for DSM Benchmark B v4 over several seeds.
#
# Usage:
#   bash scripts/run_v4_grid.sh            # all configurations
#   SEEDS="0 1" GENERATIONS=50 bash scripts/run_v4_grid.sh
#
# Then:
#   python benchmarks/dsm_benchmark_b_v4.py --mode summarize --results-dir results_v4
#
# Run `python benchmarks/dsm_benchmark_b_v4.py --mode hand` first: it compares the same
# configurations with the hand-built genome in minutes, without evolution.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="$ROOT/benchmarks/dsm_benchmark_b_v4.py"

SEEDS="${SEEDS:-0 1 2}"
GENERATIONS="${GENERATIONS:-100}"
POPULATION="${POPULATION:-64}"
LIVES="${LIVES:-32}"
CONFIGS="${CONFIGS:-fixed-16 fixed-32 fixed-64 fixed-128 random-16 imprint-16 full-16 nolearn-16 basic-16}"

for seed in $SEEDS; do
  for spec in $CONFIGS; do
    mode="${spec%-*}"
    units="${spec##*-}"
    echo ">>> $spec seed $seed"
    python "$BENCH" --structure "$mode" --units "$units" --seed "$seed" \
      --generations "$GENERATIONS" --population "$POPULATION" --lives "$LIVES" \
      --results-dir results_v4
  done
done

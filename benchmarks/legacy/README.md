# Legacy benchmarks

These versions are kept so the research log in [`docs/RESULTS.md`](../../docs/RESULTS.md)
can be reproduced. They are not maintained.

| File | What it is |
|---|---|
| `dsm_benchmark_b_v1.py` | Original dense recurrent benchmark (6 × 6 task) with the first learning-rule fixes and `easy` / `full` presets. |
| `dsm_benchmark_b_v2.py` | v2: eligibility traces, rehearsal curriculum, nonlethal errors. Reached 35.5% on 4 × 4 × 4, which a cue-blind action prior can match. |

Run them from this directory, e.g. `python dsm_benchmark_b_v2.py --help`.

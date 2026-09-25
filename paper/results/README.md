# Results used in the paper

Every number in `paper/main.tex` traces to one of these files or to the research branches named in the paper.

| File | Produced by | Paper section |
|---|---|---|
| `msra_audit.json`, `msra_audit_raw.tar.gz` (per-run records), `main_py_crosscheck.txt` | `experiments/msra_experiments.py --episodes 100 --seeds 0 1 2`, then `experiments/summarize_msra.py` | Audit of `main.py` (Table 2) |
| `v4_standard/hand_comparison.json`, `v4_standard.txt` | `python experiments/dsm_benchmark_b_v4_snapshot.py --mode hand --device cpu --seeds 0,1,2 --eval-lives 400 --results-dir results/v4_standard` | V4 standard task |
| `v4_meta/hand_comparison.json`, `v4_meta.txt` | same script with `--eval-lives 200 --steps 480 --rehearsal-fraction 0.5 --fault-probability 0 --init-repair-slot 20 --init-repair-wrong 2 --init-gate-slot 6 --reward-noise 0.15 --reversal --configs fixed-16,fixed-128,full-16,nolearn-16,basic-16` | V4.1 metacognition test |
| `learnability_validation.json`, `.txt` | `experiments/learnability_validation.py --configs fixed-16,full-16 --seeds 0,1,2 --lives 400 --boot 300` | Prospective learnability test |
| `mrc_toy.json` | `experiments/mrc_toy.py` | Anchoring simulation |

`experiments/dsm_benchmark_b_v4_snapshot.py` (MSRA repository only) is an unmodified copy of `dsm_benchmark_b_v4.py`
from branch `claude/quirky-wright-6hfp0i` (commit `26cc897`) with one comment line added at the top. In the
developmental-self-model repository, use `benchmarks/dsm_benchmark_b_v4.py` instead; it is the same code apart from
paths in its docstring.

Part I results (MSRA-L, hidden faults, evolved organisms, survival ceiling) are in branch
`claude/research-opinion-ly94jb` (commit `d185ccf`) under `results/`.

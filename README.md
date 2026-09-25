# Developmental Self-Model (DSM)

[![CI](https://github.com/ismamsha/developmental-self-model/actions/workflows/ci.yml/badge.svg)](https://github.com/ismamsha/developmental-self-model/actions/workflows/ci.yml)

A benchmark and reference implementation for **organisms that learn during their
lifetime without backpropagation and decide for themselves when to build more
neural capacity**.

Each organism is born with random wiring and has to learn a new, random set of
task mappings in every life. It learns only through local, reward-modulated
plasticity. Evolution Strategies optimise the inherited genome across
generations: learning rules, neuromodulation, self-model rates and
developmental rules.

The research question:

> Can an organism detect that its current learning machinery is insufficient and
> construct new, useful capacity, approaching a much larger fixed brain while
> using far fewer neurons?

---

## Paper

**What Can an Agent Know About Its Own Learning? Endogenous Self-Modeling, Anchored Metacognition, and the
Addressability Bottleneck in a Developmental Self-Model Research Program** —
[`paper/main.pdf`](paper/main.pdf) (LaTeX source, analysis scripts and outputs in [`paper/`](paper)).

The paper reports the whole program, including the failed versions, and labels every claim as demonstrated,
suggestive, negative, hypothesis or proposal. Beyond the results below, it adds a prospective test showing that
the learnability signal tracks past progress but does not forecast future progress, and it treats the evolved
growth results as suggestive until more seeds are run.

---

## Key results

All numbers are for 4 tasks × 4 cues × 4 actions with a one-step delay and
240-step lives. Random chance is **25%**. The **cue-blind** baseline is ~50%:
the best strategy that ignores the cue and picks the most frequent answer in
each phase. A perfect-memory learner reaches **~80%**.

### Evolved genomes (Evolution Strategies)

| Configuration | Accuracy | Avg / final units | Phase 1 → 4 | Task 0 retained |
|---|---|---|---|---|
| fixed-16 | 54.2% | 16 / 16 | 63.7 → 46.5% | 74.3% |
| fixed-64 | 62.0% | 64 / 64 | 69.0 → 57.0% | 83.8% |
| fixed-128 | 64.5% | 128 / 128 | 70.4 → 60.6% | 86.2% |
| random growth from 16 | 54.8% | 39 / 63 | 61.5 → 50.3% | 58.6% |
| **imprint growth from 16** | **67.1%** | **24 / 30** | 62.7 → **69.1%** | 85.7% |

Seeds per row: 2 for the fixed and random rows, 1 for imprint. The spread
between seeds is below 1 point. These runs used v4 before the repair and
plasticity-gate genes were added (a 53-parameter genome). Raw files are in
[`results/v4_standard/`](results/v4_standard).

- **Functional growth beats a 5× larger fixed brain.** The imprint organism
  starts with 16 units and scores 67.1%, against 64.5% for fixed-128, while
  using ~24 units on average.
- **Accuracy improves as tasks accumulate** (63% → 69%). Fixed brains degrade
  instead (70% → 61%).
- **Random growth does not help.** Adding units with random wiring gives no
  gain and hurts retention: growth has to create *useful* structure, not just
  more neurons.

### Does the self-model help?

In the standard task, removing self-model signals from the controller changes
nothing: evolved `full` = `nolearn` = `basic` ≈ 67%. The reason is that every
error there means "this pattern has no slot yet", so a simple novelty rule is
already near-optimal.

To test metacognition, the benchmark adds conditions where an error is
ambiguous: reward noise, and mid-life reversals of already-learned mappings.
It also gives the organism two levers: **repair**, which resets the slot that
owns a pattern, and **surprise-gated plasticity**. Hand-built genomes, 480-step
lives, noise 0.15 plus reversal:

| Configuration | Accuracy | Accuracy on remapped patterns |
|---|---|---|
| fixed-128 | 48.6% | 15.7% |
| **full-16** | **53.7%** | **29.4%** |
| basic-16 (no self-model signals) | 52.0% | 12.1% |

Without self-model signals the organism cannot find which of its slots are
broken, and remapped patterns stay below chance. The useful signal is per-slot
**surprise**, meaning confident errors. Global learnability adds nothing so
far (`nolearn` = `full`). Under noise alone the hand-set repair policy
over-triggers. Whether evolution tunes that away is the open experiment:
[`scripts/run_v4_meta_grid.sh`](scripts/run_v4_meta_grid.sh).

The full research log, including the three failed architectures and what each
one taught, is in [`docs/RESULTS.md`](docs/RESULTS.md).

---

## Architecture (v4)

```
 instruction step                     response step
 ────────────────                     ─────────────
 task + cue ──► sparse expansion ──► working memory ──► associative readout ──► action
                (k-winners-take-all   (holds the code     W[unit, action]
                 over active units)    across the delay)       ▲
                                                               │ three-factor rule
                                                               │ dW = η·m·code⊗(onehot(a) − π)
 self-model ───────► doubt, slot surprise, learnability ───────┤
 (predicts energy,    │                                        │ m = neuromodulator(RPE, doubt, …)
  damage, reward)     ▼                                        │ η gated by slot surprise
              structural controller ──► GROW   : imprint a new unit on this pattern
                                        REPAIR : reset the unit that owns it
                                        PRUNE  : remove low-utility units
```

- **No backpropagation during life.** All weight changes are local:
  pre-activity × post-activity × neuromodulator.
- **Evolution optimises a 70-parameter genome**: rule coefficients, learning
  rate, temperature, modulator, self-model rates, and the
  grow/prune/repair/gate controllers.
- **Common random numbers.** Every candidate genome in a generation meets the
  same lives, so ES compares genomes rather than luck.

### Structural modes

| Mode | Growth | Pruning | Controller inputs |
|---|---|---|---|
| `fixed` | – | – | – |
| `random` | random wiring | – | all |
| `imprint` | imprinted slots | – | all |
| `full` | imprinted slots | ✓ | all |
| `nolearn` | imprinted slots | ✓ | no learnability |
| `basic` | imprinted slots | ✓ | no self-model signals at all |

---

## Repository layout

```
benchmarks/
  dsm_benchmark_b_v4.py     current benchmark: sparse associative memory + development
  dsm_benchmark_b_v3.py     dense recurrent baseline with diagnostics (rule-test, probes)
  legacy/                   v1 and v2, kept for reproducibility of the research log
scripts/
  run_v4_grid.sh            capacity + structural ablations over seeds
  run_v4_meta_grid.sh       metacognition conditions (noise, reversal) over seeds
  run_v3_grid.sh            staged v3 experiments
results/v4_standard/        evolved and hand-built results behind the tables above
docs/RESULTS.md             research log: every version, number and lesson
tests/                      behavioural tests (controls at chance, learning, growth, …)
```

Each benchmark is a single standalone file with no project imports.

---

## Getting started

```bash
pip install -r requirements.txt          # numpy, torch (CUDA optional)
```

**1. Hand-built comparison.** No evolution. Takes about a minute on a GPU and
reproduces the capacity and growth results:

```bash
python benchmarks/dsm_benchmark_b_v4.py --mode hand --seeds 0,1,2
```

**2. Evolve one configuration:**

```bash
python benchmarks/dsm_benchmark_b_v4.py --structure full --units 16 --generations 100
```

**3. Full grids over seeds, then summarise:**

```bash
bash scripts/run_v4_grid.sh
python benchmarks/dsm_benchmark_b_v4.py --mode summarize --results-dir results_v4

bash scripts/run_v4_meta_grid.sh
python benchmarks/dsm_benchmark_b_v4.py --mode summarize --results-dir results_v4_meta
```

Useful flags: `--reward-noise P`, `--reversal`, `--steps N`, `--k K`,
`--pool N`, `--random-policy` and `--no-plasticity` (controls). Run with
`--help` for the full list.

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite runs on CPU in about 10 seconds. It checks that both controls score
chance, that the learner beats the cue-blind baseline, that more capacity
helps, that growth is bounded, that the reversal and noise mechanics work,
that self-model signals improve accuracy after reversals, and that the ES loop
runs.

---

## Status and open questions

- ✅ Local lifetime learning well above the cue-blind baseline.
- ✅ Capacity scaling (8 → 128 units) and functional growth that matches a
  larger fixed brain with ~5× fewer neurons.
- 🔬 **Self-model contribution.** Per-slot surprise helps after reversals.
  Whether evolution makes it robust under noise, and whether global
  learnability adds anything, is being tested.
- 🔬 The remaining gap to the perfect-learner ceiling (67% vs ~80%), mostly in
  phase 1.

# Research log

This log covers how the benchmark reached its current form: each version, what
it measured, and what the result implied for the next step. Numbers are
accuracy on response trials unless stated otherwise.

## Measurement conventions

- **Chance**: 1 / number of actions.
- **Cue-blind baseline**: the best fixed action in each phase, chosen with
  hindsight per life. A learner that only learns "which answer is most common"
  can at best reach this. Accuracy has to clearly exceed it before we can
  claim the task × cue mapping is learned.
- **Perfect-learner ceiling**: an agent with perfect memory that tries untried
  actions for each (task, cue) until rewarded, then repeats the rewarded one.
  It is computed on the same lives.
- **Controls**: `--random-policy` and `--no-plasticity` must both score chance.

| Setting | Chance | Cue-blind | Ceiling |
|---|---|---|---|
| 4 tasks × 6 cues × 6 actions, 240 steps | 16.7% | – | 62.7% |
| 4 × 4 × 4, 240 steps, 30% rehearsal | 25% | ~50% | ~80% |
| 4 × 4 × 4, 480 steps, 50% rehearsal | 25% | ~47% | ~90% |

---

## v1: dense recurrent network, Hebbian plasticity

`benchmarks/legacy/dsm_benchmark_b_v1.py` (6 × 6 task)

**Result:** 16.2% against 16.7% chance over 50 generations. Accuracy was flat,
and ES was only shrinking the network to avoid the complexity cost.

**Causes:**
- The lifetime learning rate at initialisation was about 1e-4 per step.
- The modulator had a constant bias and no dependence on reward, so
  plasticity was blind Hebbian.
- Deterministic argmax meant there was no exploration.
- Plasticity also ran on instruction steps.
- Evolution had no gradient to follow, because every genome scored chance.

The file in `legacy/` already contains the first fixes (RPE modulator,
exploration, per-trial eligibility). On an easy 1 × 3 × 3 preset these reached
76% at generation 0.

## v2: eligibility traces, curriculum, nonlethal errors

`benchmarks/legacy/dsm_benchmark_b_v2.py` (4 × 4 × 4)

**Result:** 35.5% against 25% chance, with accuracy peaking around step 60 and
then declining.

**Diagnosis:** a cue-blind strategy can score as well as this (40% for one
action per life, 50% for one action per phase). The peak-then-decline pattern
matches a phase-specific action prior being made obsolete at each phase
boundary, rather than forgetting. The main bug: eligibility decayed 0.85 per
step without a reset, so the past trials together received about 2.6× the
credit of the current one.

## v3: fixed learning rule and diagnostics

`benchmarks/dsm_benchmark_b_v3.py`

**Changes:**
- Eligibility resets every trial.
- The modulator is driven by reward-prediction error.
- The output rule uses `onehot(action) − policy`.
- The temperature is evolved.
- Recurrent gain is scaled by √(H / active).
- The cue input is amplified.
- Learnability is behavioural: fast minus slow accuracy EMA.
- A rule-test mode, cue-blind metrics and linear probes were added.

**Results** (hand-built rule, 200 lives):

| Variant | Accuracy | Cue-blind | Probe: task × cue | Probe: cue |
|---|---|---|---|---|
| base | 37.6% | 49.5% | 63.4% (baseline 8.9%) | 93.9% (baseline 24%) |
| centered readout | 40.2% | 49.5% | 60.2% | 94.4% |
| leaky units | 34.3% | 49.5% | 53.4% | 88.1% |
| no delay (diagnostic) | 40.6–43.7% | 49.5% | ~71% | ~98% |

Evolving fixed-16 for 100 generations gave 39.3%, still below cue-blind.

**Conclusion:** the cue is present in the representation, so memory is not the
bottleneck. Dense, slow, reward-modulated updates of shared weights cannot
learn 16 conjunctions from about 7 exposures each without them overwriting
one another. A different lifetime learner was needed, not a larger network or
more evolution.

## v4: sparse expansion, associative memory, imprint growth

`benchmarks/dsm_benchmark_b_v4.py`

task + cue → k-winners-take-all expansion → working memory → associative
readout (same three-factor rule). GROW imprints a new unit on the current
pattern, and only if no unit already owns it.

### Hand-built genome (3 seeds × 1000 lives, GPU)

| Config | Accuracy | Avg / final units | Task 0 retained |
|---|---|---|---|
| controls | 25.1% | 16 | – |
| fixed-8 | 46.8% | 8 | 67.1% |
| fixed-16 | 50.8% | 16 | 73.0% |
| fixed-32 | 54.0% | 32 | 78.2% |
| fixed-64 | 56.7% | 64 | 81.8% |
| fixed-128 | 58.6% | 128 | 85.0% |
| random growth from 16 | 50.5% | 42 / 68 | 56.6% |
| imprint growth from 16 | 58.5% | 24 / 30 | 85.7% |
| basic-16 (no self-model) | 56.6% | 23 / 29 | 82.1% |

Standard deviation across seeds was ≤ 0.5 everywhere.

### Evolved (ES, pop 64, 32 lives, 100 generations)

| Config | Accuracy | Gain over hand-built | Phase 1 → 4 |
|---|---|---|---|
| fixed-16 | 54.2% | +3.4 | 63.7 → 46.5% |
| fixed-128 | 64.5% | +5.9 | 70.4 → 60.6% |
| random-16 | 54.8% | +4.3 | 61.5 → 50.3% |
| imprint-16 | 67.1% | +8.6 | 62.7 → 69.1% |
| full-16 | 67.0% | +8.6 | 62.6 → 68.7% |
| nolearn-16 | 66.8% | +8.3 | 62.5 → 68.6% |
| basic-16 | 67.1% | +10.5 | 62.9 → 68.9% |

**Conclusions:**
1. Capacity helps at every size from 8 to 128.
2. Growth that allocates dedicated slots matches or beats a 5× larger fixed
   brain.
3. Growth with random wiring does not help.
4. In this task the self-model signals are redundant, because every error
   means "missing slot".

## v4.1: making errors ambiguous

Added in v4:
- `--reward-noise` (perceived feedback flipped; fitness uses the true outcome).
- `--reversal` (known mappings change mid-life).
- REPAIR: reset the owning slot.
- A plasticity gate.
- Per-slot surprise: an EMA of error × confidence in the chosen action.
- Ablations now remove self-model inputs from growth, repair, gate and
  modulator.

**What we learned along the way:**
- Noise alone cannot cause wasted growth, because the novelty gate allows one
  slot per pattern. Metacognition only matters once there is a lever that acts
  on slots that already exist.
- A plain per-slot error rate cannot separate "still learning" from "broken",
  so repair fired too often. Weighting errors by confidence (surprise) fixes
  that.
- At 240 steps a reversal leaves about one trial per remapped pattern, so no
  strategy can relearn. The metacognition tests use 480 steps and 50%
  rehearsal.
- Without repair, remapped patterns score about 10%, below chance: a
  confident slot hardly updates, because `onehot − policy ≈ 0`.

**Hand-built results** (480 steps, 50% rehearsal, 200 lives):

| Condition | full-16 | basic-16 | Remapped: full / basic |
|---|---|---|---|
| noise 0.15 + reversal | 53.7% | 52.0% | 29.4% / 12.1% |

Per condition, 40 lives × 3 seeds, full − basic:

| Condition | full − basic |
|---|---|
| standard | +1.9 |
| noise 0.15 | −2.0 |
| reversal | +5.2 |
| noise + reversal | +2.6 |

**Open:**
- Does evolution widen the full − basic gap under noise + reversal?
- Does it remove the penalty under noise alone?
- Does global learnability ever add anything beyond per-slot surprise? So far
  `nolearn` = `full`.

Run with `scripts/run_v4_meta_grid.sh`.

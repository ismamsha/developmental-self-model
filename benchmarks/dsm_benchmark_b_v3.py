"""
DSM Benchmark B v3 — Developmental Self Model
=============================================

One standalone file. No project imports.

What this tests
---------------
An organism starts with a small recurrent brain. During one lifetime it must:

1) remember delayed instructions,
2) learn random task mappings that are different in every life,
3) preserve old skills as new tasks are introduced (rehearsal curriculum),
4) adapt when hidden body dynamics change,
5) estimate its own prediction error ("doubt"),
6) estimate whether its BEHAVIOUR is still improving ("learnability"),
7) change synapses locally without backpropagation,
8) decide HOLD / GROW / PRUNE for its own neural capacity.

Evolution Strategies optimise only the inherited genome. There is NO gradient
descent during an organism's lifetime.

Changes from v2
---------------
Learning (why v2 plateaued near a cue-blind "favourite action" strategy):
  * Eligibility traces are reset at every instruction step. In v2 they decayed
    0.85/step across trials, so ~72% of each reward was credited to earlier
    trials with different cues -> only an action-frequency prior was learned.
  * Neuromodulator is driven by reward-prediction-error (reward - running
    baseline) instead of raw reward.
  * Output rule post-synaptic term is onehot(action) - policy (local,
    REINFORCE-like), evolved temperature for softmax exploration.
  * Recurrent gain is normalised by active neuron count (sqrt(H / active)),
    so fixed-8/16/32/64 comparisons measure capacity, not memory gain.
  * Task/cue inputs amplified (cue_gain) so the cue survives the delay.

Metacognition / development:
  * Learnability = predicted BEHAVIOURAL progress (fast minus slow accuracy
    EMA), updated on response steps only.
  * Capacity pressure = task error x (1 - learnability) is a structural input.
  * Structural prior HOLD=1.5, GROW=-0.5 (+pressure), PRUNE=-1.5.
  * Newborn neurons get temporarily boosted plasticity.
  * Pruning uses |activity| x |output weights| utility, not raw activity.

Measurement:
  * Accuracy per task per phase (retention of task 0 after new tasks).
  * Cue-blind hindsight baseline per life (best constant action). Accuracy must
    clearly beat this before claiming the mapping is learned.
  * Growth analysis: pressure at GROW vs HOLD ticks, and accuracy change after
    growth vs matched HOLD organisms (difference-in-differences).
  * Controls: --random-policy (exact chance), --no-plasticity (birth wiring).
  * Results saved as JSON; --mode summarize prints mean/std across seeds.

Structural modes (--structure)
------------------------------
  fixed    : constant size = --hidden N (baselines fixed-8/16/32/64)
  grow     : full controller, pruning disabled
  full     : full DSM controller, grow + prune
  nolearn  : grow + prune, controller does not see learnability
  basic    : grow + prune, controller sees no self-model / learnability
             signals (only activity, utility, accuracy, capacity, time)

Requirements
------------
    pip install torch numpy

Research order (each stage gates the next)
------------------------------------------
0. Controls: random policy must equal chance; no plasticity shows what birth
   wiring alone gives.
    python benchmarks/dsm_benchmark_b_v3.py --mode eval --structure fixed --random-policy
    python benchmarks/dsm_benchmark_b_v3.py --mode eval --structure fixed --no-plasticity

1. Rule test: hand-built genome, NO evolution, NO growth, NO doubt.
   Must beat the cue-blind baseline and ideally reach >= 60%.
    python benchmarks/dsm_benchmark_b_v3.py --mode rule-test --fault-probability 0 --sizes 16

2. Evolution on fixed-16, starting from the best rule-test setting.
   Accuracy should rise in the first ~10 generations.
    python benchmarks/dsm_benchmark_b_v3.py --structure fixed --hidden 16 --no-metacognition \\
        --init-eta-out <best> --init-temperature <best>

3. Fixed capacity 8/16/32/64 over seeds, 4. metacognition on,
5. growth / pruning ablations:
    bash scripts/run_v3_grid.sh <stage> <init-eta-out> <init-temperature>
    python benchmarks/dsm_benchmark_b_v3.py --mode summarize --results-dir results_v3

Smoke test:
    python benchmarks/dsm_benchmark_b_v3.py --generations 2 --population 8 --lives 4 --steps 80
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
import warnings
from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# =====================================================================
# CONFIG
# =====================================================================

STRUCTURE_MODES = ("fixed", "grow", "full", "nolearn", "basic")


@dataclass
class Config:
    # Benchmark B
    steps: int = 240                 # must be even
    max_tasks: int = 4
    n_cues: int = 4
    n_actions: int = 4
    fault_probability: float = 0.65
    rehearsal_fraction: float = 0.30  # share of old-task trials after phase 1
    cue_gain: float = 2.0

    # Brain
    max_hidden: int = 128
    initial_hidden: int = 16
    min_hidden: int = 8
    growth_key_dim: int = 16
    max_structural_change: int = 2
    structure: str = "full"          # see STRUCTURE_MODES

    # Development clock
    structural_warmup: int = 24
    structural_interval: int = 12
    newborn_boost: float = 2.0       # extra plasticity for new neurons
    newborn_tau: float = 40.0        # steps for the boost to decay

    # Hand-built starting genome (tune with --mode rule-test first)
    init_eta_out: float = 1.0        # logit; rate = 0.15 * sigmoid(logit)
    init_temperature: float = 0.3

    # Controls
    random_policy: bool = False
    plasticity: bool = True
    metacognition: bool = True       # False = doubt/learnability disconnected

    # Representation (rule-test variants)
    center: bool = False             # read out activity minus its own mean
    retain: float = 0.0              # leaky units: h = r*h_prev + (1-r)*tanh(.)
    no_delay: bool = False           # DIAGNOSTIC: cue also visible at response

    # Evolution
    generations: int = 100
    population: int = 64             # even
    lives: int = 24
    sigma: float = 0.05
    es_lr: float = 0.02
    weight_decay: float = 0.001

    # Fitness
    complexity_cost: float = 0.005
    structural_change_cost: float = 0.002
    survival_bonus: float = 5.0
    death_penalty: float = 5.0
    accuracy_weight: float = 0.0

    # Reproducibility
    seed: int = 0

    @property
    def run_name(self) -> str:
        name = f"{self.structure}-{self.initial_hidden}"
        return name if self.metacognition else name + "-nometa"


# =====================================================================
# DEVICE
# =====================================================================

def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def configure_torch(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


# =====================================================================
# SCENARIOS
# =====================================================================

class Scenarios:
    """
    All randomness for a group of lives is drawn before evaluation.
    Every candidate genome in one ES generation meets the same lives.
    """

    def __init__(self, cfg: Config, n: int, seed: int, device: torch.device):
        assert cfg.steps % 2 == 0, "steps must be even"

        self.n = n
        self.trials = cfg.steps // 2

        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        self.mapping = torch.randint(
            0, cfg.n_actions, (n, cfg.max_tasks, cfg.n_cues),
            generator=gen, device=device,
        )
        self.task_u = torch.rand(n, self.trials, generator=gen, device=device)
        self.cue = torch.randint(
            0, cfg.n_cues, (n, self.trials), generator=gen, device=device
        )

        has_fault = (
            torch.rand(n, generator=gen, device=device) < cfg.fault_probability
        )
        kind = torch.randint(0, 2, (n,), generator=gen, device=device)
        self.fault_kind = torch.where(has_fault, kind, torch.full_like(kind, -1))

        lo = max(20, cfg.steps // 4)
        hi = max(lo + 1, (3 * cfg.steps) // 4)
        onset = torch.randint(lo, hi, (n,), generator=gen, device=device)
        self.fault_onset = torch.where(
            has_fault, onset, torch.full_like(onset, 10**9)
        )
        self.fault_action = torch.randint(
            0, cfg.n_actions, (n,), generator=gen, device=device
        )

        self.shock_u = torch.rand(n, cfg.steps, generator=gen, device=device)
        self.structure_u = torch.rand(n, cfg.steps, generator=gen, device=device)
        self.action_u = torch.rand(n, cfg.steps, generator=gen, device=device)

        self.birth_seed = seed + 917_381


# =====================================================================
# ENVIRONMENT
# =====================================================================

class BenchmarkB:
    """
    Instruction step: task ID + cue visible, no reward, action ignored.
    Response step:    task ID + cue hidden, reward for the correct action.

    Phase 1 uses task 0. Phase p>1: (1 - rehearsal_fraction) newest task,
    rehearsal_fraction spread over older tasks.
    """

    def __init__(self, cfg: Config, scenarios: Scenarios, population: int,
                 device: torch.device):
        self.cfg = cfg
        self.device = device
        self.batch = population * scenarios.n
        B, T, A = self.batch, cfg.max_tasks, cfg.n_actions

        self.mapping = scenarios.mapping.repeat(population, 1, 1)
        self.task_u = scenarios.task_u.repeat(population, 1)
        self.cue_seq = scenarios.cue.repeat(population, 1)
        self.fault_kind = scenarios.fault_kind.repeat(population)
        self.fault_onset = scenarios.fault_onset.repeat(population)
        self.fault_action = scenarios.fault_action.repeat(population)
        self.shock_u = scenarios.shock_u.repeat(population, 1)
        self.structure_u = scenarios.structure_u.repeat(population, 1)
        self.action_u = scenarios.action_u.repeat(population, 1)

        # Energy economy was tuned for 240 steps; keep it balanced for longer
        # lives so correct play can always survive.
        self.energy_scale = min(1.0, 240.0 / cfg.steps)

        self.energy = torch.ones(B, device=device)
        self.damage = torch.zeros(B, device=device)
        self.alive = torch.ones(B, dtype=torch.bool, device=device)
        self.survived = torch.zeros(B, dtype=torch.bool, device=device)

        self.current_task = torch.zeros(B, dtype=torch.long, device=device)
        self.current_cue = torch.zeros(B, dtype=torch.long, device=device)
        self.current_target = torch.zeros(B, dtype=torch.long, device=device)

        self.last_task_reward = torch.zeros(B, device=device)
        self.correct_total = torch.zeros(B, device=device)
        self.response_total = torch.zeros(B, device=device)

        self.phase_correct = torch.zeros(B, T, device=device)
        self.phase_responses = torch.zeros(B, T, device=device)

        # [life, task, phase]: retention of each task over the life.
        self.task_correct = torch.zeros(B, T, T, device=device)
        self.task_responses = torch.zeros(B, T, T, device=device)

        # [life, phase, action]: target counts for the cue-blind baseline.
        self.target_count = torch.zeros(B, T, A, device=device)

        self.rows = torch.arange(B, device=device)
        self.step_number = 0

    @property
    def obs_dim(self) -> int:
        return 5 + self.cfg.max_tasks + self.cfg.n_cues

    def phase_index(self, t: int) -> int:
        phase_len = max(2, self.cfg.steps // self.cfg.max_tasks)
        return min(self.cfg.max_tasks - 1, t // phase_len)

    def _trial_task(self, trial: int, phase: int) -> torch.Tensor:
        if phase == 0:
            return torch.zeros(self.batch, dtype=torch.long, device=self.device)

        u = self.task_u[:, trial]
        split = 1.0 - self.cfg.rehearsal_fraction
        newest = torch.full(
            (self.batch,), phase, dtype=torch.long, device=self.device
        )
        old_u = torch.clamp(
            (u - split) / max(1e-6, self.cfg.rehearsal_fraction), 0.0, 0.999999
        )
        old_task = torch.clamp((old_u * phase).long(), min=0, max=phase - 1)
        return torch.where(u < split, newest, old_task)

    def observation(self) -> Tuple[torch.Tensor, bool, int]:
        t = self.step_number
        is_instruction = (t % 2 == 0)
        trial = t // 2
        phase = self.phase_index(t)

        task_onehot = torch.zeros(self.batch, self.cfg.max_tasks, device=self.device)
        cue_onehot = torch.zeros(self.batch, self.cfg.n_cues, device=self.device)

        if is_instruction:
            self.current_task = self._trial_task(trial, phase)
            self.current_cue = self.cue_seq[:, trial]
            self.current_target = self.mapping[
                self.rows, self.current_task, self.current_cue
            ]
            task_onehot = self.cfg.cue_gain * F.one_hot(
                self.current_task, self.cfg.max_tasks
            ).float()
            cue_onehot = self.cfg.cue_gain * F.one_hot(
                self.current_cue, self.cfg.n_cues
            ).float()
        elif self.cfg.no_delay:
            # Diagnostic only: removes the memory requirement.
            task_onehot = self.cfg.cue_gain * F.one_hot(
                self.current_task, self.cfg.max_tasks
            ).float()
            cue_onehot = self.cfg.cue_gain * F.one_hot(
                self.current_cue, self.cfg.n_cues
            ).float()

        base = torch.stack(
            [
                self.energy,
                self.damage,
                torch.full((self.batch,), t / max(1, self.cfg.steps - 1),
                           device=self.device),
                torch.full((self.batch,), 1.0 if is_instruction else 0.0,
                           device=self.device),
                self.last_task_reward,
            ],
            dim=1,
        )
        return torch.cat([base, task_onehot, cue_onehot], dim=1), is_instruction, phase

    def step(self, action: torch.Tensor, is_instruction: bool, phase: int
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self.step_number
        live = self.alive.clone()
        es = self.energy_scale

        reward = torch.zeros(self.batch, device=self.device)
        correct_mask = torch.zeros(self.batch, device=self.device)

        fault_active = live & (self.fault_kind >= 0) & (t >= self.fault_onset)
        motor_fault = fault_active & (self.fault_kind == 0)
        fragile_fault = fault_active & (self.fault_kind == 1)

        energy = self.energy.clone()
        damage = self.damage.clone()

        if is_instruction:
            energy = torch.where(
                live, torch.clamp(energy + es * 0.006 * (1.0 - damage), max=1.0),
                energy,
            )
            self.last_task_reward = torch.where(
                live, torch.zeros_like(self.last_task_reward),
                self.last_task_reward,
            )
        else:
            correct = live & (action == self.current_target)
            wrong = live & ~correct
            live_f = live.float()

            correct_mask = correct.float()
            self.correct_total += correct_mask
            self.response_total += live_f
            self.phase_correct[:, phase] += correct_mask
            self.phase_responses[:, phase] += live_f

            self.task_correct[self.rows, self.current_task, phase] += correct_mask
            self.task_responses[self.rows, self.current_task, phase] += live_f
            self.target_count[self.rows, phase, self.current_target] += live_f

            task_reward = torch.where(
                correct, torch.ones_like(reward), torch.full_like(reward, -0.20)
            )
            reward = torch.where(live, task_reward, reward)
            self.last_task_reward = torch.where(
                live, task_reward, self.last_task_reward
            )

            action_scale = action.float() / max(1, self.cfg.n_actions - 1)
            cost = es * (0.004 + 0.004 * action_scale)
            expensive = motor_fault & (action == self.fault_action)
            cost = torch.where(expensive, cost * 3.0, cost)
            energy = torch.where(live, torch.clamp(energy - cost, min=0.0), energy)

            # Cognitive errors teach before they kill: only a fragile fault
            # makes wrong answers physically costly.
            damage_add = torch.where(
                fragile_fault, torch.full_like(damage, 0.006),
                torch.zeros_like(damage),
            )
            damage = torch.where(wrong, torch.clamp(damage + damage_add, max=1.0),
                                 damage)
            damage = torch.where(correct, torch.clamp(damage - 0.002, min=0.0),
                                 damage)

        shock_p = torch.where(
            fragile_fault, torch.full_like(energy, 0.025),
            torch.full_like(energy, 0.005),
        )
        shocked = live & (self.shock_u[:, t] < shock_p)
        energy = torch.where(shocked, torch.clamp(energy - es * 0.04, min=0.0),
                             energy)
        damage = torch.where(
            shocked,
            torch.clamp(
                damage + torch.where(
                    fragile_fault, torch.full_like(damage, 0.040),
                    torch.full_like(damage, 0.015),
                ),
                max=1.0,
            ),
            damage,
        )

        died = live & ((energy <= 0.0) | (damage >= 1.0))
        finished = live & ~died & (t + 1 >= self.cfg.steps)

        reward = torch.where(died, reward - self.cfg.death_penalty, reward)
        reward = torch.where(finished, reward + self.cfg.survival_bonus, reward)

        self.energy = torch.where(live, energy, self.energy)
        self.damage = torch.where(live, damage, self.damage)

        self.survived |= finished
        self.alive = live & ~died & ~finished
        self.step_number += 1

        return reward, live, correct_mask


# =====================================================================
# GENOME
# =====================================================================

# Structural controller inputs (index -> meaning):
#  0-2 doubt, 3-5 |doubt|, 6 self-model error EMA, 7 task error,
#  8 learnability, 9 capacity pressure, 10 active fraction, 11 mean activity,
#  12 mean utility, 13 recent accuracy, 14 time, 15 bias
STRUCT_FEATURES = 16
F_PRESSURE = 9
F_BIAS = 15


def structure_input_mask(mode: str) -> List[float]:
    mask = [1.0] * STRUCT_FEATURES
    if mode == "nolearn":
        mask[8] = 0.0                      # learnability
    elif mode == "basic":
        for i in (0, 1, 2, 3, 4, 5, 6, 8, 9):
            mask[i] = 0.0                  # all self-model / meta signals
    return mask


def genome_spec(cfg: Config, obs_dim: int) -> List[Tuple[str, Tuple[int, ...]]]:
    return [
        # dW = A*pre*post + B*pre + C*post + D, per postsynaptic unit.
        ("rule_in", (4, cfg.max_hidden)),
        ("rule_rec", (4, cfg.max_hidden)),
        ("rule_out", (4, cfg.n_actions)),
        ("eta", (3,)),
        # doubt(3), |doubt|(3), reward-prediction-error, learnability, bias
        ("modulator", (9,)),
        ("temperature", (1,)),
        ("eta_self", (1,)),
        ("doubt_gain", (3,)),
        ("doubt_memory", (1,)),
        ("eta_learnability", (1,)),
        ("structure", (3, STRUCT_FEATURES)),       # HOLD / GROW / PRUNE
        ("growth_context", (STRUCT_FEATURES, cfg.growth_key_dim)),
        ("growth_keys", (cfg.max_hidden, cfg.growth_key_dim)),
        ("structural_size", (2,)),
        ("birth_scale", (3,)),
    ]


def initial_genome(cfg: Config, obs_dim: int, device: torch.device,
                   seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    parts = []

    for name, shape in genome_spec(cfg, obs_dim):
        if name.startswith("rule"):
            value = 0.02 * rng.standard_normal(shape)
            # Start inside a working three-factor learning basin.
            value[0] += 1.0 if name == "rule_out" else 0.20

        elif name == "eta":
            # in, rec slow; out fast enough to learn within one life.
            value = np.array([-2.0, -2.0, cfg.init_eta_out])

        elif name == "modulator":
            value = np.zeros(shape)
            value[6] = 2.0                # reward-prediction-error

        elif name == "temperature":
            value = np.array([math.log(cfg.init_temperature)])

        elif name == "eta_self":
            value = np.array([-1.5])

        elif name == "doubt_gain":
            value = np.log([10.0, 30.0, 1.5])

        elif name == "doubt_memory":
            value = np.array([2.0])

        elif name == "eta_learnability":
            value = np.array([-1.0])

        elif name == "structure":
            # Exploratory but not forced: ~11% GROW per structural tick at
            # zero pressure, more when capacity pressure is high.
            value = np.zeros(shape)
            value[0, F_BIAS] = 1.5
            value[1, F_BIAS] = -0.5
            value[1, F_PRESSURE] = 1.0
            value[2, F_BIAS] = -1.5

        elif name == "growth_context":
            value = 0.02 * rng.standard_normal(shape)

        elif name == "growth_keys":
            value = 0.05 * rng.standard_normal(shape)

        else:
            value = np.zeros(shape)

        parts.append(np.asarray(value, dtype=np.float64).reshape(-1))

    return torch.tensor(np.concatenate(parts).astype(np.float32), device=device)


def unpack_genomes(theta: torch.Tensor, cfg: Config, obs_dim: int
                   ) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    pos = 0
    for name, shape in genome_spec(cfg, obs_dim):
        n = int(np.prod(shape))
        out[name] = theta[:, pos:pos + n].reshape(len(theta), *shape)
        pos += n
    return out


def local_rule(coef: torch.Tensor, pre: torch.Tensor, post: torch.Tensor
               ) -> torch.Tensor:
    A = coef[:, 0, None, :]
    B = coef[:, 1, None, :]
    C = coef[:, 2, None, :]
    D = coef[:, 3, None, :]
    return (A * pre[:, :, None] * post[:, None, :] + B * pre[:, :, None]
            + C * post[:, None, :] + D)


# =====================================================================
# LIFETIME
# =====================================================================

@torch.no_grad()
def run_lives(theta: torch.Tensor, scenarios: Scenarios, cfg: Config,
              device: torch.device, record: bool = False,
              probe_lives: int = 0) -> Dict:
    population = theta.shape[0]
    lives = scenarios.n
    batch = population * lives

    env = BenchmarkB(cfg, scenarios, population, device)
    obs_dim = env.obs_dim
    n_input = obs_dim + 3 + 1 + 1

    genome = unpack_genomes(theta, cfg, obs_dim)
    g = {k: v.repeat_interleave(lives, dim=0) for k, v in genome.items()}

    H = cfg.max_hidden
    A = cfg.n_actions
    developing = cfg.structure != "fixed"
    allow_prune = cfg.structure in ("full", "nolearn", "basic")
    struct_mask = torch.tensor(structure_input_mask(cfg.structure), device=device)

    # ----------------------------------------------------------------
    # Random birth wiring (shared across genomes for the same life).
    # ----------------------------------------------------------------
    gen = torch.Generator(device=device)
    gen.manual_seed(scenarios.birth_seed)
    life_in = torch.randn(lives, n_input, H, generator=gen, device=device) / math.sqrt(n_input)
    life_rec = torch.randn(lives, H, H, generator=gen, device=device) / math.sqrt(H)
    life_out = torch.randn(lives, H, A, generator=gen, device=device) / math.sqrt(H)

    scale = torch.exp(torch.clamp(g["birth_scale"], -2.0, 2.0))
    birth_in = life_in.repeat(population, 1, 1) * scale[:, 0, None, None]
    birth_rec = life_rec.repeat(population, 1, 1) * scale[:, 1, None, None]
    birth_out = life_out.repeat(population, 1, 1) * scale[:, 2, None, None]

    # ----------------------------------------------------------------
    # Phenotype state.
    # ----------------------------------------------------------------
    hidden = torch.zeros(batch, H, device=device)
    active = torch.zeros(batch, H, dtype=torch.bool, device=device)
    active[:, : cfg.initial_hidden] = True
    utility = torch.zeros(batch, H, device=device)
    neuron_age = torch.full((batch, H), 1e4, device=device)

    plastic_in = torch.zeros_like(birth_in)
    plastic_rec = torch.zeros_like(birth_rec)
    plastic_out = torch.zeros_like(birth_out)

    # Eligibility spans exactly one instruction -> response trial.
    elig_in = torch.zeros_like(birth_in)
    elig_rec = torch.zeros_like(birth_rec)

    reward_baseline = torch.zeros(batch, device=device)

    # Per-neuron running mean of response-time activity (for --center).
    readout_mean = torch.zeros(batch, H, device=device)

    probe_n = min(probe_lives, batch)
    probe_x: List[torch.Tensor] = []
    probe_y: List[torch.Tensor] = []
    probe_m: List[torch.Tensor] = []

    SELF_OUT = 3
    feature_dim = H + n_input
    self_model = torch.zeros(batch, feature_dim, SELF_OUT * A, device=device)

    # Behavioural learnability: predicts (fast - slow) accuracy EMA.
    learnability_model = torch.zeros(batch, feature_dim, device=device)
    learnability_signal = torch.zeros(batch, device=device)
    learn_pred_prev = torch.zeros(batch, device=device)
    feat_prev = torch.zeros(batch, feature_dim, device=device)

    acc_fast = torch.full((batch,), 1.0 / A, device=device)
    acc_slow = torch.full((batch,), 1.0 / A, device=device)

    doubt = torch.zeros(batch, SELF_OUT, device=device)
    error_ema = torch.zeros(batch, device=device)

    total_reward = torch.zeros(batch, device=device)
    total_capacity = torch.zeros(batch, device=device)
    structural_changes = torch.zeros(batch, device=device)
    grow_total = torch.zeros(batch, device=device)
    prune_total = torch.zeros(batch, device=device)

    phase_capacity_sum = torch.zeros(batch, cfg.max_tasks, device=device)
    phase_capacity_n = torch.zeros(batch, cfg.max_tasks, device=device)

    temperature = torch.exp(torch.clamp(g["temperature"][:, 0], -4.0, 2.0))
    eta = 0.15 * torch.sigmoid(g["eta"])
    eta_self = 0.20 * torch.sigmoid(g["eta_self"][:, 0])
    eta_meta = 0.10 * torch.sigmoid(g["eta_learnability"][:, 0])
    memory = torch.sigmoid(g["doubt_memory"][:, 0])
    gain = torch.exp(torch.clamp(g["doubt_gain"], -3.0, 4.0))
    growth_count = 1 + (torch.sigmoid(g["structural_size"][:, 0])
                        * (cfg.max_structural_change - 1)).long()
    prune_count = 1 + (torch.sigmoid(g["structural_size"][:, 1])
                       * (cfg.max_structural_change - 1)).long()

    rows = torch.arange(batch, device=device)
    ones = torch.ones(batch, 1, device=device)
    # Keeps only the RPE and bias modulator inputs.
    meta_off_mask = torch.tensor(
        [0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=torch.float32, device=device
    )

    rec: Optional[Dict] = None
    if record:
        rec = {
            "correct": [], "capacity": [], "acc_fast": [], "doubt": [],
            "error": [], "learnability": [], "pressure": [], "grow": [],
            "prune": [], "tick_steps": [], "tick_action": [],
            "tick_pressure": [], "tick_task_error": [],
            "tick_learnability": [], "tick_error": [],
        }

    for t in range(cfg.steps):
        if not env.alive.any():
            break

        obs, is_instruction, phase = env.observation()

        active_f = active.float()
        active_count = active.sum(dim=1)
        active_fraction = active_count.float() / H

        if cfg.metacognition:
            meta_in = torch.cat([doubt, learnability_signal[:, None]], dim=1)
        else:
            meta_in = torch.zeros(batch, SELF_OUT + 1, device=device)
        x = torch.cat([obs, meta_in, ones], dim=1)
        previous_hidden = hidden

        # Recurrent gain normalised to the active population.
        rec_gain = torch.sqrt(H / active_count.float().clamp(min=1.0))
        hidden = torch.tanh(
            torch.einsum("bi,bij->bj", x, birth_in + plastic_in)
            + rec_gain[:, None] * torch.einsum("bi,bij->bj", previous_hidden, birth_rec)
            + torch.einsum("bi,bij->bj", previous_hidden, plastic_rec)
        ) * active_f
        if cfg.retain > 0.0:
            hidden = cfg.retain * previous_hidden + (1.0 - cfg.retain) * hidden

        # Readout representation: optionally centred by each neuron's own
        # running mean, so learning targets cue-specific activity instead of
        # the component shared by every cue.
        readout = (hidden - readout_mean) * active_f if cfg.center else hidden

        w_out = birth_out + plastic_out
        logits = torch.einsum("bi,bij->bj", readout, w_out)
        policy = torch.softmax(logits / temperature[:, None], dim=1)

        if cfg.random_policy:
            action = (env.action_u[:, t] * A).long().clamp(max=A - 1)
        else:
            action = (policy.cumsum(dim=1) < env.action_u[:, t, None]
                      ).sum(dim=1).clamp(max=A - 1)

        # Utility = contribution to behaviour, not raw activity.
        utility = 0.985 * utility + 0.015 * hidden.abs() * w_out.norm(dim=2)

        feature = torch.cat([hidden, x], dim=1)

        # ------------------------------------------------------------
        # Self-prediction before reality.
        # ------------------------------------------------------------
        predictions = torch.einsum("bi,bij->bj", feature, self_model)
        predicted_outcome = torch.stack(
            [predictions[rows, action], predictions[rows, A + action],
             predictions[rows, 2 * A + action]], dim=1,
        )

        energy_before = env.energy.clone()
        damage_before = env.damage.clone()

        reward, lived, correct_mask = env.step(action, is_instruction, phase)
        total_reward += reward
        lived_f = lived.float()

        real_outcome = torch.stack(
            [env.energy - energy_before, env.damage - damage_before, reward],
            dim=1,
        )
        error = (real_outcome - predicted_outcome) * lived_f[:, None]
        instant_error = error.abs().mean(dim=1)

        for channel in range(SELF_OUT):
            col = channel * A + action
            self_model[rows, :, col] += (
                eta_self[:, None] * feature * error[:, channel:channel + 1]
            )
        self_model.clamp_(-3.0, 3.0)

        surprise = torch.clamp(error, -1.0, 1.0) * gain
        doubt = memory[:, None] * doubt + (1.0 - memory)[:, None] * surprise

        # ------------------------------------------------------------
        # Behavioural statistics and learnability (response steps only).
        # ------------------------------------------------------------
        if not is_instruction:
            acc_fast = torch.where(lived, 0.80 * acc_fast + 0.20 * correct_mask, acc_fast)
            acc_slow = torch.where(lived, 0.95 * acc_slow + 0.05 * correct_mask, acc_slow)
            error_ema = torch.where(lived, 0.90 * error_ema + 0.10 * instant_error, error_ema)

            # Teach last trial's prediction with the progress observed now.
            progress_target = torch.tanh(10.0 * (acc_fast - acc_slow))
            l_err = (progress_target - learn_pred_prev) * lived_f
            learnability_model += eta_meta[:, None] * feat_prev * l_err[:, None]
            learnability_model.clamp_(-2.0, 2.0)

            learn_pred = torch.tanh((feature * learnability_model).sum(dim=1))
            learnability_signal = torch.where(lived, learn_pred, learnability_signal)
            learn_pred_prev = learn_pred
            feat_prev = feature

        # ------------------------------------------------------------
        # Three-factor plasticity.
        # ------------------------------------------------------------
        young = 1.0 + cfg.newborn_boost * torch.exp(-neuron_age / cfg.newborn_tau)

        d_in = local_rule(g["rule_in"], x, hidden)
        d_in *= (active_f * young)[:, None, :]
        d_rec = local_rule(g["rule_rec"], previous_hidden, hidden)
        d_rec *= active_f[:, :, None] * (active_f * young)[:, None, :]

        if is_instruction:
            elig_in = d_in
            elig_rec = d_rec
        else:
            elig_in.add_(d_in)
            elig_rec.add_(d_rec)

            task_reward = env.last_task_reward
            rpe = (task_reward - reward_baseline) * lived_f
            reward_baseline = torch.where(
                lived, 0.9 * reward_baseline + 0.1 * task_reward, reward_baseline
            )

            mod_input = torch.cat(
                [doubt, doubt.abs(), rpe[:, None], learnability_signal[:, None], ones],
                dim=1,
            )
            if not cfg.metacognition:
                # Pure reward-prediction-error learning.
                mod_input = mod_input * meta_off_mask
            modulator = torch.tanh((mod_input * g["modulator"]).sum(dim=1)) * lived_f

            if cfg.plasticity:
                action_onehot = F.one_hot(action, A).float()
                d_out = local_rule(g["rule_out"], readout, action_onehot - policy)
                d_out *= (active_f * young)[:, :, None]

                plastic_in += (eta[:, 0] * modulator)[:, None, None] * elig_in
                plastic_rec += (eta[:, 1] * modulator)[:, None, None] * elig_rec
                plastic_out += (eta[:, 2] * modulator)[:, None, None] * d_out
                plastic_in.clamp_(-1.0, 1.0)
                plastic_rec.clamp_(-1.0, 1.0)
                plastic_out.clamp_(-1.0, 1.0)
                del d_out

        neuron_age += 1.0

        # ------------------------------------------------------------
        # Development controller.
        # ------------------------------------------------------------
        task_error = 1.0 - acc_slow
        learn_for_pressure = (
            learnability_signal if cfg.structure not in ("nolearn", "basic")
            else torch.zeros_like(learnability_signal)
        )
        pressure = task_error * (1.0 - learn_for_pressure) / 2.0

        growth_events = torch.zeros(batch, device=device)
        prune_events = torch.zeros(batch, device=device)

        structural_tick = (
            developing
            and t >= cfg.structural_warmup
            and (t - cfg.structural_warmup) % cfg.structural_interval == 0
        )

        if structural_tick:
            count = active_count.clamp(min=1)
            structure_input = torch.cat(
                [
                    doubt, doubt.abs(), error_ema[:, None], task_error[:, None],
                    learnability_signal[:, None], pressure[:, None],
                    active_fraction[:, None],
                    (hidden.abs().sum(dim=1) / count)[:, None],
                    (utility.sum(dim=1) / count)[:, None],
                    acc_fast[:, None],
                    torch.full((batch, 1), t / max(1, cfg.steps - 1), device=device),
                    ones,
                ],
                dim=1,
            ) * struct_mask

            probs = torch.softmax(
                torch.einsum("bi,bji->bj", structure_input, g["structure"]), dim=1
            )
            u = env.structure_u[:, t]
            structural_action = torch.full((batch,), 2, dtype=torch.long, device=device)
            structural_action[u < probs[:, 0] + probs[:, 1]] = 1
            structural_action[u < probs[:, 0]] = 0
            structural_action[~lived] = -1

            grow_mask = structural_action == 1
            prune_mask = (structural_action == 2) & (t >= cfg.steps // 3)
            if not allow_prune:
                prune_mask = torch.zeros_like(prune_mask)

            growth_scores = torch.einsum(
                "bik,bk->bi", g["growth_keys"],
                torch.einsum("bi,bij->bj", structure_input, g["growth_context"]),
            )

            for slot in range(cfg.max_structural_change):
                can_grow = grow_mask & (growth_count > slot) & (active.sum(dim=1) < H)
                if not can_grow.any():
                    continue
                cand = growth_scores.masked_fill(active, float("-inf")).argmax(dim=1)
                rr = torch.where(can_grow)[0]
                cc = cand[rr]
                active[rr, cc] = True
                hidden[rr, cc] = 0.0
                utility[rr, cc] = 0.0
                neuron_age[rr, cc] = 0.0
                plastic_in[rr, :, cc] = 0.0
                plastic_rec[rr, :, cc] = 0.0
                plastic_rec[rr, cc, :] = 0.0
                plastic_out[rr, cc, :] = 0.0
                growth_events[rr] += 1.0

            for slot in range(cfg.max_structural_change):
                can_prune = (prune_mask & (prune_count > slot)
                             & (active.sum(dim=1) > cfg.min_hidden))
                if not can_prune.any():
                    continue
                cand = utility.masked_fill(~active, float("inf")).argmin(dim=1)
                rr = torch.where(can_prune)[0]
                cc = cand[rr]
                active[rr, cc] = False
                hidden[rr, cc] = 0.0
                utility[rr, cc] = 0.0
                prune_events[rr] += 1.0

            if rec is not None:
                rec["tick_steps"].append(t)
                rec["tick_action"].append(structural_action.cpu())
                rec["tick_pressure"].append(pressure.cpu())
                rec["tick_task_error"].append(task_error.cpu())
                rec["tick_learnability"].append(learnability_signal.cpu())
                rec["tick_error"].append(error_ema.cpu())

        structural_changes += growth_events + prune_events
        grow_total += growth_events
        prune_total += prune_events

        now_capacity = active.sum(dim=1).float()
        total_capacity += now_capacity * lived_f
        if not is_instruction:
            phase_capacity_sum[:, phase] += now_capacity * lived_f
            phase_capacity_n[:, phase] += lived_f

        if rec is not None:
            alive_n = lived_f.sum().clamp(min=1.0)
            if is_instruction:
                corr = torch.full((batch,), float("nan"), device=device)
            else:
                corr = torch.where(lived, correct_mask,
                                   torch.full_like(correct_mask, float("nan")))
            rec["correct"].append(corr.cpu())
            rec["capacity"].append((now_capacity * lived_f).sum().item() / alive_n.item())
            rec["acc_fast"].append((acc_fast * lived_f).sum().item() / alive_n.item())
            rec["doubt"].append(doubt.norm(dim=1).mean().item())
            rec["error"].append(error_ema.mean().item())
            rec["learnability"].append(learnability_signal.mean().item())
            rec["pressure"].append(pressure.mean().item())
            rec["grow"].append(growth_events.mean().item())
            rec["prune"].append(prune_events.mean().item())

        if not is_instruction:
            if probe_n > 0:
                probe_x.append(readout[:probe_n].clone())
                probe_y.append(env.current_task[:probe_n] * cfg.n_cues
                               + env.current_cue[:probe_n])
                probe_m.append(lived[:probe_n].clone())
            readout_mean = torch.where(
                lived[:, None], 0.95 * readout_mean + 0.05 * hidden, readout_mean
            )

        del d_in, d_rec

    # ----------------------------------------------------------------
    # Fitness.
    # ----------------------------------------------------------------
    average_capacity = total_capacity / cfg.steps
    extra_fraction = torch.clamp(average_capacity - cfg.initial_hidden, min=0.0) / H
    complexity_penalty = cfg.complexity_cost * extra_fraction * (cfg.steps / 2)
    structural_penalty = cfg.structural_change_cost * structural_changes
    lifetime_accuracy = env.correct_total / env.response_total.clamp(min=1.0)

    fitness = (total_reward - complexity_penalty - structural_penalty
               + cfg.accuracy_weight * lifetime_accuracy)

    P, L = population, lives
    T = cfg.max_tasks
    result: Dict = {
        "fitness": fitness.reshape(P, L),
        "reward": total_reward.reshape(P, L),
        "survived": env.survived.reshape(P, L),
        "correct": env.correct_total.reshape(P, L),
        "responses": env.response_total.reshape(P, L),
        "capacity": average_capacity.reshape(P, L),
        "final_capacity": active.sum(dim=1).float().reshape(P, L),
        "phase_correct": env.phase_correct.reshape(P, L, T),
        "phase_responses": env.phase_responses.reshape(P, L, T),
        "phase_capacity": (phase_capacity_sum / phase_capacity_n.clamp(min=1.0)
                           ).reshape(P, L, T),
        "task_correct": env.task_correct.reshape(P, L, T, T),
        "task_responses": env.task_responses.reshape(P, L, T, T),
        "target_count": env.target_count.reshape(P, L, T, A),
        "structural_changes": structural_changes.reshape(P, L),
        "grow_total": grow_total.reshape(P, L),
        "prune_total": prune_total.reshape(P, L),
    }
    if rec is not None:
        result["record"] = rec
    if probe_n > 0 and probe_x:
        result["probe"] = (torch.stack(probe_x, dim=1), torch.stack(probe_y, dim=1),
                           torch.stack(probe_m, dim=1))
    return result


# =====================================================================
# ES
# =====================================================================

class ESAdam:
    def __init__(self, dimension: int, lr: float, device: torch.device):
        self.lr = lr
        self.m = torch.zeros(dimension, device=device)
        self.v = torch.zeros(dimension, device=device)
        self.t = 0

    def step(self, grad: torch.Tensor) -> torch.Tensor:
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * grad.square()
        m_hat = self.m / (1.0 - 0.9 ** self.t)
        v_hat = self.v / (1.0 - 0.999 ** self.t)
        return self.lr * m_hat / (torch.sqrt(v_hat) + 1e-8)


def centered_ranks(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x)
    ranks = torch.empty_like(x)
    ranks[order] = torch.arange(len(x), device=x.device, dtype=x.dtype)
    if len(x) > 1:
        ranks /= (len(x) - 1)
    return ranks - 0.5


# =====================================================================
# ANALYSIS
# =====================================================================

def growth_analysis(rec: Dict, window: int = 8) -> Dict[str, float]:
    """
    1) Are GROW decisions made under higher capacity pressure than HOLD
       decisions at the same tick (controls for time / phase)?
    2) Does accuracy improve more after GROW than after HOLD at the same tick
       (difference-in-differences over `window` response trials)?
    """
    out: Dict[str, float] = {}
    if not rec["tick_steps"]:
        return out

    correct = torch.stack(rec["correct"]).numpy()   # [steps, batch]
    responses = correct[1::2]                         # [trials, batch]

    diffs: Dict[str, List[float]] = {
        "pressure": [], "task_error": [], "learnability": [], "error": []
    }
    did, did_w = [], []
    grow_n = hold_n = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i, t in enumerate(rec["tick_steps"]):
            act = rec["tick_action"][i].numpy()
            grow, hold = act == 1, act == 0
            grow_n += int(grow.sum())
            hold_n += int(hold.sum())
            if grow.sum() == 0 or hold.sum() == 0:
                continue

            for key in diffs:
                v = rec[f"tick_{key}"][i].numpy()
                diffs[key].append(float(v[grow].mean() - v[hold].mean()))

            k = t // 2
            before = responses[max(0, k - window):k]
            after = responses[k:k + window]
            if len(before) == 0 or len(after) == 0:
                continue
            delta = np.nanmean(after, axis=0) - np.nanmean(before, axis=0)
            dg, dh = np.nanmean(delta[grow]), np.nanmean(delta[hold])
            if np.isfinite(dg) and np.isfinite(dh):
                did.append(dg - dh)
                did_w.append(float(grow.sum()))

    out["grow_decisions"] = float(grow_n)
    out["hold_decisions"] = float(hold_n)
    for key, vals in diffs.items():
        if vals:
            out[f"grow_minus_hold_{key}"] = float(np.mean(vals))
    if did:
        out["post_growth_accuracy_gain_vs_hold"] = float(
            np.average(did, weights=did_w)
        )
    return out


@torch.no_grad()
def evaluate(theta: torch.Tensor, cfg: Config, device: torch.device, seed: int,
             lives: int = 512, record: bool = False) -> Dict:
    scenarios = Scenarios(cfg, lives, seed, device)
    r = run_lives(theta[None], scenarios, cfg, device, record=record)

    survived = r["survived"][0]
    correct = r["correct"][0]
    responses = r["responses"][0]
    fault = scenarios.fault_kind >= 0
    healthy = ~fault
    total_resp = responses.sum().clamp(min=1)

    target_count = r["target_count"][0]            # [L, P, A]
    cue_blind_life = target_count.sum(dim=1).max(dim=1).values.sum() / total_resp
    cue_blind_phase = target_count.max(dim=2).values.sum() / total_resp

    m: Dict = {
        "accuracy": (100.0 * correct.sum() / total_resp).item(),
        "chance": 100.0 / cfg.n_actions,
        "cue_blind_life": 100.0 * cue_blind_life.item(),
        "cue_blind_phase": 100.0 * cue_blind_phase.item(),
        "survival": 100.0 * survived.float().mean().item(),
        "survival_healthy": (100.0 * survived[healthy].float().mean().item()
                             if healthy.any() else float("nan")),
        "survival_fault": (100.0 * survived[fault].float().mean().item()
                           if fault.any() else float("nan")),
        "reward": r["reward"][0].mean().item(),
        "average_neurons": r["capacity"][0].mean().item(),
        "final_neurons": r["final_capacity"][0].mean().item(),
        "structural_changes": r["structural_changes"][0].mean().item(),
        "grow_events": r["grow_total"][0].mean().item(),
        "prune_events": r["prune_total"][0].mean().item(),
    }

    T = cfg.max_tasks
    for p in range(T):
        m[f"phase_{p+1}_accuracy"] = (
            100.0 * r["phase_correct"][0][:, p].sum()
            / r["phase_responses"][0][:, p].sum().clamp(min=1)
        ).item()
        m[f"phase_{p+1}_neurons"] = r["phase_capacity"][0][:, p].mean().item()

    tc = r["task_correct"][0].sum(dim=0)            # [task, phase]
    tr = r["task_responses"][0].sum(dim=0)
    task_phase = torch.where(tr > 0, 100.0 * tc / tr.clamp(min=1),
                             torch.full_like(tc, float("nan")))
    m["task_phase_accuracy"] = task_phase.cpu().tolist()
    # Retention: task 0 in its own phase vs. rehearsal trials in the last phase.
    m["task0_phase1"] = task_phase[0, 0].item()
    m["task0_last_phase"] = task_phase[0, T - 1].item()

    if record:
        rec = r["record"]
        m["trace"] = {k: rec[k] for k in (
            "capacity", "acc_fast", "doubt", "error", "learnability",
            "pressure", "grow", "prune")}
        m["growth_analysis"] = growth_analysis(rec)

    return m


# =====================================================================
# EVOLUTION
# =====================================================================

def print_header(cfg: Config, device: torch.device, dimension: int) -> None:
    print("=" * 78)
    print(f"DSM Benchmark B v3   run={cfg.run_name}   seed={cfg.seed}")
    print("=" * 78)
    print(f"Device             : {device}")
    if device.type == "cuda":
        print(f"GPU                : {torch.cuda.get_device_name(device)}")
    print(f"Genome parameters  : {dimension:,}")
    print(f"Structure          : {cfg.structure}, start {cfg.initial_hidden} "
          f"of {cfg.max_hidden} neurons")
    print(f"Benchmark          : {cfg.max_tasks} tasks x {cfg.n_cues} cues x "
          f"{cfg.n_actions} actions, {cfg.steps} steps")
    print(f"Chance accuracy    : {100.0 / cfg.n_actions:.1f}%")
    print(f"ES                 : pop={cfg.population}, lives={cfg.lives}, "
          f"generations={cfg.generations}")
    print("=" * 78)


def print_progress(generation: int, m: Dict, cfg: Config, elapsed: float) -> None:
    print(
        f"gen {generation:4d} | acc {m['accuracy']:5.1f}%"
        f" (cue-blind {m['cue_blind_phase']:4.1f}%)"
        f" | survive {m['survival']:5.1f}%"
        f" | reward {m['reward']:7.2f}"
        f" | neurons {m['average_neurons']:5.1f}->{m['final_neurons']:5.1f}"
        f" | grow {m['grow_events']:4.1f} prune {m['prune_events']:4.1f}"
        f" | {elapsed:7.1f}s"
    )
    text = "  phases:"
    for p in range(cfg.max_tasks):
        text += (f" P{p+1}={m[f'phase_{p+1}_accuracy']:.1f}%"
                 f"/{m[f'phase_{p+1}_neurons']:.1f}n")
    text += f" | task0 P1 {m['task0_phase1']:.1f}% -> last {m['task0_last_phase']:.1f}%"
    print(text)


def evolve(cfg: Config, device: torch.device) -> torch.Tensor:
    dummy = Scenarios(cfg, 1, cfg.seed + 123, device)
    obs_dim = BenchmarkB(cfg, dummy, 1, device).obs_dim

    theta = initial_genome(cfg, obs_dim, device, cfg.seed)
    dimension = len(theta)
    print_header(cfg, device, dimension)

    optimizer = ESAdam(dimension, cfg.es_lr, device)
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed + 8_888)
    half = cfg.population // 2
    start = time.time()

    for generation in range(cfg.generations + 1):
        if generation % 10 == 0 or generation == cfg.generations:
            m = evaluate(theta, cfg, device, seed=50_000 + cfg.seed, lives=256)
            print_progress(generation, m, cfg, time.time() - start)

        if generation == cfg.generations:
            break

        epsilon = torch.randn(half, dimension, generator=gen, device=device)
        candidates = torch.cat(
            [theta[None] + cfg.sigma * epsilon, theta[None] - cfg.sigma * epsilon]
        )
        scenarios = Scenarios(
            cfg, cfg.lives, seed=100_000 + cfg.seed * 10_000 + generation,
            device=device,
        )
        result = run_lives(candidates, scenarios, cfg, device)
        fitness = result["fitness"].mean(dim=1)

        if generation % 10 == 0:
            noise = result["fitness"].std(dim=1).mean()
            print(f"  ES: fitness mean {fitness.mean().item():7.2f}"
                  f" | std {fitness.std().item():6.3f}"
                  f" | best {fitness.max().item():7.2f}"
                  f" | per-life noise {noise.item():6.2f}")

        shaped = centered_ranks(fitness)
        diff = shaped[:half] - shaped[half:]
        grad = (diff[:, None] * epsilon).sum(dim=0) / (cfg.population * cfg.sigma)
        grad -= cfg.weight_decay * theta
        theta += optimizer.step(grad)

        if device.type == "cuda" and generation % 25 == 0:
            torch.cuda.empty_cache()

    return theta


# =====================================================================
# REPORTING
# =====================================================================

def print_final(m: Dict, cfg: Config) -> None:
    print("\n" + "=" * 78)
    print(f"FINAL TEST  run={cfg.run_name}  seed={cfg.seed}")
    print("=" * 78)
    for key in ("accuracy", "chance", "cue_blind_life", "cue_blind_phase",
                "survival", "survival_healthy", "survival_fault", "reward",
                "average_neurons", "final_neurons", "grow_events",
                "prune_events"):
        print(f"{key:24s}: {m[key]:.3f}")

    verdict = m["accuracy"] - m["cue_blind_phase"]
    print(f"\nAccuracy - cue-blind(phase) : {verdict:+.2f} points "
          f"({'mapping learned' if verdict > 0 else 'NOT beyond a cue-blind prior'})")

    print("\nPhases (accuracy / average active neurons):")
    for p in range(cfg.max_tasks):
        print(f"  Phase {p+1}: {m[f'phase_{p+1}_accuracy']:6.2f}%"
              f" / {m[f'phase_{p+1}_neurons']:6.2f}")

    print("\nAccuracy per task (rows) per phase (cols) — retention matrix:")
    header = "        " + "".join(f"  P{p+1:<5d}" for p in range(cfg.max_tasks))
    print(header)
    for task, row in enumerate(m["task_phase_accuracy"]):
        cells = "".join(
            f"  {v:5.1f}%" if v == v else "      - " for v in row
        )
        print(f"  task{task}{cells}")

    ga = m.get("growth_analysis") or {}
    if ga:
        print("\nGrowth analysis (GROW vs HOLD at the same structural tick):")
        for key, value in ga.items():
            print(f"  {key:36s}: {value:+.4f}")


def print_trace(trace: Dict[str, List[float]]) -> None:
    print("\nDevelopment trace (alive mean; acc = recent-accuracy EMA)")
    print("step | neurons | acc%  | doubt | error | learnab. | pressure | grow | prune")
    for t in range(len(trace["capacity"])):
        grow, prune = trace["grow"][t], trace["prune"][t]
        if t % 10 != 0 and grow == 0.0 and prune == 0.0:
            continue
        print(f"{t:4d} | {trace['capacity'][t]:7.2f}"
              f" | {100 * trace['acc_fast'][t]:5.1f}"
              f" | {trace['doubt'][t]:5.2f}"
              f" | {trace['error'][t]:5.3f}"
              f" | {trace['learnability'][t]:+8.4f}"
              f" | {trace['pressure'][t]:8.4f}"
              f" | {grow:4.2f} | {prune:5.2f}")


def save_results(path: str, m: Dict, cfg: Config) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {k: v for k, v in m.items() if k != "trace"}
    data["run_name"] = cfg.run_name
    data["config"] = asdict(cfg)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved results: {path}")


def summarize(results_dir: str) -> None:
    files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    if not files:
        print(f"No results in {results_dir}")
        return

    groups: Dict[str, List[Dict]] = {}
    for path in files:
        with open(path) as f:
            d = json.load(f)
        groups.setdefault(d["run_name"], []).append(d)

    cols = [
        ("accuracy", "acc"),
        ("cue_blind_phase", "cue-blind"),
        ("phase_4_accuracy", "P4"),
        ("task0_last_phase", "task0-last"),
        ("reward", "reward"),
        ("survival", "survive"),
        ("average_neurons", "neurons"),
        ("grow_events", "grow"),
    ]
    print(f"{'run':16s} {'n':>2s}" + "".join(f" {label:>15s}" for _, label in cols))

    def order(name: str) -> Tuple[int, int]:
        mode, _, rest = name.partition("-")
        size = rest.split("-")[0]
        rank = STRUCTURE_MODES.index(mode) if mode in STRUCTURE_MODES else 99
        return rank, int(size) if size.isdigit() else 0

    for name in sorted(groups, key=order):
        runs = groups[name]
        line = f"{name:16s} {len(runs):2d}"
        for key, _ in cols:
            vals = np.array([r.get(key, float("nan")) for r in runs], dtype=float)
            sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
            line += f" {np.nanmean(vals):8.2f}±{sd:5.2f}"
        print(line)

        ga_keys = ("grow_minus_hold_pressure", "post_growth_accuracy_gain_vs_hold")
        ga = [r.get("growth_analysis", {}) for r in runs]
        if any(ga):
            parts = []
            for key in ga_keys:
                vals = [x[key] for x in ga if key in x]
                if vals:
                    parts.append(f"{key}={np.mean(vals):+.4f}")
            if parts:
                print(f"{'':19s}" + "  ".join(parts))


# =====================================================================
# RULE TEST (no evolution)
# =====================================================================

def ideal_learner_accuracy(cfg: Config, scenarios: Scenarios,
                           device: torch.device) -> float:
    """
    Ceiling on the exact same lives: a perfect-memory agent that tries untried
    actions for each (task, cue) until one is rewarded, then repeats it.
    """
    env = BenchmarkB(cfg, scenarios, 1, device)
    L, T, C, A = scenarios.n, cfg.max_tasks, cfg.n_cues, cfg.n_actions
    rng = np.random.default_rng(cfg.seed)
    order = np.argsort(rng.random((L, T, C, A)), axis=-1)
    tried = np.zeros((L, T, C, A), dtype=bool)
    known = np.full((L, T, C), -1)
    mapping = scenarios.mapping.cpu().numpy()
    cues = scenarios.cue.cpu().numpy()
    lr = np.arange(L)
    correct = 0

    for trial in range(scenarios.trials):
        phase = env.phase_index(2 * trial)
        task = env._trial_task(trial, phase).cpu().numpy()
        cue = cues[:, trial]
        target = mapping[lr, task, cue]

        untried = ~tried[lr, task, cue]                         # [L, A]
        ranked = np.take_along_axis(untried, order[lr, task, cue], axis=1)
        first = np.argmax(ranked, axis=1)
        guess = order[lr, task, cue][lr, first]
        k = known[lr, task, cue]
        act = np.where(k >= 0, k, guess)

        ok = act == target
        correct += int(ok.sum())
        tried[lr, task, cue, act] = True
        known[lr[ok], task[ok], cue[ok]] = act[ok]

    return 100.0 * correct / (L * scenarios.trials)


RULE_VARIANTS: Dict[str, Dict] = {
    "base": {},
    "center": {"center": True},
    "retain": {"retain": None},                 # None -> --retain-value
    "center+retain": {"center": True, "retain": None},
    # Diagnostics (not organisms): cue stays visible, no memory needed.
    "nodelay": {"no_delay": True},
    "nodelay+center": {"no_delay": True, "center": True},
}


def probe_decode(x: torch.Tensor, y: torch.Tensor, m: torch.Tensor,
                 n_classes: int, lam: float = 1.0) -> Tuple[float, float]:
    """
    Per-life ridge decoder (measurement only, not part of the organism):
    can the response-time readout tell which class this trial was?
    Trains on even trials, tests on odd trials of the same life.
    Returns (decode accuracy, majority-class baseline).
    """
    B, N, H = x.shape
    X = torch.cat([x, torch.ones(B, N, 1, device=x.device)], dim=2)
    Y = F.one_hot(y, n_classes).float()
    even = (torch.arange(N, device=x.device) % 2 == 0)[None, :]
    train = (m & even).float()[..., None]
    test = m & ~even

    Xt = X * train
    A = Xt.transpose(1, 2) @ Xt + lam * torch.eye(H + 1, device=x.device)
    W = torch.linalg.solve(A, Xt.transpose(1, 2) @ (Y * train))
    pred = (X @ W).argmax(dim=2)

    n_test = test.sum().clamp(min=1)
    acc = ((pred == y) & test).sum() / n_test
    majority = (Y * train).sum(dim=1).argmax(dim=1)
    base = ((y == majority[:, None]) & test).sum() / n_test
    return 100.0 * acc.item(), 100.0 * base.item()


def rule_stats(r: Dict, cfg: Config) -> Dict[str, float]:
    resp = r["responses"][0].sum().clamp(min=1)
    pc, pr = r["phase_correct"][0], r["phase_responses"][0]
    tc = r["task_correct"][0].sum(dim=0)
    tr = r["task_responses"][0].sum(dim=0).clamp(min=1)
    out = {
        "acc": (100.0 * r["correct"][0].sum() / resp).item(),
        "blind": (100.0 * r["target_count"][0].max(dim=2).values.sum() / resp).item(),
        "t0_first": (100.0 * tc[0, 0] / tr[0, 0]).item(),
        "t0_last": (100.0 * tc[0, -1] / tr[0, -1]).item(),
    }
    for p in range(cfg.max_tasks):
        out[f"P{p+1}"] = (100.0 * pc[:, p].sum() / pr[:, p].sum().clamp(min=1)).item()
    return out


def rule_test(args: argparse.Namespace, device: torch.device) -> None:
    """
    Step 1 of the research order: prove the local learning rule works with a
    hand-built genome. No evolution, no growth, no doubt/learnability.
    Compares representation variants and diagnoses WHY a variant fails.
    """
    base_cfg = config_from_args(args)
    base_cfg.structure = "fixed"
    base_cfg.metacognition = args.metacognition
    validate_config(base_cfg)

    scenarios = Scenarios(base_cfg, args.eval_lives, 900_000 + base_cfg.seed, device)
    obs_dim = BenchmarkB(base_cfg, scenarios, 1, device).obs_dim
    ceiling = ideal_learner_accuracy(base_cfg, scenarios, device)

    sizes = [int(s) for s in args.sizes.split(",")]
    eta_logits = [float(v) for v in args.eta_out_grid.split(",")]
    temps = [float(v) for v in args.temperature_grid.split(",")]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in RULE_VARIANTS:
            raise ValueError(f"unknown variant {v}; choose from {list(RULE_VARIANTS)}")

    C, T = base_cfg.n_cues, base_cfg.max_tasks
    print("=" * 96)
    print("RULE TEST — hand-built genome, no evolution, fixed size, "
          f"metacognition {'on' if base_cfg.metacognition else 'off'}")
    print(f"{T} tasks x {C} cues x {base_cfg.n_actions} actions, {base_cfg.steps} steps, "
          f"faults p={base_cfg.fault_probability}, {args.eval_lives} lives")
    print(f"chance {100.0 / base_cfg.n_actions:.1f}%   |   perfect-learner ceiling "
          f"{ceiling:.1f}%")
    print("probe = per-life linear decoder of task x cue from the response-time "
          "readout (measurement only)")
    print("=" * 96)
    print(f"{'variant':15s} {'size':>4s} {'eta':>5s} {'temp':>4s} | {'acc':>6s} "
          f"{'blind':>6s} {'margin':>6s} |"
          + "".join(f" {'P' + str(p + 1):>5s}" for p in range(T))
          + f" | {'t0 P1':>5s} {'t0 end':>6s} | {'probe':>11s} {'cue':>11s}")

    summary = []
    for variant in variants:
        for size in sizes:
            best = None
            for eta_logit in eta_logits:
                for temp in temps:
                    cfg = Config(**asdict(base_cfg))
                    for key, value in RULE_VARIANTS[variant].items():
                        setattr(cfg, key, args.retain_value if value is None else value)
                    cfg.initial_hidden = size
                    cfg.min_hidden = min(cfg.min_hidden, size)
                    cfg.init_eta_out = eta_logit
                    cfg.init_temperature = temp
                    theta = initial_genome(cfg, obs_dim, device, cfg.seed)
                    st = rule_stats(run_lives(theta[None], scenarios, cfg, device), cfg)
                    if args.verbose:
                        print(f"  {variant:13s} {size:4d} {eta_logit:+5.1f} {temp:4.2f} | "
                              f"{st['acc']:5.1f}% {st['blind']:5.1f}% "
                              f"{st['acc'] - st['blind']:+6.1f}")
                    if best is None or st["acc"] > best[0]["acc"]:
                        best = (st, cfg)

            st, cfg = best
            # Probe the best setting of this variant.
            r = run_lives(initial_genome(cfg, obs_dim, device, cfg.seed)[None],
                          scenarios, cfg, device, probe_lives=args.probe_lives)
            x, y, m = r["probe"]
            conj, conj_base = probe_decode(x, y, m, T * C)
            cue, cue_base = probe_decode(x, y % C, m, C)

            print(f"{variant:15s} {size:4d} {cfg.init_eta_out:+5.1f} "
                  f"{cfg.init_temperature:4.2f} | {st['acc']:5.1f}% {st['blind']:5.1f}% "
                  f"{st['acc'] - st['blind']:+6.1f} |"
                  + "".join(f" {st[f'P{p+1}']:4.1f}%" for p in range(T))
                  + f" | {st['t0_first']:4.1f}% {st['t0_last']:5.1f}% |"
                  f" {conj:4.1f}/{conj_base:4.1f}% {cue:4.1f}/{cue_base:4.1f}%")
            summary.append((variant, size, st, cfg))

    print("-" * 96)
    print("probe/cue columns: decoder accuracy / majority-class baseline. If the "
          "probe is near its baseline,\nthe cue is not in the representation "
          "(memory/representation problem, not the learning rule).")

    organisms = [s for s in summary if not s[3].no_delay]
    if not organisms:
        return
    variant, size, st, cfg = max(organisms, key=lambda s: s[2]["acc"])
    margin = st["acc"] - st["blind"]
    print(f"\nbest organism variant: {variant} size={size} -> {st['acc']:.1f}% "
          f"(cue-blind {st['blind']:.1f}%, ceiling {ceiling:.1f}%)")

    flags = f"--init-eta-out {cfg.init_eta_out} --init-temperature {cfg.init_temperature}"
    if cfg.center:
        flags += " --center"
    if cfg.retain > 0:
        flags += f" --retain {cfg.retain}"
    if margin <= 0:
        print("VERDICT: rule does NOT beat a cue-blind prior. Do not run evolution yet.")
    elif st["acc"] < args.target:
        print(f"VERDICT: mapping is learned (+{margin:.1f} over cue-blind) but below the "
              f"{args.target:.0f}% target. Keep fixing the rule before evolution.")
    else:
        print(f"VERDICT: rule works (>= {args.target:.0f}%). Proceed to evolution with:")
    print(f"  python dsm_benchmark_b_v3.py --structure fixed --hidden {size} "
          f"--no-metacognition {flags}")


# =====================================================================
# CHECKPOINTS
# =====================================================================

def save_checkpoint(path: str, theta: torch.Tensor, cfg: Config) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"genome": theta.detach().cpu(), "config": asdict(cfg)}, path)


def load_checkpoint(path: str, device: torch.device) -> Tuple[torch.Tensor, Config]:
    data = torch.load(path, map_location="cpu")
    known = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in data["config"].items() if k in known})
    return data["genome"].to(device), cfg


# =====================================================================
# CLI
# =====================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DSM Benchmark B v3")
    p.add_argument("--mode", choices=["train", "eval", "rule-test", "summarize"],
                   default="train")
    p.add_argument("--device", default="auto")
    p.add_argument("--checkpoint", default=None,
                   help="default: checkpoints_v3/<run>_seed<seed>.pt")
    p.add_argument("--results", default=None,
                   help="default: <results-dir>/<run>_seed<seed>.json")
    p.add_argument("--results-dir", default="results_v3")
    p.add_argument("--eval-lives", type=int, default=1000)

    p.add_argument("--structure", choices=STRUCTURE_MODES, default="full")
    p.add_argument("--hidden", type=int, default=16,
                   help="initial neurons (fixed size in --structure fixed)")
    p.add_argument("--max-hidden", type=int, default=128)
    p.add_argument("--min-hidden", type=int, default=8)

    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--population", type=int, default=64)
    p.add_argument("--lives", type=int, default=24)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--max-tasks", type=int, default=4)
    p.add_argument("--n-cues", type=int, default=4)
    p.add_argument("--n-actions", type=int, default=4)
    p.add_argument("--fault-probability", type=float, default=0.65)
    p.add_argument("--rehearsal-fraction", type=float, default=0.30)
    p.add_argument("--cue-gain", type=float, default=2.0)

    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--es-lr", type=float, default=0.02)
    p.add_argument("--complexity-cost", type=float, default=0.005)
    p.add_argument("--accuracy-weight", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--random-policy", action="store_true",
                   help="control: uniform random actions (should equal chance)")
    p.add_argument("--no-plasticity", action="store_true",
                   help="control: disable lifetime synaptic learning")
    p.add_argument("--no-metacognition", action="store_true",
                   help="disconnect doubt/learnability from network and modulator")

    p.add_argument("--init-eta-out", type=float, default=1.0,
                   help="starting output learning-rate logit")
    p.add_argument("--init-temperature", type=float, default=0.3,
                   help="starting policy temperature")

    # rule-test options
    p.add_argument("--sizes", default="16", help="rule-test: comma list of sizes")
    p.add_argument("--eta-out-grid", default="-1,0,1,3",
                   help="rule-test: output learning-rate logits")
    p.add_argument("--temperature-grid", default="0.1,0.2,0.3,0.5",
                   help="rule-test: policy temperatures")
    p.add_argument("--metacognition", action="store_true",
                   help="rule-test: keep doubt/learnability connected")
    p.add_argument("--target", type=float, default=60.0,
                   help="rule-test: accuracy the rule must reach")
    p.add_argument("--variants",
                   default="base,center,retain,center+retain,nodelay,nodelay+center",
                   help="rule-test: representation variants to compare")
    p.add_argument("--retain-value", type=float, default=0.5,
                   help="rule-test: retain used by the 'retain' variants")
    p.add_argument("--probe-lives", type=int, default=300,
                   help="rule-test: lives used by the linear probe")
    p.add_argument("--verbose", action="store_true",
                   help="rule-test: print every grid point")

    # Representation options (use the ones rule-test recommends)
    p.add_argument("--center", action="store_true",
                   help="read out activity minus each neuron's running mean")
    p.add_argument("--retain", type=float, default=0.0,
                   help="leaky units: h = r*h_prev + (1-r)*tanh(.)")
    p.add_argument("--no-delay", action="store_true",
                   help="DIAGNOSTIC: cue also visible on response steps")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        steps=args.steps, max_tasks=args.max_tasks, n_cues=args.n_cues,
        n_actions=args.n_actions, fault_probability=args.fault_probability,
        rehearsal_fraction=args.rehearsal_fraction, cue_gain=args.cue_gain,
        max_hidden=args.max_hidden, initial_hidden=args.hidden,
        min_hidden=min(args.min_hidden, args.hidden), structure=args.structure,
        random_policy=args.random_policy, plasticity=not args.no_plasticity,
        metacognition=not args.no_metacognition,
        center=args.center, retain=args.retain, no_delay=args.no_delay,
        init_eta_out=args.init_eta_out, init_temperature=args.init_temperature,
        generations=args.generations, population=args.population,
        lives=args.lives, sigma=args.sigma, es_lr=args.es_lr,
        complexity_cost=args.complexity_cost,
        accuracy_weight=args.accuracy_weight, seed=args.seed,
    )


def validate_config(cfg: Config) -> None:
    if cfg.steps % 2 != 0:
        raise ValueError("--steps must be even")
    if cfg.population < 2 or cfg.population % 2 != 0:
        raise ValueError("--population must be an even number >= 2")
    if not (cfg.min_hidden <= cfg.initial_hidden <= cfg.max_hidden):
        raise ValueError("require min_hidden <= hidden <= max_hidden")
    if cfg.structure not in STRUCTURE_MODES:
        raise ValueError(f"--structure must be one of {STRUCTURE_MODES}")


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "summarize":
        summarize(args.results_dir)
        return

    device = choose_device(args.device)
    configure_torch(device)

    if args.mode == "rule-test":
        rule_test(args, device)
        return

    if args.mode == "train":
        cfg = config_from_args(args)
        validate_config(cfg)
        ckpt = args.checkpoint or f"checkpoints_v3/{cfg.run_name}_seed{cfg.seed}.pt"
        theta = evolve(cfg, device)
        save_checkpoint(ckpt, theta, cfg)
        print(f"\nSaved checkpoint: {ckpt}")
    else:
        if args.checkpoint:
            theta, cfg = load_checkpoint(args.checkpoint, device)
            cfg.random_policy = args.random_policy
            cfg.plasticity = not args.no_plasticity
            print(f"Loaded checkpoint: {args.checkpoint}")
        else:
            # Evaluate the initial genome (useful for controls).
            cfg = config_from_args(args)
            validate_config(cfg)
            dummy = Scenarios(cfg, 1, cfg.seed + 123, device)
            obs_dim = BenchmarkB(cfg, dummy, 1, device).obs_dim
            theta = initial_genome(cfg, obs_dim, device, cfg.seed)
            print("No checkpoint: evaluating the initial (generation 0) genome")

    final = evaluate(theta, cfg, device, seed=900_000 + cfg.seed,
                     lives=args.eval_lives, record=True)
    print_final(final, cfg)
    print_trace(final["trace"])

    if args.mode == "train" or args.results:
        suffix = ""
        if cfg.random_policy:
            suffix += "_random"
        if not cfg.plasticity:
            suffix += "_noplast"
        path = args.results or os.path.join(
            args.results_dir, f"{cfg.run_name}{suffix}_seed{cfg.seed}.json"
        )
        save_results(path, final, cfg)

    if device.type == "cuda":
        print(f"\nPeak CUDA memory: "
              f"{torch.cuda.max_memory_allocated(device) / 2**30:.2f} GB")


if __name__ == "__main__":
    main()

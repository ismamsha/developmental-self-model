"""
DSM Benchmark B v4 — Developmental Self Model with sparse associative memory
============================================================================

One standalone file. No project imports.

Why v4
------
v3 diagnostics showed the cue IS present in the recurrent state (a linear
probe decodes it at ~94%), yet dense three-factor learning only reached
34-40% on 4 tasks x 4 cues x 4 actions — below a cue-blind "favourite action"
strategy (~50%). The bottleneck was interference in dense shared weights, not
memory and not evolution.

v4 architecture (per organism, all learning local, no backprop):

    task+cue ──► sparse expansion (k-winners-take-all over active units)
             ──► gated working memory (held across the delay)
             ──► fast associative readout, three-factor rule:
                   dW = eta * modulator(RPE, doubt, learnability)
                            * code ⊗ (onehot(action) - policy)

Development = functional growth: when the structural controller decides to
GROW (driven by errors, capacity pressure = task error x (1 - learnability),
doubt, novelty), a new unit is IMPRINTED on the current task+cue pattern —
a dedicated representational slot — but only if no unit already owns that
pattern (novelty gate). Pruning removes units with low usage x |weights|.

Prototype results (hand-built rule, no evolution, 400 lives, CPU):
    fixed-16  (k=2)                          52.1%   16 units
    fixed-128 (k=2)                          62.7%   128 units
    random growth on every error (k=2)       53.4%   44 average / 72 final
    imprint growth on every error (k=2)      62.9%   40 average / 61 final
    fixed-128 (k=1)                          58.8%   128 units
    novelty-gated imprint growth (k=1)       60.7%   24 average / 30 final
    cue-blind baseline ~49.5%, perfect-learner ceiling ~80%
Run --mode hand to reproduce with this file's controller.

Structural modes (--structure)
------------------------------
  fixed    : constant size (baselines fixed-8/16/32/64/128)
  random   : controller grows, but new units have random input wiring
  imprint  : controller grows by imprinting new pattern slots (no pruning)
  full     : imprint growth + pruning (full DSM)
  nolearn  : full, controller does not see learnability
  basic    : full, controller sees no self-model / learnability signals

Metacognition tests (v4.1)
--------------------------
In the standard task every error means "this pattern has no slot yet", so the
novelty rule alone is near-optimal and self-model signals add nothing (evolved
full = nolearn = basic ~ 67%). Two conditions make an error ambiguous:
  --reward-noise p : feedback flipped with probability p (true accuracy scored)
  --reversal       : the first --reversal-tasks tasks remap mid-life
Levers that need the self-model to be used well:
  REPAIR           : reset the slot that owns the current pattern
  plasticity gate  : eta *= exp(gate . [|doubt|, slot doubt, learnability, 1])
  slot doubt       : per-unit recent CONFIDENT-error rate (surprise), so a
                     slot that is still learning is not mistaken for a broken one
Ablations remove self-model inputs from growth, repair, gate and modulator
(basic: all; nolearn: learnability only).

Hand-built check (480 steps, rehearsal 0.5, noise 0.15 + reversal, 200 lives):
    fixed-128        48.6%   remapped patterns 15.7%
    full-16          53.7%   remapped patterns 29.4%
    basic-16         52.0%   remapped patterns 12.1%  (cannot find broken slots)
    nolearn-16 = full-16 (the useful signal is slot doubt, not learnability)

    bash scripts/run_v4_meta_grid.sh

Quick start
-----------
1) Hand-built comparison (no evolution, minutes on a GPU):
    python benchmarks/dsm_benchmark_b_v4.py --mode hand --seeds 0,1,2

2) Evolve one configuration:
    python benchmarks/dsm_benchmark_b_v4.py --structure full --units 16 --generations 100

3) Full grid over seeds, then summarise:
    bash scripts/run_v4_grid.sh
    python benchmarks/dsm_benchmark_b_v4.py --mode summarize --results-dir results_v4
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

STRUCTURE_MODES = ("fixed", "random", "imprint", "full", "nolearn", "basic")


@dataclass
class Config:
    # Benchmark B
    steps: int = 240                 # must be even
    max_tasks: int = 4
    n_cues: int = 4
    n_actions: int = 4
    fault_probability: float = 0.65
    rehearsal_fraction: float = 0.30
    cue_gain: float = 2.0
    no_delay: bool = False           # diagnostic only

    # Conditions that make "an error" ambiguous (metacognition tests)
    reward_noise: float = 0.0        # P(feedback flipped); true accuracy is scored
    reversal: bool = False           # tasks < reversal_tasks remap mid-life
    reversal_tasks: int = 1

    # Sparse representation
    pool: int = 128                  # maximum units
    initial_units: int = 16
    min_units: int = 8
    k: int = 1                       # active units per pattern
    novelty: float = 0.9             # imprint only if best match < novelty*gain
    imprint_gain: float = 1.5
    structure: str = "full"
    prune_interval: int = 12

    # Controls
    random_policy: bool = False
    plasticity: bool = True
    metacognition: bool = True

    # Hand-built starting genome
    init_eta: float = 0.0            # logit; rate = sigmoid(logit) (0.5)
    init_temperature: float = 0.3
    init_repair_bias: float = -8.0   # repair off in the hand-built genome
    init_repair_slot: float = 0.0    # weight on slot doubt
    init_repair_wrong: float = 0.0   # weight on "wrong this trial"
    init_gate_slot: float = 0.0      # plasticity gate weight on slot doubt

    # Evolution
    generations: int = 100
    population: int = 64
    lives: int = 32
    sigma: float = 0.05
    es_lr: float = 0.02
    weight_decay: float = 0.001

    # Fitness
    complexity_cost: float = 0.005
    structural_change_cost: float = 0.002
    survival_bonus: float = 5.0
    death_penalty: float = 5.0
    accuracy_weight: float = 0.0

    seed: int = 0

    @property
    def run_name(self) -> str:
        name = f"{self.structure}-{self.initial_units}"
        if not self.metacognition:
            name += "-nometa"
        if self.reward_noise > 0:
            name += f"-noise{self.reward_noise:g}"
        if self.reversal:
            name += "-rev"
        return name


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

        # Drawn last so earlier random streams are unchanged.
        self.noise_u = torch.rand(n, cfg.steps, generator=gen, device=device)
        self.reversal_step = torch.randint(
            cfg.steps // 2, max(cfg.steps // 2 + 1, (3 * cfg.steps) // 4), (n,),
            generator=gen, device=device)
        shift = torch.randint(1, cfg.n_actions, self.mapping.shape,
                              generator=gen, device=device)
        self.mapping_rev = self.mapping.clone()
        rt = cfg.reversal_tasks
        self.mapping_rev[:, :rt] = (self.mapping[:, :rt] + shift[:, :rt]) % cfg.n_actions
        self.repair_u = torch.rand(n, cfg.steps, generator=gen, device=device)

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
        self.noise_u = scenarios.noise_u.repeat(population, 1)
        self.repair_u = scenarios.repair_u.repeat(population, 1)
        self.reversal_step = scenarios.reversal_step.repeat(population)
        self.mapping_rev = scenarios.mapping_rev.repeat(population, 1, 1)
        self.last_true_task_reward = torch.zeros(self.batch, device=device)
        self.current_reversed = torch.zeros(self.batch, dtype=torch.bool, device=device)
        # Accuracy on remapped patterns after the reversal.
        self.reversed_correct = torch.zeros(self.batch, device=device)
        self.reversed_responses = torch.zeros(self.batch, device=device)

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
            if self.cfg.reversal:
                self.current_reversed = (
                    (t >= self.reversal_step)
                    & (self.current_task < self.cfg.reversal_tasks))
                self.current_target = torch.where(
                    self.current_reversed,
                    self.mapping_rev[self.rows, self.current_task, self.current_cue],
                    self.current_target)
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

            self.reversed_correct += (correct & self.current_reversed).float()
            self.reversed_responses += (live & self.current_reversed).float()

            task_reward = torch.where(
                correct, torch.ones_like(reward), torch.full_like(reward, -0.20)
            )
            # Fitness uses the true outcome; the organism perceives feedback
            # that is flipped with probability reward_noise.
            reward = torch.where(live, task_reward, reward)
            flipped = self.noise_u[:, t] < self.cfg.reward_noise
            perceived = torch.where(
                correct ^ flipped, torch.ones_like(reward), torch.full_like(reward, -0.20))
            self.last_true_task_reward = torch.where(
                live, task_reward, self.last_true_task_reward)
            self.last_task_reward = torch.where(
                live, perceived, self.last_task_reward
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

# Structural controller features (index -> meaning):
#  0 |doubt|, 1 self-model error EMA, 2 task error, 3 learnability,
#  4 capacity pressure, 5 wrong this trial, 6 novelty of the pattern,
#  7 active fraction, 8 recent accuracy, 9 slot doubt (recent confident-
#  error rate of the unit that owns this pattern), 10 bias
STRUCT_FEATURES = 11
F_PRESSURE, F_WRONG, F_SLOT, F_BIAS = 4, 5, 9, 10

# Self-model signals removed by the ablations (structure, gate, modulator).
SELF_MODEL_FEATURES = (0, 1, 3, 4, F_SLOT)


def structure_mask(mode: str) -> List[float]:
    mask = [1.0] * STRUCT_FEATURES
    if mode == "nolearn":
        mask[3] = 0.0
    elif mode == "basic":
        for i in SELF_MODEL_FEATURES:
            mask[i] = 0.0
    return mask


def gate_mask(mode: str) -> List[float]:
    # Plasticity gate inputs: |doubt|, slot doubt, learnability, bias.
    if mode == "basic":
        return [0.0, 0.0, 0.0, 1.0]
    if mode == "nolearn":
        return [1.0, 1.0, 0.0, 1.0]
    return [1.0, 1.0, 1.0, 1.0]


def modulator_mask(mode: str, metacognition: bool) -> List[float]:
    # Modulator inputs: doubt(3), |doubt|(3), RPE, learnability, bias.
    if not metacognition or mode == "basic":
        return [0, 0, 0, 0, 0, 0, 1, 0, 1]
    if mode == "nolearn":
        return [1, 1, 1, 1, 1, 1, 1, 0, 1]
    return [1] * 9


def genome_spec(cfg: Config) -> List[Tuple[str, Tuple[int, ...]]]:
    return [
        # Output rule dW = A*pre*post + B*pre + C*post + D (per action).
        ("rule_out", (4, cfg.n_actions)),
        ("eta", (1,)),
        ("temperature", (1,)),
        # doubt(3), |doubt|(3), reward-prediction-error, learnability, bias
        ("modulator", (9,)),
        ("eta_self", (1,)),
        ("doubt_gain", (3,)),
        ("doubt_memory", (1,)),
        ("eta_learnability", (1,)),
        ("grow", (STRUCT_FEATURES,)),
        ("prune", (STRUCT_FEATURES,)),
        # REPAIR: reset the unit that owns this pattern so it can relearn.
        ("repair", (STRUCT_FEATURES,)),
        # Meta-plasticity: eta *= exp(gate . [|doubt|, slot doubt, learn, 1]).
        ("plasticity_gate", (4,)),
    ]


def initial_genome(cfg: Config, device: torch.device, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    parts = []
    for name, shape in genome_spec(cfg):
        value = np.zeros(shape)
        if name == "rule_out":
            value = 0.02 * rng.standard_normal(shape)
            value[0] += 1.0
        elif name == "eta":
            value[0] = cfg.init_eta
        elif name == "temperature":
            value[0] = math.log(cfg.init_temperature)
        elif name == "modulator":
            value[6] = 2.0                      # reward-prediction-error
        elif name == "eta_self":
            value[0] = -1.5
        elif name == "doubt_gain":
            value = np.log([10.0, 30.0, 1.5])
        elif name == "doubt_memory":
            value[0] = 2.0
        elif name == "eta_learnability":
            value[0] = -1.0
        elif name == "grow":
            # Grow mostly after errors, more under capacity pressure:
            # wrong & pressure 0.35 -> p ~ 0.87; correct -> p ~ 0.07.
            value[F_BIAS] = -4.0
            value[F_WRONG] = 4.5
            value[F_PRESSURE] = 4.0
        elif name == "prune":
            value[F_BIAS] = -4.0                # rare until evolution says so
        elif name == "repair":
            value[F_BIAS] = cfg.init_repair_bias
            value[F_SLOT] = cfg.init_repair_slot
            value[F_WRONG] = cfg.init_repair_wrong
        elif name == "plasticity_gate":
            value[1] = cfg.init_gate_slot
        parts.append(np.asarray(value, dtype=np.float64).reshape(-1))
    return torch.tensor(np.concatenate(parts).astype(np.float32), device=device)


def unpack_genomes(theta: torch.Tensor, cfg: Config) -> Dict[str, torch.Tensor]:
    out, pos = {}, 0
    for name, shape in genome_spec(cfg):
        n = int(np.prod(shape))
        out[name] = theta[:, pos:pos + n].reshape(len(theta), *shape)
        pos += n
    return out


def local_rule(coef: torch.Tensor, pre: torch.Tensor, post: torch.Tensor
               ) -> torch.Tensor:
    A, B = coef[:, 0, None, :], coef[:, 1, None, :]
    C, D = coef[:, 2, None, :], coef[:, 3, None, :]
    return (A * pre[:, :, None] * post[:, None, :] + B * pre[:, :, None]
            + C * post[:, None, :] + D)


# =====================================================================
# LIFETIME
# =====================================================================

@torch.no_grad()
def run_lives(theta: torch.Tensor, scenarios: Scenarios, cfg: Config,
              device: torch.device, record: bool = False) -> Dict:
    population, lives = theta.shape[0], scenarios.n
    batch = population * lives

    env = BenchmarkB(cfg, scenarios, population, device)
    n_obs = env.obs_dim
    n_input = n_obs + 3 + 1 + 1
    Pn, A = cfg.pool, cfg.n_actions

    genome = unpack_genomes(theta, cfg)
    g = {k: v.repeat_interleave(lives, dim=0) for k, v in genome.items()}

    developing = cfg.structure != "fixed"
    imprinting = cfg.structure in ("imprint", "full", "nolearn", "basic")
    pruning = cfg.structure in ("full", "nolearn", "basic")
    s_mask = torch.tensor(structure_mask(cfg.structure), device=device)

    # Random birth wiring of the expansion layer: task/cue inputs only.
    gen = torch.Generator(device=device)
    gen.manual_seed(scenarios.birth_seed)
    life_exp = torch.randn(lives, n_obs, Pn, generator=gen, device=device) / math.sqrt(n_obs)
    life_exp[:, :5] = 0.0
    W_exp = life_exp.repeat(population, 1, 1)

    active = torch.zeros(batch, Pn, dtype=torch.bool, device=device)
    active[:, : cfg.initial_units] = True
    W = torch.zeros(batch, Pn, A, device=device)
    code = torch.zeros(batch, Pn, device=device)
    x_instr = torch.zeros(batch, n_obs, device=device)
    usage = torch.zeros(batch, Pn, device=device)
    reward_baseline = torch.zeros(batch, device=device)
    # Per-unit self-model: recent (perceived) error rate of each unit's
    # association — how much the organism should trust that slot.
    slot_doubt = torch.zeros(batch, Pn, device=device)
    repair_total = torch.zeros(batch, device=device)

    SELF_OUT = 3
    feature_dim = Pn + n_input
    self_model = torch.zeros(batch, feature_dim, SELF_OUT * A, device=device)
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
    grow_total = torch.zeros(batch, device=device)
    prune_total = torch.zeros(batch, device=device)
    T = cfg.max_tasks
    phase_capacity_sum = torch.zeros(batch, T, device=device)
    phase_capacity_n = torch.zeros(batch, T, device=device)
    phase_grow = torch.zeros(batch, T, device=device)

    temperature = torch.exp(torch.clamp(g["temperature"][:, 0], -4.0, 2.0))
    eta = torch.sigmoid(g["eta"][:, 0])
    eta_self = 0.20 * torch.sigmoid(g["eta_self"][:, 0])
    eta_meta = 0.10 * torch.sigmoid(g["eta_learnability"][:, 0])
    memory = torch.sigmoid(g["doubt_memory"][:, 0])
    gain = torch.exp(torch.clamp(g["doubt_gain"], -3.0, 4.0))

    rows = torch.arange(batch, device=device)
    ones = torch.ones(batch, 1, device=device)
    mod_mask = torch.tensor(modulator_mask(cfg.structure, cfg.metacognition),
                            dtype=torch.float32, device=device)
    g_mask = torch.tensor(gate_mask(cfg.structure), device=device)

    rec: Optional[Dict] = None
    if record:
        rec = {"correct": [], "pattern": [], "act": [], "pressure": [],
               "learnability": [], "task_error": [],
               "capacity": [], "acc_fast": [], "doubt": [], "error": [],
               "learn_mean": [], "pressure_mean": [], "grow": [], "prune": []}

    for t in range(cfg.steps):
        if not env.alive.any():
            break
        obs, is_instruction, phase = env.observation()

        if cfg.metacognition:
            meta_in = torch.cat([doubt, learnability_signal[:, None]], dim=1)
        else:
            meta_in = torch.zeros(batch, SELF_OUT + 1, device=device)
        x = torch.cat([obs, meta_in, ones], dim=1)

        # Sparse pattern code, held in working memory across the delay.
        if is_instruction:
            x_instr = obs.clone()
            x_instr[:, :5] = 0.0
            z = torch.einsum("bi,bij->bj", x_instr, W_exp).masked_fill(~active, -1e9)
            code = torch.zeros(batch, Pn, device=device).scatter_(
                1, z.topk(cfg.k, dim=1).indices, 1.0)

        logits = torch.einsum("bi,bij->bj", code, W)
        policy = torch.softmax(logits / temperature[:, None], dim=1)
        if cfg.random_policy:
            action = (env.action_u[:, t] * A).long().clamp(max=A - 1)
        else:
            action = (policy.cumsum(dim=1) < env.action_u[:, t, None]
                      ).sum(dim=1).clamp(max=A - 1)

        feature = torch.cat([code, x], dim=1)

        # Self-model predicts energy change, damage change and reward.
        predictions = torch.einsum("bi,bij->bj", feature, self_model)
        predicted = torch.stack([predictions[rows, action],
                                 predictions[rows, A + action],
                                 predictions[rows, 2 * A + action]], dim=1)
        energy_before, damage_before = env.energy.clone(), env.damage.clone()

        reward, lived, correct_mask = env.step(action, is_instruction, phase)
        total_reward += reward
        lived_f = lived.float()

        # The self-model only sees perceived (possibly noisy) feedback.
        perceived = reward
        if not is_instruction:
            perceived = reward + (env.last_task_reward - env.last_true_task_reward) * lived_f
        real = torch.stack([env.energy - energy_before,
                            env.damage - damage_before, perceived], dim=1)
        error = (real - predicted) * lived_f[:, None]
        for ch in range(SELF_OUT):
            self_model[rows, :, ch * A + action] += (
                eta_self[:, None] * feature * error[:, ch:ch + 1])
        self_model.clamp_(-3.0, 3.0)
        doubt = (memory[:, None] * doubt
                 + (1.0 - memory)[:, None] * torch.clamp(error, -1.0, 1.0) * gain)

        growth_events = torch.zeros(batch, device=device)
        prune_events = torch.zeros(batch, device=device)

        if not is_instruction:
            # Behavioural statistics and learnability.
            acc_fast = torch.where(lived, 0.80 * acc_fast + 0.20 * correct_mask, acc_fast)
            acc_slow = torch.where(lived, 0.95 * acc_slow + 0.05 * correct_mask, acc_slow)
            error_ema = torch.where(lived, 0.90 * error_ema
                                    + 0.10 * error.abs().mean(dim=1), error_ema)
            target = torch.tanh(10.0 * (acc_fast - acc_slow))
            l_err = (target - learn_pred_prev) * lived_f
            learnability_model += eta_meta[:, None] * feat_prev * l_err[:, None]
            learnability_model.clamp_(-2.0, 2.0)
            learn_pred = torch.tanh((feature * learnability_model).sum(dim=1))
            learnability_signal = torch.where(lived, learn_pred, learnability_signal)
            learn_pred_prev, feat_prev = learn_pred, feature

            # Three-factor associative learning.
            task_reward = env.last_task_reward
            rpe = (task_reward - reward_baseline) * lived_f
            reward_baseline = torch.where(
                lived, 0.9 * reward_baseline + 0.1 * task_reward, reward_baseline)
            mod_input = torch.cat([doubt, doubt.abs(), rpe[:, None],
                                   learnability_signal[:, None], ones], dim=1)
            mod_input = mod_input * mod_mask
            modulator = torch.tanh((mod_input * g["modulator"]).sum(dim=1)) * lived_f

            wrong = lived & (task_reward < 0)
            slot_now = (code * slot_doubt).sum(dim=1) / code.sum(dim=1).clamp(min=1.0)

            if cfg.plasticity:
                gate_in = torch.cat([doubt.norm(dim=1, keepdim=True), slot_now[:, None],
                                     learnability_signal[:, None], ones], dim=1) * g_mask
                eta_eff = eta * torch.exp(torch.clamp(
                    (gate_in * g["plasticity_gate"]).sum(dim=1), -2.0, 2.0))
                post = F.one_hot(action, A).float() - policy
                dW = local_rule(g["rule_out"], code, post) * active[:, :, None]
                W += (eta_eff * modulator)[:, None, None] * dW
                W.clamp_(-3.0, 3.0)
            usage = 0.98 * usage + 0.02 * code
            # Surprise = an error the slot was confident it would not make
            # (confidence = policy probability of the chosen action). Errors
            # while a slot is still learning count little; a confident slot
            # that starts failing (e.g. after a reversal) counts a lot.
            confidence = policy[rows, action]
            surprise = wrong.float() * confidence
            slot_doubt = torch.where((code > 0) & lived[:, None],
                                     0.7 * slot_doubt + 0.3 * surprise[:, None],
                                     slot_doubt)

            # ---------------- development ----------------
            task_error = 1.0 - acc_slow
            learn_p = (learnability_signal if cfg.structure not in ("nolearn", "basic")
                       else torch.zeros_like(learnability_signal))
            pressure = task_error * (1.0 - learn_p) / 2.0

            v = x_instr / x_instr.norm(dim=1, keepdim=True).clamp(min=1e-6)
            match = torch.einsum("bi,bij->bj", v, W_exp).masked_fill(~active, -1e9
                                                                    ).max(dim=1).values
            novelty = torch.clamp(1.0 - match / cfg.imprint_gain, 0.0, 1.0)

            f = torch.cat([
                doubt.norm(dim=1, keepdim=True), error_ema[:, None],
                task_error[:, None], learnability_signal[:, None], pressure[:, None],
                wrong.float()[:, None], novelty[:, None],
                (active.sum(dim=1).float() / Pn)[:, None], acc_fast[:, None],
                slot_now[:, None], ones,
            ], dim=1) * s_mask

            grow = torch.zeros_like(lived)
            if developing:
                p_grow = torch.sigmoid((f * g["grow"]).sum(dim=1))
                room = active.sum(dim=1) < Pn
                grow = lived & room & (env.structure_u[:, t] < p_grow)
                if imprinting:
                    grow &= match < cfg.novelty * cfg.imprint_gain
                if grow.any():
                    rr = torch.where(grow)[0]
                    cc = (~active[rr]).float().argmax(dim=1)
                    active[rr, cc] = True
                    W[rr, cc] = 0.0
                    usage[rr, cc] = 0.0
                    slot_doubt[rr, cc] = 0.0
                    if imprinting:
                        # New slot tuned to exactly this task+cue pattern.
                        W_exp[rr, :, cc] = cfg.imprint_gain * v[rr]
                    growth_events[rr] = 1.0

                # REPAIR: the pattern already has a slot but the organism
                # decides that slot is unreliable -> reset it to relearn.
                owned = match >= cfg.novelty * cfg.imprint_gain if imprinting else lived
                p_repair = torch.sigmoid((f * g["repair"]).sum(dim=1))
                repair = lived & ~grow & owned & (env.repair_u[:, t] < p_repair)
                if repair.any():
                    reset = (code > 0) & repair[:, None]
                    W = torch.where(reset[:, :, None], torch.zeros_like(W), W)
                    slot_doubt = torch.where(reset, torch.zeros_like(slot_doubt), slot_doubt)
                    repair_total += repair.float()

                trial = t // 2
                if pruning and t >= cfg.steps // 3 and trial % max(1, cfg.prune_interval // 2) == 0:
                    p_prune = torch.sigmoid((f * g["prune"]).sum(dim=1))
                    can = (lived & (active.sum(dim=1) > cfg.min_units)
                           & (env.structure_u[:, t - 1] < p_prune))
                    if can.any():
                        util = (usage * W.abs().sum(dim=2)).masked_fill(~active, float("inf"))
                        rr = torch.where(can)[0]
                        cc = util[rr].argmin(dim=1)
                        active[rr, cc] = False
                        W[rr, cc] = 0.0
                        prune_events[rr] = 1.0

            phase_grow[:, phase] += growth_events

            if rec is not None:
                nan = torch.full_like(correct_mask, float("nan"))
                rec["correct"].append(torch.where(lived, correct_mask, nan).cpu())
                rec["pattern"].append((env.current_task * cfg.n_cues
                                       + env.current_cue).cpu())
                act = torch.where(grow, 1, 0)
                act = torch.where(lived, act, -1)
                rec["act"].append(act.cpu())
                rec["pressure"].append(pressure.cpu())
                rec["learnability"].append(learnability_signal.cpu())
                rec["task_error"].append(task_error.cpu())

        grow_total += growth_events
        prune_total += prune_events
        now_capacity = active.sum(dim=1).float()
        total_capacity += now_capacity * lived_f
        if not is_instruction:
            phase_capacity_sum[:, phase] += now_capacity * lived_f
            phase_capacity_n[:, phase] += lived_f

        if rec is not None:
            alive_n = max(1.0, lived_f.sum().item())
            rec["capacity"].append((now_capacity * lived_f).sum().item() / alive_n)
            rec["acc_fast"].append((acc_fast * lived_f).sum().item() / alive_n)
            rec["doubt"].append(doubt.norm(dim=1).mean().item())
            rec["error"].append(error_ema.mean().item())
            rec["learn_mean"].append(learnability_signal.mean().item())
            rec["pressure_mean"].append(
                ((1.0 - acc_slow) * (1.0 - learnability_signal) / 2.0).mean().item())
            rec["grow"].append(growth_events.mean().item())
            rec["prune"].append(prune_events.mean().item())

    # ---------------- fitness ----------------
    average_capacity = total_capacity / cfg.steps
    extra = torch.clamp(average_capacity - cfg.initial_units, min=0.0) / Pn
    fitness = (total_reward
               - cfg.complexity_cost * extra * (cfg.steps / 2)
               - cfg.structural_change_cost * (grow_total + prune_total + repair_total)
               + cfg.accuracy_weight * env.correct_total / env.response_total.clamp(min=1.0))

    P, L = population, lives
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
        "phase_grow": phase_grow.reshape(P, L, T),
        "task_correct": env.task_correct.reshape(P, L, T, T),
        "task_responses": env.task_responses.reshape(P, L, T, T),
        "target_count": env.target_count.reshape(P, L, T, A),
        "grow_total": grow_total.reshape(P, L),
        "prune_total": prune_total.reshape(P, L),
        "repair_total": repair_total.reshape(P, L),
        "reversed_correct": env.reversed_correct.reshape(P, L),
        "reversed_responses": env.reversed_responses.reshape(P, L),
    }
    if rec is not None:
        result["record"] = rec
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
# CEILING
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
# ANALYSIS
# =====================================================================

def growth_analysis(rec: Dict, window: int = 3) -> Dict[str, float]:
    """
    After an ERROR, compare trials where the organism grew with trials where
    it held:
      * pressure / learnability / task error at the decision (averaged per
        trial, so it is time-controlled),
      * accuracy on the next `window` occurrences of the SAME task+cue
        pattern (does the new slot fix the pattern that failed?).
    """
    out: Dict[str, float] = {}
    if not rec["act"]:
        return out
    correct = torch.stack(rec["correct"]).numpy()        # [trials, batch]
    pattern = torch.stack(rec["pattern"]).numpy()
    act = torch.stack(rec["act"]).numpy()
    eligible = (act >= 0) & (correct == 0)
    grow, hold = eligible & (act == 1), eligible & (act == 0)
    out["grow_decisions_after_error"] = float(grow.sum())
    out["hold_decisions_after_error"] = float(hold.sum())
    if grow.sum() == 0 or hold.sum() == 0:
        return out

    for key in ("pressure", "learnability", "task_error"):
        v = torch.stack(rec[key]).numpy()
        diffs = [v[i][grow[i]].mean() - v[i][hold[i]].mean()
                 for i in range(len(v)) if grow[i].any() and hold[i].any()]
        if diffs:
            out[f"grow_minus_hold_{key}"] = float(np.mean(diffs))

    n_trials, n_lives = correct.shape
    future = np.full_like(correct, np.nan)
    for b in range(n_lives):
        seen: Dict[int, List[int]] = {}
        for i in range(n_trials - 1, -1, -1):
            nxt = seen.get(int(pattern[i, b]), [])
            vals = correct[nxt[:window], b] if nxt else np.array([])
            vals = vals[~np.isnan(vals)]
            if len(vals):
                future[i, b] = vals.mean()
            seen[int(pattern[i, b])] = [i] + nxt[: window - 1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fg, fh = np.nanmean(future[grow]), np.nanmean(future[hold])
    if np.isfinite(fg) and np.isfinite(fh):
        out["next_same_pattern_acc_after_grow"] = float(100 * fg)
        out["next_same_pattern_acc_after_hold"] = float(100 * fh)
    return out


@torch.no_grad()
def evaluate(theta: torch.Tensor, cfg: Config, device: torch.device, seed: int,
             lives: int = 512, record: bool = False) -> Dict:
    scenarios = Scenarios(cfg, lives, seed, device)
    r = run_lives(theta[None], scenarios, cfg, device, record=record)

    survived, correct = r["survived"][0], r["correct"][0]
    total = r["responses"][0].sum().clamp(min=1)
    fault = scenarios.fault_kind >= 0
    tc = r["target_count"][0]
    T = cfg.max_tasks

    m: Dict = {
        "accuracy": (100.0 * correct.sum() / total).item(),
        "chance": 100.0 / cfg.n_actions,
        "cue_blind_life": (100.0 * tc.sum(dim=1).max(dim=1).values.sum() / total).item(),
        "cue_blind_phase": (100.0 * tc.max(dim=2).values.sum() / total).item(),
        "survival": 100.0 * survived.float().mean().item(),
        "survival_fault": (100.0 * survived[fault].float().mean().item()
                           if fault.any() else float("nan")),
        "reward": r["reward"][0].mean().item(),
        "average_units": r["capacity"][0].mean().item(),
        "final_units": r["final_capacity"][0].mean().item(),
        "grow_events": r["grow_total"][0].mean().item(),
        "prune_events": r["prune_total"][0].mean().item(),
        "repair_events": r["repair_total"][0].mean().item(),
        "reversed_accuracy": (
            (100.0 * r["reversed_correct"][0].sum()
             / r["reversed_responses"][0].sum()).item()
            if r["reversed_responses"][0].sum() > 0 else float("nan")),
    }
    for p in range(T):
        m[f"phase_{p+1}_accuracy"] = (100.0 * r["phase_correct"][0][:, p].sum()
                                      / r["phase_responses"][0][:, p].sum().clamp(min=1)).item()
        m[f"phase_{p+1}_units"] = r["phase_capacity"][0][:, p].mean().item()
        m[f"phase_{p+1}_grow"] = r["phase_grow"][0][:, p].mean().item()

    tcor = r["task_correct"][0].sum(dim=0)
    tres = r["task_responses"][0].sum(dim=0)
    tp = torch.where(tres > 0, 100.0 * tcor / tres.clamp(min=1),
                     torch.full_like(tcor, float("nan")))
    m["task_phase_accuracy"] = tp.cpu().tolist()
    m["task0_phase1"] = tp[0, 0].item()
    m["task0_last_phase"] = tp[0, T - 1].item()

    if record:
        rec = r["record"]
        m["trace"] = {k: rec[k] for k in ("capacity", "acc_fast", "doubt", "error",
                                          "learn_mean", "pressure_mean", "grow", "prune")}
        m["growth_analysis"] = growth_analysis(rec)
    return m


# =====================================================================
# HAND-BUILT COMPARISON (no evolution)
# =====================================================================

def parse_run(spec: str) -> Tuple[str, int]:
    mode, _, units = spec.partition("-")
    if mode not in STRUCTURE_MODES or not units.isdigit():
        raise ValueError(f"bad config '{spec}', expected e.g. fixed-16 or full-16")
    return mode, int(units)


def hand_mode(args: argparse.Namespace, device: torch.device) -> None:
    base = config_from_args(args)
    validate_config(base)
    seeds = [int(s) for s in args.seeds.split(",")]
    specs = [s.strip() for s in args.configs.split(",") if s.strip()]

    sc0 = Scenarios(base, args.eval_lives, 900_000 + seeds[0], device)
    ceiling = ideal_learner_accuracy(base, sc0, device)

    print("=" * 110)
    print("HAND-BUILT GENOME — no evolution. Same lives for every configuration.")
    print(f"{base.max_tasks} tasks x {base.n_cues} cues x {base.n_actions} actions, "
          f"{base.steps} steps, k={base.k}, pool={base.pool}, faults p={base.fault_probability}, "
          f"{args.eval_lives} lives x seeds {seeds}")
    print(f"reward noise {base.reward_noise}, reversal {base.reversal} "
          f"(tasks < {base.reversal_tasks}), rehearsal {base.rehearsal_fraction}")
    print(f"chance {100.0 / base.n_actions:.1f}%  |  perfect-learner ceiling "
          f"{ceiling:.1f}% (seed {seeds[0]})")
    print("=" * 110)
    head = (f"{'config':18s} | {'acc':>11s} {'blind':>6s} {'margin':>7s} |"
            + "".join(f" {'P' + str(p + 1):>5s}" for p in range(base.max_tasks))
            + f" | {'t0 end':>6s} | {'avg u':>6s} {'final':>6s} {'grow':>5s}"
            f" | {'rev':>5s} {'repair':>6s}")
    print(head)

    def row(label: str, runs: List[Dict]) -> None:
        def mean(key: str) -> float:
            return float(np.nanmean([r[key] for r in runs]))

        acc = np.array([r["accuracy"] for r in runs])
        sd = acc.std(ddof=1) if len(acc) > 1 else 0.0
        rev = mean("reversed_accuracy")
        rev_s = f"{rev:4.1f}%" if rev == rev else "    -"
        print(f"{label:18s} | {acc.mean():5.1f}±{sd:4.1f}% {mean('cue_blind_phase'):5.1f}%"
              f" {acc.mean() - mean('cue_blind_phase'):+6.1f} |"
              + "".join(f" {mean(f'phase_{p+1}_accuracy'):4.1f}%"
                        for p in range(base.max_tasks))
              + f" | {mean('task0_last_phase'):5.1f}% | {mean('average_units'):6.1f}"
              f" {mean('final_units'):6.1f} {mean('grow_events'):5.1f} | {rev_s}"
              f" {mean('repair_events'):6.1f}")

    results: Dict[str, List[Dict]] = {}
    for spec in ["control:random", "control:no-plasticity"] + specs:
        runs = []
        for seed in seeds:
            cfg = Config(**asdict(base))
            cfg.seed = seed
            if spec.startswith("control:"):
                cfg.structure, cfg.initial_units = "fixed", 16
                cfg.random_policy = spec.endswith("random")
                cfg.plasticity = not spec.endswith("no-plasticity")
            else:
                cfg.structure, cfg.initial_units = parse_run(spec)
            cfg.min_units = min(cfg.min_units, cfg.initial_units)
            theta = initial_genome(cfg, device, seed)
            runs.append(evaluate(theta, cfg, device, seed=900_000 + seed,
                                 lives=args.eval_lives, record=not spec.startswith("control:")))
        results[spec] = runs
        row(spec.replace("control:", "ctrl:"), runs)

    print("-" * 110)
    print("margin = accuracy - cue-blind (best fixed action per phase, hindsight). "
          "rev = accuracy on remapped\npatterns after the reversal (--reversal). "
          "repair = slot resets per life.")

    growth = [s for s in specs if parse_run(s)[0] != "fixed"]
    if growth:
        print("\nGrowth by phase (mean units added per life):")
        for spec in growth:
            per = [np.mean([r[f"phase_{p+1}_grow"] for r in results[spec]])
                   for p in range(base.max_tasks)]
            print(f"  {spec:14s}" + "".join(f"  P{p+1} {v:5.2f}" for p, v in enumerate(per)))

    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        path = os.path.join(args.results_dir, "hand_comparison.json")
        with open(path, "w") as f:
            json.dump({k: [{kk: vv for kk, vv in r.items() if kk != "trace"} for r in v]
                       for k, v in results.items()}, f, indent=2)
        print(f"\nSaved: {path}")


# =====================================================================
# EVOLUTION
# =====================================================================

def print_progress(generation: int, m: Dict, cfg: Config, elapsed: float) -> None:
    print(f"gen {generation:4d} | acc {m['accuracy']:5.1f}% (cue-blind "
          f"{m['cue_blind_phase']:4.1f}%) | survive {m['survival']:5.1f}%"
          f" | reward {m['reward']:7.2f} | units {m['average_units']:5.1f}"
          f"->{m['final_units']:5.1f} | grow {m['grow_events']:5.1f}"
          f" prune {m['prune_events']:4.1f} | {elapsed:7.1f}s")
    text = "  phases:"
    for p in range(cfg.max_tasks):
        text += (f" P{p+1}={m[f'phase_{p+1}_accuracy']:.1f}%"
                 f"/{m[f'phase_{p+1}_units']:.1f}u")
    text += (f" | task0 P1 {m['task0_phase1']:.1f}% -> last "
             f"{m['task0_last_phase']:.1f}%")
    print(text)


def evolve(cfg: Config, device: torch.device) -> torch.Tensor:
    theta = initial_genome(cfg, device, cfg.seed)
    dimension = len(theta)
    print("=" * 78)
    print(f"DSM Benchmark B v4   run={cfg.run_name}   seed={cfg.seed}")
    print("=" * 78)
    print(f"Device             : {device}")
    print(f"Genome parameters  : {dimension}")
    print(f"Structure          : {cfg.structure}, start {cfg.initial_units} of "
          f"{cfg.pool} units, k={cfg.k}")
    print(f"Benchmark          : {cfg.max_tasks} tasks x {cfg.n_cues} cues x "
          f"{cfg.n_actions} actions, {cfg.steps} steps, chance "
          f"{100.0 / cfg.n_actions:.1f}%")
    print(f"ES                 : pop={cfg.population}, lives={cfg.lives}, "
          f"generations={cfg.generations}")
    print("=" * 78)

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

        eps = torch.randn(half, dimension, generator=gen, device=device)
        candidates = torch.cat([theta[None] + cfg.sigma * eps,
                                theta[None] - cfg.sigma * eps])
        scenarios = Scenarios(cfg, cfg.lives,
                              seed=100_000 + cfg.seed * 10_000 + generation, device=device)
        result = run_lives(candidates, scenarios, cfg, device)
        fitness = result["fitness"].mean(dim=1)
        if generation % 10 == 0:
            noise = result["fitness"].std(dim=1).mean()
            print(f"  ES: fitness mean {fitness.mean().item():7.2f} | std "
                  f"{fitness.std().item():6.3f} | best {fitness.max().item():7.2f}"
                  f" | per-life noise {noise.item():6.2f}")

        shaped = centered_ranks(fitness)
        grad = ((shaped[:half] - shaped[half:])[:, None] * eps).sum(dim=0) / (
            cfg.population * cfg.sigma)
        grad -= cfg.weight_decay * theta
        theta += optimizer.step(grad)
    return theta


# =====================================================================
# REPORTING
# =====================================================================

def print_final(m: Dict, cfg: Config) -> None:
    print("\n" + "=" * 78)
    print(f"FINAL TEST  run={cfg.run_name}  seed={cfg.seed}")
    print("=" * 78)
    for key in ("accuracy", "chance", "cue_blind_phase", "survival", "survival_fault",
                "reward", "average_units", "final_units", "grow_events", "prune_events",
                "repair_events", "reversed_accuracy"):
        print(f"{key:24s}: {m[key]:.3f}")
    margin = m["accuracy"] - m["cue_blind_phase"]
    print(f"\nAccuracy - cue-blind(phase) : {margin:+.2f} points "
          f"({'mapping learned' if margin > 0 else 'NOT beyond a cue-blind prior'})")

    print("\nPhases (accuracy / average units / units grown):")
    for p in range(cfg.max_tasks):
        print(f"  Phase {p+1}: {m[f'phase_{p+1}_accuracy']:6.2f}% / "
              f"{m[f'phase_{p+1}_units']:6.2f} / {m[f'phase_{p+1}_grow']:5.2f}")

    print("\nAccuracy per task (rows) per phase (cols):")
    print("        " + "".join(f"  P{p+1:<5d}" for p in range(cfg.max_tasks)))
    for task, row in enumerate(m["task_phase_accuracy"]):
        print(f"  task{task}" + "".join(f"  {v:5.1f}%" if v == v else "      - "
                                        for v in row))

    ga = m.get("growth_analysis") or {}
    if ga:
        print("\nGrowth analysis (after an error: grew vs held):")
        for key, value in ga.items():
            print(f"  {key:36s}: {value:+.4f}")


def print_trace(trace: Dict[str, List[float]]) -> None:
    print("\nDevelopment trace (alive mean; acc = recent-accuracy EMA)")
    print("step |  units | acc%  | doubt | error | learnab. | pressure | grow | prune")
    for t in range(len(trace["capacity"])):
        if t % 10 != 0:
            continue
        print(f"{t:4d} | {trace['capacity'][t]:6.2f} | {100 * trace['acc_fast'][t]:5.1f}"
              f" | {trace['doubt'][t]:5.2f} | {trace['error'][t]:5.3f}"
              f" | {trace['learn_mean'][t]:+8.4f} | {trace['pressure_mean'][t]:8.4f}"
              f" | {trace['grow'][t]:4.2f} | {trace['prune'][t]:5.2f}")


def save_results(path: str, m: Dict, cfg: Config) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {k: v for k, v in m.items() if k != "trace"}
    data["run_name"] = cfg.run_name
    data["config"] = asdict(cfg)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved results: {path}")


def summarize(results_dir: str) -> None:
    files = sorted(p for p in glob.glob(os.path.join(results_dir, "*.json"))
                   if not p.endswith("hand_comparison.json"))
    if not files:
        print(f"No results in {results_dir}")
        return
    groups: Dict[str, List[Dict]] = {}
    for path in files:
        with open(path) as f:
            d = json.load(f)
        groups.setdefault(d["run_name"], []).append(d)

    cols = [("accuracy", "acc"), ("cue_blind_phase", "cue-blind"),
            ("phase_4_accuracy", "P4"), ("task0_last_phase", "task0-last"),
            ("average_units", "avg units"), ("final_units", "final units"),
            ("grow_events", "grow")]
    print(f"{'run':18s} {'n':>2s}" + "".join(f" {label:>14s}" for _, label in cols))

    def order(name: str) -> Tuple[int, int]:
        mode, _, rest = name.partition("-")
        size = rest.split("-")[0]
        return (STRUCTURE_MODES.index(mode) if mode in STRUCTURE_MODES else 99,
                int(size) if size.isdigit() else 0)

    for name in sorted(groups, key=order):
        runs = groups[name]
        line = f"{name:18s} {len(runs):2d}"
        for key, _ in cols:
            vals = np.array([r.get(key, float("nan")) for r in runs], dtype=float)
            sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
            line += f" {np.nanmean(vals):7.2f}±{sd:5.2f}"
        print(line)


# =====================================================================
# CLI
# =====================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DSM Benchmark B v4")
    p.add_argument("--mode", choices=["train", "eval", "hand", "summarize"],
                   default="train")
    p.add_argument("--device", default="auto")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--results", default=None)
    p.add_argument("--results-dir", default="results_v4")
    p.add_argument("--eval-lives", type=int, default=1000)

    p.add_argument("--structure", choices=STRUCTURE_MODES, default="full")
    p.add_argument("--units", type=int, default=16, help="initial units")
    p.add_argument("--pool", type=int, default=128)
    p.add_argument("--min-units", type=int, default=8)
    p.add_argument("--k", type=int, default=1, help="active units per pattern")
    p.add_argument("--novelty", type=float, default=0.9)

    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--population", type=int, default=64)
    p.add_argument("--lives", type=int, default=32)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--max-tasks", type=int, default=4)
    p.add_argument("--n-cues", type=int, default=4)
    p.add_argument("--n-actions", type=int, default=4)
    p.add_argument("--fault-probability", type=float, default=0.65)
    p.add_argument("--rehearsal-fraction", type=float, default=0.30)
    p.add_argument("--reward-noise", type=float, default=0.0,
                   help="probability that feedback is flipped (true accuracy scored)")
    p.add_argument("--reversal", action="store_true",
                   help="remap the first --reversal-tasks tasks mid-life")
    p.add_argument("--reversal-tasks", type=int, default=2)
    p.add_argument("--init-repair-bias", type=float, default=-8.0)
    p.add_argument("--init-repair-slot", type=float, default=0.0)
    p.add_argument("--init-repair-wrong", type=float, default=0.0)
    p.add_argument("--init-gate-slot", type=float, default=0.0)

    p.add_argument("--init-eta", type=float, default=0.0)
    p.add_argument("--init-temperature", type=float, default=0.3)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--es-lr", type=float, default=0.02)
    p.add_argument("--complexity-cost", type=float, default=0.005)
    p.add_argument("--accuracy-weight", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--no-metacognition", action="store_true")
    p.add_argument("--random-policy", action="store_true")
    p.add_argument("--no-plasticity", action="store_true")

    p.add_argument("--seeds", default="0,1,2", help="hand mode: seeds")
    p.add_argument("--configs",
                   default="fixed-8,fixed-16,fixed-32,fixed-64,fixed-128,"
                           "random-16,imprint-16,full-16,nolearn-16,basic-16",
                   help="hand mode: configurations to compare")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        steps=args.steps, max_tasks=args.max_tasks, n_cues=args.n_cues,
        n_actions=args.n_actions, fault_probability=args.fault_probability,
        rehearsal_fraction=args.rehearsal_fraction,
        reward_noise=args.reward_noise, reversal=args.reversal,
        reversal_tasks=args.reversal_tasks,
        init_repair_bias=args.init_repair_bias, init_repair_slot=args.init_repair_slot,
        init_repair_wrong=args.init_repair_wrong, init_gate_slot=args.init_gate_slot,
        pool=args.pool, initial_units=args.units,
        min_units=min(args.min_units, args.units), k=args.k, novelty=args.novelty,
        structure=args.structure,
        random_policy=args.random_policy, plasticity=not args.no_plasticity,
        metacognition=not args.no_metacognition,
        init_eta=args.init_eta, init_temperature=args.init_temperature,
        generations=args.generations, population=args.population, lives=args.lives,
        sigma=args.sigma, es_lr=args.es_lr, complexity_cost=args.complexity_cost,
        accuracy_weight=args.accuracy_weight, seed=args.seed,
    )


def validate_config(cfg: Config) -> None:
    if cfg.steps % 2 != 0:
        raise ValueError("--steps must be even")
    if cfg.population < 2 or cfg.population % 2 != 0:
        raise ValueError("--population must be an even number >= 2")
    if not (cfg.min_units <= cfg.initial_units <= cfg.pool):
        raise ValueError("require min_units <= units <= pool")
    if not (1 <= cfg.k <= cfg.min_units):
        raise ValueError("require 1 <= k <= min_units")


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "summarize":
        summarize(args.results_dir)
        return

    device = choose_device(args.device)
    configure_torch(device)

    if args.mode == "hand":
        hand_mode(args, device)
        return

    if args.mode == "train":
        cfg = config_from_args(args)
        validate_config(cfg)
        ckpt = args.checkpoint or f"checkpoints_v4/{cfg.run_name}_seed{cfg.seed}.pt"
        theta = evolve(cfg, device)
        save_checkpoint(ckpt, theta, cfg)
        print(f"\nSaved checkpoint: {ckpt}")
    else:
        if args.checkpoint:
            theta, cfg = load_checkpoint(args.checkpoint, device)
            print(f"Loaded checkpoint: {args.checkpoint}")
        else:
            cfg = config_from_args(args)
            validate_config(cfg)
            theta = initial_genome(cfg, device, cfg.seed)
            print("No checkpoint: evaluating the hand-built genome")

    final = evaluate(theta, cfg, device, seed=900_000 + cfg.seed,
                     lives=args.eval_lives, record=True)
    print_final(final, cfg)
    print_trace(final["trace"])

    if args.mode == "train" or args.results:
        path = args.results or os.path.join(args.results_dir,
                                            f"{cfg.run_name}_seed{cfg.seed}.json")
        save_results(path, final, cfg)

    if device.type == "cuda":
        print(f"\nPeak CUDA memory: "
              f"{torch.cuda.max_memory_allocated(device) / 2**30:.2f} GB")


if __name__ == "__main__":
    main()

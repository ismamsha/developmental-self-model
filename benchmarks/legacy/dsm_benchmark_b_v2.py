"""
DSM Benchmark B v2 — Developmental Self Model
==========================================

One standalone file. No project imports.

What this tests
---------------
An organism starts with a small recurrent brain. During one lifetime it must:

1) remember delayed instructions,
2) learn random task mappings that are different in every life,
3) preserve old skills as new tasks are introduced,
4) adapt when hidden body dynamics change,
5) estimate its own prediction error ("doubt"),
6) estimate whether its current learning is making progress ("learnability"),
7) change synapses locally without backpropagation,
8) decide HOLD / GROW / PRUNE for its own neural capacity.

Across generations, Evolution Strategies (ES) optimize only the inherited genome:
local learning rules, neuromodulation, self-model rates, and developmental rules.

There is NO gradient descent during an organism's lifetime.

Requirements
------------
    pip install torch numpy

Quick CUDA smoke test
---------------------
    python dsm_benchmark_b.py --generations 2 --population 8 --lives 4 --steps 80

First useful A5000 run
----------------------
    python dsm_benchmark_b.py --generations 100 --population 64 --lives 24

Longer run
----------
    python dsm_benchmark_b.py --generations 400 --population 128 --lives 32

Evaluate a saved genome
-----------------------
    python dsm_benchmark_b.py --mode eval --checkpoint dsm_b.pt

Notes
-----
Default max_hidden=128 is intentional. The lifetime plastic recurrent matrix is
stored per organism-life pair. Increasing max_hidden raises VRAM approximately
quadratically. Test 128 first on the A5000 before trying 192 or 256.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# =====================================================================
# CONFIG
# =====================================================================

@dataclass
class Config:
    # Benchmark B
    steps: int = 240                 # must be even
    max_tasks: int = 4               # task families introduced over life
    n_cues: int = 4                  # cue identities per task
    n_actions: int = 4               # response choices
    fault_probability: float = 0.65

    # Brain
    max_hidden: int = 128
    initial_hidden: int = 16
    min_hidden: int = 8
    growth_key_dim: int = 16
    max_structural_change: int = 2

    # Development happens on a slower clock than neural activity.
    structural_warmup: int = 24
    structural_interval: int = 12

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

    # Reproducibility
    seed: int = 0


# =====================================================================
# DEVICE
# =====================================================================

def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def configure_torch(device: torch.device) -> None:
    if device.type == "cuda":
        # A5000 supports TF32 and benefits substantially from it.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


# =====================================================================
# BENCHMARK B SCENARIOS
# =====================================================================

class Scenarios:
    """
    All randomness for a group of lives is drawn before evaluation.
    Every candidate genome in one ES generation therefore meets the same lives.
    """

    def __init__(self, cfg: Config, n: int, seed: int, device: torch.device):
        assert cfg.steps % 2 == 0, "steps must be even: instruction/response pairs"

        self.n = n
        self.trials = cfg.steps // 2

        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        # Each lifetime has its own random task dictionary:
        # mapping[life, task, cue] -> correct action.
        self.mapping = torch.randint(
            0,
            cfg.n_actions,
            (n, cfg.max_tasks, cfg.n_cues),
            generator=gen,
            device=device,
        )

        # Trial random numbers. The number of available tasks grows by phase.
        self.task_u = torch.rand(
            n, self.trials, generator=gen, device=device
        )
        self.cue = torch.randint(
            0,
            cfg.n_cues,
            (n, self.trials),
            generator=gen,
            device=device,
        )

        # Hidden body faults.
        has_fault = torch.rand(n, generator=gen, device=device) < cfg.fault_probability

        # 0 = motor-energy fault, 1 = fragile-error fault
        kind = torch.randint(0, 2, (n,), generator=gen, device=device)
        self.fault_kind = torch.where(
            has_fault, kind, torch.full_like(kind, -1)
        )

        # Fault appears after the organism has already learned something.
        lo = max(20, cfg.steps // 4)
        hi = max(lo + 1, (3 * cfg.steps) // 4)
        onset = torch.randint(lo, hi, (n,), generator=gen, device=device)
        self.fault_onset = torch.where(
            has_fault, onset, torch.full_like(onset, 10**9)
        )

        # Which response action becomes expensive under a motor fault.
        self.fault_action = torch.randint(
            0, cfg.n_actions, (n,), generator=gen, device=device
        )

        # Common stochasticity.
        self.shock_u = torch.rand(
            n, cfg.steps, generator=gen, device=device
        )
        self.structure_u = torch.rand(
            n, cfg.steps, generator=gen, device=device
        )

        # Common random numbers for stochastic policy sampling.
        # Every ES candidate sees the same exploration draws.
        self.action_u = torch.rand(
            n, cfg.steps, generator=gen, device=device
        )

        self.birth_seed = seed + 917_381


# =====================================================================
# ENVIRONMENT
# =====================================================================

class BenchmarkB:
    """
    A life consists of instruction/response pairs.

    Instruction step:
        task ID + cue are visible.
        No task reward is given.
        The action is ignored.

    Response step:
        task ID + cue disappear.
        The recurrent brain must remember them.
        Correct action is defined by a random mapping unique to this life.

    Across life:
        phase 1 uses task 0
        phase 2 uses tasks 0..1
        phase 3 uses tasks 0..2
        phase 4 uses tasks 0..3

    This forces accumulation of skills rather than replacing one task with another.
    """

    def __init__(
        self,
        cfg: Config,
        scenarios: Scenarios,
        population: int,
        device: torch.device,
    ):
        self.cfg = cfg
        self.device = device
        self.population = population
        self.lives = scenarios.n
        self.batch = population * scenarios.n

        # Repeat world specification for each candidate genome.
        self.mapping = scenarios.mapping.repeat(population, 1, 1)
        self.task_u = scenarios.task_u.repeat(population, 1)
        self.cue_seq = scenarios.cue.repeat(population, 1)
        self.fault_kind = scenarios.fault_kind.repeat(population)
        self.fault_onset = scenarios.fault_onset.repeat(population)
        self.fault_action = scenarios.fault_action.repeat(population)
        self.shock_u = scenarios.shock_u.repeat(population, 1)
        self.structure_u = scenarios.structure_u.repeat(population, 1)
        self.action_u = scenarios.action_u.repeat(population, 1)

        self.energy = torch.ones(self.batch, device=device)
        self.damage = torch.zeros(self.batch, device=device)
        self.alive = torch.ones(self.batch, dtype=torch.bool, device=device)
        self.survived = torch.zeros(self.batch, dtype=torch.bool, device=device)

        self.current_task = torch.zeros(self.batch, dtype=torch.long, device=device)
        self.current_cue = torch.zeros(self.batch, dtype=torch.long, device=device)
        self.current_target = torch.zeros(self.batch, dtype=torch.long, device=device)

        self.last_task_reward = torch.zeros(self.batch, device=device)
        self.correct_total = torch.zeros(self.batch, device=device)
        self.response_total = torch.zeros(self.batch, device=device)

        self.phase_correct = torch.zeros(
            self.batch, cfg.max_tasks, device=device
        )
        self.phase_responses = torch.zeros(
            self.batch, cfg.max_tasks, device=device
        )

        self.step_number = 0

    @property
    def obs_dim(self) -> int:
        # energy, damage, time, instruction flag, previous task reward
        # + task one-hot + cue one-hot
        return 5 + self.cfg.max_tasks + self.cfg.n_cues

    def phase_index(self, t: int) -> int:
        # Equal-length phases.
        phase_len = max(2, self.cfg.steps // self.cfg.max_tasks)
        return min(self.cfg.max_tasks - 1, t // phase_len)

    def _trial_task(self, trial: int, phase: int) -> torch.Tensor:
        """
        Curriculum with rehearsal.

        Phase 0:
            only task 0

        Later phases:
            70% newest task
            30% spread across older tasks

        This gives a newly introduced task enough repetitions to be learnable
        while still testing retention of previous skills.
        """
        if phase == 0:
            return torch.zeros(
                self.batch, dtype=torch.long, device=self.device
            )

        u = self.task_u[:, trial]
        newest = torch.full(
            (self.batch,), phase, dtype=torch.long, device=self.device
        )

        # Remap the upper 30% of u onto old tasks [0, phase-1].
        old_u = torch.clamp((u - 0.70) / 0.30, 0.0, 0.999999)
        old_task = torch.clamp(
            (old_u * phase).long(), min=0, max=phase - 1
        )

        return torch.where(u < 0.70, newest, old_task)

    def observation(self) -> Tuple[torch.Tensor, bool, int]:
        """
        Returns:
            observation [batch, obs_dim]
            is_instruction
            phase
        """
        t = self.step_number
        is_instruction = (t % 2 == 0)
        trial = t // 2
        phase = self.phase_index(t)

        task_onehot = torch.zeros(
            self.batch, self.cfg.max_tasks, device=self.device
        )
        cue_onehot = torch.zeros(
            self.batch, self.cfg.n_cues, device=self.device
        )

        if is_instruction:
            self.current_task = self._trial_task(trial, phase)
            self.current_cue = self.cue_seq[:, trial]

            rows = torch.arange(self.batch, device=self.device)
            self.current_target = self.mapping[
                rows, self.current_task, self.current_cue
            ]

            task_onehot = F.one_hot(
                self.current_task, self.cfg.max_tasks
            ).float()
            cue_onehot = F.one_hot(
                self.current_cue, self.cfg.n_cues
            ).float()

        base = torch.stack(
            [
                self.energy,
                self.damage,
                torch.full(
                    (self.batch,),
                    t / max(1, self.cfg.steps - 1),
                    device=self.device,
                ),
                torch.full(
                    (self.batch,),
                    1.0 if is_instruction else 0.0,
                    device=self.device,
                ),
                self.last_task_reward,
            ],
            dim=1,
        )

        obs = torch.cat([base, task_onehot, cue_onehot], dim=1)
        return obs, is_instruction, phase

    def step(
        self, action: torch.Tensor, is_instruction: bool, phase: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            reward
            lived_before_step
            correct_mask (response steps only, otherwise zeros)
        """
        t = self.step_number
        live = self.alive.clone()

        reward = torch.zeros(self.batch, device=self.device)
        correct_mask = torch.zeros(self.batch, device=self.device)

        fault_active = (
            live
            & (self.fault_kind >= 0)
            & (t >= self.fault_onset)
        )

        motor_fault = fault_active & (self.fault_kind == 0)
        fragile_fault = fault_active & (self.fault_kind == 1)

        energy = self.energy.clone()
        damage = self.damage.clone()

        if is_instruction:
            # Small recovery. The response is intentionally delayed by one step.
            energy = torch.where(
                live,
                torch.clamp(energy + 0.006 * (1.0 - damage), max=1.0),
                energy,
            )
            self.last_task_reward = torch.where(
                live, torch.zeros_like(self.last_task_reward), self.last_task_reward
            )

        else:
            correct = live & (action == self.current_target)
            wrong = live & ~correct

            correct_mask = correct.float()
            self.correct_total += correct.float()
            self.response_total += live.float()

            self.phase_correct[:, phase] += correct.float()
            self.phase_responses[:, phase] += live.float()

            task_reward = torch.where(
                correct,
                torch.ones_like(reward),
                torch.full_like(reward, -0.20),
            )
            reward = torch.where(live, task_reward, reward)
            self.last_task_reward = torch.where(
                live, task_reward, self.last_task_reward
            )

            # Normal energetic cost of responding.
            action_scale = action.float() / max(1, self.cfg.n_actions - 1)
            cost = 0.004 + 0.004 * action_scale

            # One actuator becomes unexpectedly expensive after a hidden fault.
            expensive = motor_fault & (action == self.fault_action)
            cost = torch.where(expensive, cost * 3.0, cost)

            energy = torch.where(
                live, torch.clamp(energy - cost, min=0.0), energy
            )

            # Cognitive errors should teach before they kill.
            # Healthy organisms take no body damage from a wrong answer.
            # A fragile hidden fault makes repeated errors physically costly,
            # but slowly enough to preserve a learning window.
            damage_add = torch.where(
                fragile_fault,
                torch.full_like(damage, 0.006),
                torch.zeros_like(damage),
            )
            damage = torch.where(
                wrong, torch.clamp(damage + damage_add, max=1.0), damage
            )

            # Correct behavior lets the body recover slightly.
            damage = torch.where(
                correct, torch.clamp(damage - 0.002, min=0.0), damage
            )

        # Rare external shock. Fragility makes it more likely and larger.
        shock_p = torch.where(
            fragile_fault,
            torch.full_like(energy, 0.025),
            torch.full_like(energy, 0.005),
        )
        shocked = live & (self.shock_u[:, t] < shock_p)

        energy = torch.where(
            shocked, torch.clamp(energy - 0.04, min=0.0), energy
        )
        damage = torch.where(
            shocked,
            torch.clamp(
                damage + torch.where(
                    fragile_fault,
                    torch.full_like(damage, 0.040),
                    torch.full_like(damage, 0.015),
                ),
                max=1.0,
            ),
            damage,
        )

        died = live & ((energy <= 0.0) | (damage >= 1.0))
        finished = live & ~died & (t + 1 >= self.cfg.steps)

        reward = torch.where(
            died, reward - self.cfg.death_penalty, reward
        )
        reward = torch.where(
            finished, reward + self.cfg.survival_bonus, reward
        )

        self.energy = torch.where(live, energy, self.energy)
        self.damage = torch.where(live, damage, self.damage)

        self.survived |= finished
        self.alive = live & ~died & ~finished
        self.step_number += 1

        return reward, live, correct_mask


# =====================================================================
# GENOME
# =====================================================================

def genome_spec(cfg: Config, obs_dim: int) -> List[Tuple[str, Tuple[int, ...]]]:
    # External observation + 3 doubt components + learnability + bias.
    n_input = obs_dim + 3 + 1 + 1

    # Structure features:
    # doubt(3), |doubt|(3), error EMA, learnability, active fraction,
    # mean activity, mean utility, recent accuracy, progress, bias
    struct_features = 14

    return [
        # General local learning equations:
        # A*pre*post + B*pre + C*post + D
        ("rule_in", (4, cfg.max_hidden)),
        ("rule_rec", (4, cfg.max_hidden)),
        ("rule_out", (4, cfg.n_actions)),

        # Layer lifetime learning rates.
        ("eta", (3,)),

        # Neuromodulator sees:
        # doubt(3), |doubt|(3), reward, learnability, bias = 9
        ("modulator", (9,)),

        # Self-model / meta-model rates.
        ("eta_self", (1,)),
        ("doubt_gain", (3,)),
        ("doubt_memory", (1,)),
        ("eta_learnability", (1,)),

        # Development.
        ("structure", (3, struct_features)),  # HOLD / GROW / PRUNE
        ("growth_context", (struct_features, cfg.growth_key_dim)),
        ("growth_keys", (cfg.max_hidden, cfg.growth_key_dim)),
        ("structural_size", (2,)),

        # Scale of random birth wiring.
        ("birth_scale", (3,)),
    ]


def genome_size(cfg: Config, obs_dim: int) -> int:
    return sum(int(np.prod(shape)) for _, shape in genome_spec(cfg, obs_dim))


def initial_genome(
    cfg: Config, obs_dim: int, device: torch.device, seed: int
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    parts = []

    for name, shape in genome_spec(cfg, obs_dim):
        if name.startswith("rule"):
            value = 0.02 * rng.standard_normal(shape)

            # Give evolution a learnable starting basin instead of asking ES
            # to discover Hebbian credit assignment from near-zero noise.
            # Row 0 is the pre*post term.
            if name == "rule_in":
                value[0] += 0.20
            elif name == "rule_rec":
                value[0] += 0.20
            elif name == "rule_out":
                value[0] += 0.80

        elif name == "eta":
            value = np.full(shape, -1.5)

        elif name == "modulator":
            # Three-factor learning starts reward-gated:
            # positive reward strengthens recent eligibility,
            # negative reward weakens it.
            value = np.zeros(shape)
            value[6] = 1.5   # reward coefficient
            value[-1] = 0.0  # no unconditional rewiring bias

        elif name == "eta_self":
            value = np.array([-1.5])

        elif name == "doubt_gain":
            # Body deltas are small; reward error is naturally order-one.
            value = np.log([10.0, 30.0, 1.5])

        elif name == "doubt_memory":
            value = np.array([2.0])

        elif name == "eta_learnability":
            value = np.array([-2.0])

        elif name == "structure":
            value = np.zeros(shape)
            # Development begins conservative. Evolution can override this.
            value[0, -1] = 3.0
            value[1, -1] = -1.5
            value[2, -1] = -1.5

        elif name == "growth_context":
            value = 0.02 * rng.standard_normal(shape)

        elif name == "growth_keys":
            value = 0.05 * rng.standard_normal(shape)

        elif name in ("structural_size", "birth_scale"):
            value = np.zeros(shape)

        else:
            value = np.zeros(shape)

        parts.append(value.reshape(-1))

    arr = np.concatenate(parts).astype(np.float32)
    return torch.tensor(arr, device=device)


def unpack_genomes(
    theta: torch.Tensor, cfg: Config, obs_dim: int
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    pos = 0

    for name, shape in genome_spec(cfg, obs_dim):
        n = int(np.prod(shape))
        out[name] = theta[:, pos : pos + n].reshape(len(theta), *shape)
        pos += n

    return out


# =====================================================================
# LOCAL PLASTICITY
# =====================================================================

def local_rule(
    coef: torch.Tensor, pre: torch.Tensor, post: torch.Tensor
) -> torch.Tensor:
    """
    Evolved local rule per postsynaptic unit:

        dW = A*pre*post + B*pre + C*post + D
    """
    A = coef[:, 0, None, :]
    B = coef[:, 1, None, :]
    C = coef[:, 2, None, :]
    D = coef[:, 3, None, :]

    return (
        A * pre[:, :, None] * post[:, None, :]
        + B * pre[:, :, None]
        + C * post[:, None, :]
        + D
    )


# =====================================================================
# LIFETIME
# =====================================================================

@torch.no_grad()
def run_lives(
    theta: torch.Tensor,
    scenarios: Scenarios,
    cfg: Config,
    device: torch.device,
    record: bool = False,
) -> Dict[str, torch.Tensor]:
    population = theta.shape[0]
    lives = scenarios.n
    batch = population * lives

    # Temporary env tells us observation dimensionality.
    env = BenchmarkB(cfg, scenarios, population, device)
    obs_dim = env.obs_dim
    n_input = obs_dim + 3 + 1 + 1

    genome = unpack_genomes(theta, cfg, obs_dim)
    g = {
        name: value.repeat_interleave(lives, dim=0)
        for name, value in genome.items()
    }

    H = cfg.max_hidden
    A = cfg.n_actions

    # -------------------------------------------------------------
    # Random phenotype at birth.
    # Same random birth wiring for every genome on corresponding life.
    # -------------------------------------------------------------
    gen = torch.Generator(device=device)
    gen.manual_seed(scenarios.birth_seed)

    life_in = torch.randn(
        lives, n_input, H, generator=gen, device=device
    ) / math.sqrt(n_input)

    life_rec = torch.randn(
        lives, H, H, generator=gen, device=device
    ) / math.sqrt(H)

    life_out = torch.randn(
        lives, H, A, generator=gen, device=device
    ) / math.sqrt(H)

    birth_in = life_in.repeat(population, 1, 1)
    birth_rec = life_rec.repeat(population, 1, 1)
    birth_out = life_out.repeat(population, 1, 1)

    birth_in *= torch.exp(
        torch.clamp(g["birth_scale"][:, 0], -2.0, 2.0)
    )[:, None, None]
    birth_rec *= torch.exp(
        torch.clamp(g["birth_scale"][:, 1], -2.0, 2.0)
    )[:, None, None]
    birth_out *= torch.exp(
        torch.clamp(g["birth_scale"][:, 2], -2.0, 2.0)
    )[:, None, None]

    # -------------------------------------------------------------
    # Phenotype / fast state.
    # -------------------------------------------------------------
    hidden = torch.zeros(batch, H, device=device)

    active = torch.zeros(batch, H, dtype=torch.bool, device=device)
    active[:, : cfg.initial_hidden] = True

    utility = torch.zeros(batch, H, device=device)

    # Lifetime plastic components.
    plastic_in = torch.zeros_like(birth_in)
    plastic_rec = torch.zeros_like(birth_rec)
    plastic_out = torch.zeros_like(birth_out)

    # Eligibility traces bridge the delayed instruction -> response reward.
    # This is the missing temporal-credit mechanism in v1.
    eligibility_in = torch.zeros_like(birth_in)
    eligibility_rec = torch.zeros_like(birth_rec)
    eligibility_out = torch.zeros_like(birth_out)
    eligibility_decay = 0.85

    # Self-model predicts:
    #   delta energy
    #   delta damage
    #   immediate reward
    # for every action.
    SELF_OUT = 3
    feature_dim = H + n_input
    self_model = torch.zeros(
        batch, feature_dim, SELF_OUT * A, device=device
    )

    # Meta-self-model: "Will I reduce my own prediction error?"
    learnability_model = torch.zeros(batch, feature_dim, device=device)
    learnability_signal = torch.zeros(batch, device=device)

    doubt = torch.zeros(batch, SELF_OUT, device=device)
    error_ema = torch.zeros(batch, device=device)
    recent_accuracy = torch.full((batch,), 0.5, device=device)

    total_reward = torch.zeros(batch, device=device)
    total_capacity = torch.zeros(batch, device=device)
    structural_changes = torch.zeros(batch, device=device)

    # Per-phase capacity, aligned to response opportunities.
    phase_capacity_sum = torch.zeros(
        batch, cfg.max_tasks, device=device
    )
    phase_capacity_n = torch.zeros(
        batch, cfg.max_tasks, device=device
    )

    trace = None
    if record:
        trace = {
            "capacity": [],
            "doubt": [],
            "error": [],
            "learnability": [],
            "accuracy": [],
            "grow": [],
            "prune": [],
        }

    rows = torch.arange(batch, device=device)

    # -------------------------------------------------------------
    # Life loop.
    # -------------------------------------------------------------
    for t in range(cfg.steps):
        if not env.alive.any():
            break

        obs, is_instruction, phase = env.observation()

        active_f = active.float()
        active_count = active.sum(dim=1)
        active_fraction = active_count.float() / H

        # Core network receives its own metacognitive state.
        x = torch.cat(
            [
                obs,
                doubt,
                learnability_signal[:, None],
                torch.ones(batch, 1, device=device),
            ],
            dim=1,
        )

        previous_hidden = hidden.clone()

        # Recurrent fast state.
        hidden = torch.tanh(
            torch.einsum("bi,bij->bj", x, birth_in + plastic_in)
            + torch.einsum("bi,bij->bj", previous_hidden, birth_rec + plastic_rec)
        )
        hidden *= active_f

        # Utility is a slow estimate of actual use.
        utility = 0.985 * utility + 0.015 * hidden.abs()

        logits = torch.einsum("bi,bij->bj", hidden, birth_out + plastic_out)

        if is_instruction:
            # Action is ignored by the world on instruction steps.
            action = logits.argmax(dim=1)
        else:
            # The mapping is new in every lifetime, so deterministic argmax
            # can get trapped forever. Softmax sampling provides within-life
            # exploration while becoming nearly greedy as preferences sharpen.
            temperature = 0.75
            probs = torch.softmax(logits / temperature, dim=1)
            cdf = probs.cumsum(dim=1)
            u_action = env.action_u[:, t:t+1]
            action = (cdf < u_action).sum(dim=1).clamp(max=A - 1)

        feature = torch.cat([hidden, x], dim=1)

        # ---------------------------------------------------------
        # Self-prediction BEFORE reality.
        # ---------------------------------------------------------
        predictions = torch.einsum("bi,bij->bj", feature, self_model)

        predicted_outcome = torch.stack(
            [
                predictions[rows, action],
                predictions[rows, A + action],
                predictions[rows, 2 * A + action],
            ],
            dim=1,
        )

        predicted_learning_progress = torch.tanh(
            (feature * learnability_model).sum(dim=1)
        )

        energy_before = env.energy.clone()
        damage_before = env.damage.clone()

        reward, lived, correct_mask = env.step(action, is_instruction, phase)
        total_reward += reward

        # Recent behavioral competence is useful to the structural controller.
        if not is_instruction:
            recent_accuracy = (
                0.92 * recent_accuracy + 0.08 * correct_mask
            )

        real_outcome = torch.stack(
            [
                env.energy - energy_before,
                env.damage - damage_before,
                reward,
            ],
            dim=1,
        )

        error = (real_outcome - predicted_outcome) * lived[:, None].float()
        instant_error = error.abs().mean(dim=1)

        old_error_ema = error_ema.clone()
        error_ema = 0.90 * error_ema + 0.10 * instant_error
        actual_learning_progress = old_error_ema - error_ema

        # ---------------------------------------------------------
        # Self-model learns locally.
        # ---------------------------------------------------------
        eta_self = 0.20 * torch.sigmoid(g["eta_self"][:, 0])

        for channel in range(SELF_OUT):
            col = channel * A + action
            self_model[rows, :, col] += (
                eta_self[:, None]
                * feature
                * error[:, channel : channel + 1]
            )

        self_model.clamp_(-3.0, 3.0)

        # ---------------------------------------------------------
        # Learnability model learns:
        # "Was I right about whether my own error would improve?"
        # ---------------------------------------------------------
        eta_meta = 0.10 * torch.sigmoid(g["eta_learnability"][:, 0])

        # Scale small EMA progress into a useful bounded teaching signal.
        progress_target = torch.tanh(20.0 * actual_learning_progress)
        learnability_error = progress_target - predicted_learning_progress

        learnability_model += (
            eta_meta[:, None] * feature * learnability_error[:, None]
        )
        learnability_model.clamp_(-2.0, 2.0)

        # This is intentionally the pre-update prediction: what the organism
        # believed about its learning ability at this moment.
        learnability_signal = predicted_learning_progress

        # ---------------------------------------------------------
        # Doubt = persistent signed self-prediction surprise.
        # ---------------------------------------------------------
        memory = torch.sigmoid(g["doubt_memory"][:, 0])
        gain = torch.exp(torch.clamp(g["doubt_gain"], -3.0, 4.0))

        surprise = torch.clamp(error, -1.0, 1.0) * gain
        doubt = (
            memory[:, None] * doubt
            + (1.0 - memory)[:, None] * surprise
        )

        # ---------------------------------------------------------
        # Neuromodulator controls lifetime synaptic plasticity.
        # ---------------------------------------------------------
        mod_input = torch.cat(
            [
                doubt,
                doubt.abs(),
                reward[:, None],
                learnability_signal[:, None],
                torch.ones(batch, 1, device=device),
            ],
            dim=1,
        )

        modulator = torch.tanh(
            (mod_input * g["modulator"]).sum(dim=1)
        ) * lived.float()

        eta = 0.15 * torch.sigmoid(g["eta"])

        action_onehot = F.one_hot(action, A).float()

        d_in = local_rule(g["rule_in"], x, hidden)
        d_in *= active_f[:, None, :]

        d_rec = local_rule(g["rule_rec"], previous_hidden, hidden)
        d_rec *= active_f[:, :, None] * active_f[:, None, :]

        d_out = local_rule(g["rule_out"], hidden, action_onehot)
        d_out *= active_f[:, :, None]

        # Eligibility remembers instruction-time neural activity until
        # response feedback arrives.
        eligibility_in.mul_(eligibility_decay).add_(d_in)
        eligibility_rec.mul_(eligibility_decay).add_(d_rec)

        if is_instruction:
            # Output action is meaningless on instruction steps.
            eligibility_out.mul_(eligibility_decay)
        else:
            eligibility_out.mul_(eligibility_decay).add_(d_out)

        # Plasticity is applied only when response feedback exists.
        if not is_instruction:
            plastic_in += (
                eta[:, 0] * modulator
            )[:, None, None] * eligibility_in

            plastic_rec += (
                eta[:, 1] * modulator
            )[:, None, None] * eligibility_rec

            plastic_out += (
                eta[:, 2] * modulator
            )[:, None, None] * eligibility_out

            plastic_in.clamp_(-1.0, 1.0)
            plastic_rec.clamp_(-1.0, 1.0)
            plastic_out.clamp_(-1.0, 1.0)

        # ---------------------------------------------------------
        # Development controller.
        # ---------------------------------------------------------
        mean_activity = hidden.abs().sum(dim=1) / active_count.clamp(min=1)
        mean_utility = utility.sum(dim=1) / active_count.clamp(min=1)

        structure_input = torch.cat(
            [
                doubt,
                doubt.abs(),
                error_ema[:, None],
                learnability_signal[:, None],
                active_fraction[:, None],
                mean_activity[:, None],
                mean_utility[:, None],
                recent_accuracy[:, None],
                torch.full(
                    (batch, 1),
                    t / max(1, cfg.steps - 1),
                    device=device,
                ),
                torch.ones(batch, 1, device=device),
            ],
            dim=1,
        )

        structural_logits = torch.einsum(
            "bi,bji->bj", structure_input, g["structure"]
        )
        structural_prob = torch.softmax(structural_logits, dim=1)

        # Stochastic common-random-number decision improves ES smoothness.
        u = env.structure_u[:, t]
        p_hold = structural_prob[:, 0]
        p_grow = structural_prob[:, 1]

        structural_action = torch.zeros(
            batch, dtype=torch.long, device=device
        )

        structural_tick = (
            t >= cfg.structural_warmup
            and (t - cfg.structural_warmup) % cfg.structural_interval == 0
        )

        if structural_tick:
            structural_action = torch.where(
                u < p_hold,
                torch.zeros(batch, dtype=torch.long, device=device),
                torch.where(
                    u < (p_hold + p_grow),
                    torch.ones(batch, dtype=torch.long, device=device),
                    torch.full((batch,), 2, dtype=torch.long, device=device),
                ),
            )

        grow_mask = (structural_action == 1) & lived

        # Do not prune before the organism has had time to establish a skill.
        prune_mask = (
            (structural_action == 2)
            & lived
            & (t >= cfg.steps // 3)
        )

        growth_count = 1 + (
            torch.sigmoid(g["structural_size"][:, 0])
            * (cfg.max_structural_change - 1)
        ).long()

        prune_count = 1 + (
            torch.sigmoid(g["structural_size"][:, 1])
            * (cfg.max_structural_change - 1)
        ).long()

        # ---------------------------------------------------------
        # Neurogenesis.
        # ---------------------------------------------------------
        growth_context = torch.einsum(
            "bi,bij->bj", structure_input, g["growth_context"]
        )
        growth_scores = torch.einsum(
            "bik,bk->bi", g["growth_keys"], growth_context
        )

        growth_events = torch.zeros(batch, device=device)

        for slot in range(cfg.max_structural_change):
            can_grow = (
                grow_mask
                & (growth_count > slot)
                & (active.sum(dim=1) < H)
            )

            if not can_grow.any():
                continue

            scores = growth_scores.masked_fill(active, float("-inf"))
            candidate = scores.argmax(dim=1)

            rr = torch.where(can_grow)[0]
            cc = candidate[rr]

            active[rr, cc] = True
            hidden[rr, cc] = 0.0
            utility[rr, cc] = 0.0

            # Newborn connection plasticity starts clean; inherited random
            # birth wiring is already present and gives it something to test.
            plastic_in[rr, :, cc] = 0.0
            plastic_rec[rr, :, cc] = 0.0
            plastic_rec[rr, cc, :] = 0.0
            plastic_out[rr, cc, :] = 0.0

            growth_events[rr] += 1.0

        # ---------------------------------------------------------
        # Pruning: remove least-used currently active units.
        # ---------------------------------------------------------
        prune_events = torch.zeros(batch, device=device)

        for slot in range(cfg.max_structural_change):
            can_prune = (
                prune_mask
                & (prune_count > slot)
                & (active.sum(dim=1) > cfg.min_hidden)
            )

            if not can_prune.any():
                continue

            prune_scores = utility.masked_fill(~active, float("inf"))
            candidate = prune_scores.argmin(dim=1)

            rr = torch.where(can_prune)[0]
            cc = candidate[rr]

            active[rr, cc] = False
            hidden[rr, cc] = 0.0
            utility[rr, cc] = 0.0
            prune_events[rr] += 1.0

        structural_changes += growth_events + prune_events

        # Capacity accounting.
        now_capacity = active.sum(dim=1).float()
        total_capacity += now_capacity * lived.float()

        if not is_instruction:
            phase_capacity_sum[:, phase] += now_capacity * lived.float()
            phase_capacity_n[:, phase] += lived.float()

        if record and trace is not None:
            accuracy_now = (
                env.correct_total.sum()
                / env.response_total.sum().clamp(min=1)
            )
            trace["capacity"].append(now_capacity.mean().item())
            trace["doubt"].append(doubt.norm(dim=1).mean().item())
            trace["error"].append(error_ema.mean().item())
            trace["learnability"].append(learnability_signal.mean().item())
            trace["accuracy"].append(accuracy_now.item())
            trace["grow"].append(growth_events.mean().item())
            trace["prune"].append(prune_events.mean().item())

        # Explicitly drop huge temporary tensors before next step.
        del d_in, d_rec, d_out

    # -------------------------------------------------------------
    # Fitness.
    # -------------------------------------------------------------
    average_capacity = total_capacity / cfg.steps

    extra_fraction = torch.clamp(
        average_capacity - cfg.initial_hidden, min=0.0
    ) / cfg.max_hidden

    # Scale capacity pressure by number of response opportunities.
    complexity_penalty = (
        cfg.complexity_cost
        * extra_fraction
        * (cfg.steps / 2)
    )

    structural_penalty = (
        cfg.structural_change_cost * structural_changes
    )

    fitness = total_reward - complexity_penalty - structural_penalty

    phase_capacity = phase_capacity_sum / phase_capacity_n.clamp(min=1.0)

    result: Dict[str, torch.Tensor] = {
        "fitness": fitness.reshape(population, lives),
        "reward": total_reward.reshape(population, lives),
        "survived": env.survived.reshape(population, lives),
        "correct": env.correct_total.reshape(population, lives),
        "responses": env.response_total.reshape(population, lives),
        "capacity": average_capacity.reshape(population, lives),
        "final_capacity": active.sum(dim=1).float().reshape(population, lives),
        "phase_correct": env.phase_correct.reshape(
            population, lives, cfg.max_tasks
        ),
        "phase_responses": env.phase_responses.reshape(
            population, lives, cfg.max_tasks
        ),
        "phase_capacity": phase_capacity.reshape(
            population, lives, cfg.max_tasks
        ),
        "structural_changes": structural_changes.reshape(population, lives),
    }

    if record and trace is not None:
        # type: ignore[assignment]
        result["trace"] = trace  # type: ignore[assignment]

    return result


# =====================================================================
# ES OPTIMIZER
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

        m_hat = self.m / (1.0 - 0.9**self.t)
        v_hat = self.v / (1.0 - 0.999**self.t)

        return self.lr * m_hat / (torch.sqrt(v_hat) + 1e-8)


def centered_ranks(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x)
    ranks = torch.empty_like(x)
    ranks[order] = torch.arange(
        len(x), device=x.device, dtype=x.dtype
    )

    if len(x) > 1:
        ranks /= (len(x) - 1)

    return ranks - 0.5


# =====================================================================
# EVALUATION
# =====================================================================

@torch.no_grad()
def evaluate(
    theta: torch.Tensor,
    cfg: Config,
    device: torch.device,
    seed: int,
    lives: int = 512,
    record: bool = False,
) -> Dict:
    scenarios = Scenarios(cfg, lives, seed, device)
    result = run_lives(theta[None], scenarios, cfg, device, record=record)

    survived = result["survived"][0]
    correct = result["correct"][0]
    responses = result["responses"][0]
    reward = result["reward"][0]
    capacity = result["capacity"][0]
    final_capacity = result["final_capacity"][0]
    changes = result["structural_changes"][0]

    fault = scenarios.fault_kind >= 0
    healthy = ~fault

    accuracy = 100.0 * correct.sum() / responses.sum().clamp(min=1)

    metrics: Dict = {
        "survival": 100.0 * survived.float().mean().item(),
        "survival_healthy": 100.0
        * survived[healthy].float().mean().item()
        if healthy.any()
        else float("nan"),
        "survival_fault": 100.0
        * survived[fault].float().mean().item()
        if fault.any()
        else float("nan"),
        "accuracy": accuracy.item(),
        "reward": reward.mean().item(),
        "average_neurons": capacity.mean().item(),
        "final_neurons": final_capacity.mean().item(),
        "structural_changes": changes.mean().item(),
    }

    phase_correct = result["phase_correct"][0]
    phase_responses = result["phase_responses"][0]
    phase_capacity = result["phase_capacity"][0]

    for p in range(cfg.max_tasks):
        acc = 100.0 * phase_correct[:, p].sum() / phase_responses[
            :, p
        ].sum().clamp(min=1)

        metrics[f"phase_{p+1}_accuracy"] = acc.item()
        metrics[f"phase_{p+1}_neurons"] = phase_capacity[:, p].mean().item()

    if record:
        metrics["trace"] = result["trace"]

    return metrics


# =====================================================================
# EVOLUTION
# =====================================================================

def evolve(cfg: Config, device: torch.device) -> torch.Tensor:
    # Build one tiny scenario only to get observation size.
    dummy = Scenarios(cfg, 1, cfg.seed + 123, device)
    dummy_env = BenchmarkB(cfg, dummy, 1, device)
    obs_dim = dummy_env.obs_dim

    theta = initial_genome(cfg, obs_dim, device, cfg.seed)
    dimension = len(theta)

    print("=" * 78)
    print("DSM Benchmark B")
    print("=" * 78)
    print(f"Device             : {device}")
    if device.type == "cuda":
        print(f"GPU                : {torch.cuda.get_device_name(device)}")
        print(
            f"VRAM               : "
            f"{torch.cuda.get_device_properties(device).total_memory / 2**30:.1f} GB"
        )
    print(f"Genome parameters  : {dimension:,}")
    print(
        f"Phenotype capacity : {cfg.initial_hidden} -> {cfg.max_hidden} neurons"
    )
    print(
        f"Benchmark          : {cfg.max_tasks} accumulating tasks, "
        f"{cfg.n_cues} cues, delayed response"
    )
    print(
        f"ES                 : pop={cfg.population}, lives={cfg.lives}, "
        f"generations={cfg.generations}"
    )
    print("=" * 78)

    if cfg.population % 2 != 0:
        raise ValueError("population must be even for antithetic ES")

    optimizer = ESAdam(dimension, cfg.es_lr, device)

    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed + 8_888)

    half = cfg.population // 2
    start = time.time()

    for generation in range(cfg.generations + 1):
        if generation % 10 == 0 or generation == cfg.generations:
            metrics = evaluate(
                theta,
                cfg,
                device,
                seed=50_000 + cfg.seed,
                lives=256,
                record=False,
            )

            elapsed = time.time() - start

            print(
                f"gen {generation:4d}"
                f" | acc {metrics['accuracy']:5.1f}%"
                f" | survive {metrics['survival']:5.1f}%"
                f" | fault {metrics['survival_fault']:5.1f}%"
                f" | reward {metrics['reward']:7.2f}"
                f" | neurons {metrics['average_neurons']:5.1f}"
                f"->{metrics['final_neurons']:5.1f}"
                f" | changes {metrics['structural_changes']:5.1f}"
                f" | {elapsed:7.1f}s"
            )

            phase_text = "  phases:"
            for p in range(cfg.max_tasks):
                phase_text += (
                    f" P{p+1}={metrics[f'phase_{p+1}_accuracy']:.1f}%"
                    f"/{metrics[f'phase_{p+1}_neurons']:.1f}n"
                )
            print(phase_text)

        if generation == cfg.generations:
            break

        epsilon = torch.randn(
            half, dimension, generator=gen, device=device
        )

        candidates = torch.cat(
            [
                theta[None] + cfg.sigma * epsilon,
                theta[None] - cfg.sigma * epsilon,
            ],
            dim=0,
        )

        scenarios = Scenarios(
            cfg,
            cfg.lives,
            seed=100_000 + cfg.seed * 10_000 + generation,
            device=device,
        )

        result = run_lives(candidates, scenarios, cfg, device)
        fitness = result["fitness"].mean(dim=1)

        shaped = centered_ranks(fitness)
        antithetic_difference = shaped[:half] - shaped[half:]

        grad = (
            antithetic_difference[:, None] * epsilon
        ).sum(dim=0) / (cfg.population * cfg.sigma)

        grad -= cfg.weight_decay * theta
        theta += optimizer.step(grad)

        if device.type == "cuda" and generation % 25 == 0:
            torch.cuda.empty_cache()

    return theta


# =====================================================================
# REPORTING
# =====================================================================

def print_final(metrics: Dict, cfg: Config) -> None:
    print("\n" + "=" * 78)
    print("FINAL TEST")
    print("=" * 78)

    for key in [
        "accuracy",
        "survival",
        "survival_healthy",
        "survival_fault",
        "reward",
        "average_neurons",
        "final_neurons",
        "structural_changes",
    ]:
        print(f"{key:24s}: {metrics[key]:.3f}")

    print("\nPhases (accuracy / average active neurons):")
    for p in range(cfg.max_tasks):
        print(
            f"  Phase {p+1}: "
            f"{metrics[f'phase_{p+1}_accuracy']:6.2f}%"
            f" / {metrics[f'phase_{p+1}_neurons']:6.2f}"
        )


def print_trace(trace: Dict[str, List[float]]) -> None:
    print("\nDevelopment trace (population mean)")
    print(
        "step | neurons | acc%  | doubt | error | learnability | grow | prune"
    )

    n = len(trace["capacity"])

    for t in range(n):
        grow = trace["grow"][t]
        prune = trace["prune"][t]

        if t % 10 != 0 and grow == 0.0 and prune == 0.0:
            continue

        print(
            f"{t:4d}"
            f" | {trace['capacity'][t]:7.2f}"
            f" | {100*trace['accuracy'][t]:5.1f}"
            f" | {trace['doubt'][t]:5.2f}"
            f" | {trace['error'][t]:5.3f}"
            f" | {trace['learnability'][t]:+11.4f}"
            f" | {grow:4.2f}"
            f" | {prune:5.2f}"
        )


# =====================================================================
# SAVE / LOAD
# =====================================================================

def save_checkpoint(path: str, theta: torch.Tensor, cfg: Config) -> None:
    torch.save(
        {
            "genome": theta.detach().cpu(),
            "config": asdict(cfg),
        },
        path,
    )


def load_checkpoint(
    path: str, device: torch.device
) -> Tuple[torch.Tensor, Config]:
    data = torch.load(path, map_location="cpu")
    cfg = Config(**data["config"])
    theta = data["genome"].to(device)
    return theta, cfg


# =====================================================================
# CLI
# =====================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Standalone Developmental Self Model Benchmark B"
    )

    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--device", default="auto")
    p.add_argument("--checkpoint", default="dsm_b.pt")

    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--population", type=int, default=64)
    p.add_argument("--lives", type=int, default=24)
    p.add_argument("--steps", type=int, default=240)

    p.add_argument("--max-hidden", type=int, default=128)
    p.add_argument("--initial-hidden", type=int, default=16)
    p.add_argument("--min-hidden", type=int, default=8)

    p.add_argument("--max-tasks", type=int, default=4)
    p.add_argument("--n-cues", type=int, default=4)
    p.add_argument("--n-actions", type=int, default=4)

    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--es-lr", type=float, default=0.02)
    p.add_argument("--complexity-cost", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=0)

    return p


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        steps=args.steps,
        max_tasks=args.max_tasks,
        n_cues=args.n_cues,
        n_actions=args.n_actions,
        max_hidden=args.max_hidden,
        initial_hidden=args.initial_hidden,
        min_hidden=args.min_hidden,
        generations=args.generations,
        population=args.population,
        lives=args.lives,
        sigma=args.sigma,
        es_lr=args.es_lr,
        complexity_cost=args.complexity_cost,
        seed=args.seed,
    )


def validate_config(cfg: Config) -> None:
    if cfg.steps % 2 != 0:
        raise ValueError("--steps must be even")
    if cfg.population < 2 or cfg.population % 2 != 0:
        raise ValueError("--population must be an even number >= 2")
    if not (cfg.min_hidden <= cfg.initial_hidden <= cfg.max_hidden):
        raise ValueError(
            "Require min_hidden <= initial_hidden <= max_hidden"
        )
    if cfg.max_tasks < 1:
        raise ValueError("max_tasks must be >= 1")
    if cfg.n_actions < 2:
        raise ValueError("n_actions must be >= 2")


def main() -> None:
    args = build_parser().parse_args()
    device = choose_device(args.device)
    configure_torch(device)

    if args.mode == "train":
        cfg = config_from_args(args)
        validate_config(cfg)

        theta = evolve(cfg, device)
        save_checkpoint(args.checkpoint, theta, cfg)
        print(f"\nSaved checkpoint: {args.checkpoint}")

    else:
        theta, cfg = load_checkpoint(args.checkpoint, device)
        validate_config(cfg)
        print(f"Loaded checkpoint: {args.checkpoint}")
        print(f"Device: {device}")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(device)}")

    final = evaluate(
        theta,
        cfg,
        device,
        seed=900_000 + cfg.seed,
        lives=1000,
        record=True,
    )

    print_final(final, cfg)
    print_trace(final["trace"])

    if device.type == "cuda":
        print(
            f"\nPeak CUDA memory: "
            f"{torch.cuda.max_memory_allocated(device) / 2**30:.2f} GB"
        )


if __name__ == "__main__":
    main()

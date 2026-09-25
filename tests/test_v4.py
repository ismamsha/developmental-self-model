"""Behavioural tests for DSM Benchmark B v4 (CPU, hand-built genome)."""

import math

import pytest
import torch

import dsm_benchmark_b_v4 as v4

LIVES = 200
SEED = 900_000


def evaluate(cpu, **overrides):
    cfg = v4.Config(fault_probability=0.0, **overrides)
    theta = v4.initial_genome(cfg, cpu, seed=0)
    return v4.evaluate(theta, cfg, cpu, seed=SEED, lives=LIVES)


@pytest.mark.parametrize("control", [{"random_policy": True}, {"plasticity": False}])
def test_controls_score_chance(cpu, control):
    m = evaluate(cpu, structure="fixed", **control)
    assert abs(m["accuracy"] - 25.0) < 3.0


def test_learner_beats_cue_blind_baseline(cpu):
    m = evaluate(cpu, structure="imprint")
    assert m["accuracy"] > m["cue_blind_phase"] + 3.0


def test_more_capacity_helps(cpu):
    small = evaluate(cpu, structure="fixed", initial_units=8, min_units=8)
    large = evaluate(cpu, structure="fixed", initial_units=64)
    assert large["accuracy"] > small["accuracy"] + 5.0


def test_imprint_growth_is_bounded_by_patterns(cpu):
    m = evaluate(cpu, structure="imprint")
    # One slot per task x cue pattern at most (4 x 4 = 16).
    assert 5.0 < m["grow_events"] <= 16.0
    assert m["final_units"] <= 16 + 16


def test_fixed_structure_never_changes(cpu):
    m = evaluate(cpu, structure="fixed")
    assert m["grow_events"] == 0.0 and m["prune_events"] == 0.0
    assert m["average_units"] == pytest.approx(16.0)


def test_ceiling_above_learner(cpu):
    cfg = v4.Config(fault_probability=0.0)
    scenarios = v4.Scenarios(cfg, LIVES, SEED, cpu)
    ceiling = v4.ideal_learner_accuracy(cfg, scenarios, cpu)
    m = evaluate(cpu, structure="imprint")
    assert m["accuracy"] < ceiling <= 100.0


def test_reversal_remaps_only_selected_tasks(cpu):
    cfg = v4.Config(reversal=True, reversal_tasks=2)
    sc = v4.Scenarios(cfg, 50, SEED, cpu)
    assert (sc.mapping_rev[:, :2] != sc.mapping[:, :2]).all()
    assert (sc.mapping_rev[:, 2:] == sc.mapping[:, 2:]).all()


def test_reward_noise_flips_perceived_feedback(cpu):
    cfg = v4.Config(fault_probability=0.0, reward_noise=0.3, steps=200)
    sc = v4.Scenarios(cfg, 400, SEED, cpu)
    env = v4.BenchmarkB(cfg, sc, 1, cpu)
    flips = total = 0
    for _ in range(cfg.steps):
        obs, is_instruction, phase = env.observation()
        env.step(torch.zeros(env.batch, dtype=torch.long), is_instruction, phase)
        if not is_instruction:
            flips += (env.last_task_reward != env.last_true_task_reward).sum().item()
            total += env.batch
    assert abs(flips / total - 0.3) < 0.03


def test_self_model_signals_help_after_reversal(cpu):
    common = dict(steps=480, rehearsal_fraction=0.5, reversal=True,
                  init_repair_slot=20.0, init_repair_wrong=2.0, init_gate_slot=6.0)
    full = evaluate(cpu, structure="full", **common)
    basic = evaluate(cpu, structure="basic", **common)
    assert full["reversed_accuracy"] > basic["reversed_accuracy"] + 5.0


def test_evolution_smoke(cpu):
    cfg = v4.Config(steps=40, generations=1, population=4, lives=2,
                    fault_probability=0.0)
    theta = v4.evolve(cfg, cpu)
    assert torch.isfinite(theta).all()
    assert len(theta) == sum(math.prod(s) for _, s in v4.genome_spec(cfg))

"""Smoke tests for DSM Benchmark B v3 (dense recurrent baseline)."""

import torch

import dsm_benchmark_b_v3 as v3


def test_random_policy_scores_chance(cpu):
    cfg = v3.Config(fault_probability=0.0, structure="fixed", random_policy=True)
    dummy = v3.Scenarios(cfg, 1, 0, cpu)
    obs_dim = v3.BenchmarkB(cfg, dummy, 1, cpu).obs_dim
    theta = v3.initial_genome(cfg, obs_dim, cpu, seed=0)
    m = v3.evaluate(theta, cfg, cpu, seed=900_000, lives=100)
    assert abs(m["accuracy"] - 25.0) < 4.0


def test_growth_run_is_finite(cpu):
    cfg = v3.Config(fault_probability=0.0, structure="full", steps=80)
    dummy = v3.Scenarios(cfg, 1, 0, cpu)
    obs_dim = v3.BenchmarkB(cfg, dummy, 1, cpu).obs_dim
    theta = v3.initial_genome(cfg, obs_dim, cpu, seed=0)
    r = v3.run_lives(theta[None], v3.Scenarios(cfg, 8, 1, cpu), cfg, cpu)
    assert torch.isfinite(r["fitness"]).all()

"""
Prospective validation of the behavioural learnability signal (DSM V4 code).

Question: does the organism's online learnability estimate L_t predict the
FUTURE change in its own task accuracy, beyond what trivially available
retrospective statistics already predict?

Pre-specified analysis (written before any result of this script was seen)
--------------------------------------------------------------------------
For every life and every response step t (alive at t and for W further
responses):
    past_acc_t   = mean correct over responses t-W+1 .. t
    future_acc_t = mean correct over responses t+1 .. t+W
    dA_t         = future_acc_t - past_acc_t                      (target)
Predictors:
    L_t          = the organism's learnability signal (learned online)
    trend_t      = tanh(10 (acc_fast - acc_slow)), the retrospective trend
                   that L_t is trained to predict (EMAs as in the code)
    err_t        = 1 - acc_slow (regression-to-the-mean / "room to improve")
Metrics per seed:
    r_raw      = Pearson corr(L_t, dA_t)
    r_partial  = corr of L_t with dA_t after linearly regressing both on
                 [trend_t, err_t, past_acc_t, phase one-hot, 1]
    r_trend    = Pearson corr(trend_t, dA_t)             (baseline)
    also r_raw restricted to steps whose windows do not cross a phase boundary.
Success criteria (fixed in advance):
    S1 "tracks progress retrospectively": corr(L_t, trend_t) > 0.3 in all seeds.
    S2 "predicts future progress":        r_raw > 0 with bootstrap 95% CI
                                          excluding 0 in all seeds.
    S3 "adds information beyond trivial statistics": r_partial > 0 with
                                          bootstrap 95% CI excluding 0 in all seeds.
Only S3 would support the claim that the signal is a genuine prospective
self-estimate rather than a smoothed restatement of recent accuracy.
Bootstrap resamples LIVES (not steps) to respect within-life dependence.
"""

import argparse
import importlib.util
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# In the developmental-self-model repository the benchmark lives in benchmarks/;
# in the MSRA repository a verbatim snapshot sits next to this script.
_CANDIDATES = [os.path.join(HERE, "..", "..", "benchmarks", "dsm_benchmark_b_v4.py"),
               os.path.join(HERE, "dsm_benchmark_b_v4_snapshot.py")]
spec = importlib.util.spec_from_file_location(
    "v4", next(p for p in _CANDIDATES if os.path.exists(p)))
v4 = importlib.util.module_from_spec(spec)
import sys
sys.modules["v4"] = v4
spec.loader.exec_module(v4)


def ema(x, a):
    out = np.empty_like(x)
    m = np.full(x.shape[1], 0.25)
    for t in range(x.shape[0]):
        m = np.where(np.isnan(x[t]), m, a * m + (1 - a) * np.nan_to_num(x[t]))
        out[t] = m
    return out


def residualize(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def analyse(correct, learn, phase_of_resp, W, n_boot, rng):
    T, L = correct.shape
    fast, slow = ema(correct, 0.80), ema(correct, 0.95)
    trend = np.tanh(10 * (fast - slow))
    rows = []  # (life, L, trend, err, past, dA, phase, crosses)
    for t in range(W - 1, T - W):
        past = correct[t - W + 1:t + 1]
        fut = correct[t + 1:t + 1 + W]
        ok = ~np.isnan(past).any(0) & ~np.isnan(fut).any(0)
        if not ok.any():
            continue
        idx = np.where(ok)[0]
        crosses = phase_of_resp[t - W + 1] != phase_of_resp[t + W]
        for i in idx:
            rows.append((i, learn[t, i], trend[t, i], 1 - slow[t, i], past[:, i].mean(),
                         fut[:, i].mean() - past[:, i].mean(), phase_of_resp[t], crosses))
    a = np.array(rows, dtype=float)
    life, Lt, tr, er, pa, dA, ph, cr = a.T
    X = np.column_stack([tr, er, pa, np.eye(int(ph.max()) + 1)[ph.astype(int)]])

    def metrics(mask):
        Xm = X[mask]
        return dict(r_raw=corr(Lt[mask], dA[mask]),
                    r_trend=corr(tr[mask], dA[mask]),
                    r_L_trend=corr(Lt[mask], tr[mask]),
                    r_partial=corr(residualize(Lt[mask], Xm), residualize(dA[mask], Xm)),
                    r_raw_within_phase=corr(Lt[mask & (cr == 0)], dA[mask & (cr == 0)]))

    point = metrics(np.ones(len(a), bool))
    lives = np.unique(life)
    by_life = {l: np.where(life == l)[0] for l in lives}
    boots = {k: [] for k in point}
    for _ in range(n_boot):
        pick = rng.choice(lives, len(lives), replace=True)
        m = np.zeros(len(a), bool)
        idx = np.concatenate([by_life[l] for l in pick])
        # duplicates matter for a bootstrap: use weights via repeated indices
        Xb = X[idx]
        boots["r_raw"].append(corr(Lt[idx], dA[idx]))
        boots["r_trend"].append(corr(tr[idx], dA[idx]))
        boots["r_L_trend"].append(corr(Lt[idx], tr[idx]))
        boots["r_partial"].append(corr(residualize(Lt[idx], Xb), residualize(dA[idx], Xb)))
        wp = idx[cr[idx] == 0]
        boots["r_raw_within_phase"].append(corr(Lt[wp], dA[wp]))
        del m
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
          for k, v in boots.items()}

    # Event study: learnability around phase transitions (response index).
    trans = np.where(np.diff(phase_of_resp) != 0)[0] + 1
    ev = {}
    for off in range(-10, 21, 2):
        vals = [np.nanmean(np.where(np.isnan(correct[t + off]), np.nan, learn[t + off]))
                for t in trans if 0 <= t + off < T]
        ev[off] = float(np.nanmean(vals))
    return point, ci, len(a), ev


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="fixed-16,full-16")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--lives", type=int, default=400)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--boot", type=int, default=500)
    p.add_argument("--out", default=os.path.join(HERE, "..", "results", "learnability_validation.json"))
    a = p.parse_args()
    torch.set_num_threads(1)
    dev = torch.device("cpu")
    results = {}
    for spec_s in a.configs.split(","):
        for seed in [int(s) for s in a.seeds.split(",")]:
            cfg = v4.Config()
            cfg.seed = seed
            cfg.structure, cfg.initial_units = v4.parse_run(spec_s)
            cfg.min_units = min(cfg.min_units, cfg.initial_units)
            theta = v4.initial_genome(cfg, dev, seed)
            sc = v4.Scenarios(cfg, a.lives, 900_000 + seed, dev)
            r = v4.run_lives(theta[None], sc, cfg, dev, record=True)
            rec = r["record"]
            correct = torch.stack(rec["correct"]).numpy()
            learn = torch.stack(rec["learnability"]).numpy()
            resp_steps = np.arange(1, cfg.steps, 2)[:correct.shape[0]]
            phase_of_resp = np.array([v4.BenchmarkB.phase_index(
                type("E", (), {"cfg": cfg})(), int(s)) for s in resp_steps])
            rng = np.random.default_rng(seed)
            point, ci, n, ev = analyse(correct, learn, phase_of_resp, a.window, a.boot, rng)
            key = f"{spec_s}/s{seed}"
            results[key] = dict(point=point, ci=ci, n_samples=n, event_study=ev,
                                accuracy=float(np.nanmean(correct)))
            print(key, "n=", n, {k: round(v, 3) for k, v in point.items()},
                  {k: [round(x, 3) for x in v] for k, v in ci.items()}, flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)

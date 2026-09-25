"""
Linear toy model of the reference gap (Minimal Reference Condition).

    true performance   T(theta) = v . theta,  |v| = 1   (latent)
    internal evaluator E_t(theta) = w_t . theta
    policy update      theta <- theta + eta_theta * w_hat + sigma * xi      (ascend own evaluator)
    evaluator update   w <- w + eta_w * theta_hat                          (self-endorsement)
    MRC                w <- Proj_{C(R, eps)}(w),  C = {w : angle(w, R) <= eps},
                       with a fixed reference R at angle delta from v.

Outputs results/mrc_toy.json and figures/mrc_toy.pdf.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_R = os.path.join(HERE, "..", "results")
OUT_F = os.path.join(HERE, "..", "figures")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"

N, STEPS, ETA_T, ETA_W, SIGMA = 20, 400, 0.05, 0.05, 0.10
KAPPA = 9.0


def unit(x):
    return x / (np.linalg.norm(x) + 1e-12)


def project_cone(w, R, eps):
    """Project w onto {u : angle(u, R) <= eps} (R unit), keeping |w|."""
    n = np.linalg.norm(w)
    u = unit(w)
    c = np.clip(u @ R, -1, 1)
    if np.arccos(c) <= eps:
        return w
    perp = u - c * R
    perp = unit(perp) if np.linalg.norm(perp) > 1e-9 else unit(np.random.randn(len(R)))
    return n * (np.cos(eps) * R + np.sin(eps) * perp)


def reference_at(v, delta, rng):
    p = unit(rng.standard_normal(len(v)) - (rng.standard_normal(len(v)) @ v) * 0 * v)
    p = unit(p - (p @ v) * v)
    return np.cos(delta) * v + np.sin(delta) * p


def run(seed, anchored, eps=np.radians(15), delta=np.radians(20), kappa=0.0):
    """kappa > 0: a 'cheap' direction h (orthogonal to v) along which the
    policy changes kappa times more easily -- a Goodhart/proxy direction."""
    rng = np.random.default_rng(seed)
    v = unit(rng.standard_normal(N))
    R = reference_at(v, delta, rng)
    h = unit(rng.standard_normal(N)); h = unit(h - (h @ v) * v)
    w = v.copy()                       # evaluator starts perfectly aligned
    theta = 0.1 * rng.standard_normal(N)
    gap, T, E = [], [], []
    for _ in range(STEPS):
        step = unit(w) + kappa * (h @ unit(w)) * h
        theta = theta + ETA_T * step + SIGMA * rng.standard_normal(N) / np.sqrt(N)
        w = w + ETA_W * unit(theta)
        if anchored:
            w = project_cone(w, R, eps)
        gap.append(np.degrees(np.arccos(np.clip(unit(w) @ v, -1, 1))))
        T.append(float(v @ theta))
        E.append(float(unit(w) @ theta))
    return np.array(gap), np.array(T), np.array(E)


if __name__ == "__main__":
    os.makedirs(OUT_R, exist_ok=True)
    os.makedirs(OUT_F, exist_ok=True)
    seeds = range(200)
    res = {}
    for name, anch, kap in [("reference_free", False, 0.0), ("mrc", True, 0.0),
                            ("reference_free_cheap", False, KAPPA), ("mrc_cheap", True, KAPPA)]:
        G, T, E = zip(*[run(s, anch, kappa=kap) for s in seeds])
        G, T, E = np.array(G), np.array(T), np.array(E)
        E_mono = float(np.mean(np.all(np.diff(E, axis=1) > -0.05, axis=1)))
        res[name] = dict(final_gap_mean=float(G[:, -1].mean()), final_gap_sd=float(G[:, -1].std()),
                         final_T_mean=float(T[:, -1].mean()), final_T_sd=float(T[:, -1].std()),
                         final_E_mean=float(E[:, -1].mean()),
                         frac_T_below_start=float(np.mean(T[:, -1] < T[:, 0])),
                         max_gap=float(G.max()), G=G, T=T, E=E)
    # Sweep of anchor quality: final T vs eps + delta, with the bound eta*cos(eps+delta)*t
    sweep = []
    for tot in [10, 30, 50, 70, 85, 95, 110]:
        d = np.radians(min(tot, 60) if tot <= 60 else tot - 10)
        e = np.radians(tot) - d
        Ts = [run(s, True, eps=e, delta=d, kappa=KAPPA)[1][-1] for s in range(60)]
        sweep.append(dict(eps_plus_delta=tot, final_T_mean=float(np.mean(Ts)),
                          final_T_sd=float(np.std(Ts)),
                          bound=float(ETA_T * np.cos(np.radians(tot)) * STEPS)))

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    t = np.arange(1, STEPS + 1)
    for sfx, a_gap, a_perf, tag in [("", 0, 1, "isotropic"), ("_cheap", 0, 2, "cheap direction")]:
        for name, col, lab in [("reference_free", ORANGE, "reference-free"), ("mrc", BLUE, "MRC anchored")]:
            G, T, E = res[name + sfx]["G"], res[name + sfx]["T"], res[name + sfx]["E"]
            ax[a_gap].plot(t, G.mean(0), color=col, lw=2, ls="-" if sfx == "" else ":",
                           label=f"{lab}, {tag}")
            ax[a_perf].plot(t, T.mean(0), color=col, lw=2, label=f"true $T$, {lab}")
            ax[a_perf].plot(t, E.mean(0), color=col, lw=2, ls="--", label=f"internal $E$, {lab}")
        ax[a_perf].set(xlabel="self-modification step", ylabel="performance",
                       title=f"({'b' if sfx == '' else 'c'}) Internal vs true: {tag}")
    ax[0].axhline(35, color=MUTED, lw=1, ls=":")
    ax[0].text(STEPS, 37, r"bound $\epsilon+\delta=35^\circ$", ha="right", color=MUTED, fontsize=8)
    ax[0].set(xlabel="self-modification step", ylabel=r"reference gap $\Delta_t$ (deg)",
              title="(a) Evaluator drift")
    xs = [s["eps_plus_delta"] for s in sweep]
    ax[3].errorbar(xs, [s["final_T_mean"] for s in sweep], yerr=[s["final_T_sd"] for s in sweep],
                   color=BLUE, marker="o", ms=6, lw=2, capsize=3, label="simulated (anchored)")
    ax[3].plot(xs, [s["bound"] for s in sweep], color=AQUA, lw=2, ls="--",
               label=r"$\eta\,t\cos(\epsilon+\delta)$")
    ax[3].axhline(0, color=MUTED, lw=1)
    ax[3].set(xlabel=r"$\epsilon+\delta$ (deg)", ylabel=r"final $T$", title="(d) Anchor quality (cheap dir.)")
    for a in ax:
        a.grid(alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
        a.legend(fontsize=7, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_F, "mrc_toy.pdf"))
    for k in res:
        for x in ("G", "T", "E"):
            res[k].pop(x)
    res["sweep"] = sweep
    res["params"] = dict(N=N, steps=STEPS, eta_theta=ETA_T, eta_w=ETA_W, sigma=SIGMA,
                         eps_deg=15, delta_deg=20, kappa=KAPPA, seeds=len(seeds))
    json.dump(res, open(os.path.join(OUT_R, "mrc_toy.json"), "w"), indent=2)
    print(json.dumps(res, indent=1))

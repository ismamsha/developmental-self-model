"""Figure for DSM V4 re-runs and the learnability validation (from results/*.json)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MUTED = "#52514e"

std = json.load(open(os.path.join(R, "v4_standard", "hand_comparison.json")))
meta = json.load(open(os.path.join(R, "v4_meta", "hand_comparison.json")))
lv = json.load(open(os.path.join(R, "learnability_validation.json")))


def m(runs, k):
    v = np.array([r[k] for r in runs], float)
    return v.mean(), v.std(ddof=1)


fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))

# (a) capacity scaling and growth type
a = ax[0]
fx = [f"fixed-{n}" for n in (8, 16, 32, 64, 128)]
xs = [m(std[k], "average_units")[0] for k in fx]
ys = [m(std[k], "accuracy")[0] for k in fx]
es = [m(std[k], "accuracy")[1] for k in fx]
a.errorbar(xs, ys, yerr=es, color=BLUE, marker="o", ms=6, lw=2, capsize=3, label="fixed capacity")
for k, col, mk, lab in [("random-16", ORANGE, "s", "random-wired growth"),
                        ("imprint-16", AQUA, "D", "imprint growth"),
                        ("basic-16", YELLOW, "^", "imprint growth, no self-model inputs")]:
    x, _ = m(std[k], "average_units")
    y, e = m(std[k], "accuracy")
    a.errorbar([x], [y], yerr=[e], color=col, marker=mk, ms=8, lw=0, elinewidth=2, capsize=3, label=lab)
blind = m(std["fixed-16"], "cue_blind_phase")[0]
a.axhline(blind, color=MUTED, ls="--", lw=1)
a.text(8.3, blind + 0.4, "cue-blind (hindsight, per phase)", ha="left", fontsize=7, color=MUTED)
a.set_xscale("log", base=2)
a.set_xticks([8, 16, 32, 64, 128])
a.set_xticklabels(["8", "16", "32", "64", "128"])
a.set(xlabel="average active units per life (log scale)", ylabel="accuracy (%)",
      title="(a) Capacity vs. addressable capacity")
a.set_ylim(44, 61)

# (b) metacognition under ambiguity
b = ax[1]
cfgs = ["fixed-16", "fixed-128", "full-16", "nolearn-16", "basic-16"]
labels = ["fixed-16", "fixed-128", "full", "no learn-\nability", "no self-\nmodel"]
x = np.arange(len(cfgs))
w = 0.38
for off, key, col, lab in [(-w / 2, "accuracy", BLUE, "all patterns"),
                           (w / 2, "reversed_accuracy", ORANGE, "remapped patterns")]:
    mu = [m(meta[c], key)[0] for c in cfgs]
    sd = [m(meta[c], key)[1] for c in cfgs]
    b.bar(x + off, mu, w - 0.04, yerr=sd, color=col, capsize=2, label=lab)
b.axhline(m(meta["fixed-16"], "cue_blind_phase")[0], color=MUTED, ls="--", lw=1)
b.axhline(25, color=MUTED, ls=":", lw=1)
b.set_xticks(x)
b.set_xticklabels(labels, fontsize=8)
b.set(ylabel="accuracy (%)", title="(b) Reward noise 0.15 + reversal", ylim=(0, 72))

# (c) learnability: retrospective vs prospective
c = ax[2]
keys = [("r_L_trend", "corr(L, past trend)"), ("r_raw", "corr(L, future ΔA)"),
        ("r_partial", "partial corr(L, future ΔA)")]
for j, (cfgname, col) in enumerate([("fixed-16", BLUE), ("full-16", AQUA)]):
    for i, (k, _) in enumerate(keys):
        vals = [lv[f"{cfgname}/s{s}"]["point"][k] for s in range(3)]
        xx = i + (j - 0.5) * 0.25
        c.scatter([xx] * 3, vals, color=col, s=30, zorder=3,
                  label=cfgname if i == 0 else None)
c.axhline(0, color=MUTED, lw=1)
c.set_xticks(range(len(keys)))
c.set_xticklabels([k[1] for k in keys], fontsize=8)
c.set(ylabel="Pearson r (each dot = one seed)", title="(c) Learnability: tracks, does not forecast")

for a_ in ax:
    a_.grid(alpha=0.25)
    a_.spines[["top", "right"]].set_visible(False)
ax[0].legend(fontsize=7, frameon=False, loc="lower right")
ax[1].legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
ax[2].legend(fontsize=7, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "..", "figures", "v4_results.pdf"))
print("ok")

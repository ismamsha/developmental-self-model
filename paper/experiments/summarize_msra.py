"""Summarize results/raw/*.json from msra_experiments.py into
results/msra_audit.json and tables/msra_audit.tex."""
import glob
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
T = os.path.join(HERE, "..", "tables")

ORDER = [("msra_full", "Full agent"),
         ("msra_no_doubt", "\\quad doubt clamped to 0"),
         ("msra_no_shield", "\\quad without rule layer"),
         ("shield_only", "Rule layer alone"),
         ("learned_only", "No rules, no coded economics"),
         ("learned_only_no_doubt", "\\quad doubt clamped to 0"),
         ("proactive_base", "Base imagination agent"),
         ("actor_critic", "Actor-critic acting"),
         ("self_graded_full", "Full, self-graded doubt"),
         ("self_graded_learned_only", "No rules, self-graded doubt"),
         ("random", "Random policy"),
         ("always_rest", "Always rest")]
DIFFS = ["normal", "hard", "expert"]


def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    # average ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    ranks = (sums / cnt)[inv]
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fmt(m, s, d=1):
    if np.isnan(m):
        return "--"
    return f"{m:.{d}f}" + (f"$\\pm${s:.{d}f}" if s == s and s > 0 else "")


if __name__ == "__main__":
    data = defaultdict(list)
    for f in glob.glob(os.path.join(R, "raw", "*.json")):
        d = json.load(open(f))
        data[(d["variant"], d["difficulty"])].append(d)
    out = {}
    for (v, diff), runs in sorted(data.items()):
        per = []
        for d in runs:
            eps = d["episodes"]
            surv = np.mean([e["survived"] for e in eps]) * 100
            last = np.mean([e["survived"] for e in eps[-50:]]) * 100
            rew = np.mean([e["reward"] for e in eps[-50:]])
            caut = np.mean([e["caution_pct"] for e in eps[-50:]])
            caut_all = np.mean([e["caution_pct"] for e in eps])
            acts = np.sum([e["actions"] for e in eps[-50:]], axis=0)
            work = acts[2:].sum() / max(1, acts.sum()) * 100
            st = d["steps"]
            sh = np.array([s["shocked"] for s in st])
            eu = np.array([s["err_used"] for s in st])
            ea = np.array([s["err_anchored"] for s in st])
            per.append(dict(survival_all=surv, survival_last50=last, reward_last50=rew,
                            caution_last50=caut, caution_all=caut_all, work_pct_last50=work,
                            auroc_err_used=auroc(eu[sh == 1], eu[sh == 0]) if len(st) else float("nan"),
                            auroc_err_anchored=auroc(ea[sh == 1], ea[sh == 0]) if len(st) else float("nan"),
                            nan_events=d.get("nan_events", 0), seed=d["seed"]))
        agg = {}
        for k in per[0]:
            vals = np.array([p[k] for p in per], float)
            agg[k] = dict(mean=float(np.nanmean(vals)), sd=float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
                          per_seed=[float(x) for x in vals])
        out[f"{v}/{diff}"] = agg
    json.dump(out, open(os.path.join(R, "msra_audit.json"), "w"), indent=1)

    os.makedirs(T, exist_ok=True)
    L = []
    L.append("\\begin{table}[t]\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{3pt}")
    L.append("\\caption{Re-execution of the original agent (\\texttt{main.py}) with ablations. Survival (\\%) over the last 50 of "
             "100 online training episodes, mean$\\pm$SD over 3 seeds. "
             "Reward, caution (\\% of steps in caution mode), work (\\% of work actions) and AUROC are for hard difficulty; "
             "AUROC measures how well the prediction error the agent \\emph{uses} separates shock steps from other steps (0.5 = no information). Source: \\texttt{paper/results/msra\\_audit.json}.}")
    L.append("\\label{tab:msra-audit}")
    L.append("\\begin{tabular}{lccccccc}\n\\toprule")
    L.append("Variant & Normal & Hard & Expert & Reward & Caution & Work \\% & AUROC\\\\\n\\midrule")
    for v, name in ORDER:
        row = [name]
        for diff in DIFFS:
            a = out.get(f"{v}/{diff}")
            row.append(fmt(a["survival_last50"]["mean"], a["survival_last50"]["sd"]) if a else "n/a")
        h = out.get(f"{v}/hard")
        row.append(fmt(h["reward_last50"]["mean"], h["reward_last50"]["sd"]) if h else "n/a")
        if h:
            row.append(fmt(h["caution_last50"]["mean"], h["caution_last50"]["sd"]) if v.startswith(("msra", "learned", "self_graded")) and "no_doubt" not in v else "--")
            row.append(fmt(h["work_pct_last50"]["mean"], h["work_pct_last50"]["sd"], 0))
            row.append(f"{h['auroc_err_used']['mean']:.2f}" if v not in ("random", "always_rest", "actor_critic") else "--")
        else:
            row += ["n/a"] * 3
        L.append(" & ".join(row) + "\\\\")
        if v in ("shield_only", "learned_only_no_doubt", "actor_critic", "self_graded_learned_only"):
            L.append("\\addlinespace[2pt]")
    L.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
    open(os.path.join(T, "msra_audit.tex"), "w").write("\n".join(L) + "\n")
    for k, a in sorted(out.items()):
        print(f"{k:38s} surv_all {a['survival_all']['mean']:6.1f} last50 {a['survival_last50']['mean']:6.1f}±{a['survival_last50']['sd']:4.1f}"
              f" rew {a['reward_last50']['mean']:6.1f} caut {a['caution_last50']['mean']:5.1f} (all {a['caution_all']['mean']:5.1f})"
              f" work {a['work_pct_last50']['mean']:5.1f} auc_used {a['auroc_err_used']['mean']:.2f} auc_anch {a['auroc_err_anchored']['mean']:.2f}"
              f" nan {a['nan_events']['mean']:.0f}")

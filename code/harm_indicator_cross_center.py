"""Label-free harm indicator across centers.

For every (center, seed, adapter) run, among patches the frozen model gets
right (initially correct), the score 1 - frozen softmax confidence is compared
with the indicator "the adapter turned this patch wrong" (C->W).  Reported per
run, then averaged over seeds:

* AUROC of the score against C->W,
* capture@10%: share of all C->W patches inside the 10% least-confident
  target patches (frozen confidence, label-free),
* bottom-decile / top-decile C->W rate: C->W patches in the least / most
  confident decile of all target patches, divided by the decile size,
* conditional prevalence: C->W share of all initially-correct patches.

Run:  python code/harm_indicator_cross_center.py
      CG_TTA_OUT=/tmp/out python code/harm_indicator_cross_center.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.dirname(HERE)
OUT = os.environ.get("CG_TTA_OUT", os.path.join(REL, "tables"))
os.makedirs(OUT, exist_ok=True)

ADAPTERS = ["tent", "flip", "eata", "shot"]
SEEDS = {"center2": (1, 2, 3, 4, 5), "center0": (1, 2, 3, 4, 5)}


def load(center, seed, name):
    p = os.path.join(REL, "predictions", center, f"seed{seed}", f"{name}.npz")
    z = np.load(p)
    return z["probabilities"].astype(np.float64), z["predictions"], z["labels"]


def auroc(score, positive):
    """Rank-sum AUROC with average ranks for ties."""
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    r = np.empty(len(s))
    r[order] = ranks
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((r[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def indicator(center, seed, adapter):
    prob0, pred0, lab = load(center, seed, "no")
    _, pred1, lab1 = load(center, seed, adapter)
    assert np.array_equal(lab, lab1)
    conf = prob0.max(1)
    correct0 = pred0 == lab
    harm_all = correct0 & (pred1 != lab)          # C->W, indexed over all patches
    # AUROC and prevalence are conditional on the frozen model being right.
    s_c = 1.0 - conf[correct0]
    h_c = harm_all[correct0]
    # Review budget: the 10% least-confident of ALL target patches (label-free).
    n = len(lab)
    k = int(np.ceil(0.10 * n))
    order = np.argsort(conf, kind="mergesort")    # least confident first
    low, high = order[:k], order[-k:]
    return dict(
        auroc=auroc(s_c, h_c),
        capture_at_10pct=float(harm_all[low].sum() / max(1, harm_all.sum())),
        bottom_decile_harm_rate=float(harm_all[low].sum() / k),
        top_decile_harm_rate=float(harm_all[high].sum() / k),
        prevalence_cond=float(h_c.mean()),
        n_harm=int(harm_all.sum()), n_correct0=int(correct0.sum()))


if __name__ == "__main__":
    summary = {"provenance": "label-free harm indicator, cross-center generalization",
               "indicator": "1 - pre-adaptation softmax confidence for C->W among initially-correct",
               "centers": {}}
    per_run = []
    for center, seeds in SEEDS.items():
        summary["centers"][center] = {}
        for ad in ADAPTERS:
            rows = []
            for seed in seeds:
                r = indicator(center, seed, ad)
                r.update(center=center, seed=seed, method=ad)
                rows.append(r)
                per_run.append(r)
            agg = {f"{k}_mean": float(np.mean([x[k] for x in rows]))
                   for k in ("auroc", "capture_at_10pct", "bottom_decile_harm_rate",
                             "top_decile_harm_rate", "prevalence_cond")}
            agg["n_runs"] = len(rows)
            name = "eata" if ad == "eata" else ("flip_tta" if ad == "flip" else ad)
            summary["centers"][center][name] = agg
    with open(os.path.join(OUT, "harm_indicator_cross_center.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    import csv
    with open(os.path.join(OUT, "harm_indicator_per_run.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_run[0].keys()))
        w.writeheader()
        w.writerows(per_run)
    for center in summary["centers"]:
        for m, v in summary["centers"][center].items():
            print(f"{center:8s} {m:9s} runs={v['n_runs']} auroc={v['auroc_mean']:.3f} "
                  f"cap10={v['capture_at_10pct_mean']:.3f} bot={v['bottom_decile_harm_rate_mean']*100:.2f}% "
                  f"top={v['top_decile_harm_rate_mean']:.2e} prev={v['prevalence_cond_mean']*100:.2f}%")

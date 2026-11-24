#!/usr/bin/env python
"""Summarize the whole-slide-held-out budget experiment (job budget_slideout).

Reads  <root>/budget_slideout/seed*/fold*/<k>pct/subset*/tent_summary.json
Writes <out>/budget_slideout_per_run.csv, _per_seed_fold.csv, _summary.csv and
       <out>/tab_budget_slideout.tex

Design: Center-2 slides are split into two fixed five-slide folds.  For fold F the
adapter sees k% of the *other* five slides and is scored once on the five unseen
slides of F.  The two folds cover every Center-2 slide exactly once, so the
"pooled" rows score all 85,054 patches, each from a model that never saw its slide.

Averaging order: subset seeds inside (source seed, fold, budget), then pooled over
folds (patch-weighted accuracy delta, summed help/harm), then mean and sd over
source seeds.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from postprocess_colab import read_json, save_tex  # same folder

ORDER = {"1pct": 1, "5pct": 5, "25pct": 25, "100pct": 100}


def load_runs(root):
    base = os.path.join(root, "budget_slideout")
    rows = []
    for p in glob.glob(os.path.join(base, "seed*", "fold*", "*pct", "subset*",
                                    "tent_summary.json")):
        met = read_json(p)
        if met:
            rows.append(met)
    return pd.DataFrame(rows)


def summarize(df, out):
    df = df.copy()
    df["delta_acc_pp"] = 100.0 * (df["accuracy"] - df["frozen_eval_acc"])
    df.to_csv(os.path.join(out, "budget_slideout_per_run.csv"), index=False)

    # subset seeds -> one value per (seed, fold, budget)
    sf = df.groupby(["budget", "seed", "fold"]).agg(
        n_eval=("n_eval", "first"), n_adapt=("n_adapt", "mean"),
        frozen=("frozen_eval_acc", "first"), accuracy=("accuracy", "mean"),
        delta_acc_pp=("delta_acc_pp", "mean"), help=("help", "mean"),
        harm=("harm", "mean"), net=("net", "mean"),
        n_subsets=("subset_seed", "nunique")).reset_index()
    sf.to_csv(os.path.join(out, "budget_slideout_per_seed_fold.csv"), index=False)

    # pooled over the two folds (each slide scored once)
    pooled = []
    for (b, s), g in sf.groupby(["budget", "seed"]):
        if set(g["fold"]) != {"A", "B"}:
            continue  # incomplete pair: do not pool
        w = g["n_eval"].to_numpy(float)
        pooled.append(dict(
            budget=b, seed=s, fold="pooled", n_eval=int(w.sum()),
            n_adapt=float(g["n_adapt"].mean()),
            frozen=float((g["frozen"] * w).sum() / w.sum()),
            accuracy=float((g["accuracy"] * w).sum() / w.sum()),
            delta_acc_pp=float((g["delta_acc_pp"] * w).sum() / w.sum()),
            help=float(g["help"].sum()), harm=float(g["harm"].sum()),
            net=float(g["net"].sum()), n_subsets=int(g["n_subsets"].min())))
    allr = pd.concat([sf, pd.DataFrame(pooled)], ignore_index=True)
    allr.to_csv(os.path.join(out, "budget_slideout_per_seed.csv"), index=False)

    summ = allr.groupby(["fold", "budget"]).agg(
        n_seeds=("seed", "nunique"), n_eval=("n_eval", "first"),
        n_adapt=("n_adapt", "mean"), frozen=("frozen", "mean"),
        accuracy=("accuracy", "mean"), delta_acc_pp=("delta_acc_pp", "mean"),
        delta_acc_pp_sd=("delta_acc_pp", "std"), help=("help", "mean"),
        harm=("harm", "mean"), net=("net", "mean"),
        seeds_positive=("delta_acc_pp", lambda x: int((x > 0).sum())),
        n_subsets=("n_subsets", "min")).reset_index()
    summ["_o"] = summ["budget"].map(ORDER).fillna(999)
    fo = {"A": 0, "B": 1, "pooled": 2}
    summ["_f"] = summ["fold"].map(fo)
    summ = summ.sort_values(["_f", "_o"]).drop(columns=["_o", "_f"])
    summ.to_csv(os.path.join(out, "budget_slideout_summary.csv"), index=False)
    return summ


def write_tex(summ, out):
    body = []
    names = {"A": "Fold A (slides 20, 21, 24, 27, 28 unseen)",
             "B": "Fold B (slides 22, 23, 25, 26, 29 unseen)",
             "pooled": "Both folds (all ten slides unseen once)"}
    for fold in ("A", "B", "pooled"):
        g = summ[summ["fold"] == fold]
        if g.empty:
            continue
        body.append(r"\multicolumn{8}{@{}l}{\textit{" + names[fold] + r"}} \\")
        body.append(r"0\% (frozen) & 0 & " + f"{g.frozen.iloc[0]:.4f}"
                    + r" & -- & -- & -- & -- & -- \\")
        for _, r in g.iterrows():
            label = r.budget.replace("pct", r"\%")
            n_adapt = f"{int(round(r.n_adapt)):,}".replace(",", "{,}")
            sd = ("" if np.isnan(r.delta_acc_pp_sd)
                  else r"\pm" + f"{r.delta_acc_pp_sd:.2f}")
            body.append(f"{label} & {n_adapt} & {r.accuracy:.4f} & "
                        + f"${r.delta_acc_pp:+.2f}{sd}$ & "
                        + f"{r.help:.0f} & {r.harm:.0f} & {r.net:+.0f} & "
                        + f"{int(r.seeds_positive)}/{int(r.n_seeds)}" + r" \\")
        body.append(r"\midrule")
    body = body[:-1]
    tex = (r"\begin{tabular}{@{}lrccrrrc@{}}" "\n" r"\toprule" "\n"
           r"Budget & Adapt $n$ & Acc. & $\Delta$Acc (pp) & Help & Harm & Net & Seeds up \\"
           "\n" r"\midrule" "\n" + "\n".join(body) + "\n" r"\bottomrule" "\n"
           r"\end{tabular}" "\n")
    save_tex(os.path.join(out, "tab_budget_slideout.tex"), tex)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="folder that contains budget_slideout/ (17_Revision_Colab)")
    ap.add_argument("--out", default=None, help="default: <root>/tables")
    args = ap.parse_args()
    out = args.out or os.path.join(args.root, "tables")
    os.makedirs(out, exist_ok=True)
    df = load_runs(args.root)
    if df.empty:
        raise SystemExit("no tent_summary.json under budget_slideout/")
    shas = df.groupby("fold")["eval_sha256"].nunique()
    if (shas > 1).any():
        print(f"WARNING: eval set drift inside a fold: {shas.to_dict()}")
    summ = summarize(df, out)
    write_tex(summ, out)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

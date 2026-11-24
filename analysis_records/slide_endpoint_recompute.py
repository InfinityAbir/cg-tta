"""Recompute slide-level endpoints and add inference the revision needs.

All inputs are saved predictions; no model is run.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"E:\Temp\Next Research\09_Temporary")
from revision_analysis import seed_frame, add_manifest, derive  # noqa: E402

OUT = Path(os.environ.get("CG_TTA_PAPER_DIR", r"E:\Temp\Next Research\paper")) / "tables"
RNG = np.random.default_rng(20261025)


def per_slide(df, pred_col):
    rows = []
    for slide, g in df.groupby("slide"):
        y = g["label"].to_numpy()
        p = g[pred_col].to_numpy()
        tp = int(((p == 1) & (y == 1)).sum())
        tn = int(((p == 0) & (y == 0)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        rows.append({
            "slide": int(slide), "n": len(g),
            "n_tumor": int((y == 1).sum()), "n_normal": int((y == 0).sum()),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "ba": 0.5 * (sens + spec),
            "acc": (tp + tn) / len(g),
            "harm": int(((g["pred_frozen"].to_numpy() == y) & (p != y)).sum()),
            "help": int(((g["pred_frozen"].to_numpy() != y) & (p == y)).sum()),
        })
    return pd.DataFrame(rows)


def paired_ci(values, B=5000, rng=None):
    rng = rng or np.random.default_rng(7)
    values = np.asarray(values, dtype=float)
    n = len(values)
    draws = rng.integers(0, n, size=(B, n))
    means = values[draws].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    endpoint_rows = []
    per_slide_all = []
    cluster_rows = []
    loso_rows = []
    for seed in range(1, 6):
        df = add_manifest(seed_frame(seed))
        df = derive(df)
        s_frozen = per_slide(df, "pred_frozen")
        s_tent = per_slide(df, "pred_tent")
        s_gate = per_slide(df, "pred_gate")
        s_frozen["seed"] = seed
        s_tent["seed"] = seed
        s_gate["seed"] = seed
        per_slide_all += [s_frozen.assign(method="frozen"),
                          s_tent.assign(method="tent"),
                          s_gate.assign(method="gate")]

        d_ba_gate = (s_gate["ba"] - s_frozen["ba"]).to_numpy()
        d_ba_gate_tent = (s_gate["ba"] - s_tent["ba"]).to_numpy()
        # Harm rate with the initially-correct denominator, per slide.
        def harm_rate(s):
            return s["harm"].to_numpy() / (s["n"].to_numpy() - s["fp"].to_numpy() - s["fn"].to_numpy())
        d_hr = harm_rate(s_gate) - harm_rate(s_tent)
        d_hr_frozen = harm_rate(s_gate) - harm_rate(s_frozen)

        m, lo, hi = paired_ci(d_ba_gate)
        endpoint_rows.append(dict(seed=seed, contrast="gate_vs_frozen", metric="ba",
                                  mean=m, lo=lo, hi=hi,
                                  wins=int((d_ba_gate > 0).sum()),
                                  losses=int((d_ba_gate < 0).sum())))
        m, lo, hi = paired_ci(d_ba_gate_tent)
        endpoint_rows.append(dict(seed=seed, contrast="gate_vs_tent", metric="ba",
                                  mean=m, lo=lo, hi=hi,
                                  wins=int((d_ba_gate_tent > 0).sum()),
                                  losses=int((d_ba_gate_tent < 0).sum())))
        m, lo, hi = paired_ci(d_hr)
        endpoint_rows.append(dict(seed=seed, contrast="gate_vs_tent", metric="harm_rate",
                                  mean=m, lo=lo, hi=hi,
                                  wins=int((d_hr < 0).sum()),
                                  losses=int((d_hr > 0).sum())))
        m, lo, hi = paired_ci(d_hr_frozen)
        endpoint_rows.append(dict(seed=seed, contrast="gate_vs_frozen", metric="harm_rate",
                                  mean=m, lo=lo, hi=hi,
                                  wins=int((d_hr_frozen < 0).sum()),
                                  losses=int((d_hr_frozen > 0).sum())))

        # Patch-weighted slide-cluster bootstrap for gate vs frozen and gate vs TENT.
        slides = df["slide"].to_numpy()
        uniq = np.array(sorted(df["slide"].unique()))
        y = df["label"].to_numpy()
        for name, pa, pb in [("gate_vs_frozen", "pred_gate", "pred_frozen"),
                             ("gate_vs_tent", "pred_gate", "pred_tent"),
                             ("tent_vs_frozen", "pred_tent", "pred_frozen")]:
            a = (df[pa].to_numpy() == y)
            b = (df[pb].to_numpy() == y)
            diff = a.astype(float) - b.astype(float)
            idx_by_slide = {s: np.flatnonzero(slides == s) for s in uniq}
            means = []
            for _ in range(5000):
                draw = RNG.choice(uniq, size=len(uniq), replace=True)
                idx = np.concatenate([idx_by_slide[s] for s in draw])
                means.append(diff[idx].mean())
            means = np.array(means)
            cluster_rows.append(dict(seed=seed, contrast=name, mean=float(diff.mean()),
                                     lo=float(np.quantile(means, 0.025)),
                                     hi=float(np.quantile(means, 0.975))))

        for slide in sorted(df["slide"].unique()):
            keep = df[df["slide"] != slide]
            yy = keep["label"].to_numpy()
            loso_rows.append(dict(seed=seed, left_out=int(slide), n=len(keep),
                                  tent_minus_frozen=float((keep["pred_tent"].to_numpy() == yy).mean()
                                                          - (keep["pred_frozen"].to_numpy() == yy).mean()),
                                  gate_minus_frozen=float((keep["pred_gate"].to_numpy() == yy).mean()
                                                          - (keep["pred_frozen"].to_numpy() == yy).mean())))

    pd.DataFrame(endpoint_rows).to_csv(OUT / "revision_endpoint_recompute.csv", index=False)
    pd.concat(per_slide_all, ignore_index=True).to_csv(OUT / "revision_per_slide_metrics.csv", index=False)
    pd.DataFrame(cluster_rows).to_csv(OUT / "revision_slide_cluster_ci.csv", index=False)
    pd.DataFrame(loso_rows).to_csv(OUT / "revision_leave_one_slide_out.csv", index=False)

    print("=== slide-level BA (per-slide mean, 10 slides) ===")
    e = pd.DataFrame(endpoint_rows)
    print(e.groupby(["contrast", "metric"])[["mean", "lo", "hi"]].mean().to_string())
    print("\n=== per-seed gate-vs-frozen BA ===")
    print(e[(e.contrast == "gate_vs_frozen") & (e.metric == "ba")].to_string(index=False))
    print("\n=== per-seed gate-vs-TENT BA ===")
    print(e[(e.contrast == "gate_vs_tent") & (e.metric == "ba")].to_string(index=False))
    print("\n=== slide-cluster patch-weighted ===")
    print(pd.DataFrame(cluster_rows).groupby("contrast")[["mean", "lo", "hi"]].mean().to_string())
    print("\n=== per-seed cluster CI ===")
    print(pd.DataFrame(cluster_rows).to_string(index=False))


if __name__ == "__main__":
    main()

"""Paired slide x seed (two-way) bootstrap sensitivity for the Center-2 endpoints.

The primary confidence intervals in the paper resample slides only: the five
source seeds are first averaged within each slide.  Source-training randomness
is a separate uncertainty source, so this script reports, for the same paired
contrasts and the same per-(slide, seed) metrics:

  slide-only : resample the 10 slides, seeds averaged (the primary CI, rebuilt)
  seed-only  : resample the 5 seeds, slides averaged
  two-way    : resample slides AND seeds independently with replacement, then
               average the resampled slide x seed cells (cluster-by-both)

Draws: 5,000 per scheme, fixed RNG seed.  Input is the canonical per-(slide,
seed) table tables/core_method_slide_metrics.csv; nothing is recomputed from raw
predictions, so the output cannot drift from the primary-endpoint table.
Output: tables/core_twoway_bootstrap.csv (override with CG_TTA_OUT).
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.dirname(HERE)
OUT = os.environ.get("CG_TTA_OUT", os.path.join(REL, "tables"))
SRC = os.path.join(REL, "tables", "core_method_slide_metrics.csv")
NBOOT = 5000
RNG_SEED = 20261110

CONTRASTS = [("safe_tent", "no", "safe_vs_frozen", "ba"),
             ("safe_tent", "tent", "safe_vs_tent", "harm_rate"),
             ("safe_tent", "no", "safe_vs_frozen", "acc"),
             ("safe_tent", "tent", "safe_vs_tent", "ba")]


def cell_matrix(df, a, b, metric):
    """Return slides x seeds matrix of paired differences (a - b)."""
    pa = df[df.method == a].pivot(index="slide", columns="seed", values=metric)
    pb = df[df.method == b].pivot(index="slide", columns="seed", values=metric)
    assert pa.shape == pb.shape and not pa.isna().any().any(), "incomplete cells"
    return (pa - pb).to_numpy()


def boot(mat, scheme, rng):
    ns, nk = mat.shape
    out = np.empty(NBOOT)
    for i in range(NBOOT):
        if scheme == "slide":
            out[i] = mat[rng.integers(0, ns, ns)].mean()
        elif scheme == "seed":
            out[i] = mat[:, rng.integers(0, nk, nk)].mean()
        else:
            out[i] = mat[np.ix_(rng.integers(0, ns, ns), rng.integers(0, nk, nk))].mean()
    return out


def main():
    df = pd.read_csv(SRC)
    df = df[df.center == "center2"]
    assert df.seed.nunique() == 5 and df.slide.nunique() == 10
    rows = []
    for a, b, label, metric in CONTRASTS:
        mat = cell_matrix(df, a, b, metric)
        est = float(mat.mean())
        rec = dict(center="center2", contrast=label, metric=metric, n_slides=mat.shape[0],
                   n_seeds=mat.shape[1], estimate=est)
        for scheme in ("slide", "seed", "twoway"):
            # slide-only scheme reuses the primary-table RNG seeds (7 for BA, 11 otherwise)
            # so it reproduces core_primary_endpoints.csv exactly.
            rng = np.random.default_rng((7 if metric == "ba" else 11) if scheme == "slide" else RNG_SEED)
            v = boot(mat, scheme, rng)
            rec[f"{scheme}_lo"] = float(np.percentile(v, 2.5))
            rec[f"{scheme}_hi"] = float(np.percentile(v, 97.5))
            rec[f"{scheme}_se"] = float(v.std(ddof=1))
        rec["twoway_over_slide_width"] = (rec["twoway_hi"] - rec["twoway_lo"]) / (
            rec["slide_hi"] - rec["slide_lo"])
        rows.append(rec)
    res = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    res.to_csv(os.path.join(OUT, "core_twoway_bootstrap.csv"), index=False)
    pd.set_option("display.width", 220)
    print(res.round(5).to_string(index=False))


if __name__ == "__main__":
    main()

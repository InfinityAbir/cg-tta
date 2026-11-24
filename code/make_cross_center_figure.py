"""Cross-center figure: paired accuracy and Brier change vs No-TTA.

Reads the release prediction files (Center-2 and Center-0, seeds 1-5) and draws
mean +/- SD over seeds for the four adapters.  Brier is the binary per-patch
mean (p_tumor - y)^2.  Writes fig_cross_center.png and a small CSV with the
plotted numbers.

    python code/make_cross_center_figure.py
    CG_TTA_OUT=/tmp/out python code/make_cross_center_figure.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.dirname(HERE)
OUT = os.environ.get("CG_TTA_OUT", os.path.join(REL, "tables"))
os.makedirs(OUT, exist_ok=True)

SEEDS = (1, 2, 3, 4, 5)
ADAPTERS = ["tent", "flip", "eata", "shot"]
LABEL = {"tent": "TENT", "flip": "Flip-TTA", "eata": "EATA-LN", "shot": "SHOT-IM"}
COLOR = {"tent": "#1f77b4", "flip": "#2ca02c", "eata": "#9467bd", "shot": "#d62728"}


def load(center, seed, name):
    z = np.load(os.path.join(REL, "predictions", center, f"seed{seed}", f"{name}.npz"))
    return z["probabilities"].astype(np.float64), z["predictions"], z["labels"]


def paired(center, seed, adapter):
    p0, y0, lab = load(center, seed, "no")
    p1, y1, _ = load(center, seed, adapter)
    d_acc = 100.0 * ((y1 == lab).mean() - (y0 == lab).mean())
    brier = lambda p: float(np.mean((p[:, 1] - lab) ** 2))
    return d_acc, brier(p1) - brier(p0)


stats = {}
rows = []
for center in ("center2", "center0"):
    for ad in ADAPTERS:
        vals = np.array([paired(center, s, ad) for s in SEEDS])
        stats[(center, ad)] = (vals[:, 0].mean(), vals[:, 0].std(ddof=1),
                               100.0 * vals[:, 1].mean(), 100.0 * vals[:, 1].std(ddof=1))
        for s, (a, b) in zip(SEEDS, vals):
            rows.append(dict(center=center, seed=s, method=ad, d_acc_pp=a, d_brier_x100=100.0 * b))
with open(os.path.join(OUT, "cross_center_paired.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

x = np.arange(len(ADAPTERS))
w = 0.38
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
for ax, (i_mu, i_sd), title in [(axes[0], (0, 1), r"$\Delta$Accuracy vs No-TTA (pp)"),
                                (axes[1], (2, 3), r"$\Delta$Brier vs No-TTA, $\times100$ (negative $=$ better)")]:
    for k, (center, off, alpha, hatch, lab) in enumerate([
            ("center2", -w / 2, 1.0, None, "Center-2, 5 seeds (mean $\\pm$ SD)"),
            ("center0", +w / 2, 0.45, "//", "Center-0, 5 seeds (mean $\\pm$ SD)")]):
        mu = [stats[(center, a)][i_mu] for a in ADAPTERS]
        sd = [stats[(center, a)][i_sd] for a in ADAPTERS]
        ax.bar(x + off, mu, w, yerr=sd, capsize=2.5, color=[COLOR[a] for a in ADAPTERS],
               edgecolor="black", linewidth=0.5, alpha=alpha, hatch=hatch, label=lab)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[a] for a in ADAPTERS], fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=7, frameon=False, loc="lower center",
           ncol=2, bbox_to_anchor=(0.5, -0.035))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_cross_center.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
for (c, a), v in stats.items():
    print(f"{c} {a:5s} dAcc {v[0]:+.3f} (SD {v[1]:.3f})  dBrier x100 {v[2]:+.3f} (SD {v[3]:.3f})")

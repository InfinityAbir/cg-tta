"""Regenerate the three paper figures from the released CSVs.

Reads tables/ (or $CG_TTA_OUT) and writes figures/*.pdf and *.png.
Requires matplotlib and the Times New Roman font used by the paper.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.dirname(HERE)
TAB = os.environ.get("CG_TTA_OUT", os.path.join(REL, "tables"))
FIG = os.path.join(REL, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_FROZEN = "#3b6ea5"
C_ADAPT = "#c26a1b"
C_FLIP = "#3f8f5a"
C_GATE = "#5a5a5a"
C_SAFE = "#8e2f4f"


# ------------------------------------------------------------------ Figure 1
def pipeline():
    fig, ax = plt.subplots(figsize=(7.16, 2.35))
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=7.2, tc="black", lw=0.7):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.2",
                           linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, zorder=3, linespacing=1.25)

    def arrow(x1, y1, x2, y2, color=C_GATE, style="-|>", ls="-", lw=0.8, rad=0.0):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=7,
                            linewidth=lw, color=color, linestyle=ls, zorder=1,
                            connectionstyle=f"arc3,rad={rad}")
        ax.add_patch(a)

    # input
    box(1.5, 15.5, 13, 9, "Target batch\n$x_k$ (unlabeled)", "#f2f2f2", "#777777")
    # three predictors
    box(20.5, 28.0, 24, 8.5, "Frozen source\n$f_{\\theta_0}$", "#e8f0f8", C_FROZEN)
    box(20.5, 15.5, 24, 8.5, "Online adapter $\\mathcal{A}$\n$\\theta_{k-1}\\!\\to\\!\\theta_k$", "#fbeedd", C_ADAPT)
    box(20.5, 3.0, 24, 8.5, "Flip-TTA critic $\\mathcal{T}$\n4 views, no updates", "#e6f2ea", C_FLIP)
    # outputs
    box(47.5, 29.0, 12, 6.5, "$p_0$", "white", C_FROZEN)
    box(47.5, 16.5, 12, 6.5, "$p_1$", "white", C_ADAPT)
    box(47.5, 4.0, 12, 6.5, "$p_f$", "white", C_FLIP)
    # gate
    gx, gy, gw, gh = 64.0, 12.0, 21.0, 17.0
    p = FancyBboxPatch((gx, gy), gw, gh, boxstyle="round,pad=0.35,rounding_size=1.2",
                       linewidth=1.0, edgecolor=C_GATE, facecolor="#f7f7f7", zorder=2)
    ax.add_patch(p)
    ax.text(gx + gw / 2, gy + gh - 3.2,
            "Consensus gate (label-free)", ha="center", va="center",
            fontsize=7.4, fontweight="bold", zorder=3)
    ax.text(gx + gw / 2, gy + 9.0,
            "accept $\\hat y_1\\!\\neq\\!\\hat y_0$ iff\n$\\arg\\max p_f=\\arg\\max p_1$",
            ha="center", va="center", fontsize=7.0, zorder=3, linespacing=1.35)
    ax.text(gx + gw / 2, gy + 2.8,
            "else keep $\\hat y_0$", ha="center", va="center",
            fontsize=6.7, style="italic", color="#444444", zorder=3)
    # output
    box(88.0, 14.5, 10.5, 11.0, "Output\n$\\hat y$\n\n$\\hat y_1$ or $\\hat y_0$", "#f3e9ee", C_SAFE, fs=7.0)

    # arrows
    arrow(14.5, 20.0, 20.5, 32.2, C_FROZEN)
    arrow(14.5, 19.7, 20.5, 19.7, C_ADAPT)
    arrow(14.5, 19.4, 20.5, 7.2, C_FLIP)
    arrow(44.5, 32.2, 47.5, 32.2, C_FROZEN)
    arrow(44.5, 19.7, 47.5, 19.7, C_ADAPT)
    arrow(44.5, 7.2, 47.5, 7.2, C_FLIP)
    arrow(59.5, 32.2, gx, 25.5, C_GATE)
    arrow(59.5, 19.7, gx, 20.0, C_GATE)
    arrow(59.5, 7.2, gx, 14.5, C_GATE)
    arrow(gx + gw, 20.5, 88.0, 20.0, C_SAFE, lw=1.0)

    # timing annotation
    ax.text(32.5, 13.1, "score batch $k$ with $\\theta_{k-1}$, then update",
            ha="center", va="center", fontsize=6.2, color=C_ADAPT, style="italic")
    ax.text(32.5, 1.0, "final $\\theta_K$ never scores the evaluated stream",
            ha="center", va="center", fontsize=6.2, color="#666666", style="italic")

    # cost badge
    ax.text(76.0, 33.4, "no labels  ·  no threshold  ·  +4 forwards\nGates the output decision; the adapter update is unchanged",
            ha="center", va="center", fontsize=6.3, color=C_GATE, linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.5))

    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    fig.savefig(os.path.join(FIG, "fig_pipeline.pdf"))
    fig.savefig(os.path.join(FIG, "fig_pipeline.png"), dpi=400)
    plt.close(fig)


# ------------------------------------------------------------------ Figure 2
def frontier():
    main = pd.read_csv(os.path.join(TAB, "core_table2_main.csv"), index_col=0)
    gate = pd.read_csv(os.path.join(TAB, "core_table_gate_ablation.csv"), index_col=0)
    rc = pd.read_csv(os.path.join(TAB, "core_risk_coverage.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.35), gridspec_kw={"width_ratios": [1.05, 1]})

    # (a) accuracy vs harm
    ax = axes[0]
    style = {
        "no": ("o", C_FROZEN, "No-TTA"),
        "tent": ("s", C_ADAPT, "TENT"),
        "flip": ("^", C_FLIP, "Flip-TTA"),
        "eata": ("D", "#7b5aa6", "EATA-LN"),
        "shot": ("v", "#b23b3b", "SHOT-IM"),
        "safe_tent": ("*", C_SAFE, "CG-TENT"),
        "safe_eata": ("P", "#7b5aa6", "CG-EATA"),
        "safe_shot": ("X", "#b23b3b", "CG-SHOT"),
    }
    for m, (mk, col, lab) in style.items():
        r = main.loc[m]
        x = r.harm if m != "no" else 0.0
        ax.scatter(x, 100 * r.acc, marker=mk, s=42 if m == "safe_tent" else 26,
                   color=col, edgecolor="white", linewidth=0.4, zorder=4, label=lab)
    ax.annotate("CG-TENT", (main.loc["safe_tent", "harm"], 100 * main.loc["safe_tent", "acc"]),
                xytext=(6, -9), textcoords="offset points", fontsize=6.6, color=C_SAFE, fontweight="bold")
    ax.annotate("TENT", (main.loc["tent", "harm"], 100 * main.loc["tent", "acc"]),
                xytext=(4, 5), textcoords="offset points", fontsize=6.6, color=C_ADAPT)
    ax.annotate("No-TTA", (0, 100 * main.loc["no", "acc"]),
                xytext=(6, -13), textcoords="offset points", fontsize=6.6, color=C_FROZEN)
    ax.set_xlabel("Correct$\\to$wrong flips (mean over 5 seeds)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(-120, 4450)
    ax.set_ylim(87.0, 94.8)
    ax.grid(True, axis="y", lw=0.3, color="#dddddd", zorder=0)
    ax.set_title("(a) Utility versus induced harm", loc="left", fontweight="bold")
    ax.legend(loc="lower right", frameon=False, ncol=2, handletextpad=0.3, columnspacing=0.8,
              borderaxespad=0.2, fontsize=6.2)

    # (b) help vs harm, iso-net
    ax = axes[1]
    for m, (mk, col, lab) in style.items():
        r = main.loc[m]
        if m == "no":
            xh, xd = 0.0, 0.0
        else:
            xh, xd = r.help, r.harm
        ax.scatter(xd, xh, marker=mk, s=36 if m == "safe_tent" else 22,
                   color=col, edgecolor="white", linewidth=0.4, zorder=4)
    for net in [500, 1000, 1500]:
        xs = np.array([0, 4200])
        ax.plot(xs, xs + net, ls="--", lw=0.5, color="#cccccc", zorder=1)
    ax.scatter([gate.loc["random_matched", "harm"]], [gate.loc["random_matched", "help"]],
               marker="o", s=20, facecolors="none", edgecolors="#777777", linewidth=0.7, zorder=5)
    ax.annotate("matched\nrandom", (gate.loc["random_matched", "harm"], gate.loc["random_matched", "help"]),
                xytext=(4, -18), textcoords="offset points", fontsize=5.6, color="#666666")
    ax.annotate("CG-TENT", (main.loc["safe_tent", "harm"], main.loc["safe_tent", "help"]),
                xytext=(8, 9), textcoords="offset points", fontsize=6.4, color=C_SAFE, fontweight="bold")
    ax.annotate("TENT", (main.loc["tent", "harm"], main.loc["tent", "help"]),
                xytext=(6, 6), textcoords="offset points", fontsize=6.4, color=C_ADAPT)
    ax.set_xlabel("Correct$\\to$wrong flips")
    ax.set_ylabel("Wrong$\\to$correct corrections")
    ax.set_xlim(-120, 4400)
    ax.set_ylim(-80, 2000)
    ax.grid(True, lw=0.3, color="#eeeeee", zorder=0)
    ax.set_title("(b) Corrections versus induced harm", loc="left", fontweight="bold")

    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.14, wspace=0.30)
    fig.savefig(os.path.join(FIG, "fig_frontier.pdf"))
    fig.savefig(os.path.join(FIG, "fig_frontier.png"), dpi=400)
    plt.close(fig)


# ------------------------------------------------------------------ Figure 3
def slide_delta():
    P = pd.read_csv(os.path.join(TAB, "core_method_slide_metrics.csv"))
    c2 = P[P.center == "center2"]
    slides = sorted(c2.slide.unique())
    # mean over seeds per slide, and per-seed dots
    d_ba, d_harm = [], []
    for s in slides:
        g = c2[c2.slide == s]
        ba = g.pivot_table(index="seed", columns="method", values="ba")
        hr = g.pivot_table(index="seed", columns="method", values="harm_rate")
        d_ba.append((ba["safe_tent"] - ba["no"]).values * 100)
        d_harm.append((hr["safe_tent"] - hr["tent"]).values * 100)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.20), sharey=True)
    y = np.arange(len(slides))
    ax = axes[0]
    means = [v.mean() for v in d_ba]
    ax.barh(y, means, height=0.62, color="#e3d3da", edgecolor=C_SAFE, linewidth=0.6, zorder=2)
    for i, v in enumerate(d_ba):
        ax.scatter(v, np.full(len(v), y[i]), s=5, color=C_SAFE, alpha=0.55, zorder=3, linewidth=0)
    ax.axvline(0, color="black", lw=0.7, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels([f"S{s}" for s in slides])
    ax.invert_yaxis()
    ax.set_xlabel("$\\Delta$ balanced accuracy (pp)\nCG-TENT $-$ frozen")
    ax.set_title("(a) Efficacy by slide (10/10 positive)", loc="left", fontweight="bold")
    ax.grid(True, axis="x", lw=0.3, color="#dddddd", zorder=0)

    ax = axes[1]
    means = [v.mean() for v in d_harm]
    ax.barh(y, means, height=0.62, color="#dce8df", edgecolor=C_FLIP, linewidth=0.6, zorder=2)
    for i, v in enumerate(d_harm):
        ax.scatter(v, np.full(len(v), y[i]), s=5, color="#2f6d45", alpha=0.55, zorder=3, linewidth=0)
    ax.axvline(0, color="black", lw=0.7, zorder=1)
    ax.set_xlabel("$\\Delta$ C$\\to$W harm rate (pp)\nCG-TENT $-$ TENT")
    ax.set_title("(b) Harm by slide (10/10 lower)", loc="left", fontweight="bold")
    ax.grid(True, axis="x", lw=0.3, color="#dddddd", zorder=0)

    fig.subplots_adjust(left=0.06, right=0.985, top=0.88, bottom=0.21, wspace=0.10)
    fig.savefig(os.path.join(FIG, "fig_slide_delta.pdf"))
    fig.savefig(os.path.join(FIG, "fig_slide_delta.png"), dpi=400)
    plt.close(fig)


if __name__ == "__main__":
    pipeline()
    frontier()
    slide_delta()
    print("figures written to", FIG)

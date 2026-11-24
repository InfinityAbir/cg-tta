"""Figures and tables added for the journal version of the paper, regenerated from the
stored predictions and CSVs in this release.

    python jbhi_revision/make_jbhi_assets.py

Reads  jbhi_revision/inputs/   (derived CSV and LaTeX fragments copied from the analysis)
       tables/, predictions/   (this release)
Writes jbhi_revision/tables/   tab_main.tex (Table I, both centers), tab_deployment_jbhi.tex,
                               tab_batchsize_compact.tex, jbhi_asset_numbers.json
       jbhi_revision/figures/  fig_decile_harm, fig_cost_curve, fig_slide_drift (PDF + PNG)
Table S1 (supp_partition.tex) also needs the Camelyon17-WILDS metadata.csv, which is not
redistributed; without it that one step is skipped and the shipped copy is kept.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.dirname(HERE)
TAB_IN = os.path.join(HERE, "inputs")
TAB_OUT = os.path.join(HERE, "tables")
FIG = os.path.join(HERE, "figures")
os.makedirs(TAB_OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, os.path.join(REL, "code"))
import cg_tta as st  # noqa: E402

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "font.family": "serif",
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})
BLUE, RED, GREY, GREEN, ORANGE = "#2b6cb0", "#c53030", "#6b7280", "#2f855a", "#dd6b20"


def minus(s):
    return s.replace("-", "$-$")


# ------------------------------------------------------------- Table I (SDs)
def table_main():
    m2 = pd.read_csv(os.path.join(TAB_IN, "core_method_seed_metrics.csv"))
    m2 = m2[m2.center == "center2"]
    m0 = pd.read_csv(os.path.join(REL, "tables", "core_center0_multiseed_seed_metrics.csv"))
    m0 = m0[m0.center == "center0"]
    names = {"no": "No-TTA (frozen)", "tent": "TENT", "flip": "Flip-TTA", "eata": "EATA-LN",
             "shot": "SHOT-IM", "safe_tent": r"\textbf{CG-TENT}", "safe_eata": "CG-EATA",
             "safe_shot": "CG-SHOT"}
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r"Method & Acc. & AUROC & ECE & Harm & Net \\", r"\midrule"]
    out = {}
    panels = (("center2", m2, r"\multicolumn{6}{@{}l}{\textit{(a) Center-2, $n=85{,}054$}} \\"),
              ("center0", m0, r"\multicolumn{6}{@{}l}{\textit{(b) Center-0, $n=59{,}436$}} \\"))
    for pi, (cen, m, head) in enumerate(panels):
        if pi:
            lines.append(r"\midrule")
        lines.append(head)
        for k in names:
            g = m[m.method == k].sort_values("seed")
            assert len(g) == 5, (cen, k)
            a, u, e = g.acc, g.auroc, g.ece
            pm = lambda s, d: f"{s.mean():.{d}f}$\\pm${s.std(ddof=1):.{d}f}"
            if k == "no":
                harm, net = "--", "--"
            else:
                harm = f"{g.harm.mean():.0f}$\\pm${g.harm.std(ddof=1):.0f}"
                net = f"{g.net.mean():+.0f}$\\pm${g.net.std(ddof=1):.0f}"
            lines.append(f"{names[k]} & {pm(a, 4)} & {pm(u, 4)} & {pm(e, 4)} & {harm} & {net} \\\\")
            out[cen + "_" + k] = dict(acc_sd=float(a.std(ddof=1)), auroc_sd=float(u.std(ddof=1)),
                                      ece_sd=float(e.std(ddof=1)))
    lines += [r"\bottomrule", r"\end{tabular}"]
    text = "\n".join(lines)
    text = text.replace("$\\pm$", "$\\pm$").replace("+", "+")
    import re
    text = re.sub(r"(?<![\w$])-(?=\d)", "$-$", text)
    open(os.path.join(TAB_OUT, "tab_main.tex"), "w", newline="\n").write(text + "\n")
    return out


# ------------------------------------------------------------ Fig. 2 deciles
def fig_decile():
    d = pd.read_csv(os.path.join(TAB_IN, "table_decile.csv"))
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    x = d.decile.to_numpy()
    w = 0.38
    lo = lambda m, sd: np.minimum(100 * sd, 100 * m)
    ax.bar(x - w / 2, 100 * d.harm_rate_mean, w, yerr=[lo(d.harm_rate_mean, d.harm_rate_sd), 100 * d.harm_rate_sd], color=RED,
           label="Harm (C$\\to$W)", error_kw=dict(lw=0.7, capsize=1.5))
    ax.bar(x + w / 2, 100 * d.help_rate_mean, w, yerr=[lo(d.help_rate_mean, d.help_rate_sd), 100 * d.help_rate_sd], color=BLUE,
           label="Help (W$\\to$C)", error_kw=dict(lw=0.7, capsize=1.5))
    ax.set_xticks(x)
    ax.set_xlabel("Confidence decile (1 = lowest)")
    ax.set_ylabel("Rate within decile (%)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIG, "fig_decile_harm.pdf"))
    fig.savefig(os.path.join(FIG, "fig_decile_harm.png"), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------- cost-vs-ratio fig
def fig_cost():
    C = pd.read_csv(os.path.join(REL, "tables", "class_counts_by_seed.csv"))
    base = C[C.method == "frozen"].set_index("seed")
    t = C[C.method == "tent"].set_index("seed")
    g = C[C.method == "gate"].set_index("seed")
    r = np.logspace(-1, np.log10(20), 400)

    def cost(sub):
        return np.array([(sub.fn - base.fn).to_numpy(float) + ri * (sub.fp - base.fp).to_numpy(float) for ri in r])

    ct, cg = cost(t), cost(g)
    dt = (t.fn - base.fn).mean(), (t.fp - base.fp).mean()
    dg = (g.fn - base.fn).mean(), (g.fp - base.fp).mean()
    be = -(dt[0] - dg[0]) / (dt[1] - dg[1])
    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    for j in range(ct.shape[1]):
        ax.plot(r, ct[:, j], color=RED, lw=0.6, alpha=0.3)
        ax.plot(r, cg[:, j], color=BLUE, lw=0.6, alpha=0.3)
    ax.plot(r, ct.mean(1), color=RED, lw=1.8, label="TENT")
    ax.plot(r, cg.mean(1), color=BLUE, lw=1.8, label="CG-TENT")
    ax.axvline(be, color=GREY, ls="--", lw=0.9)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Cost ratio $c_{FP}:c_{FN}$")
    ax.set_ylabel("Cost change vs. frozen\n(units of one false negative)")
    ax.set_ylim(-3200, 2200)
    ax.text(be * 1.1, 1900, f"break-even\n{be:.1f}:1", fontsize=8, color=GREY, va="top")
    ax.text(0.03, 0.04, "TENT cheaper\nwhen misses cost more", transform=ax.transAxes, fontsize=7.5, color=RED, va="bottom")
    ax.text(0.97, 0.04, "gate cheaper when\nalarms cost more", transform=ax.transAxes, fontsize=7.5, color=BLUE,
            va="bottom", ha="right")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["1:10", "1:5", "1:2", "1:1", "2:1", "5:1", "10:1", "20:1"])
    ax.set_xlim(0.1, 20)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIG, "fig_cost_curve.pdf"))
    fig.savefig(os.path.join(FIG, "fig_cost_curve.png"), dpi=300)
    plt.close(fig)
    return float(be)


# ------------------------------------------- per-slide + predicted-positive drift
def fig_slide_drift():
    P = pd.read_csv(os.path.join(TAB_IN, "revision_per_slide_metrics.csv"))
    P = P[P.method.isin(["tent", "gate"])]
    mean = P.groupby(["method", "slide"])[["help", "harm"]].mean().reset_index()
    slides = sorted(P.slide.unique())
    hi = {24: "24", 27: "27", 29: "29"}

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.55), gridspec_kw=dict(width_ratios=[1, 1.15]))
    x = np.arange(len(slides))
    tt = mean[mean.method == "tent"].set_index("slide").loc[slides]
    gg = mean[mean.method == "gate"].set_index("slide").loc[slides]
    w = 0.38
    a.bar(x - w / 2, tt.help, w, color=BLUE, label="TENT help")
    a.bar(x - w / 2, -tt.harm, w, color=RED, label="TENT harm")
    a.bar(x + w / 2, gg.help, w, color=BLUE, alpha=0.4, hatch="//", edgecolor="white", lw=0, label="CG-TENT help")
    a.bar(x + w / 2, -gg.harm, w, color=RED, alpha=0.4, hatch="//", edgecolor="white", lw=0, label="CG-TENT harm")
    a.axhline(0, color="black", lw=0.5)
    a.set_yscale("symlog", linthresh=10)
    a.set_yticks([-100, -10, 0, 10, 100, 1000])
    a.set_yticklabels(["100", "10", "0", "10", "100", "1000"])
    a.set_xticks(x)
    a.set_xticklabels(slides)
    for i, s in enumerate(slides):
        if s in hi:
            a.axvspan(i - 0.5, i + 0.5, color="#f6e05e", alpha=0.35, lw=0, zorder=0)
    a.set_xlabel("Center-2 slide")
    a.set_ylabel("Patches per seed (symlog)\nhelp above, harm below")
    a.legend(frameon=False, ncol=2, loc="upper left", fontsize=7, columnspacing=0.8, handlelength=1.2)
    a.set_ylim(-250, 6000)
    a.set_title("(a) Help and harm by slide", fontsize=9)

    # predicted-positive rate over the stream
    Wb = 3200
    series = {"no": [], "tent": [], "shot": []}
    lab_rate = None
    n = None
    for seed in range(1, 6):
        d = {k: st.load_npz(os.path.join(REL, "predictions", "center2", f"seed{seed}", f"{k}.npz"))
             for k in ("no", "tent", "shot")}
        n = len(d["no"]["labels"])
        edges = list(range(0, n, Wb))
        for k in series:
            series[k].append([float(d[k]["predictions"][i:i + Wb].mean()) for i in edges])
        lab_rate = [float(d["no"]["labels"][i:i + Wb].mean()) for i in edges]
    centers = np.array([(i + min(i + Wb, n)) / 2 for i in range(0, n, Wb)]) / 1000.0
    sw = int(np.flatnonzero(np.diff(d["no"]["labels"]))[0]) + 1
    a_no, a_te, a_sh = (np.array(series[k]) for k in ("no", "tent", "shot"))
    diff = 100 * (a_te - a_no)
    for k in range(diff.shape[0]):
        b.plot(centers, diff[k], color=RED, lw=0.6, alpha=0.35)
    b.plot(centers, diff.mean(0), color=RED, lw=1.8, label="TENT minus frozen (mean of 5 seeds)")
    b.axhline(0, color="black", lw=0.6)
    b.axvline(sw / 1000.0, color="black", lw=0.6, ls="--")
    b.set_xlabel("Stream position (thousand patches)")
    b.set_ylabel("Predicted-positive rate,\nTENT minus frozen (pp)")
    ymax = max(6.0, float(diff.max()) * 1.1)
    b.set_ylim(-4, ymax)
    b.text(sw / 2000.0, ymax * 0.93, "tumor patches", fontsize=8, ha="center", va="top", color=GREY)
    b.text((sw + n) / 2000.0, ymax * 0.93, "normal patches", fontsize=8, ha="center", va="top", color=GREY)
    b.set_title("(b) TENT's drift toward the tumor label", fontsize=9)
    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(os.path.join(FIG, "fig_slide_drift.pdf"))
    fig.savefig(os.path.join(FIG, "fig_slide_drift.png"), dpi=300)
    plt.close(fig)

    # numbers for the text: tail of the normal block, per seed
    def tail(arr, lo, hi_):
        m = (centers * 1000 >= lo) & (centers * 1000 < hi_)
        return arr[:, m].mean(1)

    start_norm = tail(a_te, sw, sw + 12800) - tail(a_no, sw, sw + 12800)
    end_norm = tail(a_te, n - 12800, n) - tail(a_no, n - 12800, n)
    tum_block = tail(a_te, 0, sw - 6400) - tail(a_no, 0, sw - 6400)
    return dict(switch=int(sw), n=int(n),
                tum_block_diff_pp=(100 * tum_block).round(2).tolist(),
                first_normal_12800_diff_pp=(100 * start_norm).round(2).tolist(),
                last_normal_12800_diff_pp=(100 * end_norm).round(2).tolist(),
                frozen_ppr_normal_end=float(tail(a_no, n - 12800, n).mean()),
                tent_ppr_normal_end=float(tail(a_te, n - 12800, n).mean()),
                frozen_ppr_first_normal=float(tail(a_no, sw, sw + 12800).mean()),
                tent_ppr_first_normal=float(tail(a_te, sw, sw + 12800).mean()),
                shot_ppr_normal_end=float(tail(a_sh, n - 12800, n).mean()))


# ------------------------------------------- Table II with the locked R2 rule
def table_deployment():
    base = open(os.path.join(TAB_IN, "tab_deployment.tex"), encoding="utf8").read().splitlines()
    locked = open(os.path.join(TAB_IN, "tab_deployment_locked.tex"), encoding="utf8").read().splitlines()
    row = [l for l in locked if l.startswith("Locked R2, gate margin")]
    assert len(row) == 1
    cells = [c.strip() for c in row[0].rstrip("\\").rstrip().rstrip("\\").split("&")]
    cells = cells[:7]
    cells[0] = "Gate margin, locked on Center-1 (R2)"
    new = " & ".join(cells) + " \\\\"
    out, done = [], False
    for l in base:
        out.append(l)
        if l.startswith("Probability averaging") and not done:
            out.append(new)
            done = True
    assert done
    open(os.path.join(TAB_OUT, "tab_deployment_jbhi.tex"), "w", newline="\n").write("\n".join(out) + "\n")



# ----------------------------------------------- Center-0 class-conditional cost
def cost_center0():
    rows = []
    for seed in range(1, 6):
        d = {k: st.load_npz(os.path.join(REL, "predictions", "center0", f"seed{seed}", f"{k}.npz"))
             for k in ("no", "tent", "flip", "eata", "shot")}
        pack = st.build_pack(d)
        lab = d["no"]["labels"].astype(int)
        r = {}
        for name, key in (("frozen", "no"), ("tent", "tent"), ("gate", "safe_tent")):
            pr = pack[key][1].astype(int)
            r[name] = ((int(((lab == 1) & (pr == 0)).sum())), int(((lab == 0) & (pr == 1)).sum()))
        rows.append(r)
    fz, te, ga = (np.array([r[k] for r in rows], float) for k in ("frozen", "tent", "gate"))
    dt, dg = (te - fz).mean(0), (ga - fz).mean(0)
    be_mean = float(-(dt[0] - dg[0]) / (dt[1] - dg[1]))
    be_seed = [float(-((te[i, 0] - fz[i, 0]) - (ga[i, 0] - fz[i, 0])) /
                     ((te[i, 1] - fz[i, 1]) - (ga[i, 1] - fz[i, 1]))) for i in range(5)]
    return dict(tent_dFN=float(dt[0]), tent_dFP=float(dt[1]), gate_dFN=float(dg[0]), gate_dFP=float(dg[1]),
                breakeven_mean=be_mean, breakeven_seed=be_seed)


# ------------------------------------------------- compact batch-size table
def table_batchsize():
    b = pd.read_csv(os.path.join(TAB_IN, "aggregate_batchsize.csv"))
    fm = lambda v: f"{int(v):,}".replace(",", "{,}")
    sg = lambda v: ("$+$" if v >= 0 else "$-$") + fm(abs(v))
    lines = [r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Seed & \multicolumn{3}{c}{Batch 64} & \multicolumn{3}{c}{Batch 8} \\",
             r" & Acc. & Harm & Net & Acc. & Harm & Net \\", r"\midrule"]
    for seed in range(1, 6):
        a = b[(b.seed == seed) & (b.batch == 64)].iloc[0]
        c = b[(b.seed == seed) & (b.batch == 8)].iloc[0]
        lines.append(f"{seed} & {a.tent_accuracy:.4f} & {fm(a.harm)} & {sg(a.net)} & "
                     f"{c.tent_accuracy:.4f} & {fm(c.harm)} & {sg(c.net)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB_OUT, "tab_batchsize_compact.tex"), "w", newline="\n").write("\n".join(lines) + "\n")

# ----------------------------------------------------------------- Table S1
def table_partition():
    mp = os.environ.get("WILDS_METADATA", "")
    if not mp or not os.path.isfile(mp):
        print("table_partition: set WILDS_METADATA=/path/to/metadata.csv to regenerate Table S1; skipped")
        return {}
    meta = pd.read_csv(mp, index_col=0)
    c = meta.center
    sp = meta.split

    def row(name, m, cent):
        g = meta[m]
        return (name, cent, len(g), g.slide.nunique(), g.patient.nunique(), int(g.tumor.sum()))

    r1 = [row("Source train", c.isin([0, 3, 4]) & (sp == 0), "0, 3, 4"),
          row("Source ID val", c.isin([0, 3, 4]) & (sp == 1), "0, 3, 4"),
          row("OOD validation", c == 1, "1"),
          row("Target", c == 2, "2")]
    r2 = [row("Source train", c.isin([3, 4]) & (sp == 0), "3, 4"),
          row("OOD validation", c == 1, "1"),
          row("Target", c == 0, "0")]
    # leakage checks for the second rotation
    src = meta[c.isin([3, 4]) & (sp == 0)]
    tgt = meta[c == 0]
    checks = dict(center0_in_source=int(src.center.eq(0).sum()),
                  shared_slides=len(set(src.slide) & set(tgt.slide)),
                  shared_patients=len(set(src.patient) & set(tgt.patient)),
                  center0_raw_split0=int(((c == 0) & (sp == 0)).sum()))
    assert checks["center0_in_source"] == 0 and checks["shared_slides"] == 0 and checks["shared_patients"] == 0
    fm = lambda v: f"{v:,}".replace(",", "{,}")
    lines = [r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{llcccc}", r"\toprule",
             r"Rotation & Partition & Centers & Patches & Slides & Patients* \\", r"\midrule"]
    for k, (rows, lab) in enumerate(((r1, "Center-2 target"), (r2, "Center-0 target"))):
        for j, (nm, cent, n, sl, pa, tum) in enumerate(rows):
            first = (r"\multirow{%d}{*}{%s}" % (len(rows), lab)) if j == 0 else ""
            dag = "$^\\dagger$" if nm == "Source ID val" else ""
            lines.append(f"{first} & {nm} & {cent} & {fm(n)} & {sl}{dag} & {pa}{dag} \\\\")
        lines.append(r"\midrule" if k == 0 else r"\bottomrule")
    lines.append(r"\end{tabular}")
    open(os.path.join(TAB_OUT, "supp_partition.tex"), "w", newline="\n").write("\n".join(lines) + "\n")
    checks["rotation1"] = r1
    checks["rotation2"] = r2
    return checks


if __name__ == "__main__":
    out = {}
    out["table_main_sd"] = table_main()
    fig_decile()
    out["breakeven"] = fig_cost()
    out["drift"] = fig_slide_drift()
    out["partition"] = table_partition()
    table_deployment()
    table_batchsize()
    out["center0_cost"] = cost_center0()
    json.dump(out, open(os.path.join(TAB_OUT, "jbhi_asset_numbers.json"), "w"), indent=1, default=str)
    print(json.dumps(out["drift"], indent=1))
    print(json.dumps(out["partition"], indent=1, default=str))
    print("break-even", out["breakeven"])

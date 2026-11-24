"""Generate the supplementary-material tables that used to be typed by hand.

Every table is built from stored predictions or from the CSV/JSON files shipped
in tables/ and revision_colab/, so no number in these tables is typed.  Output:
supp_<name>.tex (tabular only; the caption and label live in the manuscript).

    python code/make_supplement_tables.py
    CG_TTA_OUT=/tmp/out python code/make_supplement_tables.py

Conventions (same as the paper): sample SD (n-1); ECE with 15 equal-width bins;
Brier is the binary per-patch mean (p_tumor - y)^2; help/harm are W->C / C->W
counts against the same-seed frozen model.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import cg_tta as st  # noqa: E402

OUT = os.environ.get("CG_TTA_OUT", os.path.join(REL, "tables"))
TAB = os.path.join(REL, "tables")
REV = os.path.join(REL, "revision_colab")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- formatting
def n_(x):
    """Integer with thousands braces, text or math safe."""
    return f"{int(round(x)):,}".replace(",", "{,}")


def signed(x, dec=0):
    s = f"{abs(x):,.{dec}f}".replace(",", "{,}")
    return f"${'+' if x >= 0 else '-'}{s}$"


def cnt(x):
    """Mean count: integer when large, one decimal when small and fractional."""
    if abs(x) >= 100 or abs(x - round(x)) < 0.05:
        return n_(x)
    return f"{x:.1f}"


def pm(vals, dec):
    v = np.asarray(vals, float)
    return f"${v.mean():.{dec}f}\\pm{v.std(ddof=1):.{dec}f}$"


def write(name, lines):
    path = os.path.join(OUT, f"supp_{name}.tex")
    with open(path, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


FIT = ["\\footnotesize", "\\setlength{\\tabcolsep}{2.5pt}"]


def tab(cols, header, body, extra=""):
    return (FIT + [f"\\begin{{tabular}}{{{cols}}}", "\\toprule", header, "\\midrule"]
            + body + ["\\bottomrule", "\\end{tabular}"])


def load(center, seed, name, root=None):
    base = root or os.path.join(REL, "predictions", center)
    z = np.load(os.path.join(base, f"seed{seed}", f"{name}.npz"))
    return z["probabilities"].astype(np.float64), z["predictions"], z["labels"]


def nll(p, lab):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(lab)), lab.astype(int)], 1e-12, 1))))


def brier(p, lab):
    return float(np.mean((p[:, 1] - lab) ** 2))


seed_metrics = pd.read_csv(os.path.join(TAB, "core_method_seed_metrics.csv"))
c2 = seed_metrics[seed_metrics.center == "center2"]

# -------------------------------------------------------------- partition
P = pd.read_csv(os.path.join(TAB, "partition_counts.csv"))
body = []
for _, r in P.iterrows():
    dag = "$^\\dagger$" if r.partition == "Source ID val" else ""
    body.append(f"{r.partition} & {r.centers} & {n_(r.patches)} & {r.slides}{dag} & {r.patients}{dag} \\\\")
write("partition", tab("lcccc", "Partition & Centers & Patches & Slides & Patients* \\\\", body))

# -------------------------------------------------------- per-seed transitions
body = []
rows = {}
for s in range(1, 6):
    t = c2[(c2.seed == s) & (c2.method == "tent")].iloc[0]
    g = c2[(c2.seed == s) & (c2.method == "safe_tent")].iloc[0]
    rows[s] = (t, g)
    body.append(f"{s} & {n_(t.help)} & {n_(t.harm)} & {signed(t.net)} & "
                f"{n_(g.help)} & {n_(g.harm)} & {signed(g.net)} \\\\")
body.append("\\cmidrule(lr){1-7}")
mt = {k: np.mean([rows[s][0][k] for s in rows]) for k in ("help", "harm", "net")}
mg = {k: np.mean([rows[s][1][k] for s in rows]) for k in ("help", "harm", "net")}
body.append(f"Mean & {cnt(mt['help'])} & {cnt(mt['harm'])} & {signed(mt['net'])} & "
            f"{cnt(mg['help'])} & {cnt(mg['harm'])} & {signed(mg['net'])} \\\\")
write("per_seed_trans", FIT + [
    "\\begin{tabular}{lcccccc}", "\\toprule",
    "Seed & \\multicolumn{3}{c}{TENT vs frozen} & \\multicolumn{3}{c}{CG-TENT vs frozen} \\\\",
    "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
    " & Helped & Harmed & Net & Helped & Harmed & Net \\\\", "\\midrule"] + body
    + ["\\bottomrule", "\\end{tabular}"])

# ----------------------------------------------- abstention and risk-coverage
abst, aurc = [], []
for s in range(1, 6):
    p0, y0, lab = load("center2", s, "no")
    _, y1, _ = load("center2", s, "tent")
    conf = p0.max(1).astype(np.float32)
    harm = ((y0 == lab) & (y1 != lab)).astype(float)
    err1 = (y1 != lab).astype(float)
    n = len(lab)
    cells = []
    for thr in (0.90, 0.95, 0.99):
        keep = conf >= np.float32(thr)
        cells.append(f"${harm[keep].mean():.5f}$ ({keep.mean():.3f})")
    abst.append(f"{s} & " + " & ".join(cells) + " \\\\")
    order = np.argsort(-conf, kind="mergesort")
    k = np.arange(1, n + 1)
    h_aurc = float((np.cumsum(harm[order]) / k).mean())
    h_orac = float((np.cumsum(np.sort(harm)) / k).mean())
    e_aurc = float((np.cumsum(err1[order]) / k).mean())
    aurc.append(f"{s} & {h_aurc:.6f} & {h_orac:.6f} & {h_aurc - h_orac:.6f} & {e_aurc:.4f} \\\\")
write("abstention", tab("lccc", "Seed & $\\ge$0.90 (cov.) & $\\ge$0.95 (cov.) & $\\ge$0.99 (cov.) \\\\", abst))
write("aurc", tab("lcccc", "Seed & Harm AURC & Oracle & Excess & Error AURC \\\\", aurc))

# ------------------------------------------------------------------- latency
T = pd.read_csv(os.path.join(REV, "tables", "timing_gpu.csv"))
names = [("frozen", "Frozen"), ("tent", "TENT"), ("flip_tta", "Flip-TTA"), ("safe_tent", "CG-TENT")]
body = []
for m, lab_ in names:
    d = T[T["mode"] == m].set_index("batch_size")
    body.append(f"{lab_} & " + " & ".join(f"{d.loc[b, 'images_per_sec']:.1f}" for b in (1, 8, 32, 64))
                + f" & {d.loc[64, 'ms_per_image']:.1f} & {n_(d.loc[64, 'peak_alloc_mb'])} \\\\")
write("latency", tab("lcccccc", "Mode & B=1 & B=8 & B=32 & B=64 & ms/img (B=64) & Peak MB (B=64) \\\\", body))

# ------------------------------------------------------- Center-0 battery
C0 = pd.read_csv(os.path.join(TAB, "core_center0_multiseed_seed_metrics.csv"))
order = [("no", "No-TTA"), ("tent", "TENT"), ("flip", "Flip-TTA"), ("eata", "EATA-LN"),
         ("shot", "SHOT-IM"), ("safe_tent", "CG-TENT"), ("safe_eata", "CG-EATA"),
         ("safe_shot", "CG-SHOT")]
body = []
for m, lab_ in order:
    if m == "safe_tent":
        body.append("\\midrule")
    d = C0[C0.method == m]
    if m == "no":
        body.append(f"{lab_} & {d.acc.mean():.4f} & --- & --- & --- & --- & {d.ece.mean():.4f} \\\\")
    else:
        body.append(f"{lab_} & {d.acc.mean():.4f} & ${d.delta_pp_vs_frozen.mean():+.2f}$ & "
                    f"{n_(d['help'].mean())} & {n_(d.harm.mean())} & {signed(d.net.mean())} & {d.ece.mean():.4f} \\\\")
write("center0_battery", tab("lcccccc", "Method & Acc & $\\Delta$Acc & Help & Harm & Net & ECE-15 \\\\", body))

# ----------------------------------------------- cross-center harm indicator
H = json.load(open(os.path.join(TAB, "harm_indicator_cross_center.json")))
lab_m = [("tent", "TENT"), ("flip_tta", "Flip-TTA"), ("eata", "EATA-LN"), ("shot", "SHOT-IM")]
body = []
for ci, (center, label) in enumerate([("center2", "Center-2"), ("center0", "Center-0")]):
    if ci:
        body.append("\\midrule")
    for j, (m, ml) in enumerate(lab_m):
        v = H["centers"][center][m]
        first = (f"\\multirow{{4}}{{*}}{{{label} ({v['n_runs']} runs)}}" if j == 0 else "")
        body.append(f"{first} & {ml} & {v['auroc_mean']:.3f} & {v['capture_at_10pct_mean']:.3f} & "
                    f"{100 * v['bottom_decile_harm_rate_mean']:.2f}\\% \\\\")
write("xcenter_harm", tab("llccc", "Center & Method & AUROC & Capture@10\\% & 1st-decile harm \\\\", body))

# -------------------------------------------------------------- SHOT-IM
rows = []
for s in range(1, 6):
    p0, y0, lab = load("center2", s, "no")
    p1, y1, _ = load("center2", s, "shot")
    m = st.method_metrics(p1, y1, lab, y0)
    rows.append(dict(seed=s, acc=m["acc"], d=100 * (m["acc"] - st.accuracy(y0, lab)), help=m["help"],
                     harm=m["harm"], net=m["net"], ece=m["ece"], brier=brier(p1, lab)))
R = pd.DataFrame(rows)
body = [f"{int(r.seed)} & {r.acc:.4f} & ${r.d:+.2f}$ & {n_(r['help'])} & {n_(r.harm)} & {signed(r.net)} & "
        f"{r.ece:.4f} & {r.brier:.4f} \\\\" for _, r in R.iterrows()]
body.append("\\midrule")
body.append(f"Mean & {R.acc.mean():.4f} & ${R.d.mean():+.2f}$ & {n_(R['help'].mean())} & {n_(R.harm.mean())} & "
            f"{signed(R.net.mean())} & {R.ece.mean():.4f} & {R.brier.mean():.4f} \\\\")
body.append(f"SD & {R.acc.std(ddof=1):.4f} & {R.d.std(ddof=1):.2f} & & & & {R.ece.std(ddof=1):.4f} & {R.brier.std(ddof=1):.4f} \\\\")
write("shot", tab("lccccccc", "Seed & Acc & $\\Delta$Acc (pp) & Help & Harm & Net & ECE-15 & Brier \\\\", body))

# -------------------------------------------------------- full-parameter EATA
E = pd.read_csv(os.path.join(TAB, "core_eata_full_seed_metrics.csv"))
body = [f"{int(r.seed)} & {r.frozen_acc:.4f} & {r.eata_acc:.4f} & ${r.eata_minus_frozen_pp:+.2f}$ & "
        f"{n_(r.eata_help)} & {n_(r.eata_harm)} & {signed(r.eata_help - r.eata_harm)} \\\\" for _, r in E.iterrows()]
body.append("\\midrule")
body.append(f"Mean & {E.frozen_acc.mean():.4f} & {E.eata_acc.mean():.4f} & ${E.eata_minus_frozen_pp.mean():+.2f}$ & "
            f"{n_(E.eata_help.mean())} & {n_(E.eata_harm.mean())} & {signed((E.eata_help - E.eata_harm).mean())} \\\\")
write("eata_full", tab("lcccccc", "Seed & Frozen & EATA-full & $\\Delta$Acc & Help & Harm & Net \\\\", body))

# ---------------------------------------------------------- Phikon online
pk = []
meth = [("no_tta", "Frozen"), ("tent", "TENT"), ("eata", "EATA-LN"), ("flip_tta", "Flip-TTA"), ("safe_tent", "CG-TENT")]
for s in range(1, 6):
    base = os.path.join(REV, "phikon", f"seed{s}_tta")
    p0, y0, lab = (lambda z: (z["probabilities"].astype(float), z["predictions"], z["labels"]))(
        np.load(os.path.join(base, "no_tta_predictions.npz")))
    for m, _ in meth:
        z = np.load(os.path.join(base, f"{m}_predictions.npz"))
        p, y = z["probabilities"].astype(float), z["predictions"]
        mm = st.method_metrics(p, y, lab, y0)
        pk.append(dict(seed=s, method=m, ba=mm["ba"], auroc=mm["auroc"], nll=nll(p, lab), brier=brier(p, lab),
                       ece=mm["ece"], help=mm["help"], harm=mm["harm"], net=mm["net"]))
PK = pd.DataFrame(pk)
body = []
for m, lab_ in meth:
    d = PK[PK.method == m]
    cells = [pm(d.ba, 4), pm(d.auroc, 4), pm(d.nll, 4), pm(d.brier, 4), pm(d.ece, 4)]
    if m == "no_tta":
        cells += ["--", "--", "--"]
    else:
        cells += [pm_i for pm_i in (
            "$" + n_(d['help'].mean()) + "\\pm" + n_(d['help'].std(ddof=1)) + "$",
            "$" + n_(d.harm.mean()) + "\\pm" + n_(d.harm.std(ddof=1)) + "$",
            "$" + n_(d.net.mean()) + "\\pm" + n_(d.net.std(ddof=1)) + "$")]
    body.append(f"{lab_} & " + " & ".join(cells) + " \\\\")
write("phikon_online", tab("lcccccccc", "Method & BA & AUROC & NLL & Brier & ECE-15 & Help & Harm & Net \\\\", body))
PK.to_csv(os.path.join(OUT, "phikon_online_per_seed.csv"), index=False)

# ------------------------------------------------------------------ Macenko
M = pd.read_csv(os.path.join(TAB, "aggregate_macenko.csv")).sort_values("seed")
body = []
for _, r in M.iterrows():
    s = int(r.seed)
    raw = c2[(c2.seed == s) & (c2.method == "no")].iloc[0].acc
    tent = c2[(c2.seed == s) & (c2.method == "tent")].iloc[0].acc
    body.append(f"{s} & {raw:.4f} & {r.macenko_accuracy:.4f} & {n_(r['help'])} & {n_(r.harm)} & {signed(r.net)} & {tent:.4f} \\\\")
write("macenko", tab("lcccccc", "Seed & Raw acc & Macenko acc & Helped & Harmed & Net & TENT acc \\\\", body))

# --------------------------------------------------------------- batch size
B = pd.read_csv(os.path.join(TAB, "aggregate_batchsize.csv")).sort_values(["seed", "batch"])
body = [f"{int(r.seed)}--{int(r.batch)} & {r.tent_accuracy:.4f} & {n_(r['help'])} & {n_(r.harm)} & {signed(r.net)} & {r.tent_ece_15:.4f} \\\\"
        for _, r in B.iterrows()]
write("batchsize", tab("lcccccc".replace("lcccccc", "lccccc"), "Seed--Batch & Accuracy & Helped & Harmed & Net & ECE-15 \\\\", body))

# ----------------------------------------------------------- target order
O = pd.read_csv(os.path.join(TAB, "aggregate_orders.csv")).sort_values("order_idx")
body = [f"perm{int(r.perm_seed)} & {r.tent_accuracy:.5f} & {r.tent_ece_15:.5f} & {n_(r['help'])} & {n_(r.harm)} & {signed(r.net)} \\\\"
        for _, r in O.iterrows()]
t1 = c2[(c2.seed == 1) & (c2.method == "tent")].iloc[0]
body.append(f"manifest & {t1.acc:.5f} & {t1.ece:.5f} & {n_(t1['help'])} & {n_(t1.harm)} & {signed(t1.net)} \\\\")
write("order_sens", tab("lccccc", "Order & Accuracy & ECE & Help & Harm & Net \\\\", body))

# ----------------------------------------------------------- TENT ablations
A = pd.read_csv(os.path.join(TAB, "aggregate_ablations.csv")).set_index("arm")
p0, y0, lab = load("center2", 1, "no")
p1, y1, _ = load("center2", 1, "tent")


def row(label, acc, auroc, nl, br, ece):
    return f"{label} & {acc:.4f} & {auroc:.4f} & {nl:.4f} & {br:.4f} & {ece:.4f} \\\\"


body = [row("No-TTA", st.accuracy(y0, lab), st.auroc(p0, lab), nll(p0, lab), brier(p0, lab), st.ece15(p0, y0, lab)),
        row("TENT (default)", st.accuracy(y1, lab), st.auroc(p1, lab), nll(p1, lab), brier(p1, lab), st.ece15(p1, y1, lab))]
for arm, label in [("lr1e-4", "LR $10^{-4}$"), ("lr5e-4", "LR $5{\\times}10^{-4}$"),
                   ("noclip", "No grad clip"), ("step2", "Two steps")]:
    a = A.loc[arm]
    body.append(row(label, a.tent_accuracy, a.tent_auroc, a.tent_nll, a.tent_brier, a.tent_ece_15))
write("tent_ablations", tab("lccccc", "Experiment & Accuracy & AUROC & NLL & Brier & ECE \\\\", body))
print("wrote 14 supp_*.tex tables to", OUT)

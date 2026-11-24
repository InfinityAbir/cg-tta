"""Cost-weighted net harm of TENT and the consensus gate on Center-2.

For every source seed we count false negatives (FN: tumor predicted normal) and
false positives (FP: normal predicted tumor) of the frozen model, TENT, and the
gated TENT (stored under the key safe_tent in the predictions). A cost model with
c_FP / c_FN = r then gives the change in total cost against the frozen model,
in units of one false negative:

    delta_cost(r) = (FN - FN_frozen) + r * (FP - FP_frozen)     (negative = better)

Outputs (CG_TTA_OUT overrides tables/):
    class_counts_by_seed.csv   FN/FP per seed and method
    class_cost_summary.csv     mean and SD of delta_cost per method and ratio
    tab_cost.tex               LaTeX tabular for the main paper
    tab_cost_seed.tex          per-seed FN/FP counts for the supplement

    python code/make_cost_table.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import cg_tta as st  # noqa: E402

OUT = os.environ.get("CG_TTA_OUT", os.path.join(REL, "tables"))
PRED = os.path.join(REL, "predictions", "center2")
os.makedirs(OUT, exist_ok=True)

# c_FP / c_FN. Values below 1 mean a missed tumor costs more than a false alarm.
RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
LABELS = ["1:10", "1:5", "1:2", "1:1", "2:1", "5:1", "10:1"]

rows = []
for seed in range(1, 6):
    d = {k: st.load_npz(os.path.join(PRED, f"seed{seed}", f"{k}.npz"))
         for k in ("no", "tent", "flip", "eata", "shot")}
    pack = st.build_pack(d)
    lab = d["no"]["labels"].astype(int)
    for name, key in [("frozen", "no"), ("tent", "tent"), ("gate", "safe_tent")]:
        pred = pack[key][1].astype(int)
        rows.append(dict(seed=seed, method=name,
                         fn=int(((lab == 1) & (pred == 0)).sum()),
                         fp=int(((lab == 0) & (pred == 1)).sum())))
C = pd.DataFrame(rows)
C.to_csv(os.path.join(OUT, "class_counts_by_seed.csv"), index=False)

base = C[C.method == "frozen"].set_index("seed")
out = []
for m in ("tent", "gate"):
    sub = C[C.method == m].set_index("seed")
    dfn = (sub.fn - base.fn).to_numpy(float)
    dfp = (sub.fp - base.fp).to_numpy(float)
    for r in RATIOS:
        dc = dfn + r * dfp
        out.append(dict(method=m, ratio_fp_over_fn=r, mean=dc.mean(),
                        sd=dc.std(ddof=1), dfn_mean=dfn.mean(), dfp_mean=dfp.mean()))
S = pd.DataFrame(out)
S.to_csv(os.path.join(OUT, "class_cost_summary.csv"), index=False)

# Break-even ratio from the five-seed mean counts.
t = S[(S.method == "tent")].iloc[0]
g = S[(S.method == "gate")].iloc[0]
be = -(t.dfn_mean - g.dfn_mean) / (t.dfp_mean - g.dfp_mean)


# Per-seed break-even: the ratio at which TENT and the gate have equal cost.
tt = C[C.method == "tent"].set_index("seed")
gt = C[C.method == "gate"].set_index("seed")
BE = pd.DataFrame({"breakeven": (gt.fn - tt.fn) / (tt.fp - gt.fp)})
BE.loc["mean_counts"] = be
BE.to_csv(os.path.join(OUT, "class_breakeven_by_seed.csv"))


def fmt(x):
    return f"${'-' if x < 0 else '+'}{abs(x):,.0f}$".replace(",", "{,}")


lines = [r"\begin{tabular}{@{}l" + "c" * len(RATIOS) + r"@{}}", r"\toprule",
         r"$c_{\mathrm{FP}}\!:\!c_{\mathrm{FN}}$ & " + " & ".join(LABELS) + r" \\", r"\midrule"]
for m, nm in (("tent", "TENT"), ("gate", "CG-TENT")):
    sub = S[S.method == m]
    lines.append(nm + " & " + " & ".join(fmt(v) for v in sub["mean"]) + r" \\")
best = []
for r in RATIOS:
    a = S[(S.method == "tent") & (S.ratio_fp_over_fn == r)]["mean"].iloc[0]
    b = S[(S.method == "gate") & (S.ratio_fp_over_fn == r)]["mean"].iloc[0]
    best.append("TENT" if a < b else "CG-TENT")
tg = C[C.method == "tent"].set_index("seed")
gg = C[C.method == "gate"].set_index("seed")
wins = []
for r in RATIOS:
    ct = (tg.fn - base.fn) + r * (tg.fp - base.fp)
    cg = (gg.fn - base.fn) + r * (gg.fp - base.fp)
    wins.append(int((cg < ct).sum()))
lines += [r"\midrule", "Lower mean cost & " + " & ".join(best) + r" \\",
          r"Seeds favoring CG-TENT & " + " & ".join(f"{w}/5" for w in wins) + r" \\",
          r"\bottomrule", r"\end{tabular}"]
with open(os.path.join(OUT, "tab_cost.tex"), "w", newline="\n") as fh:
    fh.write("\n".join(lines) + "\n")
# Per-seed counts for the supplement.
piv = C.pivot(index="seed", columns="method", values=["fn", "fp"])
sl = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
      r"Seed & \multicolumn{2}{c}{Frozen} & \multicolumn{2}{c}{TENT} & \multicolumn{2}{c}{CG-TENT} \\",
      r" & FN & FP & FN & FP & FN & FP \\", r"\midrule"]
for sd in piv.index:
    cells = []
    for m in ("frozen", "tent", "gate"):
        cells += [f"{piv.loc[sd, ('fn', m)]:,}".replace(",", "{,}"),
                  f"{piv.loc[sd, ('fp', m)]:,}".replace(",", "{,}")]
    sl.append(f"{sd} & " + " & ".join(cells) + r" \\")
sl += [r"\bottomrule", r"\end{tabular}"]
with open(os.path.join(OUT, "tab_cost_seed.tex"), "w", newline="\n") as fh:
    fh.write("\n".join(sl) + "\n")
print(S.to_string())
print(f"break-even c_FP/c_FN = {be:.2f}")

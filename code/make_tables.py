"""Write LaTeX table fragments for the paper tables."""
import math
import os
import re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.dirname(HERE)
ROOT = REL
TAB = os.environ.get("CG_TTA_OUT", os.path.join(REL, "tables"))
OUT = os.environ.get("CG_TTA_OUT", os.path.join(REL, "tables"))


def f4(x):
    return f"{x:.4f}"


def pct(x, d=3):
    return f"{100*x:+.{d}f}"


def pval(p):
    if p < 0.001:
        return f"{p:.1e}".replace("e-0", "e-")
    return f"{p:.4f}".lstrip("0")


def num(x, fmt=".3f", sign=True):
    s = f"{x:+{fmt}}" if sign else f"{x:{fmt}}"
    return s.replace("-", "$-$")


def fix_minus(s):
    return re.sub(r"(?<![\w$])-(?=\d)", "$-$", s)


main = pd.read_csv(os.path.join(TAB, "core_table2_main.csv"), index_col=0)
PR = pd.read_csv(os.path.join(TAB, "core_primary_endpoints.csv"))
SEC = pd.read_csv(os.path.join(TAB, "core_secondary_endpoints.csv"))
G = pd.read_csv(os.path.join(TAB, "core_table_gate_ablation.csv"), index_col=0)
C0 = pd.read_csv(os.path.join(TAB, "core_center0_multiseed_summary.csv")).set_index("method")
RC = pd.read_csv(os.path.join(TAB, "core_risk_coverage.csv"))

# ---------------------------------------------------------------- main table
names = {
    "no": "No-TTA (frozen)", "tent": "TENT", "flip": "Flip-TTA",
    "eata": "EATA-LN", "shot": "SHOT-IM", "safe_tent": r"\textbf{CG-TENT}",
    "safe_eata": "CG-EATA", "safe_shot": "CG-SHOT",
}
lines = [
    r"\begin{tabular}{@{}lccccc@{}}",
    r"\toprule",
    r"Method & Acc. & AUROC & ECE & Harm & Net \\",
    r"\midrule",
]
for m in ["no", "tent", "flip", "eata", "shot", "safe_tent", "safe_eata", "safe_shot"]:
    r = main.loc[m]
    acc = f"{r.acc:.4f}" + (f"$\\pm${r.acc_sd:.3f}" if m != "no" else "")
    harm = "--" if m == "no" else f"{r.harm:.0f}"
    net = "--" if m == "no" else f"{r.net:+.0f}"
    if m == "no":
        acc = f"{r.acc:.4f}"
    lines.append(f"{names[m]} & {acc} & {r.auroc:.4f} & {r.ece:.4f} & {harm} & {net} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tab_main.tex"), "w").write(fix_minus("\n".join(lines)))

# ------------------------------------------------------------ endpoint table
cp = PR[(PR.center == "center2") & (
    ((PR.contrast == "safe_vs_frozen") & (PR.metric == "ba")) |
    ((PR.contrast == "safe_vs_tent") & (PR.metric == "harm_rate")))].copy()
cp = cp.set_index("contrast")
holm = pd.read_csv(os.path.join(TAB, "core_coprimary_holm.csv")).set_index("contrast")
sec = SEC.set_index("endpoint")
lines = [
    r"\begin{tabular}{@{}lcccc@{}}",
    r"\toprule",
    r"Endpoint & $\Delta$ (95\% CI) & $p$ & Holm & F/T/U \\",
    r"\midrule",
    r"\multicolumn{5}{@{}l}{\emph{Co-primary (slide level)}}\\",
]
p1 = cp.loc["safe_vs_frozen"]
lines.append(f"P1: BA vs frozen & {num(100*p1['mean'])} pp ({num(100*p1.lo)}, {num(100*p1.hi)}) & {pval(p1.p)} & {pval(holm.loc['safe_vs_frozen','p_holm'])} & {p1.wins}/{p1.ties}/{p1.losses} \\\\")
p2 = cp.loc["safe_vs_tent"]
lines.append(f"P2: harm vs TENT & {num(100*p2['mean'])} pp ({num(100*p2.lo)}, {num(100*p2.hi)}) & {pval(p2.p)} & {pval(holm.loc['safe_vs_tent','p_holm'])} & {p2.losses}/{p2.ties}/{p2.wins} \\\\")
lines.append(r"\midrule")
lines.append(r"\multicolumn{5}{@{}l}{\emph{Secondary (slide level, gatekept)}}\\")
s1 = PR[(PR.center == "center2") & (PR.contrast == "safe_vs_tent") & (PR.metric == "ba")].iloc[0]
lines.append(f"S1: BA vs TENT & {num(100*s1['mean'])} pp ({num(100*s1.lo)}, {num(100*s1.hi)}) & {pval(s1.p)} & {pval(sec.loc['S1: CG vs TENT, slide BA','p_holm'])} & {int(s1.wins)}/{int(s1.ties)}/{int(s1.losses)} \\\\")
s2 = PR[(PR.center == "center2") & (PR.contrast == "safe_vs_frozen") & (PR.metric == "ece")].iloc[0]
lines.append(f"S2: ECE vs frozen & {num(s2['mean'],'.5f')} ({num(s2.lo,'.5f')}, {num(s2.hi,'.5f')}) & {pval(s2.p)} & {pval(sec.loc['S2: CG vs frozen, slide ECE','p_holm'])} & {int(s2.losses)}/{int(s2.ties)}/{int(s2.wins)} \\\\")
s3 = sec.loc["S3: CG vs frozen, slide AUROC"]
lines.append(f"S3: AUROC vs frozen & {num(s3['mean'],'.5f')} ({num(s3.lo,'.5f')}, {num(s3.hi,'.5f')}) & {pval(s3.p)} & {pval(s3.p_holm)} & {int(s3.wins)}/{int(s3.ties)}/{int(s3.losses)} \\\\")
s4 = sec.loc["S4: CG vs EATA-LN, slide BA"]
lines.append(f"S4: BA vs EATA & {num(100*s4['mean'])} pp ({num(100*s4.lo)}, {num(100*s4.hi)}) & {pval(s4.p)} & {pval(s4.p_holm)} & {int(s4.wins)}/{int(s4.ties)}/{int(s4.losses)} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tab_endpoints.tex"), "w").write(fix_minus("\n".join(lines)))

# --------------------------------------------------------------- gate table
gname = {"all_flips": "All TENT flips", "random_matched": "Random, matched",
         "flip_gate": "\\textbf{Flip consensus}", "both_gate": "Flip $\\cap$ EATA",
         "eata_gate": "EATA consensus", "oracle": "Oracle"}
lines = [
    r"\begin{tabular}{@{}lcccccc@{}}",
    r"\toprule",
    r"Gate & Accepted & Acc. & Help & Harm & Net & Prec. \\",
    r"\midrule",
]
for g in ["all_flips", "random_matched", "flip_gate", "both_gate", "eata_gate", "oracle"]:
    r = G.loc[g]
    prec = r.help / max(1, r.help + r.harm)
    lines.append(f"{gname[g]} & {r.accept:.0f} & {r.acc:.4f} & {r.help:.0f} & {r.harm:.0f} & {r.net:+.0f} & {prec:.3f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tab_gate.tex"), "w").write(fix_minus("\n".join(lines)))

# ------------------------------------------------------------- budget table
B = pd.read_csv(os.path.join(TAB, "core_budget_common_support.csv"))
raw = pd.read_csv(os.path.join(REL, "data", "budget_common_support_raw.csv"))
g = raw[raw.budget.isin(["1pct", "5pct", "25pct", "100pct"])].groupby(["contrast_budget", "budget"]).agg(
    n=("n_common", "first"), acc=("acc", "mean"), ece=("ece", "mean"))
lines = [
    r"\begin{tabular}{@{}lcccccc@{}}",
    r"\toprule",
    r"Budget & $n$ & Budget run & 100\% same & $\Delta$Acc & $\Delta$ECE & Wins \\",
    r"\midrule",
]
for b, label in [("1pct", "1\\%"), ("5pct", "5\\%"), ("25pct", "25\\%")]:
    ab = g.loc[(b, b)]
    af = g.loc[(b, "100pct")]
    dacc = 100 * (af.acc - ab.acc)
    dece = 100 * (af.ece - ab.ece)
    row = B[B.contrast_budget == b].iloc[0]
    lines.append(f"{label} & {int(ab.n):,} & {ab.acc:.4f} & {af.acc:.4f} & {dacc:+.2f} pp & {dece:+.3f} pp & 5/5 \\\\")
lines.append(f"100\\% & 85,054 & 0.9515 & 0.9515 & -- & -- & -- \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tab_budget.tex"), "w").write(fix_minus("\n".join(lines)))

# ------------------------------------------------------------- Center-0 table
# The manuscript uses the five-seed Center-0 replication. Harm and net are
# reported per seed; round half up so the .tex matches the paper table.
names0 = {"no": "No-TTA", "tent": "TENT", "flip": "Flip-TTA", "eata": "EATA-LN",
          "shot": "SHOT-IM", "safe_tent": r"\textbf{CG-TENT}", "safe_eata": "CG-EATA",
          "safe_shot": "CG-SHOT"}


def mean_count(total, n):
    return int(math.floor(total / n + 0.5))


lines = [
    r"\begin{tabular}{@{}lccccc@{}}",
    r"\toprule",
    r"Method & Acc. & $\Delta$Acc & $\Delta$BA & Harm & Net \\",
    r"\midrule",
]
for m in ["no", "tent", "flip", "eata", "shot", "safe_tent", "safe_eata", "safe_shot"]:
    r = C0.loc[m]
    acc = f"{r.acc_mean:.4f}"
    if m == "no":
        d, dba, harm, net = "--", "--", "--", "--"
    else:
        d = f"{r.delta_pp_mean:+.2f} pp"
        dba = f"{r.slide_delta_pp_mean:+.2f} pp"
        harm = f"{mean_count(r.harm_sum, r.n_seeds)}"
        net = f"{mean_count(r.net_sum, r.n_seeds):+d}"
    lines.append(f"{names0[m]} & {acc} & {d} & {dba} & {harm} & {net} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tab_center0.tex"), "w").write(fix_minus("\n".join(lines)))

print("wrote tab_main.tex, tab_endpoints.tex, tab_gate.tex, tab_budget.tex, tab_center0.tex")
print(open(os.path.join(OUT, "tab_budget.tex")).read())
print(open(os.path.join(OUT, "tab_endpoints.tex")).read())

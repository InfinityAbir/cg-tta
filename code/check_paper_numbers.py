"""Check that headline numbers quoted in the manuscript match the canonical CSVs.

Every value is read from the CSVs written by reproduce_tables.py /
twoway_bootstrap.py, formatted the way the prose formats it, and searched for
in main.tex and sections/*.tex.  A missing string means the prose and the
tables have drifted.  Usage:  python code/check_paper_numbers.py [paper_dir]
"""
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.dirname(HERE)
PAPER = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(REL)
TAB = os.path.join(REL, "tables")

text = ""
for f in [os.path.join(PAPER, "main.tex")] + glob.glob(os.path.join(PAPER, "sections", "*.tex")):
    text += open(f, encoding="utf8").read() + "\n"

main = pd.read_csv(os.path.join(TAB, "core_table2_main.csv")).set_index("method")
prim = pd.read_csv(os.path.join(TAB, "core_primary_endpoints.csv"))
two = pd.read_csv(os.path.join(TAB, "core_twoway_bootstrap.csv"))
holm = pd.read_csv(os.path.join(TAB, "core_coprimary_holm.csv"))

checks = []


def need(label, s):
    checks.append((label, s, s in text))


def pp(x, nd=3):
    return f"{abs(x) * 100:.{nd}f}"


need("frozen acc", f"{main.loc['no', 'acc']:.4f}")
need("TENT acc", f"{main.loc['tent', 'acc']:.4f}")
need("TENT harm", f"{main.loc['tent', 'harm']:.0f}")
need("CG-TENT harm", f"{main.loc['safe_tent', 'harm']:.1f}")
need("CG-SHOT harm", f"{main.loc['safe_shot', 'harm']:,.0f}".replace(",", "{,}"))
need("SHOT acc", f"{main.loc['shot', 'acc']:.4f}")

p1 = prim[(prim.center == "center2") & (prim.contrast == "safe_vs_frozen") & (prim.metric == "ba")].iloc[0]
p2 = prim[(prim.center == "center2") & (prim.contrast == "safe_vs_tent") & (prim.metric == "harm_rate")].iloc[0]
need("P1 mean", pp(p1["mean"]))
need("P1 lo", pp(p1.lo))
need("P1 hi", pp(p1.hi))
need("P2 mean", pp(p2["mean"]))
need("P2 lo", pp(p2.hi))      # abs of the upper (less negative) bound
need("P2 hi", pp(p2.lo))
need("Holm p", f"{holm.p_holm.iloc[0]:.4f}")

t1 = two[(two.contrast == "safe_vs_frozen") & (two.metric == "ba")].iloc[0]
t2 = two[(two.contrast == "safe_vs_tent") & (two.metric == "harm_rate")].iloc[0]
need("P1 two-way lo", pp(t1.twoway_lo))
need("P1 two-way hi", pp(t1.twoway_hi))
need("P2 two-way lo", pp(t2.twoway_hi))
need("P2 two-way hi", pp(t2.twoway_lo))

bad = [c for c in checks if not c[2]]
for label, s, ok in checks:
    print(("OK   " if ok else "MISS ") + f"{label:16s} {s}")
print(f"{len(checks) - len(bad)}/{len(checks)} headline numbers found in the manuscript")
sys.exit(1 if bad else 0)

"""Second benchmark (NCT-CRC-HE-100K -> CRC-VAL-HE-7K): per-seed and mean +- SD numbers.

    python jbhi_revision/analyze_crc.py

Reads crc_second_dataset/tta (fixed random stream) and crc_second_dataset/tta_sorted (class-sorted
stream), writes jbhi_revision/tables/crc_numbers.json and supp_crc.tex.
Help = frozen wrong -> adapted right (W->C). Harm = frozen right -> adapted wrong (C->W).
CG-TENT: accept a TENT flip only when the Flip-TTA critic agrees with TENT, otherwise keep the frozen label.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "crc_second_dataset")
OUT = os.path.join(HERE, "tables")
SEEDS = [1, 2, 3, 4, 5]
METHODS = ["no_tta", "tent", "flip_tta", "eata", "shot"]
NAMES = {"no_tta": "No-TTA", "tent": "TENT", "flip_tta": "Flip-TTA", "eata": "EATA-LN", "shot": "SHOT-IM",
         "cg_tent": "CG-TENT"}
BS = chr(92)  # backslash, kept out of string literals on purpose


def bal_acc(pred, lab, k=9):
    r = []
    for c in range(k):
        m = lab == c
        if m.any():
            r.append((pred[m] == c).mean())
    return float(np.mean(r))


def load(stream, seed, method):
    z = np.load(os.path.join(SRC, stream, "seed%d" % seed, method + "_predictions.npz"))
    return z["predictions"].astype(int), z["labels"].astype(int), z["probabilities"]


def evaluate(stream):
    rows = {m: [] for m in METHODS + ["cg_tent"]}
    extra = {"tent_flips": [], "accepted": []}
    for s in SEEDS:
        p0, lab, _ = load(stream, s, "no_tta")
        preds = {"no_tta": p0}
        for m in METHODS[1:]:
            preds[m], _, _ = load(stream, s, m)
        acc_flip = (preds["tent"] != p0) & (preds["flip_tta"] == preds["tent"])
        preds["cg_tent"] = np.where(acc_flip, preds["tent"], p0)
        extra["tent_flips"].append(int((preds["tent"] != p0).sum()))
        extra["accepted"].append(int(acc_flip.sum()))
        for m, p in preds.items():
            c0, c1 = p0 == lab, p == lab
            rows[m].append(dict(acc=float((p == lab).mean()), ba=bal_acc(p, lab),
                                help=int((~c0 & c1).sum()), harm=int((c0 & ~c1).sum())))
    return rows, extra


def ms(vals):
    v = np.asarray(vals, float)
    return float(v.mean()), float(v.std(ddof=1))


def write_table(res):
    """One-column supplement table: mean +- SD over five source seeds, both stream orders."""
    def pm(d, scale=1.0, nd=2):
        return "%.*f$%spm$%.*f" % (nd, scale * d["mean"], BS, nd, scale * d["sd"])

    top = [BS + "begin{tabular}{lcccccc}", BS + "toprule",
           " & " + BS + "multicolumn{3}{c}{Random order} & " + BS + "multicolumn{3}{c}{Sorted order}" + BS + BS,
           BS + "cmidrule(lr){2-4}" + BS + "cmidrule(lr){5-7}",
           "Method & Acc (" + BS + "%) & Help & Harm & Acc (" + BS + "%) & Help & Harm" + BS + BS,
           BS + "midrule"]
    lines = list(top)
    for m in METHODS + ["cg_tent"]:
        row = [NAMES[m]]
        for tag in ("shuffled", "sorted"):
            s = res[tag]["summary"][m]
            row += [pm(s["acc"], 100)]
            row += ["--", "--"] if m == "no_tta" else [pm(s["help"], 1, 1), pm(s["harm"], 1, 1)]
        lines.append(" & ".join(row) + BS + BS)
        if m == "shot":
            lines.append(BS + "midrule")
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    with open(os.path.join(OUT, "supp_crc.tex"), "w") as f:
        f.write(chr(10).join(lines) + chr(10))


def main():
    res = {}
    for tag, stream in (("shuffled", "tta"), ("sorted", "tta_sorted")):
        rows, extra = evaluate(stream)
        summ = {}
        for m, r in rows.items():
            summ[m] = {k: dict(zip(("mean", "sd"), ms([x[k] for x in r]))) for k in ("acc", "ba", "help", "harm")}
            summ[m]["net_mean"] = float(np.mean([x["help"] - x["harm"] for x in r]))
            summ[m]["per_seed"] = r
        res[tag] = dict(summary=summ, extra=extra)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "crc_numbers.json"), "w") as f:
        json.dump(res, f, indent=1)
    for tag in res:
        print("==", tag)
        for m in METHODS + ["cg_tent"]:
            s = res[tag]["summary"][m]
            print("%-9s acc %.2f+-%.2f  ba %.2f+-%.2f  help %.1f+-%.1f  harm %.1f+-%.1f  net %.1f" % (
                NAMES[m], 100 * s["acc"]["mean"], 100 * s["acc"]["sd"], 100 * s["ba"]["mean"], 100 * s["ba"]["sd"],
                s["help"]["mean"], s["help"]["sd"], s["harm"]["mean"], s["harm"]["sd"], s["net_mean"]))
        e = res[tag]["extra"]
        print("tent flips per seed", e["tent_flips"], "accepted by critic", e["accepted"])
    write_table(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())

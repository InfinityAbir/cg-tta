"""Regenerate the Center-0, full-EATA and Phikon CSVs.

These extension tables are included in the 10-page manuscript:

* Center-0 replication on five source seeds.
* Full-parameter EATA on Center-2, including the CG-TTA gate.
* The frozen Phikon linear probe.

Every row is derived from the npz streams with the same gate and metric code
used by the paper (`cg_tta.py`).  Run from anywhere:

    python code/make_postsubmission_tables.py
    CG_TTA_OUT=/tmp/out python code/make_postsubmission_tables.py
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
os.makedirs(OUT, exist_ok=True)

CENTER0_SEEDS = (1, 2, 3, 4, 5)
METHODS = ["no", "tent", "flip", "eata", "shot",
           "safe_tent", "safe_eata", "safe_shot"]


def paths_center0(seed):
    d = os.path.join(REL, "predictions", "center0", f"seed{seed}")
    return {m: os.path.join(d, f"{m}.npz") for m in ("no", "tent", "flip", "eata", "shot")}


def paths_eata_full(seed):
    d = os.path.join(REL, "predictions", "eata_full", f"seed{seed}")
    return {"no": os.path.join(d, "no.npz"), "eata": os.path.join(d, "eata.npz")}


def check_same(ref, other, what):
    for key in ("patch_ids", "labels", "slides"):
        if not np.array_equal(ref[key], other[key]):
            raise AssertionError(f"{what}: {key} mismatch")


def seed_rows(center, seed, pack, lab, slides, frozen_acc):
    rows, per_slide = [], []
    y0 = pack["no"][1]
    for name in METHODS:
        prob, pred = pack[name]
        m = st.method_metrics(prob, pred, lab, y0)
        rows.append(dict(center=center, seed=seed, method=name, n=int(len(lab)),
                         **m, delta_pp_vs_frozen=100.0 * (m["acc"] - frozen_acc)))
        for s in np.unique(slides):
            mask = slides == s
            ms = st.method_metrics(prob[mask], pred[mask], lab[mask], y0[mask])
            per_slide.append(dict(center=center, seed=seed, slide=int(s), method=name,
                                  n=int(mask.sum()), **ms))
    return rows, per_slide


# ---------------------------------------------------------------- Center-0
seed_metrics, slide_metrics = [], []
ref = None
for seed in CENTER0_SEEDS:
    d = {m: st.load_npz(p) for m, p in paths_center0(seed).items()}
    for m in d:
        check_same(d["no"], d[m], f"center0 seed{seed} {m}")
    if ref is None:
        ref = d["no"]
    else:
        check_same(ref, d["no"], f"center0 seed{seed} target set")
    assert len(np.unique(d["no"]["slides"])) == 10
    pack = st.build_pack(d)
    frozen_acc = st.accuracy(pack["no"][1], d["no"]["labels"])
    r, p = seed_rows("center0", seed, pack, d["no"]["labels"], d["no"]["slides"], frozen_acc)
    seed_metrics += r
    slide_metrics += p

S = pd.DataFrame(seed_metrics)
P = pd.DataFrame(slide_metrics)
S["method"] = pd.Categorical(S["method"], METHODS, ordered=True)
P["method"] = pd.Categorical(P["method"], METHODS, ordered=True)
S = S.sort_values(["seed", "method"]).reset_index(drop=True)
P = P.sort_values(["seed", "slide", "method"]).reset_index(drop=True)
# Slide deltas must be within the same slide, not against the patch-level
# frozen accuracy: slides differ substantially in difficulty.
frozen_slide = (P[P.method == "no"][["seed", "slide", "acc"]]
                .rename(columns={"acc": "frozen_slide_acc"}))
P = P.merge(frozen_slide, on=["seed", "slide"], how="left")
P["delta_pp_vs_frozen"] = 100.0 * (P.acc - P.frozen_slide_acc)
S.to_csv(os.path.join(OUT, "core_center0_multiseed_seed_metrics.csv"), index=False)
P.to_csv(os.path.join(OUT, "core_center0_multiseed_slide_metrics.csv"), index=False)

frozen = S[S.method == "no"][["seed", "acc"]].set_index("seed")["acc"]
rows = []
for m in METHODS:
    g = S[S.method == m]
    pg = P[P.method == m]
    rows.append(dict(
        method=m, n_seeds=int(g.seed.nunique()),
        acc_mean=float(g.acc.mean()), acc_sd=float(g.acc.std(ddof=1)),
        delta_pp_mean=float(g.delta_pp_vs_frozen.mean()),
        slide_acc_mean=float(pg.acc.mean()),
        slide_delta_pp_mean=float(pg.delta_pp_vs_frozen.mean()),
        help_sum=int(g.help.sum()), harm_sum=int(g.harm.sum()),
        net_sum=int(g.net.sum()),
        positive_seeds=int((g.delta_pp_vs_frozen > 0).sum()),
    ))
C0 = pd.DataFrame(rows)
C0.to_csv(os.path.join(OUT, "core_center0_multiseed_summary.csv"), index=False)
print("CENTER-0, 4 seeds, patch level")
print(C0.round(5).to_string(index=False))
print("\nCENTER-0, 4 seeds, slide level")
print(C0[["method", "slide_acc_mean", "slide_delta_pp_mean"]].round(5).to_string(index=False))

# ------------------------------------------------------------ full EATA
rows = []
for seed in (1, 2, 3):
    d0 = st.load_npz(paths_eata_full(seed)["no"])
    d1 = st.load_npz(paths_eata_full(seed)["eata"])
    df = st.load_npz(os.path.join(REL, "predictions", "center2", f"seed{seed}", "flip.npz"))
    check_same(d0, d1, f"eata_full seed{seed} adapter")
    check_same(d0, df, f"eata_full seed{seed} critic")
    y0 = d0["predictions"].astype(int)
    y1 = d1["predictions"].astype(int)
    yf = df["predictions"].astype(int)
    lab = d0["labels"].astype(int)
    probs, accept, _y0, _y1 = st.consensus_gate(
        d0["probabilities"], d1["probabilities"], df["probabilities"])
    y_safe = probs.argmax(1)
    gate_labels, accept2 = st.apply_gate(y0, y1, yf)
    assert np.array_equal(accept, accept2)
    assert np.array_equal(y_safe, gate_labels)
    base = st.method_metrics(d0["probabilities"], y0, lab, y0)
    eata = st.method_metrics(d1["probabilities"], y1, lab, y0)
    safe = st.method_metrics(probs, y_safe, lab, y0)
    accepted_correct = int((accept & (y1 == lab)).sum())
    rows.append(dict(
        seed=seed,
        frozen_acc=base["acc"], eata_acc=eata["acc"], safe_acc=safe["acc"],
        eata_minus_frozen_pp=100.0 * (eata["acc"] - base["acc"]),
        safe_minus_frozen_pp=100.0 * (safe["acc"] - base["acc"]),
        safe_minus_eata_pp=100.0 * (safe["acc"] - eata["acc"]),
        eata_help=eata["help"], eata_harm=eata["harm"],
        safe_help=safe["help"], safe_harm=safe["harm"],
        accepted=int(accept.sum()), accepted_correct=accepted_correct,
        precision=float(accepted_correct / max(1, accept.sum())),
        help_recall=float(safe["help"] / max(1, eata["help"])),
        harm_reject=float(1.0 - safe["harm"] / max(1, eata["harm"])),
    ))
E = pd.DataFrame(rows)
E.to_csv(os.path.join(OUT, "core_eata_full_seed_metrics.csv"), index=False)
summary = dict(
    n_seeds=int(len(E)),
    frozen_acc_mean=float(E.frozen_acc.mean()),
    eata_acc_mean=float(E.eata_acc.mean()),
    safe_acc_mean=float(E.safe_acc.mean()),
    eata_minus_frozen_pp=float(E.eata_minus_frozen_pp.mean()),
    safe_minus_frozen_pp=float(E.safe_minus_frozen_pp.mean()),
    safe_minus_eata_pp=float(E.safe_minus_eata_pp.mean()),
    eata_help_sum=int(E.eata_help.sum()), eata_harm_sum=int(E.eata_harm.sum()),
    safe_help_sum=int(E.safe_help.sum()), safe_harm_sum=int(E.safe_harm.sum()),
    accepted_sum=int(E.accepted.sum()),
    accepted_correct_sum=int(E.accepted_correct.sum()),
    precision=float(E.accepted_correct.sum() / max(1, E.accepted.sum())),
    help_recall=float(E.safe_help.sum() / max(1, E.eata_help.sum())),
    harm_reject=float(1.0 - E.safe_harm.sum() / max(1, E.eata_harm.sum())),
)
ES = pd.DataFrame([summary])
ES.to_csv(os.path.join(OUT, "core_eata_full_summary.csv"), index=False)
print("\nFULL-PARAMETER EATA, per seed")
print(E.round(4).to_string(index=False))
print("\nFULL-PARAMETER EATA, mean")
print(ES.round(4).to_string(index=False))

# ------------------------------------------------------------------ Phikon
probe_path = os.path.join(REL, "logs", "phikon_probe.json")
with open(probe_path) as fh:
    probe = json.load(fh)
PH = pd.DataFrame([{
    "encoder": probe["encoder"], "head": probe["head"], "stage": probe["stage"],
    "seed": probe["seed"], "best_epoch": probe["best_epoch"],
    "n": probe["n"], "accuracy": probe["accuracy"],
    "balanced_accuracy": probe["balanced_accuracy"], "auroc": probe["auroc"],
    "nll": probe["nll"], "brier": probe["brier"], "ece_15": probe["ece_15"],
    "val_ba": probe["val_ba"],
}])
PH.to_csv(os.path.join(OUT, "core_phikon_probe.csv"), index=False)
print("\nPHIKON PROBE")
print(PH.round(4).to_string(index=False))
print(f"\nwrote 6 CSVs to {OUT}")

"""Revision analyses for the CG-TTA paper (archived analysis record).

Everything here is derived from the saved per-seed prediction files and the
fixed Center-2 manifest. No new model runs are performed.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"E:\Temp\Next Research")
PAPER = Path(os.environ.get("CG_TTA_PAPER_DIR", str(ROOT / "paper")))  # directory holding tables/
EXP = ROOT / "03_Experiments" / "06_Five_Seed_Evaluation_Fixed"
MANIFEST = ROOT / "Camelyon17_WILDS_Local_Archive" / "03_Experiments" / "05_TENT_Target_Stream" / "target_budgets" / "target_100pct_seed1.csv"
OUT = PAPER / "tables"


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    out = {}
    for k in ["probabilities", "predictions", "labels", "patch_ids", "confidences", "entropies"]:
        out[k] = z[k]
    out["patch_ids"] = np.array([int(x) for x in out["patch_ids"]], dtype=np.int64)
    return out


def seed_frame(seed):
    base = EXP / f"seed{seed}"
    frames = {}
    for method, fname in [("frozen", "no_tta_predictions.npz"),
                          ("tent", "tent_predictions.npz"),
                          ("flip", "flip_tta_predictions.npz")]:
        d = load_npz(base / fname)
        df = pd.DataFrame({
            "patch_id": d["patch_ids"],
            "label": d["labels"].astype(int),
            "pred": d["predictions"].astype(int),
            "p1": d["probabilities"][:, 1].astype(float),
            "conf": d["confidences"].astype(float),
        })
        frames[method] = df
    m = frames["frozen"].merge(frames["tent"], on="patch_id", suffixes=("_frozen", "_tent"))
    m = m.merge(frames["flip"], on="patch_id")
    m = m.rename(columns={"label": "label", "pred": "pred_flip", "p1": "p1_flip", "conf": "conf_flip"})
    # The merge above leaves label_frozen and label_tent; they must agree.
    assert (m["label_frozen"] == m["label_tent"]).all()
    m["label"] = m["label_frozen"].astype(int)
    m = m.drop(columns=["label_frozen", "label_tent"])
    m["seed"] = seed
    return m


def add_manifest(df):
    man = pd.read_csv(MANIFEST, usecols=["patch_id", "slide", "tumor", "patient"])
    man["patch_id"] = man["patch_id"].astype(np.int64)
    out = df.merge(man, on="patch_id", how="left", validate="many_to_one")
    assert out["slide"].notna().all(), "unmapped patches"
    return out


def derive(df):
    y0 = df["pred_frozen"].to_numpy()
    y1 = df["pred_tent"].to_numpy()
    yf = df["pred_flip"].to_numpy()
    p0 = df["p1_frozen"].to_numpy()
    p1 = df["p1_tent"].to_numpy()
    pf = df["p1_flip"].to_numpy()
    conf0 = df["conf_frozen"].to_numpy()
    label = df["label"].to_numpy()

    df["accept_gate"] = (y0 != y1) & (yf == y1)
    df["pred_gate"] = np.where(df["accept_gate"], y1, y0)
    df["p1_gate"] = np.where(df["accept_gate"], p1, p0)

    # 2-of-3 majority, frozen breaks a three-way disagreement.
    maj = np.where(y1 == yf, y1, np.where(y0 == y1, y1, np.where(y0 == yf, y0, y0)))
    df["pred_majority"] = maj

    # Probability averaging.
    pbar = (p0 + p1 + pf) / 3.0
    df["pred_probsum"] = (pbar >= 0.5).astype(int)

    # Confidence-threshold gate at the same acceptance count as the Flip gate.
    flips = np.flatnonzero(y0 != y1)
    k = int(df["accept_gate"].sum())
    order = flips[np.argsort(-conf0[flips], kind="stable")]
    accept_thr = np.zeros(len(df), dtype=bool)
    accept_thr[order[:k]] = True
    df["accept_thr"] = accept_thr
    df["pred_thr"] = np.where(accept_thr, y1, y0)

    # Random gate, matched per seed; use a fixed RNG for reproducibility.
    rng = np.random.default_rng(20261025 + int(df["seed"].iloc[0]))
    accept_rand = np.zeros(len(df), dtype=bool)
    accept_rand[rng.choice(flips, size=k, replace=False)] = True
    df["accept_rand"] = accept_rand
    df["pred_rand"] = np.where(accept_rand, y1, y0)

    # Oracle gate: accept only helpful flips.
    accept_oracle = (y0 != y1) & (y1 == label)
    df["accept_oracle"] = accept_oracle
    df["pred_oracle"] = np.where(accept_oracle, y1, y0)

    return df


def method_stats(df, pred_col, name):
    y = df["label"].to_numpy()
    p = df[pred_col].to_numpy()
    correct = p == y
    frozen_correct = df["pred_frozen"].to_numpy() == y
    tent_correct = df["pred_tent"].to_numpy() == y
    help_ = (~frozen_correct) & correct
    harm = frozen_correct & (~correct)
    return {
        "method": name,
        "n": len(df),
        "acc": float(correct.mean()),
        "help": int(help_.sum()),
        "harm": int(harm.sum()),
        "net": int(help_.sum() - harm.sum()),
        "tp": int(((p == 1) & (y == 1)).sum()),
        "tn": int(((p == 0) & (y == 0)).sum()),
        "fp": int(((p == 1) & (y == 0)).sum()),
        "fn": int(((p == 0) & (y == 1)).sum()),
    }


def class_harm(df, pred_col):
    y = df["label"].to_numpy()
    p = df[pred_col].to_numpy()
    f = df["pred_frozen"].to_numpy()
    frozen_correct = f == y
    harm = frozen_correct & (p != y)
    help_ = (~frozen_correct) & (p == y)
    return {
        "harm_tumor_to_normal": int((harm & (y == 1)).sum()),
        "harm_normal_to_tumor": int((harm & (y == 0)).sum()),
        "help_normal_to_tumor": int((help_ & (y == 0)).sum()),
        "help_tumor_to_normal": int((help_ & (y == 1)).sum()),
        "n_tumor": int((y == 1).sum()),
        "n_normal": int((y == 0).sum()),
    }


def slide_table(df):
    rows = []
    for slide, g in df.groupby("slide"):
        label = int(g["tumor"].max())
        rows.append({
            "slide": int(slide),
            "n": len(g),
            "tumor_frac": float(g["tumor"].mean()),
            "slide_label": label,
            "p_frozen": float(g["p1_frozen"].mean()),
            "p_tent": float(g["p1_tent"].mean()),
            "p_gate": float(g["p1_gate"].mean()),
            "pred_frozen": int(float(g["p1_frozen"].mean()) >= 0.5),
            "pred_tent": int(float(g["p1_tent"].mean()) >= 0.5),
            "pred_gate": int(float(g["p1_gate"].mean()) >= 0.5),
        })
    return pd.DataFrame(rows)


def main():
    all_stats = []
    all_class = []
    slide_stats = []
    for seed in range(1, 6):
        df = add_manifest(seed_frame(seed))
        df = derive(df)
        for col, name in [("pred_frozen", "frozen"), ("pred_tent", "tent"),
                          ("pred_gate", "gate"), ("pred_majority", "majority"),
                          ("pred_probsum", "probsum"), ("pred_thr", "threshold"),
                          ("pred_rand", "random"), ("pred_oracle", "oracle")]:
            s = method_stats(df, col, name)
            s["seed"] = seed
            all_stats.append(s)
        for col, name in [("pred_tent", "tent"), ("pred_gate", "gate")]:
            c = class_harm(df, col)
            c["seed"] = seed
            c["method"] = name
            all_class.append(c)
        st = slide_table(df)
        st["seed"] = seed
        slide_stats.append(st)

    stats = pd.DataFrame(all_stats)
    cls = pd.DataFrame(all_class)
    slides = pd.concat(slide_stats, ignore_index=True)

    # Aggregate over seeds.
    agg = stats.groupby("method").agg(
        acc_mean=("acc", "mean"), acc_sd=("acc", "std"),
        help_mean=("help", "mean"), harm_mean=("harm", "mean"),
        net_mean=("net", "mean"), net_sd=("net", "std"),
        tp_mean=("tp", "mean"), tn_mean=("tn", "mean"),
        fp_mean=("fp", "mean"), fn_mean=("fn", "mean"),
    ).reset_index()
    agg["precision"] = agg["help_mean"] / (agg["help_mean"] + agg["harm_mean"])
    agg["sensitivity"] = agg["tp_mean"] / (agg["tp_mean"] + agg["fn_mean"])
    agg["specificity"] = agg["tn_mean"] / (agg["tn_mean"] + agg["fp_mean"])
    agg["ppv"] = agg["tp_mean"] / (agg["tp_mean"] + agg["fp_mean"])
    agg["npv"] = agg["tn_mean"] / (agg["tn_mean"] + agg["fn_mean"])

    cls_agg = cls.groupby("method").agg(
        harm_t2n_mean=("harm_tumor_to_normal", "mean"),
        harm_n2t_mean=("harm_normal_to_tumor", "mean"),
        help_n2t_mean=("help_normal_to_tumor", "mean"),
        help_t2n_mean=("help_tumor_to_normal", "mean"),
    ).reset_index()
    cls_agg["harm_t2n_sd"] = cls.groupby("method")["harm_tumor_to_normal"].std().values
    cls_agg["harm_n2t_sd"] = cls.groupby("method")["harm_normal_to_tumor"].std().values

    # Slide-level diagnostic performance (mean probability per slide, threshold .5).
    slide_rows = []
    for seed, g in slides.groupby("seed"):
        for method in ["frozen", "tent", "gate"]:
            y = g["slide_label"].to_numpy()
            p = g[f"pred_{method}"].to_numpy()
            tp = int(((p == 1) & (y == 1)).sum())
            tn = int(((p == 0) & (y == 0)).sum())
            fp = int(((p == 1) & (y == 0)).sum())
            fn = int(((p == 0) & (y == 1)).sum())
            slide_rows.append({"seed": seed, "method": method, "n_slides": len(g),
                               "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                               "sens": tp / max(tp + fn, 1), "spec": tn / max(tn + fp, 1),
                               "acc": (tp + tn) / len(g)})
    slide_perf = pd.DataFrame(slide_rows)

    # Leave-one-slide-out patch accuracy differences.
    loso = []
    for seed in range(1, 6):
        df = add_manifest(seed_frame(seed))
        df = derive(df)
        for slide in sorted(df["slide"].unique()):
            keep = df[df["slide"] != slide]
            y = keep["label"].to_numpy()
            loso.append({
                "seed": seed, "left_out": int(slide), "n": len(keep),
                "tent_minus_frozen": float((keep["pred_tent"].to_numpy() == y).mean() - (keep["pred_frozen"].to_numpy() == y).mean()),
                "gate_minus_frozen": float((keep["pred_gate"].to_numpy() == y).mean() - (keep["pred_frozen"].to_numpy() == y).mean()),
            })
    loso = pd.DataFrame(loso)

    stats.to_csv(OUT / "revision_method_stats.csv", index=False)
    agg.to_csv(OUT / "revision_method_summary.csv", index=False)
    cls.to_csv(OUT / "revision_class_harm.csv", index=False)
    cls_agg.to_csv(OUT / "revision_class_harm_summary.csv", index=False)
    slides.to_csv(OUT / "revision_slide_diagnosis.csv", index=False)
    slide_perf.to_csv(OUT / "revision_slide_performance.csv", index=False)
    loso.to_csv(OUT / "revision_leave_one_slide_out.csv", index=False)

    print("=== METHOD SUMMARY ===")
    print(agg.to_string(index=False))
    print("\n=== CLASS-CONDITIONAL HARM ===")
    print(cls_agg.to_string(index=False))
    print("\n=== SLIDE-LEVEL DIAGNOSIS (per-seed means) ===")
    print(slide_perf.groupby("method")[["sens", "spec", "acc"]].agg(["mean", "std"]).to_string())
    print("\n=== LEAVE-ONE-SLIDE-OUT RANGE ===")
    print(loso.groupby("seed")[["tent_minus_frozen", "gate_minus_frozen"]].agg(["min", "max"]).to_string())


if __name__ == "__main__":
    main()

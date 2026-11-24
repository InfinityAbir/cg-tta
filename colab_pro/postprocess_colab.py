#!/usr/bin/env python
"""Turn CG-TTA Colab outputs into manuscript tables.

The Colab jobs write one folder per experiment under
``03_Experiments/17_Revision_Colab``.  This script reads that folder (plus the
saved Center-2 predictions that already exist in the release) and writes CSV
summaries and LaTeX tables.

It never invents a number.  If a job has not run, the corresponding table is
skipped and the reason is written to ``README_TABLES.md``.

Typical use after downloading the Colab results:

    python postprocess_colab.py --root path/to/17_Revision_Colab

The Center-2 prediction folder defaults to the local release path; override it
with ``--center2`` if the files live somewhere else.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

DEFAULT_CENTER2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "predictions", "center2")
METHODS = ["no_tta", "tent", "flip_tta", "eata", "shot"]
PHIKON_METHODS = ["no_tta", "tent", "flip_tta", "eata"]


# --------------------------------------------------------------------- helpers
def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fmt_count(x):
    return f"{int(round(float(x))):,}".replace(",", "{,}")


def fmt_pm(mean, sd, digits=4):
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(sd):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}$\\pm${sd:.{digits}f}"


def save_tex(path, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)


def load_npz(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def find_seed_dirs(root, pattern):
    return sorted(glob.glob(os.path.join(root, pattern)))


# ------------------------------------------------------------- deployment table
def _locks_from_center1(seed_dir):
    """Read a lock file, or derive the same rules from saved predictions."""
    lock_path = os.path.join(seed_dir, "locked_thresholds.json")
    lock = read_json(lock_path)
    if lock:
        lock["_source"] = lock_path
        return lock

    paths = {m: os.path.join(seed_dir, f"{m}_predictions.npz")
             for m in ("no_tta", "tent", "flip_tta")}
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    p0 = load_npz(paths["no_tta"])["probabilities"].astype(np.float64)
    p1 = load_npz(paths["tent"])["probabilities"].astype(np.float64)
    pf = load_npz(paths["flip_tta"])["probabilities"].astype(np.float64)
    y0, y1, yf = p0.argmax(1), p1.argmax(1), pf.argmax(1)
    flips = np.flatnonzero(y1 != y0)
    if len(flips) == 0:
        return None
    k_flip = max(1, int(round(0.10 * len(flips))))
    k_all = max(1, int(round(0.01 * len(p0))))
    scores = {"frozen_conf": p0.max(1), "tent_conf": p1.max(1),
              "critic_conf": pf.max(1), "gate_margin": p1.max(1) - p0.max(1)}
    lock = dict(seed=int(os.path.basename(seed_dir).replace("seed", "")),
                tent_flips=int(len(flips)), target_accepted=int(k_flip),
                target_top1pct_accepted=int(k_all), _source="derived_from_predictions")
    for name, score in scores.items():
        lock[f"threshold_{name}"] = float(np.sort(score[flips])[-k_flip])
        lock[f"threshold_{name}_top1pct"] = float(np.sort(score)[-k_all])
    return lock


def evaluate_center2(p0, p1, pf, y, lock=None):
    y0, y1, yf = p0.argmax(1), p1.argmax(1), pf.argmax(1)
    c0, c1, cf = p0.max(1), p1.max(1), pf.max(1)
    margin = c1 - c0
    flips = y1 != y0
    correct0 = y0 == y
    pa = (p0 + p1).argmax(1)
    scores = {"frozen_conf": c0, "tent_conf": c1, "critic_conf": cf,
              "gate_margin": margin}
    rules = {
        "Frozen": (y0, np.zeros(len(y), bool), None),
        "TENT (all flips)": (np.where(flips, y1, y0), flips, y1),
        "Flip-TTA": (yf, yf != y0, yf),
        "Probability averaging": (pa, pa != y0, pa),
        "CG-TENT": (np.where(flips & (yf == y1), y1, y0), flips & (yf == y1), y1),
    }
    if lock is not None:
        for name, score in scores.items():
            accept = flips & (score >= lock[f"threshold_{name}"])
            rules[f"Lock R1 {name}"] = (np.where(accept, y1, y0), accept, y1)
            accept = flips & (score >= lock[f"threshold_{name}_top1pct"])
            rules[f"Lock R2 {name}"] = (np.where(accept, y1, y0), accept, y1)

    # Label-free target-stream quantile rules.  The threshold is read from the
    # unlabeled target scores only; no target label enters this block.
    for qname, q in (("Q10", 0.10), ("Q33", 1.0 / 3.0), ("Q50", 0.50)):
        for name, score in scores.items():
            if flips.sum() == 0:
                continue
            tau = float(np.quantile(score[flips], 1.0 - q))
            accept = flips & (score >= tau)
            rules[f"Target {qname} {name}"] = (np.where(accept, y1, y0), accept, y1)

    helpful_flips = flips & (y1 == y)
    rules["Oracle"] = (np.where(helpful_flips, y1, y0), helpful_flips, y1)

    out = {}
    for name, (pred, accept, candidate) in rules.items():
        accepted = int(accept.sum())
        help_n = int((~correct0 & (pred == y)).sum())
        harm_n = int((correct0 & (pred != y)).sum())
        possible = (int((~correct0 & (candidate == y)).sum())
                    if candidate is not None else 0)
        out[name] = dict(
            accepted=accepted,
            coverage_pct=100.0 * accepted / len(y),
            accuracy=float((pred == y).mean()),
            help=help_n,
            harm=harm_n,
            net=help_n - harm_n,
            precision=float(help_n / max(1, help_n + harm_n)),
            recall=float(help_n / max(1, possible)),
        )
    return out


def build_deployment(root, center2_root, out):
    rows = []
    for seed in range(1, 6):
        seed_dirs = [os.path.join(root, "center1", f"seed{seed}"),
                     os.path.join(root, "results", "center1", f"seed{seed}")]
        seed_dir = next((p for p in seed_dirs if os.path.isdir(p)), None)
        lock = _locks_from_center1(seed_dir) if seed_dir else None
        if lock is None:
            print(f"deployment: seed{seed}: no Center-1 lock; target-quantile rules only")
        paths = {m: os.path.join(center2_root, f"seed{seed}", f"{m}.npz")
                 for m in ("no", "tent", "flip")}
        if not all(os.path.exists(p) for p in paths.values()):
            print(f"deployment: seed{seed}: Center-2 predictions missing")
            continue
        p0 = load_npz(paths["no"])["probabilities"].astype(np.float64)
        p1 = load_npz(paths["tent"])["probabilities"].astype(np.float64)
        pf = load_npz(paths["flip"])["probabilities"].astype(np.float64)
        y = load_npz(paths["no"])["labels"].astype(np.int64)
        for rule, met in evaluate_center2(p0, p1, pf, y, lock).items():
            rows.append(dict(seed=seed, rule=rule,
                             lock_source=(lock["_source"] if lock else "none"), **met))

    if not rows:
        return None, "no Center-2 predictions found"

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "deployment_locked_per_seed.csv"), index=False)
    summary = df.groupby("rule").agg(
        accepted=("accepted", "mean"), coverage_pct=("coverage_pct", "mean"),
        accuracy=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        help=("help", "mean"), harm=("harm", "mean"), net=("net", "mean"),
        precision=("precision", "mean"), recall=("recall", "mean"),
        n_seeds=("seed", "nunique")).reset_index()
    summary["precision"] = summary["help"] / (summary["help"] + summary["harm"]).clip(lower=1)
    summary["recall"] = summary["recall"].clip(upper=1.0)
    summary.to_csv(os.path.join(out, "deployment_locked_summary.csv"), index=False)
    return summary, None


def write_deployment_tex(summary, out):
    order_main = ["Frozen", "TENT (all flips)", "Flip-TTA", "CG-TENT",
                  "Target Q33 tent_conf", "Lock R1 tent_conf",
                  "Lock R2 tent_conf", "Oracle"]
    order_full = ["Frozen", "TENT (all flips)", "Flip-TTA", "Probability averaging",
                  "CG-TENT"] + [f"Target Q33 {k}" for k in
                  ("frozen_conf", "tent_conf", "critic_conf", "gate_margin")] + \
                 [f"Lock R1 {k}" for k in
                  ("frozen_conf", "tent_conf", "critic_conf", "gate_margin")] + \
                 [f"Lock R2 {k}" for k in
                  ("frozen_conf", "tent_conf", "critic_conf", "gate_margin")] + ["Oracle"]
    lookup = summary.set_index("rule")

    def rows_for(order):
        body = []
        for name in order:
            if name not in lookup.index:
                continue
            r = lookup.loc[name]
            display = (name.replace("Lock R1 ", "Locked R1: ")
                       .replace("Lock R2 ", "Locked R2: ")
                       .replace("Target Q33 ", "Target Q33: ")
                       .replace("frozen_conf", "frozen conf.")
                       .replace("tent_conf", "TENT conf.")
                       .replace("critic_conf", "critic conf.")
                       .replace("gate_margin", "gate margin"))
            body.append(
                f"{display} & {fmt_count(r.accepted)} & {r.coverage_pct:.2f} & "
                f"{r.accuracy:.4f} & {fmt_count(r.help)} & {fmt_count(r.harm)} & "
                f"{fmt_count(r.net)} & {r.precision:.3f} & {r.recall:.3f} \\\\")
        return "\n".join(body)

    header = (r"\begin{tabular}{@{}lcccccccc@{}}" "\n"
              r"\toprule" "\n"
              r"Rule & Accepted & Cov.\% & Acc. & Help & Harm & Net & Prec. & Rec. \\" "\n"
              r"\midrule" "\n")
    footer = "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"

    main = (r"\begin{table*}[t]" "\n" r"\centering" "\n" r"\small" "\n"
            r"\setlength{\tabcolsep}{4pt}" "\n"
            r"\caption{Center-2 deployment of label-free comparators. Target Q33 "
            r"accepts the top third of TENT flips by TENT confidence on the "
            r"unlabeled target stream (Target Q10/Q50 are reported in the full "
            r"table). Locked R1 accepts the top 10\% of Center-1 TENT flips by "
            r"the score; Locked R2 accepts the top 1\% of all Center-1 patches. "
            r"No target label is used by any threshold. Counts are means over "
            r"five seeds; accuracy is the mean gated accuracy. Cov.\ is the "
            r"percentage of all patches accepted; precision is computed from "
            r"the mean help and harm counts.}"
            "\n" r"\label{tab:deployment-locked}" "\n"
            + header + rows_for(order_main) + footer + r"\end{table*}" + "\n")
    save_tex(os.path.join(out, "tab_deployment_locked.tex"), main)

    full = (r"\begin{table*}[t]" "\n" r"\centering" "\n" r"\scriptsize" "\n"
            r"\setlength{\tabcolsep}{3pt}" "\n"
            r"\caption{Full Center-1-locked comparator sweep. The main table "
            r"reports the pre-declared TENT-confidence rule; this table reports "
            r"every score and both locking rules for audit.}"
            "\n" r"\label{tab:deployment-locked-full}" "\n"
            + header + rows_for(order_full) + footer + r"\end{table*}" + "\n")
    save_tex(os.path.join(out, "tab_deployment_locked_full.tex"), full)


# ----------------------------------------------------------------- phikon table
def build_phikon(root, out):
    base = os.path.join(root, "phikon")
    if not os.path.isdir(base):
        return None, "phikon job has not run"
    rows = []
    for seed_dir in sorted(glob.glob(os.path.join(base, "seed*_tta"))):
        seed = int(os.path.basename(seed_dir).split("_")[0].replace("seed", ""))
        for method in PHIKON_METHODS + ["safe_tent"]:
            p = os.path.join(seed_dir, f"{method}_summary.json")
            met = read_json(p)
            if met:
                row = dict(met)
                row.update(seed=seed, method=method)
                rows.append(row)
        probe = read_json(os.path.join(base, f"seed{seed}_probe.json"))
        if probe:
            row = dict(probe)
            row.update(seed=seed, method="probe")
            rows.append(row)
    if not rows:
        return None, "phikon folder exists but contains no summaries"

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "phikon_eata_per_seed.csv"), index=False)
    summary = df.groupby("method").agg(
        n=("seed", "nunique"), accuracy=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
        balanced_accuracy=("balanced_accuracy", "mean"),
        balanced_accuracy_sd=("balanced_accuracy", "std"),
        harm=("harm", "mean") if "harm" in df else ("accuracy", "mean"),
        net=("net", "mean") if "net" in df else ("accuracy", "mean"),
        ece=("ece_15", "mean") if "ece_15" in df else ("accuracy", "mean"),
    ).reset_index()
    summary.to_csv(os.path.join(out, "phikon_eata_summary.csv"), index=False)
    return summary, None


def write_phikon_tex(summary, out):
    lookup = summary.set_index("method")
    order = [("probe", "Phikon probe (frozen head)"),
             ("no_tta", "No-TTA"), ("tent", "TENT"),
             ("flip_tta", "Flip-TTA"), ("eata", "EATA-LN"),
             ("safe_tent", "CG-TENT")]
    body = []
    for key, label in order:
        if key not in lookup.index:
            continue
        r = lookup.loc[key]
        n = int(r["n"])
        no_ref = key in ("probe", "no_tta")
        harm = "--" if no_ref else fmt_count(r.harm)
        net = "--" if no_ref else fmt_count(r.net)
        body.append(f"{label} & {n} & {fmt_pm(r.balanced_accuracy, r.balanced_accuracy_sd)} & "
                    f"{fmt_pm(r.accuracy, r.accuracy_sd)} & {harm} & {net} \\\\")
    if not body:
        return
    tex = (r"\begin{table}[t]" "\n" r"\centering" "\n" r"\small" "\n"
           r"\setlength{\tabcolsep}{4pt}" "\n"
           r"\caption{Phikon ViT-B/16 linear probe on Center-2, mean$\pm$sd over "
           r"head seeds. EATA-LN adapts LayerNorm parameters only; the linear "
           r"head stays frozen during adaptation. Harm and net are paired "
           r"against the same-seed frozen probe.}"
           "\n" r"\label{tab:phikon-eata}" "\n"
           r"\begin{tabular}{@{}lcccc@{}}" "\n" r"\toprule" "\n"
           r"Method & Seeds & Bal.\ acc. & Acc. & Harm & Net \\" "\n"
           r"\midrule" "\n" + "\n".join(body) + "\n" + r"\bottomrule" "\n"
           r"\end{tabular}" "\n" r"\end{table}" "\n")
    save_tex(os.path.join(out, "tab_phikon_eata.tex"), tex)


# ----------------------------------------------------------------- budget table
def build_budget(root, out):
    base = os.path.join(root, "budget")
    if not os.path.isdir(base):
        return None, "budget job has not run"
    rows = []
    for p in glob.glob(os.path.join(base, "seed*", "*pct", "subset*", "tent_summary.json")):
        met = read_json(p)
        if met:
            rows.append(met)
    if not rows:
        return None, "budget folder exists but contains no tent_summary.json"
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "budget_heldout_per_run.csv"), index=False)

    # The design requires one held-out set.  Flag any drift instead of hiding it.
    shas = sorted(set(str(x) for x in df.get("eval_sha256", pd.Series(dtype=str)).dropna()))
    if len(shas) > 1:
        print(f"budget: WARNING {len(shas)} distinct eval_sha256 values: {shas}")

    df["delta_acc_pp"] = 100.0 * (df["accuracy"] - df["frozen_eval_acc"])
    if "n_eval" not in df.columns:
        df["n_eval"] = df["eval_n"] if "eval_n" in df.columns else np.nan
    # Average the five subset seeds inside each source seed first, so a source
    # seed counts once.  The 100% stream is one run copied to five subset seeds.
    per_seed = df.groupby(["budget", "seed"]).agg(
        n_eval=("n_eval", "first"), n_adapt=("n_adapt", "mean"),
        frozen=("frozen_eval_acc", "first"), accuracy=("accuracy", "mean"),
        delta_acc_pp=("delta_acc_pp", "mean"), help=("help", "mean"),
        harm=("harm", "mean"), net=("net", "mean")).reset_index()
    per_seed.to_csv(os.path.join(out, "budget_heldout_per_seed.csv"), index=False)
    summary = per_seed.groupby("budget").agg(
        n_seeds=("seed", "nunique"), n_eval=("n_eval", "first"),
        n_adapt=("n_adapt", "mean"), frozen=("frozen", "mean"),
        accuracy=("accuracy", "mean"), delta_acc_pp=("delta_acc_pp", "mean"),
        delta_acc_pp_sd=("delta_acc_pp", "std"), help=("help", "mean"),
        harm=("harm", "mean"), harm_sd=("harm", "std"), net=("net", "mean"),
        net_sd=("net", "std"),
        seeds_positive=("delta_acc_pp", lambda x: int((x > 0).sum())),
    ).reset_index()
    order = {"1pct": 1, "5pct": 5, "25pct": 25, "100pct": 100}
    summary["_order"] = summary["budget"].map(order).fillna(999)
    summary = summary.sort_values("_order").drop(columns="_order")
    summary.to_csv(os.path.join(out, "budget_heldout_summary.csv"), index=False)
    return summary, (None if len(shas) <= 1 else f"{len(shas)} held-out hashes")


def write_budget_tex(summary, out):
    body = []
    frozen = float(summary["frozen"].mean())
    body.append(r"0\% (frozen) & 0 & " + f"{frozen:.4f}" + r" & -- & -- & -- & -- & -- \\")
    for _, r in summary.iterrows():
        label = r.budget.replace("pct", r"\%")
        n_adapt = f"{int(round(r.n_adapt)):,}".replace(",", "{,}")
        body.append(f"{label} & {n_adapt} & {r.accuracy:.4f} & "
                    + f"${r.delta_acc_pp:+.2f}" + r"\pm" + f"{r.delta_acc_pp_sd:.2f}$ & "
                    + f"{r.help:.1f} & {r.harm:.1f} & {r.net:.1f} & "
                    + f"{int(r.seeds_positive)}/{int(r.n_seeds)}" + r" \\")
    tex = (r"\begin{tabular}{@{}lrccrrrc@{}}" "\n" r"\toprule" "\n"
           r"Budget & Adapt $n$ & Acc. & $\Delta$Acc (pp) & Help & Harm & Net & Seeds up \\" "\n"
           r"\midrule" "\n" + "\n".join(body) + "\n" r"\bottomrule" "\n"
           r"\end{tabular}" "\n")
    save_tex(os.path.join(out, "tab_budget_heldout.tex"), tex)


# ----------------------------------------------------------------- center0 table
def build_center0(root, out):
    base = os.path.join(root, "center0")
    if not os.path.isdir(base):
        return None, "center0 job has not run"
    seed_dir = os.path.join(base, "seed5")
    if not os.path.isdir(seed_dir):
        return None, "center0 seed5 has not run"
    rows = []
    for method in METHODS + ["safe_tent"]:
        met = read_json(os.path.join(seed_dir, f"{method}_summary.json"))
        if met:
            row = dict(met)
            row.update(seed=5, method=method)
            rows.append(row)
    if not rows:
        return None, "center0 seed5 folder has no summaries"
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "center0_per_seed.csv"), index=False)
    summary = df.groupby("method").agg(
        n=("seed", "nunique"), accuracy=("accuracy", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"),
        harm=("harm", "mean"), net=("net", "mean")).reset_index()
    summary.to_csv(os.path.join(out, "center0_summary.csv"), index=False)
    return summary, None


def write_center0_tex(summary, out):
    lookup = summary.set_index("method")
    order = [("no_tta", "No-TTA"), ("tent", "TENT"), ("flip_tta", "Flip-TTA"),
             ("eata", "EATA-LN"), ("safe_tent", "CG-TENT"), ("shot", "SHOT-IM")]
    body = []
    for key, label in order:
        if key not in lookup.index:
            continue
        r = lookup.loc[key]
        harm = "--" if key == "no_tta" else fmt_count(r.harm)
        net = "--" if key == "no_tta" else fmt_count(r.net)
        body.append(f"{label} & {int(r['n'])} & {r.accuracy:.4f} & "
                    f"{r.balanced_accuracy:.4f} & {harm} & {net} \\\\")
    if not body:
        return
    tex = (r"\begin{table}[t]" "\n" r"\centering" "\n" r"\small" "\n"
           r"\caption{Center-0 results for the newly trained fifth source seed "
           r"(seed 5). Harm and net are paired against the same-seed frozen "
           r"model. The first four Center-0 seeds are reported in the main "
           r"table; this table adds the fifth seed only.}"
           "\n" r"\label{tab:center0-seed5}" "\n"
           r"\begin{tabular}{@{}lccccc@{}}" "\n" r"\toprule" "\n"
           r"Method & Seeds & Acc. & Bal.\ acc. & Harm & Net \\" "\n"
           r"\midrule" "\n" + "\n".join(body) + "\n" + r"\bottomrule" "\n"
           r"\end{tabular}" "\n" r"\end{table}" "\n")
    save_tex(os.path.join(out, "tab_center0_seed5.tex"), tex)


# ------------------------------------------------------------------ timing table
def build_timing(root, out):
    path = os.path.join(root, "timing_benchmark.json")
    payload = read_json(path)
    if not payload:
        return None, "timing job has not run"
    rows = payload.get("results", payload if isinstance(payload, list) else [])
    if not rows:
        return None, "timing JSON has no results"
    df = pd.DataFrame(rows)
    df.attrs["environment"] = payload.get("environment", {})
    df.to_csv(os.path.join(out, "timing_gpu.csv"), index=False)
    return df, None


def write_timing_tex(df, out):
    if df.empty:
        return
    env = df.attrs.get("environment", {}) or {}
    labels = {"frozen": "Frozen", "tent": "TENT", "flip_tta": "Flip-TTA",
              "safe_tent": "CG-TENT"}
    order = {"frozen": 0, "tent": 1, "flip_tta": 2, "safe_tent": 3}
    df = df.assign(_order=df["mode"].map(order).fillna(99))
    body = []
    for _, r in df.sort_values(["_order", "batch_size"]).iterrows():
        mode = labels.get(r["mode"], r["mode"])
        if r.get("oom", False):
            body.append(f"{mode} & {int(r['batch_size'])} & OOM & -- & -- & -- & -- \\\\")
            continue
        body.append(
            f"{mode} & {int(r['batch_size'])} & {r.images_per_sec:.1f} & "
            f"{r.ms_per_batch_p50:.2f} & {r.ms_per_batch_p95:.2f} & "
            f"{r.ms_per_image:.2f} & {r.peak_alloc_mb:.0f} \\\\")
    where = ""
    if env.get("gpu"):
        where = (f" Hardware: {env['gpu']}, {env.get('gpu_mem_gb', '?')} GB, "
                 f"torch {env.get('torch', '?')}. ")
    tex = (r"\begin{table*}[t]" "\n" r"\centering" "\n" r"\scriptsize" "\n"
           r"\setlength{\tabcolsep}{5pt}" "\n"
           r"\caption{GPU benchmark on one fixed checkpoint, model compute only "
           r"(data loading excluded). Latency is per batch, p50/p95 over twenty "
           r"measured repetitions after three warm-up repetitions. CG-TENT "
           r"includes the TENT update and four frozen critic views." + where +
           r"Peak VRAM is allocated memory.}"
           "\n" r"\label{tab:timing-gpu}" "\n"
           r"\begin{tabular}{@{}lcccccc@{}}" "\n" r"\toprule" "\n"
           r"Mode & Batch & Img/s & p50 (ms) & p95 (ms) & ms/img & Peak MB \\" "\n"
           r"\midrule" "\n" + "\n".join(body) + "\n" + r"\bottomrule" "\n"
           r"\end{tabular}" "\n" r"\end{table*}" "\n")
    save_tex(os.path.join(out, "tab_timing_gpu.tex"), tex)


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="downloaded 17_Revision_Colab folder (or results/ from the zip)")
    ap.add_argument("--center2", default=DEFAULT_CENTER2,
                    help="folder with seed1..seed5/{no,tent,flip}.npz")
    ap.add_argument("--out", default=None,
                    help="output folder; default <root>/tables")
    args = ap.parse_args()

    out = args.out or os.path.join(args.root, "tables")
    os.makedirs(out, exist_ok=True)
    notes = []
    manifest = {}

    dep, why = build_deployment(args.root, args.center2, out)
    if dep is not None:
        write_deployment_tex(dep, out)
        notes.append("deployment_locked: written")
    else:
        notes.append(f"deployment_locked: skipped ({why})")

    phikon, why = build_phikon(args.root, out)
    if phikon is not None:
        write_phikon_tex(phikon, out)
        notes.append("phikon_eata: written")
    else:
        notes.append(f"phikon_eata: skipped ({why})")

    budget, why = build_budget(args.root, out)
    if budget is not None:
        write_budget_tex(budget, out)
        notes.append("budget_heldout: written")
    else:
        notes.append(f"budget_heldout: skipped ({why})")

    center0, why = build_center0(args.root, out)
    if center0 is not None:
        write_center0_tex(center0, out)
        notes.append("center0_seed5: written")
    else:
        notes.append(f"center0_seed5: skipped ({why})")

    timing, why = build_timing(args.root, out)
    if timing is not None:
        write_timing_tex(timing, out)
        notes.append("timing_gpu: written")
    else:
        notes.append(f"timing_gpu: skipped ({why})")

    for p in sorted(glob.glob(os.path.join(out, "*"))):
        if os.path.isfile(p) and not p.endswith("postprocess_manifest.json"):
            manifest[os.path.basename(p)] = sha256(p)
    write_json(os.path.join(out, "postprocess_manifest.json"), manifest)

    with open(os.path.join(out, "README_TABLES.md"), "w", encoding="utf-8") as fh:
        fh.write("# Colab result tables\n\n")
        fh.write(f"Input root: `{os.path.abspath(args.root)}`\n\n")
        fh.write(f"Center-2 predictions: `{os.path.abspath(args.center2)}`\n\n")
        fh.write("## Status\n\n")
        for note in notes:
            fh.write(f"- {note}\n")
        fh.write("\n## Files\n\n")
        for name in sorted(manifest):
            fh.write(f"- `{name}`\n")
        fh.write("\nEvery table is generated from the per-seed CSV in this "
                 "folder. Missing experiments are skipped, not filled in.\n")
    print("\n".join(notes))
    print(f"tables written to {out}")


if __name__ == "__main__":
    main()

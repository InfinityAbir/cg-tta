#!/usr/bin/env python3
"""Reference implementation of CG-TTA.

CG-TTA is a label-free consensus gate wrapped around any online test-time
adapter.  For every target patch it computes

    p0  frozen source probabilities                f_{theta_0}(x)
    p1  adapter probabilities from theta_{k-1}     f_{theta_{k-1}}(x)   (pre-update)
    pf  four-view Flip-TTA critic                  mean_T f_{theta_0}(T x)

and returns

    y_safe = y1  if y1 != y0 and yf == y1
             y0  otherwise

so a label change proposed by the adapter is accepted only when a parameter-free
predictor independently reaches the same label.  No target label is used and the
gate has no free threshold.

What is in this file
--------------------
1. The gate (`consensus_gate`, `consensus_gate_torch`, `four_view_probs`).
2. The three adapters of the paper as losses over LayerNorm-only parameters
   (TENT, EATA-LN, SHOT-IM) plus the online `CG-TTA` wrapper that enforces the
   pre-update scoring convention.
3. The paper's metrics (accuracy, balanced accuracy, AUROC, ECE-15, help/harm).
4. A self-test that rebuilds Table II and the gate-ablation table from the
   released per-seed predictions and checks Proposition 1 on real data.

Usage
-----
    python code/cg_tta.py --self-test
    python code/cg_tta.py --demo          # needs torch

The self-test needs only numpy, scikit-learn and the released `predictions/`
and `tables/` directories.  It is independent of the GPU training code: if it
passes, the numbers in the paper follow from the released predictions and the
metric definitions below.
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE = os.path.dirname(HERE)

# Paper hyperparameters (Section "Adapters").
LR = 1e-3
MOMENTUM = 0.9
CLIP = 1.0
EATA_E0 = 0.4 * math.log(2.0)
EATA_BETA = 1.0
SHOT_LAMBDA = 1.0
FISHER_N = 1024

METHODS = ["no", "tent", "flip", "eata", "shot", "safe_tent", "safe_eata", "safe_shot"]


# --------------------------------------------------------------------------
# 1. Metrics -- definitions used for every number in the paper
# --------------------------------------------------------------------------
def accuracy(pred, lab):
    return float((pred == lab).mean())


def balanced_accuracy(pred, lab):
    """Balanced accuracy, i.e. the mean of sensitivity and specificity."""
    tpr = ((pred == 1) & (lab == 1)).sum() / max(1, (lab == 1).sum())
    tnr = ((pred == 0) & (lab == 0)).sum() / max(1, (lab == 0).sum())
    return float(0.5 * (tpr + tnr))


def ece15(prob, pred, lab):
    """Expected calibration error with 15 equal-width confidence bins."""
    p = np.asarray(prob)[:, 1].astype(float)
    conf = np.maximum(p, 1.0 - p)
    bins = np.linspace(0.0, 1.0, 16)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, 14)
    e = 0.0
    for b in range(15):
        m = idx == b
        if m.any():
            e += m.mean() * abs(float((pred[m] == lab[m]).mean()) - float(conf[m].mean()))
    return float(e)


def auroc(prob, lab):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(lab, np.asarray(prob)[:, 1]))
    except ValueError:
        return float("nan")


def help_harm(pred, lab, ref):
    """C->W flips relative to the frozen prediction.

    harm  = frozen correct, new prediction wrong
    help  = frozen wrong,  new prediction correct
    """
    pred = np.asarray(pred).astype(int)
    lab = np.asarray(lab).astype(int)
    ref = np.asarray(ref).astype(int)
    harm = int(((pred != lab) & (ref == lab)).sum())
    help_ = int(((pred == lab) & (ref != lab)).sum())
    return {"help": help_, "harm": harm, "net": help_ - harm}


def method_metrics(prob, pred, lab, ref_pred):
    """All per-run metrics reported for one method in one seed."""
    hh = help_harm(pred, lab, ref_pred)
    n_correct_ref = max(1, int((np.asarray(ref_pred) == np.asarray(lab)).sum()))
    return dict(
        acc=accuracy(pred, lab),
        ba=balanced_accuracy(pred, lab),
        auroc=auroc(prob, lab),
        ece=ece15(prob, pred, lab),
        help=hh["help"],
        harm=hh["harm"],
        net=hh["net"],
        harm_rate=float(hh["harm"] / n_correct_ref),
    )


# --------------------------------------------------------------------------
# 2. The gate
# --------------------------------------------------------------------------
def four_view_probs(model, x):
    """Mean softmax over {identity, H, V, HV} of a frozen torch model."""
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        acc = F.softmax(model(x), dim=1)
        for axes in ([3], [2], [2, 3]):
            acc = acc + F.softmax(model(torch.flip(x, axes)), dim=1)
    return acc / 4.0


def consensus_gate(p0, p1, pf):
    """Numpy form of the gate.

    p0 : frozen probabilities            (N, C)
    p1 : adapted probabilities, pre-update (N, C)
    pf : four-view critic probabilities  (N, C)

    Returns (probs, accept, harm_subset_of_adapter).
    """
    p0, p1, pf = (np.asarray(a) for a in (p0, p1, pf))
    y0 = p0.argmax(1)
    y1 = p1.argmax(1)
    yf = pf.argmax(1)
    accept = (y0 != y1) & (yf == y1)
    probs = np.where(accept[:, None], p1, p0)
    return probs, accept, y0, y1


def consensus_gate_torch(p0, p1, pf):
    """Torch form of the gate, for use inside an online loop."""
    import torch
    y0, y1, yf = p0.argmax(1), p1.argmax(1), pf.argmax(1)
    accept = (y0 != y1) & (yf == y1)
    probs = torch.where(accept[:, None], p1, p0)
    return probs, accept


def apply_gate(y0, y1, yf):
    """Label-level form: returns the CG-TTA label and the accept mask."""
    accept = (y0 != y1) & (yf == y1)
    return np.where(accept, y1, y0), accept


# --------------------------------------------------------------------------
# 3. Adapters
# --------------------------------------------------------------------------
def entropy(prob):
    """Per-sample Shannon entropy of a probability tensor."""
    import torch
    return -(prob * torch.log(prob + 1e-12)).sum(1)


def tent_loss(prob):
    """TENT: mean entropy of the batch."""
    return entropy(prob).mean()


def shot_loss(prob, lam=SHOT_LAMBDA):
    """SHOT-IM: L = H_mean + lam * sum(pbar log pbar) = H_mean - lam * H(pbar)."""
    import torch
    pbar = prob.mean(0)
    return tent_loss(prob) + lam * (pbar * torch.log(pbar + 1e-12)).sum()


def eata_loss(prob, theta, theta0, fisher, beta=EATA_BETA, e0=EATA_E0):
    """EATA-LN: entropy on low-entropy samples plus a Fisher penalty.

    `theta` and `theta0` map parameter name -> tensor; `fisher` maps the same
    names to a normalised diagonal Fisher estimate.
    """
    import torch
    ent = entropy(prob)
    mask = ent < e0
    if int(mask.sum()) == 0:
        return None, 0
    pf = prob[mask]
    loss = -(pf * torch.log(pf + 1e-12)).sum(1).mean()
    reg, cnt = 0.0, 0
    for name, f in fisher.items():
        reg = reg + (f * (theta[name] - theta0[name]) ** 2).sum()
        cnt += f.numel()
    loss = loss + beta * reg / max(1, cnt)
    return loss, int(mask.sum())


def estimate_fisher(model, loader, names, n=FISHER_N, seed=0):
    """Diagonal Fisher of the entropy objective on `n` target patches.

    Identical to the GPU script: accumulate squared gradients, divide by the
    number of samples, then normalise by the global mean so that the
    regulariser is scale-free.
    """
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    sd = dict(model.named_parameters())
    for name in names:
        sd[name].requires_grad_(True)
    fisher = {name: torch.zeros_like(sd[name]) for name in names}
    dev = next(model.parameters()).device
    seen = 0
    for batch in loader:
        x = batch[0]
        if seen >= n:
            break
        take = min(len(x), n - seen)
        x = x[:take].to(dev)
        p = F.softmax(model(x), 1)
        loss = -(p * torch.log(p + 1e-12)).sum(1).mean()
        model.zero_grad()
        loss.backward()
        for name in names:
            if sd[name].grad is not None:
                fisher[name] += (sd[name].grad.detach() ** 2) * take
        seen += take
    total = max(1, seen)
    flat = []
    for name in names:
        fisher[name] = (fisher[name] / total).detach().cpu()
        flat.append(fisher[name].flatten())
    mu = float(torch.cat(flat).mean()) + 1e-12
    for name in names:
        fisher[name] = fisher[name] / mu
    return fisher, seen


class CGTTA:
    """Online CG-TTA around a LayerNorm-only adapter.

    The wrapper owns the frozen copy, the adapted model, the optimizer and the
    pre-update bookkeeping.  `step(x)` scores the batch with theta_{k-1}, then
    updates, so the returned probabilities never come from theta_K.  That is the
    convention used for every number in the paper and the one TENT requires.

    Example
    -------
        safe = CGTTA(model, adapter="tent")
        for x, y in loader:
            probs, accept = safe.step(x)        # scored before the update
            preds = probs.argmax(1)
    """

    ADAPTERS = ("tent", "eata", "shot")

    def __init__(self, model, adapter="tent", lr=LR, momentum=MOMENTUM, clip=CLIP):
        import torch
        import torch.nn as nn

        if adapter not in self.ADAPTERS:
            raise ValueError(f"adapter must be one of {self.ADAPTERS}")
        self.adapter = adapter
        self.lr, self.momentum, self.clip = lr, momentum, clip
        self.model = model
        self.frozen = copy.deepcopy(model)
        self.frozen.eval()
        for p in self.frozen.parameters():
            p.requires_grad_(False)

        for p in model.parameters():
            p.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.train()
                if m.weight is not None:
                    m.weight.requires_grad_(True)
                if m.bias is not None:
                    m.bias.requires_grad_(True)
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.names = [n for n, p in model.named_parameters() if p.requires_grad]
        self.opt = torch.optim.SGD(self.params, lr=lr, momentum=momentum)
        self.theta0 = {n: p.detach().clone()
                       for n, p in model.named_parameters() if n in self.names}
        self.fisher = None
        self.steps = 0

    def attach_fisher(self, fisher):
        """Required for adapter='eata'.  See `estimate_fisher`."""
        import torch
        dev = next(self.model.parameters()).device
        self.fisher = {k: torch.as_tensor(v, dtype=torch.float32, device=dev).clone()
                       for k, v in fisher.items()}

    def score(self, x):
        """Score one batch with theta_{k-1}; returns (probs, accept, p0, p1)."""
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            p0 = F.softmax(self.frozen(x), 1)
            pf = four_view_probs(self.frozen, x)
        p1 = F.softmax(self.model(x), 1)
        probs, accept = consensus_gate_torch(p0, p1, pf)
        return probs, accept, p0, p1

    def update(self, p1):
        """One SGD step on the adapter objective, using the pre-update p1."""
        import torch
        import torch.nn as nn

        if self.adapter == "tent":
            loss = tent_loss(p1)
        elif self.adapter == "shot":
            loss = shot_loss(p1)
        else:
            if self.fisher is None:
                raise RuntimeError("EATA-LN needs a Fisher estimate: see estimate_fisher()")
            theta = dict(self.model.named_parameters())
            loss, _ = eata_loss(p1, theta, self.theta0, self.fisher)
            if loss is None:
                self.steps += 1
                return 0.0
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, self.clip)
        self.opt.step()
        self.steps += 1
        return float(loss.detach())

    def step(self, x):
        """Score with theta_{k-1}, then update.  Returns (probs, accept)."""
        probs, accept, _, p1 = self.score(x)
        self.update(p1)
        return probs.detach(), accept


# --------------------------------------------------------------------------
# 4. Self-test: rebuild Table II and the gate ablation from the release
# --------------------------------------------------------------------------
def load_npz(path):
    z = np.load(path)
    return {k: z[k] for k in ("probabilities", "predictions", "labels", "patch_ids", "slides")}


def build_pack(d):
    """Apply the gate to the released streams to obtain the CG-TTA rows."""
    p0 = d["no"]["probabilities"]
    y0 = d["no"]["predictions"].astype(int)
    pf = d["flip"]["probabilities"]
    yf = pf.argmax(1)
    out = {
        "no": (p0, y0),
        "tent": (d["tent"]["probabilities"], d["tent"]["predictions"].astype(int)),
        "flip": (pf, yf),
        "eata": (d["eata"]["probabilities"], d["eata"]["predictions"].astype(int)),
        "shot": (d["shot"]["probabilities"], d["shot"]["predictions"].astype(int)),
    }
    for name, key in [("safe_tent", "tent"), ("safe_eata", "eata"), ("safe_shot", "shot")]:
        p1 = d[key]["probabilities"]
        y1 = d[key]["predictions"].astype(int)
        probs, _accept, _y0, _y1 = consensus_gate(p0, p1, pf)
        out[name] = (probs, probs.argmax(1))
    return out


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def close(a, b, tol=5e-9):
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def gate_ablation_rows(d):
    """Reproduce core_table_gate_ablation.csv for one seed."""
    p0 = d["no"]["probabilities"]
    y0 = d["no"]["predictions"].astype(int)
    lab = d["no"]["labels"].astype(int)
    yt = d["tent"]["predictions"].astype(int)
    yf = d["flip"]["probabilities"].argmax(1)
    ye = d["eata"]["predictions"].astype(int)
    flips = y0 != yt
    safe_accept = flips & (yf == yt)
    eata_accept = flips & (ye == yt)
    both = safe_accept & eata_accept
    rng = np.random.default_rng(100 + int(d["no"]["seed"]))
    k = int(safe_accept.sum())
    idx = rng.choice(np.where(flips)[0], size=k, replace=False)
    rand_accept = np.zeros_like(flips)
    rand_accept[idx] = True
    good = flips & (yt == lab)
    rows = []
    for name, m in [("all_flips", flips), ("random_matched", rand_accept),
                    ("flip_gate", safe_accept), ("eata_gate", eata_accept),
                    ("both_gate", both), ("oracle", good)]:
        pred = np.where(m, yt, y0)
        hh = help_harm(pred, lab, y0)
        rows.append(dict(gate=name, seed=int(d["no"]["seed"]),
                         accept=int(m.sum()),
                         accept_rate=float(m.sum() / max(1, flips.sum())),
                         acc=accuracy(pred, lab), **hh))
    return rows


def self_test(pred_dir, table_dir, verbose=True):
    ok = True

    # ---- Table II (center-2, five seeds) ---------------------------------
    per_seed = {}
    gate_rows = []
    prop_checks = 0
    for seed in range(1, 6):
        d = {}
        for m in ("no", "tent", "flip", "eata", "shot"):
            d[m] = load_npz(os.path.join(pred_dir, "center2", f"seed{seed}", f"{m}.npz"))
            d[m]["seed"] = seed
        ref = d["no"]
        for m in ("tent", "flip", "eata", "shot"):
            assert np.array_equal(d[m]["patch_ids"], ref["patch_ids"]), f"patch order differs: {m}"
            assert np.array_equal(d[m]["labels"], ref["labels"]), f"labels differ: {m}"
        pack = build_pack(d)
        lab = ref["labels"].astype(int)
        y0 = pack["no"][1]

        # Proposition 1 on real data: harm(CG-TTA) subset of harm(adapter).
        for safe_name, base in [("safe_tent", "tent"), ("safe_eata", "eata"),
                                ("safe_shot", "shot")]:
            y_safe = pack[safe_name][1]
            y_base = pack[base][1]
            harm_safe = (y_safe != lab) & (y0 == lab)
            harm_base = (y_base != lab) & (y0 == lab)
            if not np.all(harm_safe <= harm_base):
                ok = False
                if verbose:
                    print(f"FAIL Proposition 1: {safe_name} seed{seed} has harm outside {base}")
            prop_checks += 1

        for name, (prob, pred) in pack.items():
            per_seed.setdefault(name, []).append(
                method_metrics(prob, pred, lab, y0))
        gate_rows.extend(gate_ablation_rows(d))

    table = read_csv(os.path.join(table_dir, "core_table2_main.csv"))
    fields = ["acc", "ba", "auroc", "ece", "help", "harm", "net", "harm_rate"]
    worst = 0.0
    for row in table:
        name = row["method"]
        runs = per_seed[name]
        for f in fields:
            vals = np.array([r[f] for r in runs], dtype=float)
            mean = float(vals.mean())
            sd = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
            worst = max(worst, abs(mean - float(row[f])))
            if not close(mean, row[f]):
                ok = False
                if verbose:
                    print(f"FAIL Table II {name}.{f}: rebuilt {mean:.12g} vs csv {row[f]}")
            key = f + "_sd"
            if key in row and len(vals) > 1:
                worst = max(worst, abs(sd - float(row[key])))
                if not close(sd, row[key]):
                    ok = False
                    if verbose:
                        print(f"FAIL Table II {name}.{key}: rebuilt {sd:.12g} vs csv {row[key]}")
    if verbose:
        print(f"Table II: 8 methods x 5 seeds rebuilt, max deviation {worst:.3e}")

    # ---- gate ablation ----------------------------------------------------
    table = read_csv(os.path.join(table_dir, "core_table_gate_ablation.csv"))
    by_gate = {}
    for r in gate_rows:
        by_gate.setdefault(r["gate"], []).append(r)
    worst = 0.0
    for row in table:
        runs = by_gate[row["gate"]]
        for f in ("accept", "accept_rate", "acc", "help", "harm", "net"):
            mean = float(np.mean([r[f] for r in runs]))
            worst = max(worst, abs(mean - float(row[f])))
            if not close(mean, row[f]):
                ok = False
                if verbose:
                    print(f"FAIL gate {row['gate']}.{f}: rebuilt {mean:.12g} vs csv {row[f]}")
    if verbose:
        print(f"Gate ablation: 6 gates x 5 seeds rebuilt, max deviation {worst:.3e}")

    # ---- center-0 rotation ------------------------------------------------
    d = {m: load_npz(os.path.join(pred_dir, "center0", "seed1", f"{m}.npz"))
         for m in ("no", "tent", "flip", "eata", "shot")}
    pack = build_pack(d)
    lab = d["no"]["labels"].astype(int)
    y0 = pack["no"][1]
    for safe_name, base in [("safe_tent", "tent"), ("safe_eata", "eata"),
                            ("safe_shot", "shot")]:
        harm_safe = (pack[safe_name][1] != lab) & (y0 == lab)
        harm_base = (pack[base][1] != lab) & (y0 == lab)
        if not np.all(harm_safe <= harm_base):
            ok = False
            if verbose:
                print(f"FAIL Proposition 1: {safe_name} center0")
        prop_checks += 1
    rebuilt = {name: method_metrics(prob, pred, lab, y0)
               for name, (prob, pred) in pack.items()}
    table = read_csv(os.path.join(table_dir, "core_table_center0.csv"))
    worst = 0.0
    for row in table:
        r = rebuilt[row["method"]]
        for f in ("acc", "ba", "auroc", "ece", "help", "harm", "net", "harm_rate"):
            worst = max(worst, abs(r[f] - float(row[f])))
            if not close(r[f], row[f]):
                ok = False
                if verbose:
                    print(f"FAIL center0 {row['method']}.{f}: rebuilt {r[f]:.12g} vs csv {row[f]}")
    if verbose:
        print(f"Center-0: 8 methods rebuilt, max deviation {worst:.3e}")
        print(f"Proposition 1 checked on {prop_checks} (seed, adapter) pairs: "
              f"{'PASS' if ok else 'FAIL'}")
        print("SELF-TEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# --------------------------------------------------------------------------
# 5. Demo: one synthetic online run through the real torch code path
# --------------------------------------------------------------------------
def demo(seed=0):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    n, d = 64, 8

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(1, 4, 3, padding=1)
            self.ln = nn.LayerNorm([4, d, d])
            self.fc = nn.Linear(4 * d * d, 2)

        def forward(self, x):
            return self.fc(self.ln(self.c(x)).flatten(1))

    x = torch.randn(n, 1, d, d)
    y = (x.mean(dim=(1, 2, 3)) > 0).long()
    loader = DataLoader(TensorDataset(x, y), 16, shuffle=False)

    for adapter in ("tent", "shot"):
        net = Net()
        safe = CGTTA(net, adapter=adapter)
        outs = []
        for bx, _ in loader:
            probs, _accept = safe.step(bx)
            outs.append(probs)
        p = torch.cat(outs)
        print(f"demo {adapter:4s}: steps={safe.steps} rows={len(p)} "
              f"sum-to-one max error={float((p.sum(1) - 1).abs().max()):.2e}")

    # EATA needs a Fisher estimate.
    net = Net()
    safe = CGTTA(net, adapter="eata")
    fisher, seen = estimate_fisher(net, loader, safe.names, n=32)
    net2 = Net()
    net2.load_state_dict(net.state_dict())
    safe = CGTTA(net2, adapter="eata")
    safe.attach_fisher(fisher)
    outs = [safe.step(bx)[0] for bx, _ in loader]
    p = torch.cat(outs)
    print(f"demo eata: fisher_samples={seen} steps={safe.steps} rows={len(p)}")

    # Numpy and torch gates must agree.
    p0 = torch.softmax(torch.randn(500, 2), 1)
    p1 = torch.softmax(torch.randn(500, 2), 1)
    pf = torch.softmax(torch.randn(500, 2), 1)
    pt, at = consensus_gate_torch(p0, p1, pf)
    pn, an, _, _ = consensus_gate(p0.numpy(), p1.numpy(), pf.numpy())
    assert np.array_equal(pt.numpy(), pn)
    assert np.array_equal(at.numpy(), an)
    print("demo gate: numpy and torch implementations agree on 500 random rows")
    print("DEMO PASS")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true",
                    help="rebuild Table II and the gate ablation from predictions/")
    ap.add_argument("--demo", action="store_true",
                    help="run the torch gate + adapters on a synthetic batch")
    ap.add_argument("--predictions", default=os.path.join(RELEASE, "predictions"))
    ap.add_argument("--tables", default=os.path.join(RELEASE, "tables"))
    args = ap.parse_args(argv)
    if args.demo:
        demo()
        return 0
    return self_test(args.predictions, args.tables)


if __name__ == "__main__":
    sys.exit(main())

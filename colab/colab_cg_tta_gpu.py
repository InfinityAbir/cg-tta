#!/usr/bin/env python
"""CG-TTA GPU jobs: resumable, self-verifying, one file.

JOBS
  center0         Train source {3,4}(split 0) -> val center 1, target center 0,
                  seeds 2..5, then the five-method battery.   ~4-5 h/seed on T4.
  budget_heldout  Clean budget design: adapt on k% of a 90% adaptation pool,
                  evaluate the SAME fixed 10% held-out set every time.
                  4 budgets x 5 subset seeds x 5 source seeds = 100 runs; the
                  100% budget is run once per source seed and copied to the
                  other subset seeds (identical stream).  ~4 h per source seed,
                  ~20 h for five; pilot with --budgets 1 5 --subset-seeds 1 2.
  phikon          Pathology-pretrained encoder (Phikon ViT-B/16) + linear head;
                  frozen probe, and optionally the TTA battery.  ~1-2 h + battery.
  eata_full       Full-parameter EATA on Center-2 (published-method baseline),
                  5 seeds.  Default step 2.5e-4 (published EATA value); the
                  calibrated stable runs use `--lr-full 5e-5` (post hoc).
                  Use `--probe` for a 200-batch memory/time check first.
  preflight       Check GPU, disk, archive, checkpoints. No compute.
                  Add --check-phikon to verify transformers + HF access too.
  verify          Re-audit every artifact on disk. No GPU, no forward passes.
  smoke           CPU end-to-end test of the engine, including a forced
                  interruption + resume, compared against an uninterrupted run.

EVERY job is resumable at two levels
  * unit level   : runs/<job>/<unit>/status.json exists => skip.
  * batch level  : <unit>/state.pkl holds batch index, optimizer, RNG and the
                   adapting parameters, mirrored to Drive every SAVE_EVERY batches.
                   A recycled Colab VM resumes mid-run; nothing is recomputed.
  * --max-hours  : stop cleanly between batches, save state, exit 0. Re-run the
                   same command to continue.

RECORDING CONVENTION
  Main-paper numbers (TENT/EATA-LN/SHOT-IM) score each batch BEFORE its own
  update, using theta_{k-1}. The budget job's held-out evaluation is the one
  documented exception: it scores the fixed eval set once with the final theta_K,
  because the eval set is never part of the adaptation stream.

ARTIFACTS (per unit)
  <method>_predictions.npz  probabilities, predictions, labels, patch_ids, slides
  <method>_summary.json     metrics + help/harm vs the same-seed frozen run
  status.json               stage ledger + SHA-256 of every artifact

USAGE (Colab, T4)
  !python colab_cg_tta_gpu.py preflight
  !python colab_cg_tta_gpu.py center0 --seeds 2 3 4 5 --max-hours 11
  !python colab_cg_tta_gpu.py verify
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import sys
import tarfile
import time

import numpy as np

BATCH = 64
SAVE_EVERY = 200
LR, MOM, CLIP = 1e-3, 0.9, 1.0
# LayerNorm-only adapters need a large step; full-parameter EATA uses the
# published EATA value (SGD 2.5e-4) so the baseline is not a straw man.
# NOTE (post-submission): 2.5e-4 diverges over a full 85k-patch Center-2 stream
# (seed 1: acc 0.7429, 19,543 harms). The 5e-5 runs reported as the calibrated
# baseline pass `--lr-full 5e-5`; the collapse is preserved in the release.
LR_FULL = 2.5e-4
LR_FULL_CALIBRATED = 5e-5
E0 = 0.4 * math.log(2.0)
BETA = 1.0
FISHER_N = 1024
FISHER_SEED = 0
IM_W = 1.0
MAX_EPOCHS, PATIENCE = 10, 3
LR_MAX, LR_MIN, WD = 1e-4, 1e-6, 0.05
METHODS = ["no_tta", "tent", "flip_tta", "eata", "shot"]
T0 = time.time()


# --------------------------------------------------------------------- utils
def log(*a):
    print(*a, flush=True)


def fmt(sec):
    sec = int(sec)
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return (f"{h}h" if h else "") + (f"{m}m" if m or h else "") + f"{s}s"


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return default


def free_gb(path):
    st = shutil.disk_usage(path)
    return st.free / 1e9


def pick_first(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def seed_all(seed):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# ------------------------------------------------------------------- metrics
def metrics_from_probs(probs, labels):
    from sklearn.metrics import roc_auc_score
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pred = probs.argmax(1)
    conf = probs.max(1)
    ok = pred == labels
    ece = 0.0
    edges = np.linspace(0, 1, 16)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += abs(ok[m].mean() - conf[m].mean()) * m.mean()
    tpr = ((pred == 1) & (labels == 1)).sum() / max(1, (labels == 1).sum())
    tnr = ((pred == 0) & (labels == 0)).sum() / max(1, (labels == 0).sum())
    onehot = np.eye(2)[labels]
    return dict(
        n=int(len(labels)),
        accuracy=float(ok.mean()),
        balanced_accuracy=float(0.5 * (tpr + tnr)),
        auroc=float(roc_auc_score(labels, probs[:, 1])),
        nll=float(-np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)))),
        brier=float(np.mean((probs[:, 1] - labels) ** 2)),
        ece_15=float(ece),
        mean_confidence=float(conf.mean()),
        mean_entropy=float(-(probs * np.log(probs + 1e-12)).sum(1).mean()),
    )


def help_harm(ref_pred, ref_lab, pred, lab):
    assert np.array_equal(lab, ref_lab), "labels differ from frozen reference"
    pre, post = ref_pred == lab, pred == lab
    help_n = int((~pre & post).sum())
    harm_n = int((pre & ~post).sum())
    return dict(help=help_n, harm=harm_n, net=help_n - harm_n,
                harm_rate=float(harm_n / max(1, int(pre.sum()))))


def apply_gate(p_no, y_no, p_adapt, y_adapt, p_flip, y_flip):
    """CG-TTA decision rule. Acceptance needs a proposed flip + critic agreement."""
    flips = y_adapt != y_no
    accept = flips & (y_flip == y_adapt)
    out = np.where(accept, y_adapt, y_no)
    probs = np.where(accept[:, None], p_adapt, p_no)
    return probs, out, dict(proposed=int(flips.sum()), accepted=int(accept.sum()),
                            acceptance_rate=float(accept.sum() / max(1, flips.sum())))


# --------------------------------------------------------------- environment
def write_env(out):
    """Record the runtime so a later reader can tell what produced the files."""
    import platform
    import subprocess

    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cudnn"] = str(torch.backends.cudnn.version())
            props = torch.cuda.get_device_properties(0)
            info["gpu_mem_gb"] = round(props.total_memory / 1e9, 1)
    except Exception as exc:  # torch is optional for non-GPU jobs
        info["torch_error"] = str(exc)
    for mod in ("timm", "transformers", "numpy", "pandas"):
        try:
            info[mod] = getattr(__import__(mod), "__version__", None)
        except Exception:
            pass
    path = os.path.join(out, "environment.json")
    if not os.path.exists(path):
        atomic_json(path, info)
    freeze = os.path.join(out, "pip-freeze.txt")
    if not os.path.exists(freeze):
        try:
            txt = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                 capture_output=True, text=True, timeout=120).stdout
            with open(freeze, "w", encoding="utf-8") as fh:
                fh.write(txt)
        except Exception:
            pass
    return path


class Env:
    """Resolved paths and runtime facts for one job."""

    def __init__(self, args):
        self.args = args
        self.drive = None
        try:
            from google.colab import drive  # noqa: F401
            drive.mount("/content/drive", force_remount=False)
            self.drive = "/content/drive/MyDrive/Next Research"
        except Exception as exc:
            log("drive: " + str(exc))
            self.drive = args.drive_root
        self.roots = [r for r in [self.drive, args.drive_root,
                                  "/content/Next Research",
                                  os.path.expanduser("~/Next Research")] if r]
        self.local = args.local_dir
        os.makedirs(self.local, exist_ok=True)
        self.meta = pick_first([os.path.join(r, p) for r in self.roots for p in [
            "Camelyon17-WILDS/camelyon17_v1.0/metadata.csv",
            "Camelyon17_WILDS_Local_Archive/01_Data/metadata.csv",
            "04_Datasets/Dataset/metadata.csv"]])
        self.archive = pick_first([os.path.join(r, p) for r in self.roots for p in [
            "Camelyon17-WILDS/camelyon17_v1.0/archive.tar.gz",
            "Camelyon17-WILDS/archive.tar.gz"]])
        self.extracted = pick_first([os.path.join(r, p) for r in self.roots for p in [
            "Camelyon17-WILDS/camelyon17_v1.0/patches",
            "Camelyon17_WILDS_Local_Archive/01_Data/patches"]])
        self.out = os.path.join(self.roots[0], args.out_subdir)
        if args.canary:
            self.out += "_canary"
        os.makedirs(self.out, exist_ok=True)
        self.env_file = write_env(self.out)
        self.checkpoints = {
            "center2": {s: os.path.join(self.roots[0], "Camelyon17_WILDS_Local_Archive",
                                        "02_Source_Models", f"seed{s}_best.pth")
                        for s in range(1, 6)},
        }
        pilot = [os.path.join(r, "03_Experiments", "11_Pilot_Rotation",
                              "source34_val1_target0", f"seed{s}", "seed_best.pth")
                 for r in self.roots for s in [1]]
        self.checkpoints["center0"] = {1: pick_first(pilot)}

    def describe(self):
        log(f"drive        : {self.drive}")
        log(f"metadata     : {self.meta}")
        log(f"archive      : {self.archive}")
        log(f"extracted    : {self.extracted}")
        log(f"output       : {self.out}")
        log(f"local scratch: {self.local}")


# ---------------------------------------------------------------------- data
def load_metadata(path):
    import pandas as pd
    meta = pd.read_csv(path)
    assert {"patient", "node", "x_coord", "y_coord", "tumor", "slide", "center"}.issubset(meta.columns)
    if "Unnamed: 0" in meta.columns:
        meta = meta.rename(columns={"Unnamed: 0": "patch_id"})
    else:
        meta = meta.reset_index().rename(columns={"index": "patch_id"})
    meta["patch_id"] = meta["patch_id"].astype(int)
    meta["slide"] = meta["slide"].astype(int)
    return meta


def relpath(row):
    return ("patient_%03d_node_%d/patch_patient_%03d_node_%d_x_%d_y_%d.png"
            % (int(row["patient"]), int(row["node"]), int(row["patient"]),
               int(row["node"]), int(row["x_coord"]), int(row["y_coord"])))


def partitions(meta, rotation):
    if rotation == "center2":
        src = meta[(meta.center.isin([0, 3, 4])) & (meta.split == 0)]
        val = meta[(meta.center.isin([0, 3, 4])) & (meta.split == 1)]
        tgt = meta[meta.center == 2]
    elif rotation == "center0":
        src = meta[(meta.center.isin([3, 4])) & (meta.split == 0)]
        val = meta[meta.center == 1]
        tgt = meta[meta.center == 0]
    else:
        raise ValueError(rotation)
    assert not (set(src.center) & set(tgt.center)), "source and target centers overlap"
    return src.reset_index(drop=True), val.reset_index(drop=True), tgt.reset_index(drop=True)


class PatchDS:
    """Images for a frame; returns (x, y, patch_id, slide)."""

    def __init__(self, frame, root):
        self.df = frame.reset_index(drop=True)
        self.root = root

    def __len__(self):
        return len(self.df)

    def _path(self, row):
        return os.path.join(self.root, relpath(row))

    def __getitem__(self, i):
        import torch
        from PIL import Image
        row = self.df.iloc[i]
        img = Image.open(self._path(row)).convert("RGB").resize((224, 224), Image.BICUBIC)
        x = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        return (x - 0.5) / 0.5, int(row["tumor"]), int(row["patch_id"]), int(row["slide"])


def stage(frames, env, tag):
    """Materialise images on local disk once. Resumable and verified.

    The archive is preferred: one sequential read of a single large file is far
    faster on Drive FUSE than opening hundreds of thousands of small files. The
    extracted tree is a fallback for members the archive does not contain.
    """
    import pandas as pd
    dest = os.path.join(env.local, tag)
    os.makedirs(dest, exist_ok=True)
    manifest = pd.concat(frames, ignore_index=True)
    paths = [(r, os.path.join(dest, relpath(r))) for _, r in manifest.iterrows()]
    todo = [(r, p) for r, p in paths if not (os.path.exists(p) and os.path.getsize(p) > 0)]
    log(f"stage[{tag}]: {len(paths) - len(todo)}/{len(paths)} present, {len(todo)} to copy")
    if not todo:
        return dest

    def from_tar(tar_path, items):
        left = {os.path.basename(p): p for _, p in items}
        copied = 0
        with tarfile.open(tar_path, "r") as tf:
            for m in tf:
                if not left:
                    break
                key = os.path.basename(m.name)
                if key not in left or not m.isfile():
                    continue
                p = left.pop(key)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with tf.extractfile(m) as f, open(p + ".tmp", "wb") as out:
                    shutil.copyfileobj(f, out)
                os.replace(p + ".tmp", p)
                copied += 1
                if copied % 20000 == 0:
                    log(f"  staged {copied}/{len(items)} from archive")
        return copied, [(r, p) for r, p in items if not os.path.exists(p)]

    remaining = todo
    if env.archive:
        local_tar = os.path.join(
            env.local, "_dataset" + (".tar.gz" if env.archive.endswith(".gz") else ".tar"))
        if not os.path.exists(local_tar):
            total = os.path.getsize(env.archive)
            log(f"stage: copying archive to local scratch once ({total / 1e9:.1f} GB)")
            done = 0
            with open(env.archive, "rb") as f, open(local_tar + ".tmp", "wb") as g:
                while True:
                    buf = f.read(64 << 20)
                    if not buf:
                        break
                    g.write(buf)
                    done += len(buf)
                    if done // (1 << 30) != (done - len(buf)) // (1 << 30):
                        log(f"  archive {done / 1e9:.1f}/{total / 1e9:.1f} GB")
            os.replace(local_tar + ".tmp", local_tar)
        log(f"stage[{tag}]: extracting {len(remaining)} files from the local archive")
        n, remaining = from_tar(local_tar, remaining)
        log(f"stage[{tag}]: {n} from archive, {len(remaining)} still missing")
        if not remaining:
            return dest

    if env.extracted:
        by_name = {}
        for r, p in remaining:
            by_name.setdefault(os.path.basename(p), (r, p))
        copied = 0
        for base, _, files in os.walk(env.extracted):
            for f in files:
                if f in by_name:
                    r, p = by_name[f]
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    shutil.copy2(os.path.join(base, f), p + ".tmp")
                    os.replace(p + ".tmp", p)
                    copied += 1
                    if copied % 5000 == 0:
                        log(f"  copied {copied}/{len(remaining)} from extracted tree")
        remaining = [(r, p) for r, p in remaining if not os.path.exists(p)]
        log(f"stage[{tag}]: {copied} from extracted tree, {len(remaining)} still missing")
        if not remaining:
            return dest

    raise SystemExit(f"stage[{tag}]: {len(remaining)} files absent from archive and extracted tree; "
                     f"first: {remaining[0][1]}")


# --------------------------------------------------------------------- model
def tent_tensors(model):
    import torch.nn as nn
    named = dict(model.named_parameters())
    out = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.LayerNorm):
            for p in ["weight", "bias"]:
                key = f"{name}.{p}" if name else p
                if key in named:
                    out.append(key)
    return out


def load_vit(checkpoint, seed, num_classes=2):
    import timm
    import torch
    seed_all(seed)
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = blob.get("model_state_dict") or blob.get("model") or blob.get("state_dict") or blob
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model


def phikon_encoder():
    """Phikon ViT-B/16 (owkin). Requires transformers and HF access."""
    import torch
    from transformers import AutoModel
    kw = {}
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        kw["token"] = tok
    model = AutoModel.from_pretrained("owkin/phikon", add_pooling_layer=False, **kw)
    model.eval()
    return model


def forward_logits(model_kind, model, x):
    import torch
    if model_kind == "phikon":
        out = model(pixel_values=x).last_hidden_state[:, 0]
        return model.head(out) if hasattr(model, "head") else out
    return model(x)


# ------------------------------------------------------------------- runner
def load_state(path, model, opt, tnames):
    import torch
    with open(path, "rb") as fh:
        st = pickle.load(fh)
    if st.get("theta") and tnames:
        sd = dict(model.named_parameters())
        with torch.no_grad():
            for n in tnames:
                sd[n].copy_(st["theta"][n].to(sd[n].device))
    if opt is not None and st.get("opt_state"):
        opt.load_state_dict(st["opt_state"])
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        for group in opt.state.values():
            for k, v in group.items():
                if torch.is_tensor(v):
                    group[k] = v.to(dev)
    return st


def save_state(path, bi, model, opt, tnames, extra=None):
    import torch
    st = {"batch_idx": bi,
          "rng": {"torch": torch.get_rng_state(),
                  "numpy": np.random.get_state(), "py": random.getstate()}}
    if torch.cuda.is_available():
        st["rng"]["cuda"] = torch.cuda.get_rng_state_all()
    if tnames:
        sd = dict(model.named_parameters())
        st["theta"] = {n: sd[n].detach().cpu().clone() for n in tnames}
    if opt is not None:
        osd = copy.deepcopy(opt.state_dict())
        for group in osd["state"].values():
            for k, v in group.items():
                if torch.is_tensor(v):
                    group[k] = v.cpu()
        st["opt_state"] = osd
    if extra:
        st.update(extra)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(st, fh)
    os.replace(tmp, path)


def run_stream(model, kind, mode, loader, unit_dir, tnames, budget_seconds=None,
               preupdate=True, full_params=False, lr_full=None):
    """One adaptation/evaluation pass. Resumable at batch granularity.

    Returns probabilities, labels, ids, slides in stream order.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_f = os.path.join(unit_dir, "state.pkl")
    mirror = os.path.join(unit_dir, "state_drive.pkl")
    if not os.path.exists(state_f) and os.path.exists(mirror):
        shutil.copy2(mirror, state_f)
        log(f"{mode}: restored state from Drive mirror")
    adaptive = mode in ("tent", "eata", "shot")
    params, opt, sd = [], None, dict(model.named_parameters())
    if adaptive:
        if full_params:
            for p in model.parameters():
                p.requires_grad = True
            model.train()
        else:
            for p in model.parameters():
                p.requires_grad = False
            for m in model.modules():
                if isinstance(m, nn.LayerNorm):
                    m.train()
                    if m.weight is not None:
                        m.weight.requires_grad = True
                    if m.bias is not None:
                        m.bias.requires_grad = True
        params = [sd[n] for n in tnames]
        assert len(params) > 0, "no trainable tensors selected"
        step = LR_FULL if lr_full is None else lr_full
        opt = torch.optim.SGD(params, lr=(step if full_params else LR), momentum=MOM)
    else:
        model.eval()

    fisher = extra = None
    if mode == "eata":
        fpath = os.path.join(unit_dir, "fisher.npz")
        if os.path.exists(fpath):
            z = np.load(fpath, allow_pickle=True)
            fisher = {k: torch.from_numpy(z[k]).to(device) for k in z.files}
            extra = read_json(os.path.join(unit_dir, "fisher_meta.json"), {})
            log(f"eata: fisher loaded ({len(fisher)} tensors)")
        else:
            raise RuntimeError("EATA requires fisher.npz; call compute_fisher() first")
        with torch.no_grad():
            extra = {"theta0": {n: sd[n].detach().clone() for n in tnames},
                     "fisher": fisher, "sel_total": 0, "sel_seen": 0, "skip": 0}

    st = None
    if os.path.exists(state_f):
        st = load_state(state_f, model, opt, tnames)
        if full_params and "lr_full" in st and abs(float(st["lr_full"]) - step) > 1e-12:
            raise SystemExit(
                f"{mode}: state was saved with lr_full={st['lr_full']}, requested {step}. "
                f"Re-run with the same --lr-full value.")
        start = st["batch_idx"]
        if mode == "eata":
            for k in ("sel_total", "sel_seen", "skip"):
                if k in st:
                    extra[k] = st[k]
        log(f"{mode}: RESUME at batch {start}/{len(loader)}")
    else:
        start = 0
        log(f"{mode}: START 0/{len(loader)}")

    probs_path = os.path.join(unit_dir, "probs.npy")
    have = os.path.exists(probs_path)
    P = np.load(probs_path) if have else np.zeros((0, 2), np.float32)
    L = np.load(os.path.join(unit_dir, "labels.npy")) if have else np.zeros(0, np.int64)
    I = list(np.load(os.path.join(unit_dir, "ids.npy"))) if have else []
    S = list(np.load(os.path.join(unit_dir, "slides.npy"))) if have else []
    if st is not None:
        n_done = min(len(I), start * loader.batch_size)
        if n_done < start * loader.batch_size - loader.batch_size:
            raise SystemExit(
                f"{mode}: state says {start} batches but only {len(I)} predictions exist. "
                f"Delete {unit_dir}/state.pkl and re-run (this unit restarts from 0).")
        P, L, I, S = P[:n_done], L[:n_done], I[:n_done], S[:n_done]
        log(f"{mode}: {n_done} predictions already on disk")
        start = n_done // loader.batch_size

    newP, newL, newI, newS = [], [], [], []
    t0 = time.time()
    ctx = torch.enable_grad() if adaptive else torch.no_grad()
    last_save = start
    with ctx:
        for bi, (x, y, pid, sl) in enumerate(loader):
            if bi < start:
                continue
            x = x.to(device)
            if mode == "flip_tta":
                acc = 0
                for f in (None, [3], [2], [2, 3]):
                    acc = acc + F.softmax(forward_logits(kind, model, torch.flip(x, f) if f else x), 1)
                p = acc / 4
            else:
                p = F.softmax(forward_logits(kind, model, x), 1)
            if mode == "tent":
                loss = -(p * torch.log(p + 1e-12)).sum(1).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, CLIP); opt.step()
            elif mode == "shot":
                pbar = p.mean(0)
                loss = (-(p * torch.log(p + 1e-12)).sum(1).mean()
                        + IM_W * (pbar * torch.log(pbar + 1e-12)).sum())
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, CLIP); opt.step()
            elif mode == "eata":
                ent = -(p * torch.log(p + 1e-12)).sum(1)
                mask = ent < E0
                extra["sel_total"] += int(mask.sum())
                extra["sel_seen"] += int(x.size(0))
                if int(mask.sum()) == 0:
                    extra["skip"] += 1
                else:
                    pf = p[mask]
                    loss = -(pf * torch.log(pf + 1e-12)).sum(1).mean()
                    reg, cnt = 0.0, 0
                    for n in tnames:
                        f = extra["fisher"][n].to(device)
                        reg = reg + (f * (sd[n] - extra["theta0"][n].to(device)) ** 2).sum()
                        cnt += f.numel()
                    loss = loss + BETA * reg / max(1, cnt)
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, CLIP); opt.step()
            newP.append(p.detach().float().cpu().numpy())
            newL.append(y.numpy().astype(np.int64))
            newI.extend(int(v) for v in pid)
            newS.extend(int(v) for v in sl)
            done = bi + 1
            if done % 50 == 0 or done == len(loader):
                el = time.time() - t0
                log(f"{mode} {done}/{len(loader)} {100*done/len(loader):.1f}% "
                    f"spent {fmt(el)} eta {fmt((len(loader)-done)*el/max(1, done-start))}")
            if done % SAVE_EVERY == 0 or done == len(loader):
                P = np.concatenate([P] + newP); newP = []
                L = np.concatenate([L] + newL); newL = []
                I = I + newI; newI = []
                S = S + newS; newS = []
                np.save(probs_path, P); np.save(os.path.join(unit_dir, "labels.npy"), L)
                np.save(os.path.join(unit_dir, "ids.npy"), np.asarray(I, np.int64))
                np.save(os.path.join(unit_dir, "slides.npy"), np.asarray(S, np.int64))
                ex = {}
                if mode == "eata":
                    ex = {"sel_total": extra["sel_total"], "sel_seen": extra["sel_seen"],
                          "skip": extra["skip"]}
                if full_params:
                    ex["lr_full"] = float(step)
                save_state(state_f, done, model, opt, tnames if adaptive else None, ex)
                shutil.copy2(state_f, mirror)
                last_save = done
                if budget_seconds and time.time() - T0 > budget_seconds and done < len(loader):
                    log(f"{mode}: time budget reached at batch {done}; state saved. "
                        f"Re-run the same command to resume.")
                    sys.exit(0)
    if newP:
        P = np.concatenate([P] + newP)
        L = np.concatenate([L] + newL)
        I = I + newI
        S = S + newS
        np.save(probs_path, P); np.save(os.path.join(unit_dir, "labels.npy"), L)
        np.save(os.path.join(unit_dir, "ids.npy"), np.asarray(I, np.int64))
        np.save(os.path.join(unit_dir, "slides.npy"), np.asarray(S, np.int64))
    assert len(P) == len(L) == len(I) == len(S), "artifact length mismatch"
    return P, L, np.asarray(I, np.int64), np.asarray(S, np.int64), st


def finalize(unit_dir, method, P, L, I, S, rotation, seed, extra=None):
    ref = os.path.join(unit_dir, "no_tta_predictions.npz")
    if method != "no_tta":
        z = np.load(ref)
        assert np.array_equal(z["patch_ids"], I), "patch order differs from frozen run"
        hh = help_harm(z["predictions"].astype(int), z["labels"].astype(int),
                       P.argmax(1), L)
    else:
        hh = dict(help=-1, harm=-1, net=-1, harm_rate=-1.0)
    met = metrics_from_probs(P, L)
    out = dict(method=method, rotation=rotation, seed=int(seed), **met, **hh)
    if extra:
        out.update(extra)
    atomic_npz(os.path.join(unit_dir, f"{method}_predictions.npz"),
               probabilities=P.astype(np.float32), predictions=P.argmax(1).astype(np.uint8),
               labels=L.astype(np.uint8), patch_ids=I.astype(np.int64), slides=S.astype(np.int16))
    atomic_json(os.path.join(unit_dir, f"{method}_summary.json"), out)
    for f in ("state.pkl", "state_drive.pkl", "probs.npy", "labels.npy", "ids.npy", "slides.npy"):
        try:
            os.remove(os.path.join(unit_dir, f))
        except OSError:
            pass
    log(f"{method}: DONE acc={met['accuracy']:.4f} ba={met['balanced_accuracy']:.4f} "
        f"ece={met['ece_15']:.4f} help={hh['help']} harm={hh['harm']}")
    return out


def unit_done(unit_dir, method):
    return os.path.exists(os.path.join(unit_dir, f"{method}_summary.json"))


def battery(model, kind, train_loader, unit_dir, methods=METHODS, tnames=None,
            budget_seconds=None):
    """Run the five methods in order, freezing no_tta as the reference first."""
    results = {}
    for mode in methods:
        if unit_done(unit_dir, mode):
            log(f"{mode}: SKIP (already done)")
            results[mode] = read_json(os.path.join(unit_dir, f"{mode}_summary.json"))
            continue
        seed_all(int(read_json(os.path.join(unit_dir, "unit.json"), {}).get("seed", 0)))
        m = copy.deepcopy(model)
        P, L, I, S, _ = run_stream(m, kind, mode, train_loader, unit_dir,
                                   tnames or [], budget_seconds)
        results[mode] = finalize(unit_dir, mode, P, L, I, S,
                                 read_json(os.path.join(unit_dir, "unit.json"), {}).get("rotation", "?"),
                                 read_json(os.path.join(unit_dir, "unit.json"), {}).get("seed", 0))
        del m
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    return results


# ------------------------------------------------------------------- fishing
def compute_fisher(model, kind, loader, tnames, n=FISHER_N, seed=FISHER_SEED, unit_dir=None):
    import torch
    import torch.nn.functional as F
    seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    sd = dict(model.named_parameters())
    for name in tnames:
        sd[name].requires_grad = True
    fisher = {name: torch.zeros_like(sd[name]) for name in tnames}
    seen = 0
    for x, _, _, _ in loader:
        if seen >= n:
            break
        take = min(loader.batch_size, n - seen)
        x = x[:take].to(device)
        p = F.softmax(forward_logits(kind, model, x), 1)
        loss = -(p * torch.log(p + 1e-12)).sum(1).mean()
        model.zero_grad(); loss.backward()
        for name in tnames:
            if sd[name].grad is not None:
                fisher[name] += (sd[name].grad.detach() ** 2) * take
        seen += take
    for name in tnames:
        fisher[name] = (fisher[name] / max(1, seen)).detach().cpu()
        sd[name].requires_grad = False
    flat = torch.cat([fisher[n].flatten() for n in tnames])
    mu = float(flat.mean()) + 1e-12
    for name in tnames:
        fisher[name] = fisher[name] / mu
    if unit_dir:
        np.savez_compressed(os.path.join(unit_dir, "fisher.npz"),
                            **{k: v.numpy() for k, v in fisher.items()})
        atomic_json(os.path.join(unit_dir, "fisher_meta.json"),
                    dict(n_samples=int(seen), seed=int(seed), normalisation="mean"))
    return fisher, seen


# ---------------------------------------------------------------------- jobs
def job_center0(env, args):
    import torch
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    meta = load_metadata(env.meta)
    src, val, tgt = partitions(meta, "center0")
    if args.canary:
        src, val, tgt = src.iloc[:2560], val.iloc[:640], tgt.iloc[:640]
        log(f"CANARY: truncated src={len(src)} val={len(val)} tgt={len(tgt)}")
    else:
        assert len(tgt) == 59436 and tgt.slide.nunique() == 10
    log(f"center0: src={len(src)} val={len(val)} tgt={len(tgt)} slides={tgt.slide.nunique()}")
    root = stage([src, val, tgt], env, "center0")

    def loader_for(df, shuffle=False):
        return DataLoader(PatchDS(df, root), BATCH, shuffle=shuffle, num_workers=2,
                          pin_memory=True, drop_last=False)

    def lr_at(gs, spe, total):
        warm = spe
        if gs < warm:
            return 1e-5 + (LR_MAX - 1e-5) * (gs + 1) / warm
        t = (gs - warm) / max(1, total - warm)
        return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * t))

    @torch.no_grad()
    def evaluate(model, df):
        model.eval()
        P, L = [], []
        for x, y, _, _ in loader_for(df):
            P.append(F.softmax(forward_logits("vit", model, x.to("cuda")), 1).cpu().numpy())
            L.append(y.numpy())
        P, L = np.concatenate(P), np.concatenate(L)
        pred = P.argmax(1)
        return dict(accuracy=float(accuracy_score(L, pred)),
                    balanced_accuracy=float(balanced_accuracy_score(L, pred)),
                    nll=float(-np.mean(np.log(np.clip(P[np.arange(len(L)), L], 1e-12, 1.0)))),
                    n=int(len(L)))

    for seed in args.seeds:
        unit = os.path.join(env.out, "center0", f"seed{seed}")
        os.makedirs(unit, exist_ok=True)
        atomic_json(os.path.join(unit, "unit.json"),
                    dict(rotation="center0", seed=int(seed), n=int(len(tgt)),
                         source_n=int(len(src)), val_n=int(len(val))))
        ckpt = os.path.join(unit, "seed_best.pth")
        latest = os.path.join(unit, "seed_latest.pth")
        done = os.path.join(unit, "target0_noTTA_summary.json")
        if os.path.exists(done) and os.path.exists(ckpt):
            log(f"seed{seed}: training SKIP (done)")
        else:
            seed_all(seed)
            model = __import__("timm").create_model("vit_base_patch16_224",
                                                    pretrained=True, num_classes=2).cuda()
            opt = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WD,
                                    betas=(0.9, 0.999), eps=1e-8)
            scaler = torch.cuda.amp.GradScaler()
            start_ep, gs = 0, 0
            best = dict(ba=-1.0, nll=1e9, epoch=-1, pat=0, hist=[])
            if os.path.exists(latest):
                ck = torch.load(latest, map_location="cuda", weights_only=False)
                model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
                scaler.load_state_dict(ck["scaler"])
                start_ep, gs = ck["epoch"] + 1, ck["global_step"]
                best = ck["best"]
                log(f"seed{seed}: RESUME at epoch {start_ep + 1} (best ep {best['epoch']} "
                    f"ba={best['ba']:.4f})")
            train_loader = loader_for(src, shuffle=True)
            spe = len(train_loader)
            epochs = 1 if args.canary else MAX_EPOCHS
            total = epochs * spe
            for ep in range(start_ep, epochs):
                model.train()
                tl, n, t0 = 0.0, 0, time.time()
                for x, y, _, _ in train_loader:
                    lr = lr_at(gs, spe, total)
                    for g in opt.param_groups:
                        g["lr"] = lr
                    x, y = x.cuda(), y.cuda()
                    opt.zero_grad()
                    with torch.cuda.amp.autocast():
                        loss = F.cross_entropy(model(x), y)
                    scaler.scale(loss).backward()
                    scaler.step(opt); scaler.update()
                    tl += float(loss.item()) * len(x); n += len(x); gs += 1
                vm = evaluate(model, val)
                better = (vm["balanced_accuracy"] > best["ba"] + 1e-9) or (
                    abs(vm["balanced_accuracy"] - best["ba"]) <= 1e-9 and vm["nll"] < best["nll"])
                if better:
                    best.update(ba=vm["balanced_accuracy"], nll=vm["nll"], epoch=ep + 1, pat=0)
                    torch.save({"model": model.state_dict()}, ckpt + ".tmp")
                    os.replace(ckpt + ".tmp", ckpt)
                else:
                    best["pat"] += 1
                best["hist"].append(dict(epoch=ep + 1, train_loss=tl / max(1, n),
                                         val_acc=vm["accuracy"], val_ba=vm["balanced_accuracy"],
                                         val_nll=vm["nll"], lr=float(lr)))
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "scaler": scaler.state_dict(), "epoch": ep, "global_step": gs,
                            "best": best}, latest + ".tmp")
                os.replace(latest + ".tmp", latest)
                atomic_json(os.path.join(unit, "history.json"), best["hist"])
                log(f"seed{seed} ep {ep+1}/{MAX_EPOCHS} loss={tl/max(1,n):.4f} "
                    f"val_ba={vm['balanced_accuracy']:.4f} best={best['epoch']} "
                    f"pat={best['pat']}/{PATIENCE} {fmt(time.time()-t0)}")
                if best["pat"] >= PATIENCE:
                    log(f"seed{seed}: early stop at epoch {ep+1}")
                    break
            blob = torch.load(ckpt, map_location="cuda", weights_only=False)
            model.load_state_dict(blob.get("model", blob))
            tm = evaluate(model, tgt)
            out = dict(seed=int(seed), source_n=int(len(src)), val_n=int(len(val)),
                       target_n=int(len(tgt)), best_epoch=int(best["epoch"]),
                       best_val_ba=float(best["ba"]), best_val_nll=float(best["nll"]),
                       target0_noTTA_acc=float(tm["accuracy"]),
                       target0_noTTA_ba=float(tm["balanced_accuracy"]),
                       target0_noTTA_nll=float(tm["nll"]))
            atomic_json(done, out)
            log(f"seed{seed}: training DONE best={best['epoch']} "
                f"target acc={tm['accuracy']:.4f} ba={tm['balanced_accuracy']:.4f}")
        if args.train_only:
            continue

        model = load_vit(ckpt, seed).cuda().eval()
        tnames = tent_tensors(model)
        assert len(tnames) == 50, f"expected 50 LayerNorm tensors, got {len(tnames)}"
        unit_loader = loader_for(tgt)
        if not unit_done(unit, "eata") and args.with_eata:
            fpath = os.path.join(unit, "fisher.npz")
            if not os.path.exists(fpath):
                fm = load_vit(ckpt, FISHER_SEED).cuda()
                compute_fisher(fm, "vit", unit_loader, tent_tensors(fm),
                               unit_dir=unit)
                del fm
                torch.cuda.empty_cache()
        battery(model, "vit", unit_loader, unit,
                methods=METHODS if args.with_eata else [m for m in METHODS if m != "eata"],
                tnames=tnames, budget_seconds=args.budget_seconds)

    verify_job(env, "center0")


def job_budget_heldout(env, args):
    """Adapt on k% of a 90% pool; score one fixed 10% held-out set every run."""
    import torch
    from torch.utils.data import DataLoader, Subset

    meta = load_metadata(env.meta)
    _, _, tgt = partitions(meta, "center2")
    rng = np.random.default_rng(20261020)
    ids = tgt["patch_id"].to_numpy()
    eval_mask = np.array([int(hashlib.md5(str(i).encode()).hexdigest(), 16) % 10 == 0 for i in ids])
    eval_df = tgt[eval_mask].reset_index(drop=True)
    pool_df = tgt[~eval_mask].reset_index(drop=True)
    log(f"budget: pool={len(pool_df)} eval={len(eval_df)} "
        f"eval slides={eval_df.slide.nunique()} tumor={eval_df.tumor.mean():.3f}")
    assert abs(eval_df.tumor.mean() - 0.5) < 0.02, "held-out set is not balanced"
    root = stage([pool_df, eval_df], env, "center2")
    eval_loader = DataLoader(PatchDS(eval_df, root), BATCH, shuffle=False, num_workers=2)

    for seed in args.seeds:
        ckpt = env.checkpoints["center2"][seed]
        if not os.path.exists(ckpt):
            log(f"seed{seed}: checkpoint missing, skipping")
            continue
        base = load_vit(ckpt, seed).cuda().eval()
        base_unit = os.path.join(env.out, "budget", f"seed{seed}")
        os.makedirs(base_unit, exist_ok=True)
        atomic_json(os.path.join(base_unit, "unit.json"),
                    dict(rotation="center2", seed=int(seed), design="heldout-10pct",
                         eval_n=int(len(eval_df)), pool_n=int(len(pool_df))))
        if not unit_done(base_unit, "no_tta"):
            seed_all(seed)
            P, L, I, S, _ = run_stream(copy.deepcopy(base), "vit", "no_tta", eval_loader,
                                       base_unit, [])
            finalize(base_unit, "no_tta", P, L, I, S, "center2", seed)
        ref = np.load(os.path.join(base_unit, "no_tta_predictions.npz"))
        ref_acc = float((ref["predictions"] == ref["labels"]).mean())
        for budget in args.budgets:
            bname = f"{budget}pct"
            n_take = max(1, int(round(len(pool_df) * budget / 100.0)))
            full_stream = n_take >= len(pool_df)
            for sseed in args.subset_seeds:
                unit = os.path.join(env.out, "budget", f"seed{seed}", bname, f"subset{sseed}")
                os.makedirs(unit, exist_ok=True)
                # A 100% stream is the whole pool: every subset seed draws the
                # same patches and TENT is deterministic, so copy the one run
                # instead of repeating it (saves ~4 h per source seed).
                if full_stream and sseed != args.subset_seeds[0]:
                    src = os.path.join(env.out, "budget", f"seed{seed}", bname,
                                       f"subset{args.subset_seeds[0]}", "tent_summary.json")
                    dst = os.path.join(unit, "tent_summary.json")
                    if os.path.exists(src) and not os.path.exists(dst):
                        out = read_json(src)
                        out["subset_seed"] = int(sseed)
                        out["copied_from"] = int(args.subset_seeds[0])
                        atomic_json(dst, out)
                        atomic_json(os.path.join(unit, "unit.json"),
                                    dict(rotation="center2", seed=int(seed), budget=bname,
                                         subset_seed=int(sseed), design="heldout-10pct",
                                         eval_n=int(len(eval_df)), pool_n=int(len(pool_df)),
                                         copied_from=int(args.subset_seeds[0])))
                        log(f"seed{seed}/{bname}/{sseed}: identical full-pool stream, "
                            f"copied from subset{args.subset_seeds[0]}")
                    else:
                        log(f"seed{seed}/{bname}/{sseed}: waiting for "
                            f"subset{args.subset_seeds[0]} to finish; re-run to copy")
                    continue
                atomic_json(os.path.join(unit, "unit.json"),
                            dict(rotation="center2", seed=int(seed), budget=bname,
                                 subset_seed=int(sseed), design="heldout-10pct",
                                 eval_n=int(len(eval_df)), pool_n=int(len(pool_df))))
                if unit_done(unit, "tent"):
                    log(f"seed{seed}/{bname}/{sseed}: SKIP")
                    continue
                r = np.random.default_rng(10000 * seed + 100 * budget + sseed)
                take = np.sort(r.choice(len(pool_df), size=n_take, replace=False))
                sub_loader = DataLoader(Subset(PatchDS(pool_df, root), take.tolist()),
                                        BATCH, shuffle=False, num_workers=2)
                model = copy.deepcopy(base)
                tnames = tent_tensors(model)
                seed_all(seed * 100 + sseed)
                P, L, I, S, _ = run_stream(model, "vit", "tent", sub_loader, unit, tnames,
                                           budget_seconds=args.budget_seconds)
                # The eval set is disjoint from the stream and is scored once with
                # theta_K, so it lives in its own directory and is never overwritten.
                eval_dir = os.path.join(unit, "eval")
                os.makedirs(eval_dir, exist_ok=True)
                P2, L2, I2, S2, _ = run_stream(model, "vit", "no_tta", eval_loader, eval_dir,
                                               [], budget_seconds=args.budget_seconds)
                assert np.array_equal(I2, ref["patch_ids"]), "eval patch order drifted"
                met = metrics_from_probs(P2, L2)
                hh2 = help_harm(ref["predictions"].astype(int), ref["labels"].astype(int),
                                P2.argmax(1), L2)
                out = dict(seed=int(seed), budget=bname, subset_seed=int(sseed),
                           n_adapt=int(n_take), n_stream=int(len(P)),
                           n_eval=int(len(eval_df)), frozen_eval_acc=ref_acc,
                           adaptation="tent", eval_model="theta_K", **met, **hh2)
                atomic_json(os.path.join(unit, "tent_summary.json"), out)
                log(f"seed{seed}/{bname}/{sseed}: eval acc={met['accuracy']:.4f} "
                    f"(frozen {ref_acc:.4f}) harm={hh2['harm']}")
                for f in ("state.pkl", "state_drive.pkl"):
                    try:
                        os.remove(os.path.join(unit, f))
                    except OSError:
                        pass
                for f in ("state.pkl", "state_drive.pkl"):
                    try:
                        os.remove(os.path.join(eval_dir, f))
                    except OSError:
                        pass
        del base
        torch.cuda.empty_cache()

    verify_job(env, "budget")


def job_phikon(env, args):
    """Frozen linear probe on Phikon; optional TTA battery on the same backbone."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    meta = load_metadata(env.meta)
    src, val, tgt = partitions(meta, "center2")
    root = stage([src, val, tgt], env, "center2")
    enc = phikon_encoder().cuda()
    dim = 768

    def feats(df, cache_name):
        cache = os.path.join(env.local, "phikon_feats", cache_name + ".npz")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        if os.path.exists(cache):
            z = np.load(cache)
            return z["x"], z["y"], z["ids"]
        loader = DataLoader(PatchDS(df, root), 128, shuffle=False, num_workers=2)
        X, Y, I = [], [], []
        enc.eval()
        with torch.no_grad():
            for i, (x, y, pid, _) in enumerate(loader):
                h = enc(pixel_values=x.cuda()).last_hidden_state[:, 0]
                X.append(h.half().cpu().numpy())
                Y.append(y.numpy()); I.extend(int(v) for v in pid)
                if i % 100 == 0:
                    log(f"phikon feats {cache_name} {i}/{len(loader)}")
        X = np.concatenate(X); Y = np.concatenate(Y).astype(np.int64)
        atomic_npz(cache, x=X, y=Y, ids=np.asarray(I, np.int64))
        return X, Y, np.asarray(I, np.int64)

    Xs, Ys, Is = feats(src, "source")
    Xv, Yv, Iv = feats(val, "val")
    Xt, Yt, It = feats(tgt, "target")
    log(f"phikon: features source={Xs.shape} val={Xv.shape} target={Xt.shape}")
    out = os.path.join(env.out, "phikon")
    os.makedirs(out, exist_ok=True)

    for seed in args.seeds:
        seed_all(seed)
        head = nn.Linear(dim, 2).cuda()
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
        best = dict(ba=-1, sd=None)
        for ep in range(20):
            head.train()
            perm = np.random.default_rng(seed + ep).permutation(len(Xs))
            for i in range(0, len(perm), 4096):
                idx = perm[i:i + 4096]
                x = torch.from_numpy(Xs[idx].astype(np.float32)).cuda()
                y = torch.from_numpy(Ys[idx]).cuda()
                opt.zero_grad()
                loss = nn.functional.cross_entropy(head(x), y)
                loss.backward(); opt.step()
            head.eval()
            with torch.no_grad():
                pv = head(torch.from_numpy(Xv.astype(np.float32)).cuda()).argmax(1).cpu().numpy()
            ba = 0.5 * (((pv == 1) & (Yv == 1)).sum() / max(1, (Yv == 1).sum())
                        + ((pv == 0) & (Yv == 0)).sum() / max(1, (Yv == 0).sum()))
            if ba > best["ba"]:
                best = dict(ba=float(ba), sd=copy.deepcopy(head.state_dict()), epoch=ep + 1)
        head.load_state_dict(best["sd"]); head.eval()
        with torch.no_grad():
            logits = head(torch.from_numpy(Xt.astype(np.float32)).cuda())
            pt = torch.softmax(logits, 1).cpu().numpy()
        met = metrics_from_probs(pt, Yt)
        met.update(seed=int(seed), best_epoch=int(best["epoch"]), val_ba=float(best["ba"]),
                   encoder="owkin/phikon", head="linear", stage="frozen-probe")
        atomic_json(os.path.join(out, f"seed{seed}_probe.json"), met)
        torch.save({"head": head.state_dict(), "val_ba": best["ba"]}, os.path.join(out, f"seed{seed}_head.pth"))
        log(f"phikon seed{seed}: probe target acc={met['accuracy']:.4f} ba={met['balanced_accuracy']:.4f}")
        if not args.with_tta:
            continue
        enc.head = head
        unit = os.path.join(out, f"seed{seed}_tta")
        os.makedirs(unit, exist_ok=True)
        atomic_json(os.path.join(unit, "unit.json"),
                    dict(rotation="center2", seed=int(seed), encoder="owkin/phikon"))
        loader = DataLoader(PatchDS(tgt, root), BATCH, shuffle=False, num_workers=2)
        battery(enc, "phikon", loader, unit, tnames=tent_tensors(enc),
                budget_seconds=args.budget_seconds)
        del enc.head

    verify_job(env, "phikon")


def job_eata_full(env, args):
    """Full-parameter EATA. Run --probe first; it prints selection/loss diagnostics."""
    import torch
    from torch.utils.data import DataLoader

    meta = load_metadata(env.meta)
    _, _, tgt = partitions(meta, "center2")
    root = stage([tgt], env, "center2")
    loader = DataLoader(PatchDS(tgt, root), BATCH, shuffle=False, num_workers=2)

    for seed in args.seeds:
        ckpt = env.checkpoints["center2"][seed]
        unit = os.path.join(env.out, "eata_full", f"seed{seed}")
        os.makedirs(unit, exist_ok=True)
        atomic_json(os.path.join(unit, "unit.json"),
                    dict(rotation="center2", seed=int(seed), adapter="EATA-full",
                         lr_full=float(args.lr_full)))
        if unit_done(unit, "eata"):
            log(f"eata_full seed{seed}: SKIP")
            continue
        model = load_vit(ckpt, seed).cuda()
        if args.probe:
            from torch.utils.data import Subset
            sub = Subset(PatchDS(tgt, root), list(range(min(len(tgt), BATCH * 200))))
            probe = DataLoader(sub, BATCH, shuffle=False, num_workers=2)
            run_probe(model, probe, unit, lr=args.lr_full)
            continue
        # finalize() scores the adapted stream against the frozen one, so the
        # frozen reference must exist in this unit before the EATA run starts.
        if not os.path.exists(os.path.join(unit, "no_tta_predictions.npz")):
            frozen = os.path.join(unit, "frozen")
            os.makedirs(frozen, exist_ok=True)
            seed_all(seed)
            P0, L0, I0, S0, _ = run_stream(model, "vit", "no_tta", loader, frozen, [])
            finalize(frozen, "no_tta", P0, L0, I0, S0, "center2", seed)
            shutil.copy2(os.path.join(frozen, "no_tta_predictions.npz"),
                         os.path.join(unit, "no_tta_predictions.npz"))
            torch.cuda.empty_cache()
        tnames = list(dict(model.named_parameters()).keys())
        fpath = os.path.join(unit, "fisher.npz")
        if not os.path.exists(fpath):
            compute_fisher(model, "vit", DataLoader(PatchDS(tgt, root), BATCH), tnames,
                           unit_dir=unit)
            model = load_vit(ckpt, seed).cuda()
        P, L, I, S, _ = run_stream(model, "vit", "eata", loader, unit, tnames,
                                   budget_seconds=args.budget_seconds, full_params=True,
                                   lr_full=args.lr_full)
        finalize(unit, "eata", P, L, I, S, "center2", seed)
        del model
        torch.cuda.empty_cache()

    verify_job(env, "eata_full")


def run_probe(model, loader, unit_dir, lr=LR):
    """200-batch diagnostic: selection rate, entropy, loss magnitude, accuracy."""
    import torch
    import torch.nn.functional as F
    model.eval()
    ent, sel, losses = [], [], []
    correct = total = 0
    tnames = list(dict(model.named_parameters()).keys())
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=MOM)
    with torch.enable_grad():
        for i, (x, y, _, _) in enumerate(loader):
            x = x.cuda()
            p = F.softmax(model(x), 1)
            e = -(p * torch.log(p + 1e-12)).sum(1)
            mask = e < E0
            correct += int((p.argmax(1) == y.cuda()).sum())
            total += int(len(y))
            loss = -(p[mask] * torch.log(p[mask] + 1e-12)).sum(1).mean() if mask.any() else None
            if loss is not None:
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(float(loss))
            ent.append(float(e.mean())); sel.append(float(mask.float().mean()))
            if i % 25 == 0:
                log(f"probe {i}: entropy={ent[-1]:.3f} selected={sel[-1]:.3f} "
                    f"acc={correct / max(1, total):.4f} "
                    f"loss={losses[-1] if losses else float('nan'):.4f}")
    out = dict(mean_entropy=float(np.mean(ent)), mean_selected=float(np.mean(sel)),
               mean_accuracy=float(correct / max(1, total)),
               mean_loss=float(np.mean(losses)) if losses else None, batches=int(len(ent)),
               e0=float(E0), lr=float(lr))
    atomic_json(os.path.join(unit_dir, "probe.json"), out)
    log("PROBE RESULT " + json.dumps(out))
    log("Sanity: mean_selected should be well above 0 (typ. 0.5-0.9), mean loss "
        "finite, and mean_accuracy should stay near the frozen baseline. If "
        "accuracy collapses, stop and lower --max-hours; the run is diverging.")


# --------------------------------------------------------------------- verify
def verify_job(env, which):
    root = os.path.join(env.out, which)
    if not os.path.isdir(root):
        log(f"verify[{which}]: nothing on disk")
        return True
    ok, seen = True, 0
    for dirpath, _, files in os.walk(root):
        summaries = [f for f in files if f.endswith("_summary.json")]
        for s in summaries:
            unit = dirpath
            method = s[:-len("_summary.json")]
            if method == "target0_noTTA":
                continue  # training summary, not a prediction stream
            p = os.path.join(unit, f"{method}_predictions.npz")
            if not os.path.exists(p):
                log(f"FAIL missing {p}")
                ok = False
                continue
            seen += 1
            z = np.load(p)
            P = z["probabilities"].astype(np.float64)
            pred = z["predictions"].astype(np.int64)
            lab = z["labels"].astype(np.int64)
            summ = read_json(os.path.join(unit, s))
            checks = {
                "probabilities sum to 1": np.allclose(P.sum(1), 1, atol=2e-5),
                "predictions are argmax": np.array_equal(pred, P.argmax(1)),
                "n matches summary": int(summ.get("n", -1)) == len(lab),
                "accuracy matches": abs(summ.get("accuracy", -1) - float((pred == lab).mean())) < 1e-9,
            }
            ref = os.path.join(unit, "no_tta_predictions.npz")
            if method != "no_tta" and os.path.exists(ref):
                r = np.load(ref)
                checks["ids match frozen"] = np.array_equal(z["patch_ids"], r["patch_ids"])
                checks["labels match frozen"] = np.array_equal(lab, r["labels"])
                hh = help_harm(r["predictions"].astype(int), r["labels"].astype(int), pred, lab)
                checks["help/harm match"] = (hh["help"] == summ.get("help")
                                             and hh["harm"] == summ.get("harm"))
            bad = [k for k, v in checks.items() if not v]
            if bad:
                ok = False
                log(f"FAIL {method} in {unit}: {bad}")
            else:
                log(f"ok   {method:9s} n={len(lab):6d} acc={summ.get('accuracy', float('nan')):.4f} "
                    f"{os.path.relpath(unit, root)}")
    log(f"verify[{which}]: {seen} prediction files, {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------- smoke
def smoke(args):
    """CPU test of the engine: fresh run vs forced interruption + resume."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    torch.manual_seed(0)
    global E0, SAVE_EVERY
    old_e0 = E0
    # A randomly initialised toy net is maximally uncertain, so the real EATA
    # entropy threshold would reject every sample and no parameter would move.
    # Raise it for this test so the update + resume path is actually exercised.
    E0 = 5.0
    tmp = os.path.join(args.local_dir, "smoke")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    n, d = 96, 8
    X = torch.randn(n, 1, d, d)
    y = (X.mean(dim=(1, 2, 3)) > 0).long()
    ids = np.arange(n)

    class Toy(Dataset):
        def __len__(self):
            return n
        def __getitem__(self, i):
            return X[i], int(y[i]), int(ids[i]), int(i // 24)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(1, 4, 3, padding=1)
            self.ln = nn.LayerNorm([4, d, d])
            self.fc = nn.Linear(4 * d * d, 2)
        def forward(self, x):
            return self.fc(self.ln(self.c(x)).flatten(1))

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def make():
        torch.manual_seed(0)
        m = Net().to(dev)
        return m

    # ---- fresh vs interrupted+resumed, for every adapter -----------------
    def prep(unit, mode):
        os.makedirs(unit, exist_ok=True)
        if mode == "eata" and not os.path.exists(os.path.join(unit, "fisher.npz")):
            fm = make()
            compute_fisher(fm, "toy", DataLoader(Toy(), 8, shuffle=False),
                           tent_tensors(fm), n=32, unit_dir=unit)

    def one(unit, mode, interrupt_at=None):
        prep(unit, mode)
        m = make()
        loader = DataLoader(Toy(), 8, shuffle=False)
        tnames = tent_tensors(m)
        init = {n: p.detach().clone() for n, p in m.named_parameters() if n in tnames}
        if interrupt_at is not None:
            global SAVE_EVERY
            old = SAVE_EVERY
            SAVE_EVERY = interrupt_at
            try:
                try:
                    # negative budget => save at the next checkpoint and exit(0)
                    run_stream(m, "toy", mode, loader, unit, tnames, budget_seconds=-1)
                except SystemExit:
                    pass
            finally:
                SAVE_EVERY = old
            st = os.path.exists(os.path.join(unit, "state.pkl"))
            log(f"smoke[{mode}]: interrupted run saved state={st}")
            if not st:
                log(f"SMOKE FAIL: {mode} interruption produced no resumable state")
            return None
        P, L, I, S, _ = run_stream(m, "toy", mode, loader, unit, tnames)
        sd = dict(m.named_parameters())
        drift = max(float((sd[n].detach() - init[n]).abs().max()) for n in tnames)
        return P, L, I, S, drift

    ok = True
    for mode in ("no_tta", "tent", "eata", "shot"):
        uf = os.path.join(tmp, mode, "fresh")
        ur = os.path.join(tmp, mode, "resumed")
        a = one(uf, mode)
        one(ur, mode, interrupt_at=4)
        b = one(ur, mode)
        if a is None or b is None:
            log(f"SMOKE FAIL[{mode}]: a run produced no output")
            ok = False
            continue
        for name, x, z in [("probs", a[0], b[0]), ("ids", a[2], b[2]),
                           ("slides", a[3], b[3]), ("labels", a[1], b[1])]:
            same = np.array_equal(x, z)
            if not same:
                log(f"smoke[{mode}] {name}: DIFFERS")
            ok = ok and same
        diff = float(np.abs(a[0] - b[0]).max())
        log(f"smoke[{mode}]: max prob difference fresh vs resumed {diff:.3e} "
            f"({'identical' if diff == 0 else 'DIFFERS'})")
        ok = ok and diff < 1e-6
        if mode != "no_tta":
            log(f"smoke[{mode}]: parameter drift over the stream {a[4]:.3e}")
            if a[4] <= 0:
                log(f"SMOKE FAIL[{mode}]: adapter never changed a parameter, "
                    f"so the resume test proves nothing")
                ok = False

    # ---- artifact + summary path, through the same finalize used on GPU --
    for mode in ("no_tta", "tent", "eata", "shot"):
        unit = os.path.join(tmp, mode, "fresh")
        if mode != "no_tta":
            fz = os.path.join(tmp, mode, "frozen")
            os.makedirs(fz, exist_ok=True)
            m = make()
            loader = DataLoader(Toy(), 8, shuffle=False)
            P, L, I, S, _ = run_stream(m, "toy", "no_tta", loader, fz, [])
            finalize(fz, "no_tta", P, L, I, S, "toy", 0)
            shutil.copy2(os.path.join(fz, "no_tta_predictions.npz"),
                         os.path.join(unit, "no_tta_predictions.npz"))
        z = np.load(os.path.join(unit, "probs.npy"))
        L = np.load(os.path.join(unit, "labels.npy"))
        I = np.load(os.path.join(unit, "ids.npy"))
        S = np.load(os.path.join(unit, "slides.npy"))
        finalize(unit, mode, z, L, I, S, "toy", 0)
    env = type("E", (), {"out": tmp})()
    ok = verify_job(env, "") and ok
    E0 = old_e0
    log("SMOKE " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# ------------------------------------------------------------------ preflight
def preflight(env, args):
    import torch
    ok = True
    log("== preflight ==")
    log(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        log("WARN: no GPU. Jobs will be very slow or refuse to run.")
    else:
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        log(f"gpu {name} {mem:.1f} GB")
        if mem < 14:
            log("WARN: less than 14 GB VRAM; batch 64 on ViT-B/16 may OOM.")
    log(f"disk free (scratch): {free_gb(env.local):.1f} GB "
        f"(dataset needs ~12 GB when staging center-2 + center-0)")
    for label, path in [("metadata", env.meta), ("archive", env.archive),
                        ("extracted", env.extracted)]:
        log(f"{label:10s}: {path} {'OK' if path else 'MISSING'}")
        ok = ok and (path or label == "extracted" or label == "archive")
    if not (env.archive or env.extracted):
        log("ACTION: need either archive.tar.gz or an extracted patches/ tree.")
        ok = False
    needs = {"center0": [], "budget_heldout": args.seeds, "phikon": args.seeds,
             "eata_full": args.seeds, "verify": list(range(1, 6)),
             "preflight": args.seeds, "smoke": []}[args.job]
    for seed in needs:
        p = env.checkpoints["center2"].get(seed)
        exists = bool(p and os.path.exists(p))
        log(f"ckpt center2 seed{seed}: {p if p else 'MISSING'} {'OK' if exists else ''}")
        if not exists:
            log(f"ACTION: seed {seed} checkpoint not found; pass --seeds with the ones you have.")
            ok = False
    if args.job == "phikon" or args.check_phikon:
        try:
            from transformers import AutoModel  # noqa: F401
        except Exception as exc:
            log(f"phikon: transformers not importable -> {exc}")
            log("ACTION: !pip install -q transformers  (then set HF_TOKEN)")
            ok = False
        else:
            try:
                kw = {}
                tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                if tok:
                    kw["token"] = tok
                AutoModel.from_pretrained("owkin/phikon", add_pooling_layer=False, **kw)
                log("phikon: weights accessible")
            except Exception as exc:
                log(f"phikon: NOT accessible -> {exc}")
                log("ACTION: accept the terms on huggingface.co/owkin/phikon and set HF_TOKEN.")
                ok = False
    elif args.job == "preflight":
        log("phikon: not checked (add --check-phikon before running that job)")
    log("preflight " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", choices=["center0", "budget_heldout", "phikon", "eata_full",
                                    "preflight", "verify", "smoke"])
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="source seeds; default 2 3 4 5 for center0 (seed 1 exists), "
                         "1 2 3 4 5 for the other jobs")
    ap.add_argument("--subset-seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 5, 25, 100])
    ap.add_argument("--drive-root", default="/content/drive/MyDrive/Next Research")
    ap.add_argument("--local-dir", default="/content/cg_tta_work")
    ap.add_argument("--out-subdir", default="03_Experiments/15_SafeTTA_TMI")
    ap.add_argument("--archive", default=None)
    ap.add_argument("--extracted", default=None)
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--with-eata", action="store_true", default=True)
    ap.add_argument("--no-eata", dest="with_eata", action="store_false")
    ap.add_argument("--with-tta", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--lr-full", type=float, default=LR_FULL,
                    help="SGD step for full-parameter EATA; default is the published "
                         f"2.5e-4. The calibrated stable runs use --lr-full "
                         f"{LR_FULL_CALIBRATED} (post-hoc; see README).")
    ap.add_argument("--canary", action="store_true",
                    help="center0 only: 1 epoch on 2.5k/0.6k/0.6k patches, full pipeline, "
                         "writes to --out-subdir (use a separate one)")
    ap.add_argument("--check-phikon", action="store_true",
                    help="in preflight, also verify transformers + HF access to owkin/phikon")
    args = ap.parse_args()
    if args.seeds is None:
        args.seeds = [2, 3, 4, 5] if args.job == "center0" else [1, 2, 3, 4, 5]
    args.budget_seconds = args.max_hours * 3600 if args.max_hours > 0 else None

    if args.job == "smoke":
        return smoke(args)

    env = Env(args)
    if args.archive:
        env.archive = args.archive
    if args.extracted:
        env.extracted = args.extracted
    env.describe()

    if args.job == "preflight":
        return preflight(env, args)
    if args.job == "verify":
        which = sys.argv[sys.argv.index("verify") + 1] if len(sys.argv) > sys.argv.index("verify") + 1 and not sys.argv[sys.argv.index("verify") + 1].startswith("-") else ""
        ok = True
        for job in (["center0", "budget", "phikon", "eata_full"] if not which else [which]):
            ok = verify_job(env, job) and ok
        return 0 if ok else 1
    if args.job == "center0":
        job_center0(env, args)
    elif args.job == "budget_heldout":
        job_budget_heldout(env, args)
    elif args.job == "phikon":
        job_phikon(env, args)
    elif args.job == "eata_full":
        job_eata_full(env, args)
    log(f"total wall {fmt(time.time() - T0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

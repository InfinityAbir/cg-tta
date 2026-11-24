#!/usr/bin/env python3
"""Second-benchmark TTA audit on Colab: NCT-CRC-HE-100K -> CRC-VAL-HE-7K.

Standalone and resumable. Nothing large ever goes to Google Drive:

  * The Zenodo zips are downloaded to Colab scratch (/content/crc_work) and
    packed into one uint8 memmap per dataset. The 11.7 GB zip is deleted after
    packing, so peak local disk use is about 30 GB.
  * Drive only receives small things: source checkpoints (one 0.34 GB best.pth
    per seed, plus one 1 GB last.pt for the seed being trained, deleted when
    that seed finishes), prediction files (a few MB), logs.
  * A new Colab session only needs `prepare` again for the training step. The
    TTA step needs just CRC-VAL-HE-7K (0.8 GB download).

Jobs (each is idempotent: run the same command again and it continues):

  smoke         CPU self-test with tiny synthetic data. Includes an interrupted
                and resumed run that must reproduce the uninterrupted run.
  prepare       download + md5 check + pack to memmap  (--only train|val)
  train         source ViT-B/16, 9-class, per seed. Resumes mid-epoch.
  tta           frozen, TENT, Flip-TTA, EATA-LN, SHOT-IM on CRC-VAL. Resumes per batch.
  verify        re-reads every prediction file and re-checks it
  status        shows what is finished and what is missing
  pack_results  zips predictions/summaries (no checkpoints) for download

Typical Colab session:

  from google.colab import drive; drive.mount('/content/drive')
  !pip -q install timm scikit-learn
  !python crc_colab.py status
  !python crc_colab.py prepare
  !python crc_colab.py train --seeds 1 2 3 --max-hours 10.5
  !python crc_colab.py tta --seeds 1 2 3
  !python crc_colab.py verify
  !python crc_colab.py pack_results
"""
import argparse
import copy
import hashlib
import io
import json
import math
import os
import pickle
import queue
import random
import shutil
import subprocess
import sys
import threading
import time
import zipfile

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

VERSION = "2026-11-24-crc-v1"
T0 = time.time()

CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
TUM = CLASSES.index("TUM")
IMG_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
SIDE = 224

ZENODO_URLS = ["https://zenodo.org/api/records/1214456/files/{file}/content",
               "https://zenodo.org/records/1214456/files/{file}?download=1"]
SOURCES = {
    "train_norm": dict(file="NCT-CRC-HE-100K.zip", size=11690284003,
                       md5="6fd702d11df6292bc054397ae038a464", n=100000),
    "train_nonorm": dict(file="NCT-CRC-HE-100K-NONORM.zip", size=11726652899,
                         md5="035777cf327776a71a05c95da6d6325f", n=100000),
    "val": dict(file="CRC-VAL-HE-7K.zip", size=800276929,
                md5="2fd1651b4f94ebd818ebf90ad2b6ce06", n=7180),
}

# Same recipe as the Camelyon17 experiments (cgtta_colab.py).
BATCH = 64
LR, MOM, CLIP = 1e-3, 0.9, 1.0          # TTA optimiser, LayerNorm tensors only
BETA = 1.0
FISHER_N = 1024
IM_W = 1.0
LR_MAX, LR_MIN, LR_START, WD = 1e-4, 1e-6, 1e-5, 0.05
METHODS = ["no_tta", "tent", "flip_tta", "eata", "shot"]


# ---------------------------------------------------------------- utilities
def log(*a):
    print(*a, flush=True)


def fmt(sec):
    sec = int(sec)
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return (f"{h}h" if h else "") + (f"{m}m" if m or h else "") + f"{s}s"


def md5_file(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return default


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


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


def safe_copy_to(src, dst):
    """Copy src to dst through a temp name, so an interrupted copy never leaves a
    half-written file under the real name (important on Drive FUSE)."""
    tmp = dst + ".tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def clean_tmp(folder):
    for name in os.listdir(folder):
        if name.endswith(".tmp"):
            try:
                os.remove(os.path.join(folder, name))
            except OSError:
                pass


class Env:
    def __init__(self, args):
        self.args = args
        drive = "/content/drive/MyDrive"
        if args.out:
            self.out = os.path.abspath(args.out)
        elif os.path.isdir(drive):
            self.out = os.path.join(drive, "Next Research", "CRC_second_dataset")
        else:
            self.out = os.path.abspath("crc_out")
        if args.work:
            self.work = os.path.abspath(args.work)
        elif os.path.isdir("/content"):
            self.work = "/content/crc_work"
        else:
            self.work = os.path.abspath("crc_work")
        os.makedirs(self.out, exist_ok=True)
        os.makedirs(self.work, exist_ok=True)
        self.on_drive = self.out.startswith("/content/drive")
        ov = read_json(os.path.join(self.work, "sources_override.json"))
        if ov:
            for k, v in ov.items():
                SOURCES[k].update(v)

    def drive_guard(self, need_gb):
        free = free_gb(self.out)
        log(f"free space where outputs go: {free:.1f} GB (this job needs about {need_gb:.1f} GB)")
        if self.on_drive and free < need_gb:
            raise SystemExit(f"Not enough free Drive space ({free:.1f} GB < {need_gb:.1f} GB). "
                             "Free up Drive or use a smaller --seeds list; nothing was written.")


# ------------------------------------------------------------------ prepare
def download(fname, dest, size, md5, skip_md5):
    ok = dest + ".ok"
    if os.path.exists(ok):
        log(f"download: {os.path.basename(dest)} already verified")
        return
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if have > size:
        os.remove(dest)
        have = 0
    # No curl --retry: on a dropped connection curl's own retry restarted the file from byte 0 and
    # threw away 3 GB. Here curl just exits, and this loop resumes from the bytes on disk (-C -).
    for attempt in range(1, 41):
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if have == size:
            break
        url = ZENODO_URLS[(attempt - 1) % len(ZENODO_URLS)].format(file=fname)
        log(f"download attempt {attempt}: {have / 1e9:.2f}/{size / 1e9:.2f} GB from {url}")
        cmd = ["curl", "-L", "--fail", "-C", "-", "--speed-time", "90", "--speed-limit", "20000",
               "--connect-timeout", "30", "-sS", "-o", dest, url]
        proc = subprocess.Popen(cmd)
        t_start, b_start = time.time(), have
        while proc.poll() is None:
            time.sleep(30)
            now = os.path.getsize(dest) if os.path.exists(dest) else 0
            rate = (now - b_start) / max(1e-6, time.time() - t_start) / 1e6
            log(f"  {now / 1e9:.2f}/{size / 1e9:.2f} GB  {rate:.1f} MB/s  "
                f"eta {fmt((size - now) / max(1.0, rate * 1e6))}")
        if proc.returncode != 0:
            time.sleep(10)
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if have != size:
        raise SystemExit(f"download incomplete: {have} of {size} bytes. Run the command again to resume.")
    if not skip_md5:
        t = time.time()
        got = md5_file(dest)
        log(f"md5 {got} ({fmt(time.time() - t)})")
        if got != md5:
            os.remove(dest)
            raise SystemExit(f"md5 mismatch for {dest}: expected {md5}. The file was deleted; run again.")
    open(ok, "w").write("ok\n")


def class_of(name):
    parts = name.replace("\\", "/").split("/")
    cls = parts[-2]
    if cls not in CLASSES:
        raise SystemExit(f"unexpected class folder '{cls}' in {name}")
    return CLASSES.index(cls)


def decode(blob):
    from PIL import Image
    im = Image.open(io.BytesIO(blob)).convert("RGB")
    resized = 0
    if im.size != (SIDE, SIDE):
        im = im.resize((SIDE, SIDE), Image.BICUBIC)
        resized = 1
    return np.asarray(im, dtype=np.uint8), resized


def pack(zip_path, key, work, expect_n, workers=4):
    from concurrent.futures import ThreadPoolExecutor
    ready = os.path.join(work, f"{key}_ready.json")
    if os.path.exists(ready):
        log(f"pack: {key} already packed")
        return read_json(ready)
    z = zipfile.ZipFile(zip_path)
    infos = [i for i in z.infolist()
             if not i.is_dir() and i.filename.lower().endswith(IMG_EXT) and "__MACOSX" not in i.filename]
    infos.sort(key=lambda i: i.filename)
    n = len(infos)
    if n != expect_n:
        raise SystemExit(f"{key}: found {n} images, expected {expect_n}. The download may be wrong.")
    names = [i.filename for i in infos]
    labels = np.array([class_of(nm) for nm in names], dtype=np.uint8)
    list_sha = sha256_bytes("\n".join(names).encode())
    xp = os.path.join(work, f"{key}_x.npy")
    prog = os.path.join(work, f"{key}_pack_progress.json")
    pr = read_json(prog)
    start = 0
    if os.path.exists(xp) and pr and pr.get("list_sha") == list_sha and pr.get("n") == n:
        mm = np.lib.format.open_memmap(xp, mode="r+")
        start = int(pr["done"])
        log(f"pack: {key} RESUME at {start}/{n}")
    else:
        mm = np.lib.format.open_memmap(xp, mode="w+", dtype=np.uint8, shape=(n, SIDE, SIDE, 3))
        log(f"pack: {key} START 0/{n}")
    resized = int(pr.get("resized", 0)) if (pr and start) else 0
    chunk, t0 = 500, time.time()
    with ThreadPoolExecutor(workers) as pool:
        for s in range(start, n, chunk):
            e = min(n, s + chunk)
            blobs = [z.read(infos[i]) for i in range(s, e)]
            res = list(pool.map(decode, blobs))
            mm[s:e] = np.stack([r[0] for r in res])
            resized += sum(r[1] for r in res)
            mm.flush()
            atomic_json(prog, dict(done=e, n=n, list_sha=list_sha, resized=resized))
            if (e // chunk) % 10 == 0 or e == n:
                el = time.time() - t0
                log(f"pack {key} {e}/{n} ({100 * e / n:.1f}%) eta {fmt((n - e) * el / max(1, e - start))}")
    del mm
    np.save(os.path.join(work, f"{key}_y.npy"), labels)
    atomic_json(os.path.join(work, f"{key}_names.json"), names)
    counts = {CLASSES[c]: int((labels == c).sum()) for c in range(len(CLASSES))}
    info = dict(key=key, n=n, class_counts=counts, list_sha256=list_sha,
                labels_sha256=sha256_bytes(labels.tobytes()), resized=resized, version=VERSION)
    atomic_json(ready, info)
    log(f"pack: {key} done, class counts {counts}, resized {resized}")
    return info


def job_prepare(env, args, only=None):
    keys = []
    train_key = "train_" + args.variant
    if only in (None, "train"):
        keys.append(train_key)
    if only in (None, "val"):
        keys.append("val")
    need = 0.0
    for k in keys:
        if not os.path.exists(os.path.join(env.work, f"{k}_ready.json")):
            need += SOURCES[k]["size"] / 1e9 + SOURCES[k]["n"] * SIDE * SIDE * 3 / 1e9
    free = free_gb(env.work)
    log(f"local scratch free: {free:.1f} GB, prepare needs about {need:.1f} GB")
    if need and free < need + 3:
        raise SystemExit(f"Not enough local disk ({free:.1f} GB). Use a runtime with more disk.")
    manifest = {}
    for k in keys:
        src = SOURCES[k]
        if os.path.exists(os.path.join(env.work, f"{k}_ready.json")):
            manifest[k] = read_json(os.path.join(env.work, f"{k}_ready.json"))
            log(f"prepare: {k} ready")
            continue
        dest = os.path.join(env.work, src["file"])
        download(src["file"], dest, src["size"], src["md5"], args.skip_md5)
        manifest[k] = pack(dest, k, env.work, src["n"])
        if not args.keep_zips:
            for f in (dest, dest + ".ok"):
                try:
                    os.remove(f)
                except OSError:
                    pass
    old = read_json(os.path.join(env.out, "dataset_manifest.json"), {})
    old.update(manifest)
    atomic_json(os.path.join(env.out, "dataset_manifest.json"), old)
    log("prepare: done")


def load_data(env, key):
    ready = read_json(os.path.join(env.work, f"{key}_ready.json"))
    if not ready:
        raise SystemExit(f"{key} is not packed in {env.work}. Run: python crc_colab.py prepare"
                         + (" --only val" if key == "val" else ""))
    x = np.load(os.path.join(env.work, f"{key}_x.npy"), mmap_mode="r")
    y = np.load(os.path.join(env.work, f"{key}_y.npy"))
    assert len(x) == len(y) == ready["n"], "packed arrays disagree with the manifest"
    return x, y


# ------------------------------------------------------------------- model
def build_model(arch, seed, pretrained):
    import timm
    seed_all(seed)
    return timm.create_model(arch, pretrained=pretrained, num_classes=len(CLASSES))


def to_input(xb, device):
    """uint8 NHWC -> float NCHW, mean/std 0.5 (the same normalisation as Camelyon17)."""
    import torch
    t = xb if torch.is_tensor(xb) else torch.from_numpy(xb)
    t = t.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0).sub_(0.5).div_(0.5)
    return t.contiguous()


class Prefetch:
    """Background thread that gathers batches from the memmap."""

    def __init__(self, x, y, batches, depth=4):
        import torch
        self.q = queue.Queue(maxsize=depth)
        self.n = len(batches)

        def work():
            try:
                for idx in batches:
                    self.q.put((torch.from_numpy(np.ascontiguousarray(x[idx])),
                                torch.from_numpy(y[idx].astype(np.int64)), idx))
            except BaseException as exc:  # surface errors in the main thread
                self.q.put(exc)

        self.t = threading.Thread(target=work, daemon=True)
        self.t.start()

    def __iter__(self):
        for _ in range(self.n):
            item = self.q.get()
            if isinstance(item, BaseException):
                raise item
            yield item


# ------------------------------------------------------------------ metrics
def ece15(conf, ok):
    edges = np.linspace(0, 1, 16)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += abs(ok[m].mean() - conf[m].mean()) * m.mean()
    return float(e)


def multiclass_metrics(P, L):
    P = np.asarray(P, np.float64)
    L = np.asarray(L, np.int64)
    pred = P.argmax(1)
    ok = pred == L
    recalls = [float((pred[L == c] == c).mean()) for c in range(len(CLASSES)) if (L == c).any()]
    f1 = []
    for c in range(len(CLASSES)):
        tp = float(((pred == c) & (L == c)).sum())
        fp = float(((pred == c) & (L != c)).sum())
        fn = float(((pred != c) & (L == c)).sum())
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    onehot = np.eye(len(CLASSES))[L]
    out = dict(n=int(len(L)), accuracy=float(ok.mean()), balanced_accuracy=float(np.mean(recalls)),
               macro_f1=float(np.mean(f1)),
               nll=float(-np.mean(np.log(np.clip(P[np.arange(len(L)), L], 1e-12, 1.0)))),
               brier=float(np.mean(((P - onehot) ** 2).sum(1))),
               ece_15=ece15(P.max(1), ok), mean_confidence=float(P.max(1).mean()),
               mean_entropy=float(-(P * np.log(P + 1e-12)).sum(1).mean()))
    try:
        from sklearn.metrics import roc_auc_score
        out["auroc_ovr_macro"] = float(roc_auc_score(L, P, multi_class="ovr", average="macro",
                                                     labels=list(range(len(CLASSES)))))
    except Exception:
        out["auroc_ovr_macro"] = None
    return out


def help_harm(ref_pred, pred, lab):
    pre, post = ref_pred == lab, pred == lab
    h, a = int((~pre & post).sum()), int((pre & ~post).sum())
    return dict(help=h, harm=a, net=h - a, harm_rate=float(a / max(1, int(pre.sum()))))


def tumor_view(ref_pred, pred, lab):
    """Tumour (TUM) versus the other eight classes: the false-positive / false-negative
    accounting used in the cost analysis."""
    t, r, p = lab == TUM, ref_pred == TUM, pred == TUM
    return dict(
        sensitivity=float(p[t].mean()) if t.any() else None,
        specificity=float((~p[~t]).mean()) if (~t).any() else None,
        fp_harm=int((~t & ~r & p).sum()),         # frozen right (non-tumour), now tumour
        fn_harm=int((t & r & ~p).sum()),          # frozen right (tumour), now non-tumour
        fn_recovered=int((t & ~r & p).sum()),     # frozen missed tumour, now found
        fp_corrected=int((~t & r & ~p).sum()),    # frozen false alarm, now cleared
        n_tumor=int(t.sum()))


# ------------------------------------------------------------------- train
def lr_at(step, total, warm):
    if step < warm:
        return LR_START + (LR_MAX - LR_START) * step / max(1, warm)
    prog = (step - warm) / max(1, total - warm)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * min(1.0, prog)))


def get_split(env, args, y):
    path = os.path.join(env.out, f"split_{args.variant}.npz")
    if os.path.exists(path):
        z = np.load(path)
        tr, va = z["train_idx"], z["val_idx"]
        assert len(tr) + len(va) == len(y), "stored split does not match the packed data"
        return tr, va
    rs = np.random.RandomState(0)
    va = []
    for c in range(len(CLASSES)):
        idx = np.where(y == c)[0]
        k = max(1, int(round(args.val_frac * len(idx))))
        va.extend(rs.choice(idx, size=k, replace=False).tolist())
    va = np.array(sorted(va), dtype=np.int64)
    mask = np.ones(len(y), bool)
    mask[va] = False
    tr = np.where(mask)[0].astype(np.int64)
    np.savez(path + ".tmp.npz", train_idx=tr, val_idx=va)
    os.replace(path + ".tmp.npz", path)
    return tr, va


def evaluate_val(model, x, y, idx, device, bs=128):
    import torch
    import torch.nn.functional as F
    model.eval()
    P = []
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            sel = idx[s:s + bs]
            xb = to_input(np.ascontiguousarray(x[sel]), device)
            P.append(F.softmax(model(xb).float(), 1).cpu().numpy())
    model.train()
    return multiclass_metrics(np.concatenate(P), y[idx])


def train_seed(seed, env, args, deadline):
    import torch
    import torch.nn.functional as F
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = device == "cuda"
    d = os.path.join(env.out, "ckpt", f"seed{seed}")
    os.makedirs(d, exist_ok=True)
    clean_tmp(d)
    if os.path.exists(os.path.join(d, "done.json")):
        log(f"train seed {seed}: SKIP (done)")
        return
    x, y = load_data(env, "train_" + args.variant)
    tr_idx, va_idx = get_split(env, args, y)
    spe = len(tr_idx) // args.batch
    total = spe * args.epochs
    warm = max(1, int(round(0.1 * total)))
    model = build_model(args.arch, seed, not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WD, betas=(0.9, 0.999), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    st = dict(epoch=0, step_in_epoch=0, gstep=0, best=None, history=[], elapsed=0.0)
    last = os.path.join(d, "last.pt")
    if os.path.exists(last):
        blob = torch.load(last, map_location="cpu", weights_only=False)
        if blob["spe"] != spe or blob["epochs"] != args.epochs or blob["batch"] != args.batch:
            raise SystemExit("last.pt was written with different --epochs/--batch/--val-frac. "
                             "Delete it (and best.pth) to restart this seed.")
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        scaler.load_state_dict(blob["scaler"])
        st = blob["st"]
        log(f"train seed {seed}: RESUME epoch {st['epoch'] + 1}/{args.epochs} step {st['step_in_epoch']}/{spe} "
            f"(global {st['gstep']}/{total})")
        del blob
    else:
        log(f"train seed {seed}: START ({len(tr_idx)} train, {len(va_idx)} val, {spe} steps/epoch, "
            f"{total} steps, arch {args.arch}, device {device})")
    try:
        cfg = getattr(model, "pretrained_cfg", {})
        log(f"pretrained tag: {cfg.get('tag')} / {cfg.get('hf_hub_id', cfg.get('url'))}")
    except Exception:
        pass
    model.train()

    def save_last():
        tmp = os.path.join(env.work, f"tmp_last_seed{seed}.pt")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
                        st=st, spe=spe, epochs=args.epochs, batch=args.batch), tmp)
        safe_copy_to(tmp, last)
        os.remove(tmp)

    def save_best(m):
        tmp = os.path.join(env.work, f"tmp_best_seed{seed}.pth")
        torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                        epoch=st["epoch"], val=m, arch=args.arch, seed=seed, classes=CLASSES,
                        version=VERSION), tmp)
        safe_copy_to(tmp, os.path.join(d, "best.pth"))
        os.remove(tmp)

    t_run, last_save, run_loss, run_n = time.time(), st["gstep"], 0.0, 0
    for epoch in range(st["epoch"], args.epochs):
        perm = np.random.RandomState(100003 * seed + epoch).permutation(tr_idx)
        batches = [perm[i * args.batch:(i + 1) * args.batch] for i in range(spe)]
        start = st["step_in_epoch"] if epoch == st["epoch"] else 0
        for si, (xb, yb, _) in zip(range(start, spe), Prefetch(x, y, batches[start:])):
            lr = lr_at(st["gstep"], total, warm)
            for g in opt.param_groups:
                g["lr"] = lr
            xin, yin = to_input(xb, device), yb.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                logits = model(xin)
            loss = F.cross_entropy(logits.float(), yin)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            st["gstep"] += 1
            st["step_in_epoch"] = si + 1
            run_loss += float(loss.detach())
            run_n += 1
            if st["gstep"] % 50 == 0:
                el = time.time() - t_run
                done_here = st["gstep"] - last_save if last_save else st["gstep"]
                log(f"seed {seed} ep {epoch + 1}/{args.epochs} step {si + 1}/{spe} "
                    f"loss {run_loss / max(1, run_n):.4f} lr {lr:.2e} spent {fmt(el)}")
                run_loss, run_n = 0.0, 0
            due = st["gstep"] % args.save_every == 0 and st["step_in_epoch"] < spe
            over = time.time() > deadline
            stop = bool(args.debug_stop_after_steps) and st["gstep"] == args.debug_stop_after_steps
            if due or ((over or stop) and st["step_in_epoch"] < spe):
                save_last()
                last_save = st["gstep"]
                if over or stop:
                    log(f"train seed {seed}: stopping at step {st['gstep']} ({'time budget' if over else 'debug'}); "
                        f"state saved. Run the same command again to resume.")
                    raise SystemExit(0)
        m = evaluate_val(model, x, y, va_idx, device)
        row = dict(epoch=epoch + 1, **m)
        st["history"].append(row)
        log(f"seed {seed} epoch {epoch + 1} val: acc {m['accuracy']:.4f} ba {m['balanced_accuracy']:.4f} "
            f"nll {m['nll']:.4f}")
        better = (st["best"] is None or m["balanced_accuracy"] > st["best"]["balanced_accuracy"] + 1e-12
                  or (abs(m["balanced_accuracy"] - st["best"]["balanced_accuracy"]) <= 1e-12
                      and m["nll"] < st["best"]["nll"]))
        st["epoch"], st["step_in_epoch"] = epoch + 1, 0
        if better:
            st["best"] = dict(epoch=epoch + 1, **m)
            save_best(m)
            log(f"seed {seed}: new best at epoch {epoch + 1}")
        if epoch + 1 < args.epochs:
            save_last()
        if time.time() > deadline and epoch + 1 < args.epochs:
            log(f"train seed {seed}: time budget reached after epoch {epoch + 1}; run again to resume.")
            raise SystemExit(0)
    atomic_json(os.path.join(d, "train_log.json"), dict(seed=seed, arch=args.arch, history=st["history"],
                                                         best=st["best"], steps=st["gstep"],
                                                         val_split_n=int(len(va_idx)), version=VERSION))
    atomic_json(os.path.join(d, "done.json"), dict(seed=seed, best_epoch=st["best"]["epoch"],
                                                    best_val_ba=st["best"]["balanced_accuracy"]))
    try:
        os.remove(last)
    except OSError:
        pass
    log(f"train seed {seed}: DONE best epoch {st['best']['epoch']} val BA {st['best']['balanced_accuracy']:.4f}")


def job_train(env, args):
    env.drive_guard(3.5 + 0.4 * len(args.seeds))
    if not (os.path.exists(os.path.join(env.work, f"train_{args.variant}_ready.json"))):
        job_prepare(env, args, only="train")
    deadline = T0 + args.max_hours * 3600
    write_env(env.out)
    for seed in args.seeds:
        train_seed(seed, env, args, deadline)


# --------------------------------------------------------------------- TTA
def ln_names(model):
    import torch.nn as nn
    named = dict(model.named_parameters())
    out = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.LayerNorm):
            for p in ("weight", "bias"):
                key = f"{name}.{p}" if name else p
                if key in named:
                    out.append(key)
    return out


def tta_dir(args):
    """Shuffled stream -> tta/, class-sorted stream -> tta_sorted/ (so the two never mix)."""
    return "tta" if args.stream == "shuffled" else "tta_sorted"


def stream_order(env, n, mode="shuffled"):
    """shuffled: one fixed random permutation (seed 0), saved so every run uses the same stream.
    sorted: the packed order itself, which is class-grouped (ADI, BACK, ..., TUM) because the
    files are sorted by path. It mirrors the Camelyon17 stream, where all tumor patches come first."""
    if mode == "sorted":
        return np.arange(n, dtype=np.int64)
    path = os.path.join(env.out, "stream_order.npy")
    if os.path.exists(path):
        order = np.load(path)
        assert len(order) == n and sorted(order.tolist()) == list(range(n)), "stored stream order is not a permutation"
        return order
    order = np.random.RandomState(0).permutation(n).astype(np.int64)
    np.save(path + ".tmp.npy", order)
    os.replace(path + ".tmp.npy", path)
    return order


def make_batches(order, bs):
    return [order[i:i + bs] for i in range(0, len(order), bs)]


def compute_fisher(model, x, y, batches, tnames, device, n=FISHER_N):
    import torch
    import torch.nn.functional as F
    seed_all(0)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    sd = dict(model.named_parameters())
    for nm in tnames:
        sd[nm].requires_grad = True
    fisher = {nm: torch.zeros_like(sd[nm]) for nm in tnames}
    seen = 0
    for idx in batches:
        if seen >= n:
            break
        take = min(len(idx), n - seen)
        xb = to_input(np.ascontiguousarray(x[idx[:take]]), device)
        p = F.softmax(model(xb), 1)
        loss = -(p * torch.log(p + 1e-12)).sum(1).mean()
        model.zero_grad()
        loss.backward()
        for nm in tnames:
            if sd[nm].grad is not None:
                fisher[nm] += (sd[nm].grad.detach() ** 2) * take
        seen += take
    for nm in tnames:
        fisher[nm] = (fisher[nm] / max(1, seen)).detach().cpu()
        sd[nm].requires_grad = False
    mu = float(torch.cat([fisher[nm].flatten() for nm in tnames]).mean()) + 1e-12
    return {nm: fisher[nm] / mu for nm in tnames}, seen


def run_method(method, base_model, x, y, order, unit, tnames, args, deadline, device, fisher=None):
    """One adaptation/evaluation pass over the stream, resumable every few batches.

    The state file holds parameters, optimizer momentum, EATA counters, and the
    probabilities recorded so far, so a restart can never disagree with itself.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    state_f = os.path.join(unit, f"{method}_state.pkl")
    model = copy.deepcopy(base_model).to(device)
    seed_all(0)
    adaptive = method in ("tent", "eata", "shot")
    sd = dict(model.named_parameters())
    params, opt = [], None
    if adaptive:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.train()
        for nm in tnames:
            sd[nm].requires_grad = True
        params = [sd[nm] for nm in tnames]
        opt = torch.optim.SGD(params, lr=LR, momentum=MOM)
    else:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    e0 = 0.4 * math.log(len(CLASSES))
    theta0 = {nm: sd[nm].detach().clone() for nm in tnames} if method == "eata" else None
    fish = {nm: fisher[nm].to(device) for nm in tnames} if method == "eata" else None
    batches = make_batches(order, args.batch)
    nb = len(batches)
    P, bi = np.zeros((0, len(CLASSES)), np.float32), 0
    ex = dict(sel_total=0, sel_seen=0, skip=0)
    if os.path.exists(state_f):
        with open(state_f, "rb") as fh:
            s = pickle.load(fh)
        bi, P, ex = s["bi"], s["P"], s["ex"]
        if adaptive:
            with torch.no_grad():
                for nm in tnames:
                    sd[nm].copy_(s["theta"][nm].to(device))
            opt.load_state_dict(s["opt"])
            for grp in opt.state.values():
                for k, v in grp.items():
                    if torch.is_tensor(v):
                        grp[k] = v.to(device)
        want = min(len(order), bi * args.batch)
        assert len(P) == want, f"{method}: state has {len(P)} predictions for {bi} batches"
        log(f"{method}: RESUME at batch {bi}/{nb}")
    else:
        log(f"{method}: START 0/{nb}")
    t0 = time.time()
    new = []
    ctx = torch.enable_grad() if adaptive else torch.no_grad()

    def save():
        full = np.concatenate([P] + new) if new else P
        s = dict(bi=bi, P=full, ex=ex)
        if adaptive:
            s["theta"] = {nm: sd[nm].detach().cpu().clone() for nm in tnames}
            o = copy.deepcopy(opt.state_dict())
            for grp in o["state"].values():
                for k, v in grp.items():
                    if torch.is_tensor(v):
                        grp[k] = v.cpu()
            s["opt"] = o
        tmp = state_f + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(s, fh)
        os.replace(tmp, state_f)

    with ctx:
        for b in range(bi, nb):
            idx = batches[b]
            xin = to_input(np.ascontiguousarray(x[idx]), device)
            if method == "flip_tta":
                acc = 0
                for f in (None, [3], [2], [2, 3]):
                    acc = acc + F.softmax(model(torch.flip(xin, f) if f else xin), 1)
                p = acc / 4
            else:
                p = F.softmax(model(xin), 1)
            if method == "tent":
                loss = -(p * torch.log(p + 1e-12)).sum(1).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, CLIP); opt.step()
            elif method == "shot":
                pbar = p.mean(0)
                loss = (-(p * torch.log(p + 1e-12)).sum(1).mean() + IM_W * (pbar * torch.log(pbar + 1e-12)).sum())
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, CLIP); opt.step()
            elif method == "eata":
                ent = -(p * torch.log(p + 1e-12)).sum(1)
                mask = ent < e0
                ex["sel_total"] += int(mask.sum())
                ex["sel_seen"] += int(xin.size(0))
                if int(mask.sum()) == 0:
                    ex["skip"] += 1
                else:
                    pf = p[mask]
                    loss = -(pf * torch.log(pf + 1e-12)).sum(1).mean()
                    reg, cnt = 0.0, 0
                    for nm in tnames:
                        reg = reg + (fish[nm] * (sd[nm] - theta0[nm]) ** 2).sum()
                        cnt += fish[nm].numel()
                    loss = loss + BETA * reg / max(1, cnt)
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, CLIP); opt.step()
            new.append(p.detach().float().cpu().numpy())
            bi = b + 1
            stop = bool(args.debug_stop_after_batches) and bi == args.debug_stop_after_batches and bi < nb
            if bi % args.tta_save_every == 0 or bi == nb or stop or time.time() > deadline:
                save()
                P, new = np.concatenate([P] + new), []
                if bi % 20 == 0 or bi == nb:
                    el = time.time() - t0
                    log(f"{method} {bi}/{nb} spent {fmt(el)}")
                if (stop or time.time() > deadline) and bi < nb:
                    log(f"{method}: stopping at batch {bi} ({'debug' if stop else 'time budget'}); state saved.")
                    raise SystemExit(0)
    P = np.concatenate([P] + new) if new else P
    assert len(P) == len(order), "prediction count differs from stream length"
    return P, ex


def finalize_method(method, P, order, y, unit, seed, ref, ex):
    L = y[order].astype(np.int64)
    pred = P.argmax(1)
    met = multiclass_metrics(P, L)
    out = dict(method=method, seed=int(seed), version=VERSION, **met)
    if method != "no_tta":
        out.update(help_harm(ref["pred"], pred, L))
        out["tumor_view"] = tumor_view(ref["pred"], pred, L)
    else:
        out.update(help=-1, harm=-1, net=-1, harm_rate=-1.0)
        out["tumor_view"] = tumor_view(pred, pred, L)
    if method == "eata":
        out["selected_fraction"] = ex["sel_total"] / max(1, ex["sel_seen"])
        out["skipped_batches"] = ex["skip"]
    tmp = os.path.join(unit, f"{method}_predictions.tmp.npz")
    np.savez_compressed(tmp, probabilities=P.astype(np.float32), predictions=pred.astype(np.uint8),
                        labels=L.astype(np.uint8), patch_ids=order.astype(np.int64),
                        stream_position=np.arange(len(order), dtype=np.int32))
    os.replace(tmp, os.path.join(unit, f"{method}_predictions.npz"))
    atomic_json(os.path.join(unit, f"{method}_summary.json"), out)
    try:
        os.remove(os.path.join(unit, f"{method}_state.pkl"))
    except OSError:
        pass
    log(f"{method}: DONE acc {out['accuracy']:.4f} ba {out['balanced_accuracy']:.4f} ece {out['ece_15']:.4f} "
        f"help {out['help']} harm {out['harm']}")
    return out


def tta_seed(seed, env, args, deadline):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = os.path.join(env.out, "ckpt", f"seed{seed}", "best.pth")
    if not os.path.exists(os.path.join(env.out, "ckpt", f"seed{seed}", "done.json")):
        log(f"tta seed {seed}: training is not finished (no done.json); skipping")
        return False
    x, y = load_data(env, "val")
    order = stream_order(env, len(y), args.stream)
    if args.limit_val:
        order = order[order < args.limit_val]      # debug subset: first N patches, stream order kept
    unit = os.path.join(env.out, tta_dir(args), f"seed{seed}")
    os.makedirs(unit, exist_ok=True)
    clean_tmp(unit)
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    model = build_model(blob["arch"], seed, False)
    model.load_state_dict(blob["model"])
    model.to(device)
    tnames = ln_names(model)
    log(f"tta seed {seed}: checkpoint epoch {blob['epoch']}, {len(tnames)} LayerNorm tensors, "
        f"{len(order)} stream patches, {len(make_batches(order, args.batch))} batches")
    ref, fisher = None, None
    for method in METHODS:
        summ = os.path.join(unit, f"{method}_summary.json")
        if os.path.exists(summ):
            log(f"{method}: SKIP (done)")
            if method == "no_tta":
                z = np.load(os.path.join(unit, "no_tta_predictions.npz"))
                ref = dict(pred=z["predictions"].astype(np.int64))
            continue
        if method != "no_tta" and ref is None:
            z = np.load(os.path.join(unit, "no_tta_predictions.npz"))
            ref = dict(pred=z["predictions"].astype(np.int64))
        if method == "eata":
            fp = os.path.join(unit, "fisher.npz")
            if os.path.exists(fp):
                z = np.load(fp)
                import torch as _t
                fisher = {nm: _t.from_numpy(z[nm]) for nm in tnames}
            else:
                fm = copy.deepcopy(model)
                fisher, seen = compute_fisher(fm, x, y, make_batches(order, args.batch), tnames, device)
                del fm
                np.savez_compressed(fp + ".tmp.npz", **{k: v.numpy() for k, v in fisher.items()})
                os.replace(fp + ".tmp.npz", fp)
                log(f"eata: Fisher from {seen} stream patches")
        P, ex = run_method(method, model, x, y, order, unit, tnames, args, deadline, device, fisher)
        out = finalize_method(method, P, order, y, unit, seed, ref or dict(pred=P.argmax(1)), ex)
        if method == "no_tta":
            ref = dict(pred=P.argmax(1))
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return True


def gate_preview(env, args, seed):
    """Print CG-TENT help/harm for a quick sanity read (the real analysis is local)."""
    unit = os.path.join(env.out, tta_dir(args), f"seed{seed}")
    try:
        z0, z1, zf = (np.load(os.path.join(unit, f"{m}_predictions.npz")) for m in ("no_tta", "tent", "flip_tta"))
    except Exception:
        return
    y0, y1, yf, lab = z0["predictions"], z1["predictions"], zf["predictions"], z0["labels"]
    acc = (y1 != y0) & (yf == y1)
    yg = np.where(acc, y1, y0)
    hh_t, hh_g = help_harm(y0.astype(int), y1.astype(int), lab.astype(int)), help_harm(y0.astype(int), yg.astype(int), lab.astype(int))
    log(f"seed {seed} preview: TENT help {hh_t['help']} harm {hh_t['harm']} | CG-TENT help {hh_g['help']} "
        f"harm {hh_g['harm']} (accepted {int(acc.sum())} of {int((y1 != y0).sum())} flips)")


def job_tta(env, args):
    if not os.path.exists(os.path.join(env.work, "val_ready.json")):
        job_prepare(env, args, only="val")
    deadline = T0 + args.max_hours * 3600
    write_env(env.out)
    for seed in args.seeds:
        if tta_seed(seed, env, args, deadline):
            gate_preview(env, args, seed)


# ------------------------------------------------------- verify / status / pack
def job_verify(env, args):
    x, y = load_data(env, "val")
    order = stream_order(env, len(y), args.stream)
    bad, checked = 0, 0
    for seed in args.seeds:
        unit = os.path.join(env.out, tta_dir(args), f"seed{seed}")
        zr = None
        for m in METHODS:
            f = os.path.join(unit, f"{m}_predictions.npz")
            if not os.path.exists(f):
                continue
            z = np.load(f)
            P, pred, lab, pid = z["probabilities"], z["predictions"], z["labels"], z["patch_ids"]
            ok = (np.allclose(P.sum(1), 1.0, atol=1e-4) and np.array_equal(P.argmax(1), pred)
                  and np.array_equal(pid, order[:len(pid)]) and np.array_equal(lab, y[pid])
                  and np.isfinite(P).all() and len(pid) == len(order))
            if m == "no_tta":
                zr = pred.astype(int)
            else:
                s = read_json(os.path.join(unit, f"{m}_summary.json"))
                if zr is not None:
                    hh = help_harm(zr, pred.astype(int), lab.astype(int))
                    ok = ok and hh["help"] == s["help"] and hh["harm"] == s["harm"]
            checked += 1
            if not ok:
                bad += 1
                log(f"VERIFY FAIL seed {seed} {m}")
    log(f"verify: {checked} prediction files checked, {bad} failed")
    if bad:
        raise SystemExit(1)


def job_status(env, args):
    log(f"outputs: {env.out}\nscratch: {env.work}")
    for k in ("train_" + args.variant, "val"):
        log(f"  scratch {k}: {'packed' if os.path.exists(os.path.join(env.work, k + '_ready.json')) else 'missing'}")
    for seed in range(1, 6):
        d = os.path.join(env.out, "ckpt", f"seed{seed}")
        tr = ("done" if os.path.exists(os.path.join(d, "done.json")) else
              "in progress (last.pt)" if os.path.exists(os.path.join(d, "last.pt")) else "not started")
        parts = []
        for tag, sub in (("shuffled", "tta"), ("sorted", "tta_sorted")):
            u = os.path.join(env.out, sub, f"seed{seed}")
            have = [m for m in METHODS if os.path.exists(os.path.join(u, f"{m}_summary.json"))]
            parts.append(f"tta {tag} {len(have)}/{len(METHODS)}")
        log(f"  seed {seed}: train {tr}; " + "; ".join(parts))


def job_pack_results(env, args):
    path = os.path.join(env.out, "CRC_results.zip")
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for top in ("tta", "tta_sorted", "ckpt"):
            for root, _, files in os.walk(os.path.join(env.out, top)):
                for f in files:
                    if f.endswith((".pth", ".pt", ".tmp", "_state.pkl")):
                        continue
                    full = os.path.join(root, f)
                    zf.write(full, os.path.relpath(full, env.out))
        for f in ("dataset_manifest.json", "environment.json", "pip-freeze.txt", "stream_order.npy"):
            full = os.path.join(env.out, f)
            if os.path.exists(full):
                zf.write(full, f)
        for f in os.listdir(env.out):
            if f.startswith("split_") and f.endswith(".npz"):
                zf.write(os.path.join(env.out, f), f)
    os.replace(tmp, path)
    log(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB; no checkpoints inside)")


def write_env(out):
    import platform
    info = {"python": platform.python_version(), "platform": platform.platform(), "script": VERSION}
    try:
        import torch
        info.update(torch=torch.__version__, cuda_available=bool(torch.cuda.is_available()), cuda=torch.version.cuda)
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        info["torch_error"] = str(exc)
    for mod in ("timm", "numpy", "sklearn", "PIL"):
        try:
            info[mod] = getattr(__import__(mod), "__version__", None)
        except Exception:
            pass
    p = os.path.join(out, "environment.json")
    if not os.path.exists(p):
        atomic_json(p, info)
    freeze = os.path.join(out, "pip-freeze.txt")
    if not os.path.exists(freeze):
        try:
            txt = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True,
                                 timeout=120).stdout
            open(freeze, "w", encoding="utf-8").write(txt)
        except Exception:
            pass


# ------------------------------------------------------------------- smoke
def make_fake_zip(path, per_class, seed):
    from PIL import Image
    rs = np.random.RandomState(seed)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for ci, cn in enumerate(CLASSES):
            base = np.array([(ci * 28) % 256, (ci * 57 + 40) % 256, (ci * 91 + 90) % 256], np.float32)
            for j in range(per_class):
                arr = np.clip(base + rs.randn(SIDE, SIDE, 3) * 40, 0, 255).astype(np.uint8)
                buf = io.BytesIO()
                Image.fromarray(arr).save(buf, format="TIFF")
                zf.writestr(f"FAKE/{cn}/{cn}-{j:04d}.tif", buf.getvalue())


def smoke(args):
    """CPU self-test: prepare -> train -> tta, plus interrupted-and-resumed runs that
    must match the uninterrupted ones exactly."""
    import tempfile
    import torch
    root = tempfile.mkdtemp(prefix="crc_smoke_")
    work, out_a, out_b = (os.path.join(root, n) for n in ("work", "out_a", "out_b"))
    os.makedirs(work)
    n_tr, n_va = 20, 8
    make_fake_zip(os.path.join(work, "NCT-CRC-HE-100K.zip"), n_tr, 1)
    make_fake_zip(os.path.join(work, "CRC-VAL-HE-7K.zip"), n_va, 2)
    ov = {"train_norm": dict(size=os.path.getsize(os.path.join(work, "NCT-CRC-HE-100K.zip")), n=n_tr * 9),
          "val": dict(size=os.path.getsize(os.path.join(work, "CRC-VAL-HE-7K.zip")), n=n_va * 9)}
    atomic_json(os.path.join(work, "sources_override.json"), ov)
    for z in ("NCT-CRC-HE-100K.zip", "CRC-VAL-HE-7K.zip"):
        open(os.path.join(work, z + ".ok"), "w").write("ok")
    me = os.path.abspath(__file__)
    common = ["--work", work, "--arch", "vit_tiny_patch16_224", "--no-pretrained", "--epochs", "2",
              "--batch", "16", "--val-frac", "0.15", "--save-every", "3", "--tta-save-every", "2",
              "--skip-md5", "--keep-zips"]

    def run(out, *extra):
        cmd = [sys.executable, me, *extra[:1], "--out", out, *common, *extra[1:]]
        log("$ " + " ".join(os.path.basename(c) if c == me else c for c in cmd[1:6]) + " ...")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            log(r.stdout[-3000:]); log(r.stderr[-3000:])
            raise SystemExit(f"smoke: command failed: {extra}")
        return r.stdout

    run(out_a, "prepare")
    run(out_a, "train", "--seeds", "1")
    # On a GPU, two identical uninterrupted runs already differ slightly (nondeterministic CUDA
    # kernels, mixed precision). Measure that noise with a second uninterrupted run, and only
    # require the resumed run to stay within a small multiple of it. On CPU the noise is zero.
    out_c = os.path.join(root, "out_c")
    gpu = torch.cuda.is_available()
    if gpu:
        run(out_c, "train", "--seeds", "1")
    run(out_b, "train", "--seeds", "1", "--debug-stop-after-steps", "5")
    assert os.path.exists(os.path.join(out_b, "ckpt", "seed1", "last.pt")), "no last.pt after interrupted run"
    assert not os.path.exists(os.path.join(out_b, "ckpt", "seed1", "done.json")), "interrupted run finished?"
    run(out_b, "train", "--seeds", "1")
    ba = torch.load(os.path.join(out_a, "ckpt", "seed1", "best.pth"), weights_only=False)
    bb = torch.load(os.path.join(out_b, "ckpt", "seed1", "best.pth"), weights_only=False)
    assert ba["epoch"] == bb["epoch"], "best epoch differs after resume"
    worst = max(float((ba["model"][k] - bb["model"][k]).abs().max()) for k in ba["model"])
    noise = 0.0
    if gpu:
        bc = torch.load(os.path.join(out_c, "ckpt", "seed1", "best.pth"), weights_only=False)
        noise = max(float((ba["model"][k] - bc["model"][k]).abs().max()) for k in ba["model"])
        log(f"smoke: two identical uninterrupted GPU runs differ by {noise:.2e} (noise floor)")
    tol_train = max(1e-5, 20 * noise)
    log(f"smoke: resumed training vs uninterrupted, max weight difference {worst:.2e} (allowed {tol_train:.2e})")
    assert worst <= tol_train, "resumed training does not match the uninterrupted run"
    assert not os.path.exists(os.path.join(out_b, "ckpt", "seed1", "last.pt")), "last.pt not cleaned up"

    run(out_a, "tta", "--seeds", "1")
    if gpu:
        run(out_c, "tta", "--seeds", "1")
    run(out_b, "tta", "--seeds", "1", "--debug-stop-after-batches", "3")
    assert os.path.exists(os.path.join(out_b, "tta", "seed1", "no_tta_state.pkl")), "no state after interruption"
    # The debug stop fires at batch 3 of the first method only (no_tta); restart it, then it
    # stops again at batch 3 of the next method, and so on. Keep restarting until all methods
    # are done: every method is therefore interrupted once and resumed.
    for _ in range(len(METHODS) + 2):
        r = subprocess.run([sys.executable, me, "tta", "--out", out_b, *common, "--seeds", "1",
                            "--debug-stop-after-batches", "3"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        if all(os.path.exists(os.path.join(out_b, "tta", "seed1", f"{m}_summary.json")) for m in METHODS):
            break
    else:
        raise SystemExit("smoke: TTA restarts did not finish all methods")
    for m in METHODS:
        za = np.load(os.path.join(out_a, "tta", "seed1", f"{m}_predictions.npz"))
        zb = np.load(os.path.join(out_b, "tta", "seed1", f"{m}_predictions.npz"))
        d = float(np.abs(za["probabilities"] - zb["probabilities"]).max())
        noise_p = 0.0
        if gpu:
            zc = np.load(os.path.join(out_c, "tta", "seed1", f"{m}_predictions.npz"))
            noise_p = float(np.abs(za["probabilities"] - zc["probabilities"]).max())
        tol_p = max(1e-5, 20 * noise_p)
        log(f"smoke: {m}: resumed vs uninterrupted max prob difference {d:.2e} (noise {noise_p:.2e}, allowed {tol_p:.2e})")
        assert d <= tol_p, f"{m}: resume mismatch"
        assert np.array_equal(za["patch_ids"], zb["patch_ids"])
    run(out_a, "verify", "--seeds", "1")
    run(out_a, "tta", "--seeds", "1", "--stream", "sorted")
    run(out_a, "verify", "--seeds", "1", "--stream", "sorted")
    zs = np.load(os.path.join(out_a, "tta_sorted", "seed1", "no_tta_predictions.npz"))
    assert np.array_equal(zs["patch_ids"], np.arange(len(zs["patch_ids"]))), "sorted stream is not in packed order"
    assert (np.diff(zs["labels"].astype(int)) >= 0).all(), "sorted stream is not class-grouped"
    run(out_a, "pack_results")
    run(out_a, "status")
    log("SMOKE OK: prepare, train, interrupted/resumed train, tta, interrupted/resumed tta, verify, pack_results")
    shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", choices=["smoke", "prepare", "train", "tta", "verify", "status", "pack_results"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--variant", choices=["norm", "nonorm"], default="norm",
                    help="norm = NCT-CRC-HE-100K (Macenko-normalised, the pair used in the paper); "
                         "nonorm = NCT-CRC-HE-100K-NONORM (stain shift to the normalised CRC-VAL)")
    ap.add_argument("--out", default=None, help="outputs (default: Drive 'Next Research/CRC_second_dataset')")
    ap.add_argument("--work", default=None, help="scratch for zips and packed arrays (default /content/crc_work)")
    ap.add_argument("--only", choices=["train", "val"], default=None, help="prepare only")
    ap.add_argument("--arch", default="vit_base_patch16_224")
    ap.add_argument("--no-pretrained", action="store_true", help="smoke tests only")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--val-frac", type=float, default=0.05,
                    help="fraction of NCT-CRC-HE-100K held out (stratified) for checkpoint selection")
    ap.add_argument("--save-every", type=int, default=600, help="train steps between last.pt saves")
    ap.add_argument("--tta-save-every", type=int, default=10, help="TTA batches between state saves")
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--keep-zips", action="store_true")
    ap.add_argument("--skip-md5", action="store_true")
    ap.add_argument("--stream", choices=["shuffled", "sorted"], default="shuffled",
                    help="CRC-VAL stream order for tta/verify: shuffled (default, fixed random permutation) or "
                         "sorted (class by class, like the Camelyon17 stream). Results go to tta/ or tta_sorted/")
    ap.add_argument("--limit-val", type=int, default=0, help="debug: use a subset of CRC-VAL")
    ap.add_argument("--debug-stop-after-steps", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--debug-stop-after-batches", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.job == "smoke":
        return smoke(args)
    env = Env(args)
    log(f"crc_colab {VERSION}; job={args.job}; outputs={env.out}; scratch={env.work}")
    if args.job == "prepare":
        job_prepare(env, args, only=args.only)
    elif args.job == "train":
        job_train(env, args)
    elif args.job == "tta":
        job_tta(env, args)
    elif args.job == "verify":
        job_verify(env, args)
    elif args.job == "status":
        job_status(env, args)
    elif args.job == "pack_results":
        job_pack_results(env, args)


if __name__ == "__main__":
    main()

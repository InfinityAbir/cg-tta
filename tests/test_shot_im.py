"""Unit test: SHOT-IM equation, implementation and saved run config agree.

Paper Eq. (shot):   L = L_TENT + lam * sum_c pbar_c log pbar_c = L_TENT - lam * H(pbar)

A sign error would silently turn the diversity term into *marginal-entropy
minimisation* (collapse toward one class).  This test checks, in order:

  1. the reference implementation equals L_TENT - lam * H(pbar) numerically;
  2. the loss is LOWER for a balanced batch marginal than for a collapsed one
     at equal per-sample entropy (so minimising it favours diversity);
  3. one gradient-descent step on the logits INCREASES H(pbar) when the batch
     marginal is skewed (the wrong sign would decrease it);
  4. every executed GPU/Colab runner that implements SHOT-IM uses the same
     expression `+ IM_W * (pbar * torch.log(pbar + 1e-12)).sum()` (source check);
  5. every saved per-seed run summary records im_w == 1.0.

Run:  python tests/test_shot_im.py        (CPU only; exits non-zero on failure)
"""
import glob
import json
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REL, "code"))
from cg_tta import shot_loss, tent_loss  # noqa: E402


def marginal_entropy(p):
    pbar = p.mean(0)
    return float(-(pbar * torch.log(pbar + 1e-12)).sum())


def test_equals_tent_minus_marginal_entropy():
    torch.manual_seed(0)
    p = torch.softmax(torch.randn(64, 2), 1)
    for lam in (0.5, 1.0, 2.0):
        got = float(shot_loss(p, lam))
        want = float(tent_loss(p)) - lam * marginal_entropy(p)
        assert abs(got - want) < 1e-6, (lam, got, want)


def test_balanced_marginal_has_lower_loss_than_collapsed():
    # same per-sample entropy (each row is [0.9, 0.1] or [0.1, 0.9])
    collapsed = torch.tensor([[0.9, 0.1]] * 64)
    balanced = torch.tensor([[0.9, 0.1]] * 32 + [[0.1, 0.9]] * 32)
    assert abs(float(tent_loss(collapsed)) - float(tent_loss(balanced))) < 1e-6
    assert float(shot_loss(balanced)) < float(shot_loss(collapsed)) - 0.1


def test_gradient_step_increases_marginal_entropy():
    # skewed batch: 56 of 64 rows lean to class 0
    logits = torch.zeros(64, 2)
    logits[:56, 0] = 1.0
    logits[56:, 1] = 1.0
    logits.requires_grad_(True)
    h0 = marginal_entropy(torch.softmax(logits, 1))
    # isolate the diversity term (entropy term removed) so the check is exact
    pbar = torch.softmax(logits, 1).mean(0)
    div = (pbar * torch.log(pbar + 1e-12)).sum()
    div.backward()
    stepped = (logits - 0.5 * logits.grad).detach()
    h1 = marginal_entropy(torch.softmax(stepped, 1))
    assert h1 > h0, (h0, h1)


SIGN_OK = re.compile(r"\+\s*(?:IM_W|lam)\s*\*\s*\(\s*pbar\s*\*\s*torch\.log\(\s*pbar\s*\+\s*1e-12\s*\)\s*\)\s*\.sum\(\)")
SIGN_BAD = re.compile(r"-\s*(?:IM_W|lam)\s*\*\s*\(\s*pbar\s*\*\s*torch\.log\(\s*pbar")


def test_runner_sources_use_same_sign():
    candidates = [os.path.join(REL, "code", "cg_tta.py"),
                  os.path.join(REL, "colab", "colab_cg_tta_gpu.py")]
    candidates += glob.glob(os.path.join(REL, "colab_pro", "*.py"))
    checked = 0
    for path in candidates:
        if not os.path.exists(path):
            continue
        src = open(path, encoding="utf8").read().replace("\n", " ")
        if "pbar" not in src:
            continue
        assert not SIGN_BAD.search(src), f"wrong sign in {path}"
        assert SIGN_OK.search(src), f"expected sign not found in {path}"
        checked += 1
    assert checked >= 2, "fewer than two runners checked"


def test_saved_run_configs_record_im_w_one():
    files = sorted(glob.glob(os.path.join(REL, "logs", "shot_run_config", "seed*_summary.json")))
    assert len(files) == 5, files
    for f in files:
        cfg = json.load(open(f))
        assert cfg["im_w"] == 1.0 and cfg["batch"] == 64 and abs(cfg["lr"] - 1e-3) < 1e-12, (f, cfg)


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError as e:
                fails += 1
                print("FAIL", name, e)
    sys.exit(1 if fails else 0)

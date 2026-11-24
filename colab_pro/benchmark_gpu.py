#!/usr/bin/env python
"""Standalone GPU timing benchmark for the CG-TTA revision.

This is a thin wrapper around ``cgtta_colab.job_timing`` so the timing
experiment can be launched without the job subcommand.  It measures, on the
same GPU and the same checkpoint:

  * frozen inference
  * TENT (one entropy step per batch)
  * Flip-TTA (four frozen views)
  * CG-TENT (TENT step + four frozen critic views)

for batch sizes 1, 8, 16, 32, and 64.  It reports p50/p95 latency per batch,
per-image latency, throughput, and peak allocated/reserved VRAM.

Run in Colab after mounting Drive:

    !python benchmark_gpu.py

The JSON and CSV are written to
``<Drive>/Next Research/03_Experiments/17_Revision_Colab/``.
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import cgtta_colab as S


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive-root", default="/content/drive/MyDrive/Next Research")
    ap.add_argument("--local-dir", default="/content/cg_tta_work")
    ap.add_argument("--out-subdir", default="03_Experiments/17_Revision_Colab")
    ap.add_argument("--archive", default=None)
    ap.add_argument("--extracted", default=None)
    ap.add_argument("--seed", type=int, default=1, dest="timing_seed")
    ap.add_argument("--reps", type=int, default=20, dest="timing_reps")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16, 32, 64])
    ns = ap.parse_args()

    # Env only needs these fields; job_timing reads the three timing fields.
    args = SimpleNamespace(
        job="timing",
        drive_root=ns.drive_root,
        local_dir=ns.local_dir,
        out_subdir=ns.out_subdir,
        canary=False,
        timing_seed=ns.timing_seed,
        timing_reps=ns.timing_reps,
        batch_sizes=ns.batch_sizes,
    )
    env = S.Env(args)
    if ns.archive:
        env.archive = ns.archive
    if ns.extracted:
        env.extracted = ns.extracted
    env.describe()
    S.job_timing(env, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

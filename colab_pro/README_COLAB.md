# Run the revision experiments on Colab Pro

This folder is self-contained. Upload the whole `colab_pro` folder to Google
Drive, then run the commands below in Colab. Every job writes to Drive and is
resumable: if Colab recycles the VM, run the same command again and it continues
from the last saved batch.

## 1. What must already be in Drive

Your Drive does not need one exact folder layout. The script searches the
following under `MyDrive/Next Research/` and, if a file is still missing, does
a bounded four-level scan that skips the PNG patch trees:

| Input | Places it checks |
|---|---|
| `metadata.csv` | `Camelyon17-WILDS/camelyon17_v1.0/`, `Camelyon17_WILDS_Local_Archive/01_Data/`, `04_Datasets/Dataset/`, `Camelyon17-WILDS/`, root |
| `archive.tar.gz` | `Camelyon17-WILDS/camelyon17_v1.0/`, `Camelyon17-WILDS/`, root |
| checkpoints | `Camelyon17_WILDS_Local_Archive/02_Source_Models/`, `source_models/`, `cg_tta/source_models/`, `09_Temporary/CGTTA_release/checkpoints/`, `CGTTA_release/checkpoints/` |

A minimal layout that definitely works:

```
Next Research/
  Camelyon17-WILDS/
    camelyon17_v1.0/
      archive.tar.gz                  # full WILDS archive, ~9.9 GB
  Camelyon17_WILDS_Local_Archive/
    01_Data/metadata.csv
    02_Source_Models/seed{1..5}_best.pth
  colab_pro/                          # upload this folder here
```

The five checkpoints are each ~982 MB; the sizes are printed by `discover`, so a
truncated copy is visible immediately.

You do **not** need to upload the existing `predictions/` folder. Those files
stay on your laptop for the post-processing step.

## 2. Colab setup (once per session)

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content
!rm -rf colab_pro
!cp -r "/content/drive/MyDrive/Next Research/colab_pro" .
!pip -q install timm transformers scikit-learn pandas

# Phikon is a gated Hugging Face model. Put your token here once.
import os
os.environ["HF_TOKEN"] = "hf_xxx"

!python colab_pro/cgtta_colab.py discover
!python colab_pro/cgtta_colab.py preflight --check-phikon
```

`discover` prints the exact metadata, archive, and checkpoint paths it will use,
plus the top-level listing of the Drive root and the checkpoint sizes. Check
this before spending GPU time. If a path is wrong, override it explicitly:

```python
!python colab_pro/cgtta_colab.py discover \
    --metadata "/content/drive/MyDrive/Next Research/.../metadata.csv" \
    --archive  "/content/drive/MyDrive/Next Research/.../archive.tar.gz" \
    --checkpoints-dir "/content/drive/MyDrive/Next Research/source_models"
```

`preflight` then checks the GPU, free disk, archive, metadata, checkpoints, and
Phikon access. It does not run a forward pass.

## 3. Jobs, in priority order

Run them in separate Colab sessions if needed. `--max-hours 11` stops cleanly
between batches before a 12-hour session limit. Rerunning the same command
resumes.

### A. Center-1 threshold locking (critical, ~4–6 h on T4)

```python
!python colab_pro/cgtta_colab.py lock_thresholds --seeds 1 2 3 4 5 --max-hours 11
```

This runs frozen, TENT, and Flip-TTA on Center-1 for all five source seeds.
It writes two label-free locks per seed:

- **R1:** accept the top 10% of Center-1 TENT flips by a score.
- **R2:** accept the top 1% of all Center-1 patches by a score.

Scores are frozen confidence, TENT confidence, critic confidence, and gate
margin. No Center-1 label is read by the locking code. The script also copies
`no.npz`, `tent.npz`, `flip.npz`, and `locked_thresholds.json` into
`CGTTA_release/predictions/center1/seed{S}/`.

### B. Phikon with EATA-LN (high, ~10–15 h total)

```python
!python colab_pro/cgtta_colab.py phikon --seeds 1 2 3 --with-tta --max-hours 11
!python colab_pro/cgtta_colab.py phikon --seeds 4 5 --with-tta --max-hours 11
```

For each seed this trains the linear head on source id-train, selects it on
id-val, then runs No-TTA, TENT, Flip-TTA, EATA-LN, and CG-TENT on Center-2.
EATA-LN adapts LayerNorm parameters only; the head stays frozen. Fisher is
estimated on 1,024 unlabeled target patches.

If you only have time for the second-backbone check, this is the job to run.

### C. Fixed-held-out budget experiment (high, ~4 h per source seed)

```python
!python colab_pro/cgtta_colab.py budget_heldout --seeds 1 --budgets 1 5 25 100 --subset-seeds 1 2 3 4 5 --max-hours 11
!python colab_pro/cgtta_colab.py budget_heldout --seeds 2 --budgets 1 5 25 100 --subset-seeds 1 2 3 4 5 --max-hours 11
!python colab_pro/cgtta_colab.py budget_heldout --seeds 3 --budgets 1 5 25 100 --subset-seeds 1 2 3 4 5 --max-hours 11
!python colab_pro/cgtta_colab.py budget_heldout --seeds 4 --budgets 1 5 25 100 --subset-seeds 1 2 3 4 5 --max-hours 11
!python colab_pro/cgtta_colab.py budget_heldout --seeds 5 --budgets 1 5 25 100 --subset-seeds 1 2 3 4 5 --max-hours 11
```

All budgets use the same 90% adaptation pool and the same fixed 10% held-out
set. The held-out set is excluded from every adaptation stream. The 100%
budget is run once per source seed and copied to the other subset seeds because
the full-pool stream is identical. Each summary records the held-out SHA-256,
the pool SHA-256, and the stream SHA-256.

### C2. Whole-slide held-out budget experiment (~3.7–4.8 h per source seed)

```python
!python colab_pro/cgtta_colab.py budget_slideout --seeds 2 --budgets 1 5 25 100 --subset-seeds 1 2 3 --max-hours 11
```

Run it once per source seed (the paper uses seeds 2, 3, and 5). The ten Center-2
slides are split into two fixed folds, A (slides 20, 21, 24, 27, 28 held out) and B
(slides 22, 23, 25, 26, 29 held out). TENT adapts on k% of the other five slides and
the final parameters are scored once on the held-out slides, which never enter an
adaptation stream (the job asserts this). `--folds A` or `--folds B` runs one fold.
`python postprocess_slideout.py --root path	o_Revision_Colab` writes
`budget_slideout_*.csv` and `tab_budget_slideout.tex` into `tables/`.

### D. Center-0 fifth seed (medium, ~5–7 h)

```python
!python colab_pro/cgtta_colab.py center0 --seeds 5 --max-hours 11
```

This trains a source model on Centers 3 and 4, selects on Center-1, then runs
the five-method battery on Center-0. It is a new seed, not a reuse of the
Center-2 checkpoint.

### E. GPU timing (medium, ~10–20 min)

```python
!python colab_pro/benchmark_gpu.py --reps 20 --batch-sizes 1 8 16 32 64
```

Measures frozen, TENT, Flip-TTA, and CG-TENT on the same checkpoint and GPU.
Reports images/s, p50/p95 latency per batch, ms/image, and peak allocated VRAM.
Run this on the same GPU class you want to report. If a batch size runs out of
memory, the JSON records `oom: true` and the job continues.

### F. Verify and package

```python
!python colab_pro/cgtta_colab.py verify
!python colab_pro/cgtta_colab.py pack_results
```

`verify` re-reads every summary and prediction file and checks probability
sums, argmax consistency, patch order, and help/harm counts. `pack_results`
writes `MyDrive/Next Research/CGTTA_colab_results.zip`, which contains only
the summaries, predictions, locks, and timing files — not the feature caches.

Download that zip and run the local post-processing:

```powershell
python postprocess_colab.py --root path\to\17_Revision_Colab
```

The script writes `tab_deployment_locked.tex`, `tab_phikon_eata.tex`,
`tab_budget_heldout.tex`, `tab_center0_seed5.tex`, and `tab_timing_gpu.tex`,
plus the per-seed CSVs and a manifest.

## 4. What each job guarantees

- **No target labels in any threshold.** Center-1 locking reads predictions
  only. The Target Q33 comparator in the post-processing reads unlabeled
  target confidences only.
- **No held-out adaptation.** The budget job builds the evaluation loader from
  a disjoint set of patch IDs and never passes it to an optimizer.
- **Same prediction schema everywhere.** Every method writes
  `probabilities`, `predictions`, `labels`, `patch_ids`, and `slides`.
- **Resume is exact.** State includes the batch index, optimizer, RNG, and the
  adapting parameters. The CPU smoke test compares a fresh run against an
  interrupted-and-resumed run and requires identical probabilities.
- **No invented numbers.** If a job has not run, the post-processing script
  skips that table and says so in `README_TABLES.md`.

## 5. Time budget and session strategy

| Job | T4 estimate | Sessions |
|---|---|---|
| `lock_thresholds` (5 seeds) | 4–6 h | 1 |
| `phikon` probe + TTA (5 seeds) | 10–15 h | 2 |
| `budget_heldout` (5 seeds) | ~20 h | 5 (one seed each) |
| `budget_slideout` (per seed) | ~4–5 h | 1 per seed |
| `center0 --seeds 5` | 5–7 h | 1 |
| `benchmark_gpu.py` | 10–20 min | 1 |

On an A100 or L4, the compute-heavy jobs are faster; the batch sizes and
checkpoint stay the same. Report the GPU name from `timing_benchmark.json`.

## 6. Common problems

- **Nothing resolves.** Run `discover` first. It prints every path it found and
  the Drive listing, so you can override only what is wrong.
- **`metadata.csv` not found.** Pass `--metadata ".../metadata.csv"`.
- **`archive.tar.gz` not found.** Pass `--archive "/content/drive/MyDrive/.../archive.tar.gz"`
  or `--extracted "/content/drive/MyDrive/.../patches"`.
- **Checkpoints not found.** Pass `--checkpoints-dir ".../folder-with-seed1_best.pth"`.
  `discover` prints each checkpoint size; ~0.98 GB each is correct.
- **Phikon 401/403.** Accept the model terms at
  `https://huggingface.co/owkin/phikon`, then set `HF_TOKEN`.
- **Colab disconnects.** Rerun the same command. The state is mirrored to
  Drive every 200 batches.
- **Timing OOM at batch 64.** That is a valid result. Keep the `oom` row and
  report the largest batch that fits.
- **Drive fills up.** `stage` copies the archive to `/content` scratch, not
  Drive. Outputs are small, but the Phikon feature cache lives in
  `/content/cg_tta_work` and is deleted when the VM resets.

> **Untested option.** `phikon --rotation center0` (refit the heads on the Center-0 source and run the online battery on Center-0) was added but never run, so it has not been validated end to end. No Phikon result on Center-0 is reported in the papers.

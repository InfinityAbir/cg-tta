# CG-TTA: auditing test-time adaptation of pathology ViTs

Code, per-seed predictions and table/figure scripts for

> **Help Versus Harm: Auditing Test-Time Adaptation of Pathology Vision Transformers on Camelyon17
> with a Label-Free Consensus Gate**
> Abir Hasan, Sadia Yasmin, Md. Abdul Awal (IUBAT, Dhaka, Bangladesh). Manuscript under review.

Archive (Zenodo): [10.5281/zenodo.24508277](https://doi.org/10.5281/zenodo.24508277)

Test-time adaptation (TTA) is usually reported as a change in mean accuracy. This repository splits
that change into the predictions an adapter **helps** (wrong to right) and the ones it **harms**
(right to wrong), by error type, and adds a label-free **consensus gate** that accepts an adapter's
flip only when a four-view Flip-TTA critic agrees.

## What the study found

* **Camelyon17-WILDS, Center-2 target (85,054 patches, five source seeds).** TENT raises mean accuracy
  by 1.69 pp and still flips 210-659 correct patches per seed. Almost all of that harm is false
  positives (343.8 against 2.2 false negatives per seed), while most of its corrections are missed
  tumors. Whether adapting pays off therefore depends on the false-positive to false-negative cost
  ratio (break-even about 5.2 on Center-2, about 1.5 on Center-0).
* **The gate** is exactly a 2-of-3 majority vote over the frozen, adapted and Flip-TTA labels, so its
  harmed set is always a subset of the adapter's. It cuts TENT's harm by 86% and keeps most of the
  gain, at a cost of 1.02 pp against TENT.
* **Second target center (Center-0)** shows the same direction for every effect.
* **Second benchmark (NCT-CRC-HE-100K to CRC-VAL-HE-7K, colorectal, nine classes).** The shift is mild:
  entropy-based adapters change only 1-7 of 7,180 predictions per seed, so there is almost no harm to
  audit. The Camelyon17 harm pattern is therefore not universal. Results and code are in
  `crc_second_dataset/` and `jbhi_revision/`.

## Quick start (CPU only, no images needed)

```bash
pip install -r requirements.txt
python rebuild_all.py                     # self-test, regenerate every table, byte-compare with tables/
python jbhi_revision/make_jbhi_assets.py  # Table I (both centers), cost / decile / drift figures
python jbhi_revision/analyze_crc.py       # second-benchmark numbers and supplement table
```

`rebuild_all.py` prints `REBUILD PASS` and exits 0 only when every regenerated file matches the archived
copy. The figure script uses Times New Roman as in the paper; without it matplotlib falls back to a
default serif and the PNGs will not be byte-identical.

## Layout

| path | contents |
|---|---|
| `predictions/` | per-seed, per-method target predictions (npz): `center2`, `center0` (seeds 1-5), `eata_full` |
| `crc_second_dataset/` | second benchmark: `tta/` (fixed random stream) and `tta_sorted/` (class-sorted stream) predictions for five methods and five seeds, training logs, split and stream order, environment record |
| `tables/`, `data/`, `figures/` | CSVs, LaTeX fragments and figures of the Camelyon17 analysis |
| `jbhi_revision/` | scripts and outputs for the journal version: Table I with both centers, added figures, second-benchmark analysis |
| `revision_colab/` | outputs of the later Colab jobs (Center-1 threshold locking, Phikon online TTA, held-out budget, a fifth Center-0 seed, GPU timing) |
| `code/` | `cg_tta.py` (gate, metrics, self-test) and the table and figure scripts |
| `colab/`, `colab_pro/` | resumable GPU jobs; `colab_pro/crc_colab.py` is the second-benchmark job (see `colab_pro/README_CRC.md`) |
| `logs/`, `analysis_records/`, `tests/` | run records, analysis scripts, a SHOT-IM unit test |
| `docs/DETAILS.md` | long technical reference: file formats, GPU jobs, scope and caveats |
| `MANIFEST.csv` | path, size and SHA-256 of every file |

## The gate in four lines

```python
# y0, y1, yf: frozen, adapted and four-view-critic labels
accept = (y0 != y1) & (yf == y1)
y_cg = np.where(accept, y1, y0)
```

Probabilities follow the same rule. `code/cg_tta.py` has the numpy and torch versions and checks that
they agree. The critic is always the frozen model, and each batch is scored before its own update.

## Rerunning the GPU jobs

The predictions are included, so the GPU scripts are only needed to adapt from images. Camelyon17 jobs
are described in `docs/DETAILS.md`. The colorectal job runs end to end on a free-tier or Pro Colab T4 and
resumes after a disconnect:

```python
!python crc_colab.py smoke                                   # CPU/GPU self-test, must end with SMOKE OK
!python crc_colab.py train --seeds 1 2 3 4 5 --save-every 100000
!python crc_colab.py tta --seeds 1 2 3 4 5
!python crc_colab.py tta --seeds 1 2 3 4 5 --stream sorted
```

## Data

The Camelyon17-WILDS patches and the NCT-CRC-HE-100K / CRC-VAL-HE-7K images are **not** redistributed here.
Camelyon17-WILDS is available through the [WILDS benchmark](https://wilds.stanford.edu/). The colorectal
sets are on Zenodo (record 1214456, Kather et al., 2019) and `crc_colab.py` downloads and checks them.
The `patch_ids` in the predictions index the datasets' own metadata, and are meaningless without them.
The predictions are enough to recompute every number in the paper.

## Scope

The evidence comes from Camelyon17-WILDS (binary 96x96 patches, ten target slides per center), one
supervised ViT-B/16 recipe and one small colorectal benchmark with a mild shift. The Camelyon17 stream
is sorted, with all tumor patches first, and random orders halve the harm on one seed, so the harm counts
describe that protocol. The gate's harm-subset property assumes a fixed adapter trajectory and is not a
safety guarantee. The paper's limitations section and `docs/DETAILS.md` list the rest.

## License and citation

Code: MIT (`LICENSE`). Predictions, tables and figures: CC BY 4.0 (`LICENSE-DATA.md`). The datasets keep
their own licenses. If you use this work, please cite the paper (`CITATION.cff`). This code and data archive is also on
Zenodo: https://doi.org/10.5281/zenodo.24508277.

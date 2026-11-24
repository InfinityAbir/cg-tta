# Extension GPU runs

These runs were completed after the first full draft and are now included
in the 10-page manuscript: the four-seed Center-0 replication, the
full-parameter EATA boundary, and the frozen Phikon probe. Numbers here were
recomputed from the released npz streams with `code/cg_tta.py` and
`code/make_postsubmission_tables.py`; they are not copied from console logs.

- GPU: Colab T4, 16 GB.
- Script: `colab/colab_cg_tta_gpu.py`, SHA-256
  `b7bcf9be930b90add6d231de37f94d1dff6d406c54090ed480ffba876e74cc52`.
- Environment: the release's `environment.txt` plus the Colab session's
  `environment.json` / `pip-freeze.txt` on Drive. Those two Colab files were not
  included in this release.
- Scope: Center-0 seeds 1-4 (seeds 2-4 new), full-parameter EATA seeds 1-3 at
  5e-5, and the frozen Phikon linear probe.
- Not run: Center-0 seed 5, Phikon TTA battery, full-EATA seeds 4-5, budget
  held-out.

## 1. Center-0 replication, four source seeds

Center-0 has 59,436 patches from 10 slides. All four seeds are used in the
manuscript's Center-0 table. Seeds 2-4 are new; each is a separate source model
trained on centers 3 and 4 with center 1 as validation, then adapted to center
0. The 15 new prediction files (five streams x three seeds) pass the Colab
`verify[center0]: 15 prediction files, PASS` audit and the offline
patch-id/label alignment check.

Patch-level accuracy by seed:

| seed | frozen | TENT | Flip-TTA | EATA-LN | SHOT-IM | CG-TENT | CG-EATA | CG-SHOT |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.9517 | 0.9535 | 0.9609 | 0.9528 | 0.9475 | 0.9534 | 0.9529 | 0.9529 |
| 2 | 0.9542 | 0.9611 | 0.9590 | 0.9595 | 0.9213 | 0.9589 | 0.9580 | 0.9524 |
| 3 | 0.9627 | 0.9642 | 0.9691 | 0.9642 | 0.9632 | 0.9648 | 0.9644 | 0.9664 |
| 4 | 0.9688 | 0.9695 | 0.9733 | 0.9693 | 0.9671 | 0.9694 | 0.9692 | 0.9705 |

Four-seed means (patch level; delta in percentage points vs frozen):

| method | accuracy | delta | harm/seed | positive seeds |
|---|---|---|---|---|
| Frozen | 0.9593 | - | - | - |
| TENT | 0.9621 | +0.27 | 65 | 4/4 |
| Flip-TTA | 0.9656 | +0.62 | 275 | 4/4 |
| EATA-LN | 0.9614 | +0.21 | 41 | 4/4 |
| SHOT-IM | 0.9498 | -0.96 | 799 | 1/4 |
| CG-TENT | 0.9616 | +0.23 | 10 | 4/4 |
| CG-EATA | 0.9612 | +0.18 | 7 | 4/4 |
| CG-SHOT | 0.9606 | +0.12 | 79 | 3/4 |

Slide-level means (10 slides x 4 seeds):

| method | accuracy | delta |
|---|---|---|
| Frozen | 0.95330 | - |
| TENT | 0.95534 | +0.205 |
| CG-TENT | 0.95525 | +0.195 |
| Flip-TTA | 0.96047 | +0.717 |
| EATA-LN | 0.95491 | +0.161 |
| CG-EATA | 0.95486 | +0.157 |
| SHOT-IM | 0.94926 | -0.404 |
| CG-SHOT | 0.95565 | +0.235 |

Interpretation:

* TENT, EATA-LN and Flip-TTA are positive on all four seeds. The four-seed
  Center-0 result is not a seed-1 artifact.
* SHOT-IM is seed-unstable, not uniformly collapsing: -3.28 pp on seed 2,
  +0.05 pp on seed 3, and negative on seeds 1 and 4. It is positive on 1/4
  seeds. The manuscript's "unstable" wording is the defensible one.
* At slide level, CG-TENT and CG-EATA are within 0.01 pp of their adapters
  (0.95525 vs 0.95534; 0.95486 vs 0.95491) while cutting per-seed harm from 65
  to 10 and from 41 to 7 patches. This is the manuscript's
  accuracy-preservation claim.
* CG-SHOT recovers +0.64 pp over SHOT at slide level (+1.08 pp patch level)
  and cuts per-seed harm from 799 to 79, but remains 0.17 pp below frozen on
  seed 2, so it is not a uniform win.
* Seed 5 was deliberately not run: n=4 closes the replication item without
  spending another session. There is no seed-5 prediction in the release.

## 2. Full-parameter EATA, Center-2

The pre-specified 2.5e-4 step collapsed on the full stream (seed 1): accuracy
0.7429, 19,543 harms, ECE 0.2463, against a frozen baseline of 0.9490. The
200-batch probe had looked healthy (0.9655), so the probe is a memory/time
check, not a stability proof. The collapse run is preserved at
`predictions/eata_full/seed1_lr2.5e-4/` and in `logs/eata_full_lr.md`.

The multi-seed runs use 5e-5, a post-hoc stability calibration chosen after the
collapse. This is target-label-informed and is reported as post hoc, not as the
pre-specified protocol.

| seed | frozen | EATA-full | CG-EATA-full | EATA vs frozen | CG vs frozen | CG vs EATA | EATA harm | CG harm |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.9490 | 0.9543 | 0.9547 | +0.54 | +0.58 | +0.04 | 975 | 58 |
| 2 | 0.9182 | 0.9517 | 0.9290 | +3.35 | +1.08 | -2.27 | 651 | 76 |
| 3 | 0.9099 | 0.9319 | 0.9191 | +2.20 | +0.93 | -1.28 | 2,172 | 173 |
| mean | 0.9257 | 0.9460 | 0.9343 | +2.03 | +0.86 | -1.17 | 1,266 | 102 |

The gate accepts 2,811 of the adapter's changed predictions; 2,504 are correct
(precision 89.1%). It keeps 27.9% of the adapter's helps and rejects 91.9% of
its harms. Mean harm falls from 1,266 to 102 patches (-92%), but accuracy falls
1.17 pp below the adapter. On seed 1 the gate is neutral-to-slightly-positive;
on seeds 2 and 3 it is clearly conservative.

**This is a safety-accuracy trade-off, not accuracy preservation.** It is the
boundary case for the paper's claim: LayerNorm-only adapters (TENT, EATA-LN,
SHOT-IM) change fewer predictions and their correct flips are more often
confirmed by the frozen four-view critic, so the gate preserves their accuracy.
Full-parameter EATA produces many correct flips that the critic does not
confirm; rejecting them is what buys the 92% harm reduction, and it costs
accuracy. The manuscript reports this boundary explicitly.

## 3. Phikon frozen linear probe

`owkin/phikon`, frozen features, linear head, seed 1, best epoch 20, 85,054
target patches:

| metric | value |
|---|---|
| target accuracy | 0.9432 |
| balanced accuracy | 0.9432 |
| AUROC | 0.9915 |
| ECE (15 bins) | 0.0247 |
| validation balanced accuracy | 0.9893 |

The supervised ViT-B seed-1 baseline is 0.9490, so Phikon is -0.58 pp. The
feature arrays were 302,436 x 768 (source), 33,560 x 768 (validation) and
85,054 x 768 (target); the Colab `verify` audit passed. The Phikon TTA battery
was not run; session 2 was disconnected to stop burning compute units. The
released probe is therefore descriptive, not a second TTA result.

## 4. What is in the release

- `predictions/center0/seed{2,3,4}/{no,tent,flip,eata,shot}.npz`
- `predictions/eata_full/seed{1,2,3}/{no,eata}.npz`
- `predictions/eata_full/seed1_lr2.5e-4/{no,eata}.npz`
- `tables/core_center0_multiseed_{seed_metrics,slide_metrics,summary}.csv`
- `tables/core_eata_full_{seed_metrics,summary}.csv`
- `tables/core_phikon_probe.csv`
- `logs/phikon_probe.json`, `logs/postsubmission_runs.json`
- `code/make_postsubmission_tables.py` (regenerates all six CSVs)

Regenerate:

```bash
python code/make_postsubmission_tables.py
```

The script asserts patch-id, label and slide alignment before computing
anything, and it derives CG-TTA rows with the same `cg_tta.py` gate used by
the paper.

## 5. Caveats

* Center-0 n=4, seed 5 not run. The center0 source checkpoints for seeds 2-4
  were not archived in the artifact zip, so `logs/seeds.csv` records them as
  "not archived locally"; the predictions are the authoritative artifact.
  The Center-2 checkpoint hashes for the EATA-full runs are archived.
* The 5e-5 full-EATA calibration used target accuracy after the 2.5e-4 collapse.
  It is post hoc and must not be presented as a pre-specified hyperparameter.
* Full-EATA results are on Center-2 only; there is no full-EATA run on Center-0.
* Phikon TTA was not run. The probe is one seed, and its -0.58 pp difference
  from the supervised ViT-B is descriptive, not a powered comparison.
* These extensions are included in the 10-page manuscript with their post-hoc
  labels and caveats. Any further extension must be checked against the page
  limit and the paper's pre-specified primary endpoints.


## Revision runs (Colab Pro, T4), November 2026

| Job | Scope | Wall time | verify |
|---|---|---|---|
| `lock_thresholds` | Center-1 locked thresholds, seeds 1-5 | 4 h 30 min | PASS |
| `phikon` | five heads, online TTA incl. EATA-LN, Center-2 | 4 h 48 min | PASS (25 prediction files) |
| `budget_heldout` seed 1 | 20 TENT runs, held-out 8,491 patches | 1 h 55 min | summary-only (older script version) |
| `budget_heldout` seed 2 | same | 1 h 50 min | PASS |
| `budget_heldout` seed 3 | same | 1 h 50 min | PASS |
| `budget_heldout` seed 4 | same | 2 h 03 min | summary-only (older script version) |
| `budget_heldout` seed 5 | same | 1 h 50 min | PASS |
| `center0 --seeds 5` | Center-0 rotation, source seed 5 (early stop epoch 7, best epoch 4) | 7 h 58 min | PASS (5 prediction files) |

Budget seeds 1 and 4 ran with a script version that saved summary metrics but not
per-patch held-out predictions; the later version saves `tent_predictions.npz` and
`verify` accepts summary-only units, so seeds 1 and 4 were not rerun.

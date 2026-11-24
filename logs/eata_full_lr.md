# Full-parameter EATA: learning-rate record

This note records one negative result and one post-hoc calibration. It is a
provenance record for the full-parameter EATA runs now reported in the
manuscript's safety-accuracy boundary paragraph.

## Pre-specified configuration

The published EATA step for full-parameter adaptation is SGD 2.5e-4. It was the
default (`LR_FULL`) when the full-parameter job was written. No target label was
used to choose it.

## What happened at 2.5e-4

The 200-batch probe at 2.5e-4 looked healthy (Center-2, seed 1):
`mean_accuracy` 0.9655, `mean_selected` 0.931, `mean_loss` 0.0155. It was not:
over the full 85,054-patch stream the same configuration collapsed.

Full stream, Center-2, seed 1, 2.5e-4:

- accuracy 0.7429 (frozen same-seed baseline 0.9490)
- help 2,013, harm 19,543
- ECE 0.2463

The run is preserved on Drive at
`03_Experiments/15_SafeTTA_TMI/eata_full/seed1_lr2.5e-4/` and is reported as the
pre-specified negative result. It is not discarded or overwritten.

## Post-hoc calibration to 5e-5

After the collapse, the step was reduced by a factor of five to 5e-5. This
choice used target accuracy, so it is a post-hoc stability calibration, not a
blind pre-specified hyperparameter. The adaptive objective, gate, selection
rule, Fisher estimate and evaluation protocol were unchanged.

Full stream, Center-2, 5e-5:

| seed | accuracy | help | harm | net |
|------|----------|------|------|-----|
| 1 | 0.9543 | 1,431 | 975 | +456 |
| 2 | 0.9517 | 3,501 | 651 | +2,850 |
| 3 | 0.9319 | 4,044 | 2,172 | +1,872 |
| mean | 0.9460 | 2,992 | 1,266 | +1,726 |

The frozen same-seed mean is 0.9257, so the calibrated full-parameter EATA gains
+2.03 pp on average. All three runs pass the release verifier.

The CG-TTA gate applied to these runs is recorded in
`logs/postsubmission_runs.md`. It cuts mean harm from 1,266 to 102 patches
(-92%) but costs 1.17 pp relative to the adapter, so the full-parameter case is
a safety-accuracy trade-off, not an accuracy-preservation result.

## Provenance

The 5e-5 runs were produced before `--lr-full` existed, by editing `LR_FULL` in
the Colab copy used for that session. Their `unit.json` therefore does not
contain an `lr_full` field. The canonical script now exposes `--lr-full`, and

```bash
python colab_cg_tta_gpu.py eata_full --seeds 1 2 3 --lr-full 5e-5
```

reproduces the configuration. A state file written by the current script records
`lr_full`; resuming with a different value stops with an error instead of
silently changing the step.

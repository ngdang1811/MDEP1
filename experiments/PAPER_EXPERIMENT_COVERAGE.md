# Paper Experiment Coverage

This file maps experiments described in `main_text.tex` to runnable Kaggle
commands in the `experiments/` folder.

## One-Command Kaggle Suite

Clone the GitHub repository into `/kaggle/working` first. The complete notebook
setup order is documented in `experiments/KAGGLE_GITHUB_SETUP.md`.

Run the whole paper-facing suite:

```bash
python experiments/run_kaggle_paper_suite.py --isic_suite all --no_save_model --keep_going
```

For a quick smoke test before spending GPU hours:

```bash
python experiments/run_kaggle_paper_suite.py --smoke
```

All logs and metrics are written under:

```text
paper_experiment_outputs/
```

## ISIC 2024 Main Tables

The main ISIC tables in `main_text.tex` are covered by:

```bash
python experiments/isic_paper_experiments.py --suite main_tables
```

| Paper row | Runnable experiment |
| --- | --- |
| Fisher EDL | `fisher_edl` |
| Flexible EDL | `flexible_edl` |
| R-EDL | `r_edl` |
| GUDS-EDL (Ours) | `full_guds` |

The broader baseline suite can be run with:

```bash
python experiments/isic_paper_experiments.py --suite baselines
```

It includes Standard CE, Focal Loss, Logit Adjustment, Class-Balanced CE,
Balanced Softmax, LDAM-DRW, cRT-style classifier retraining, Dense EDL, Fisher
EDL, Flexible EDL, R-EDL, Static 2:4 EDL, and a RigL-style 2:4 proxy.

## Appendix C Ablations

Run:

```bash
python experiments/isic_paper_experiments.py --suite ablations
```

| Paper target | Runnable experiment |
| --- | --- |
| Full GUDS-EDL | `full_guds` |
| w/o uncertainty pruner | `guds_without_pruner` |
| w/o evidence regrower | `guds_without_regrower` |
| w/o asymmetric KL | `guds_symmetric_kl` |
| w/o Evidential Focal Loss | `guds_without_efl` |
| w/o anti-crystallization | `guds_without_anticryst` |
| absolute-gradient pruning ablation | `guds_absolute_pruner` |
| KL-to-uniform regrowth ablation | `guds_kl_uniform_regrower` |
| w/o topology cache | `guds_without_topology_cache` |
| temperature-only calibration | `guds_temperature_only` |
| w/o post-hoc calibration | `guds_no_posthoc_calibration` |

## Planned Generalization Protocols

The CIFAR-100-LT protocol can be run by the all-in-one suite, or manually:

```bash
python experiments/generalization_paper_suite.py --benchmark cifar --ratio 10 --epochs 100 --seeds 42 43 44
python experiments/generalization_paper_suite.py --benchmark cifar --ratio 50 --epochs 100 --seeds 42 43 44
python experiments/generalization_paper_suite.py --benchmark cifar --ratio 100 --epochs 100 --seeds 42 43 44
```

Each CIFAR run includes CE, Focal Loss, Logit Adjustment, Class-Balanced CE,
Balanced Softmax, LDAM-DRW, cRT-style classifier retraining, Dense EDL, Static
2:4 EDL, RigL-style 2:4, and full GUDS-EDL. The CIFAR metrics include top-1,
top-5, balanced accuracy, macro-F1, macro-AUROC, macro-PR-AUC, NLL, multiclass
Brier, AURC/E-AURC, failure-detection AUROC/AUPR, many/medium/few-shot
accuracy, group ECE, classwise ECE, and worst-group accuracy.

The MVTec AD image-level protocol can be run by:

```bash
python experiments/generalization_paper_suite.py --benchmark mvtec --category hazelnut --epochs 20 --seeds 42 43 44
python experiments/generalization_paper_suite.py --benchmark mvtec --category bottle --epochs 20 --seeds 42 43 44
```

`mvtec_ad_runner.py` now searches for a real MVTec category under `MVTEC_ROOT`,
`./data/mvtec_ad`, `./data/mvtec`, or `/kaggle/input`. If no real category is
found, it fails fast by default. Dummy tensors require the explicit
`--allow_dummy_data` flag and should never be used for paper results.

For an anomaly-detection reference baseline on the same real MVTec categories,
run the PatchCore-lite protocol:

```bash
python experiments/mvtec_patchcore_reference.py --category hazelnut --seeds 42 43 44
python experiments/mvtec_patchcore_reference.py --category bottle --seeds 42 43 44
```

The all-in-one suite runs this reference by default for each selected MVTec
category. Pass `--skip_mvtec_reference` to skip it. The reference metrics include
image-level AUROC, image-level AP, F1-max, NLL/Brier on normalized anomaly
scores, risk-at-coverage, and failure-detection AUROC/AUPR.

## Hardware, Quality Gates, and Backbones

Hardware profiling is included in the all-in-one suite and can also be run
manually:

```bash
python experiments/hardware_profile.py
```

The hardware profiler now writes per-mode `metrics.json` files so
`summarize_results.py` can aggregate active density, valid 2:4 block fraction,
masked-PyTorch throughput, peak CUDA memory, and the theoretical 2:4 sparse
Tensor Core speedup upper bound. These are structural/profiling metrics unless
you export to a real cuSPARSELt/TensorRT sparse kernel.

The ISIC runner saves quality-gated failure-detection metrics inside each
`metrics.json` file. Additional backbone experiments are implemented as an
optional heavyweight protocol:

```bash
python experiments/backbone_generalization_runner.py --backbones resnet18 convnext_tiny swin_t --epochs 40 --seeds 42 43 44
```

After all runs, aggregate seed statistics:

```bash
python experiments/summarize_results.py
```

## Remaining Caveats

- `Flexible EDL` and `R-EDL` are implemented as reproducible in-repo baseline
  variants so the paper table can be regenerated on the same split. They are
  not official external code releases.
- `RigL-style 2:4` is a proxy using the available structured sparse update
  surface. A fully faithful RigL implementation would require a separate update
  engine.
- Real speedups for 2:4 sparsity still require sparse Tensor Core kernels; these
  scripts report training/evaluation metrics and structural sparsity, not
  guaranteed hardware acceleration.

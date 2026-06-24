# GUDS-EDL Experiments Setup Guide

This folder contains the experimental pipeline for evaluating the **GUDS-EDL** (Generalized Uncertainty-Guided Dynamic Sparsification for Evidential Long-Tailed Learning) framework across different extreme imbalance and long-tailed benchmarks.

## 1. Prerequisites
Ensure your environment has the following installed:
- Python 3.8+
- PyTorch (with CUDA support)
- torchvision, numpy, pandas, scikit-learn, matplotlib, jupyter
- wandb (Optional, for online experiment tracking)

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib jupyter wandb
```

## 2. Supported Benchmarks & Runners

We provide targeted runners to evaluate GUDS-EDL on different benchmarks. Each runner handles dataset loading, model initialization, dynamic sparse training, bias-corrected temperature calibration, and final evaluation.

### Group A: Controlled Long-Tailed Recognition
- **`cifar_lt_runner.py`**: Runs CIFAR-100 with exponential class imbalance (Ratios 1:10, 1:50, 1:100). Automatically downloads the dataset if not present.
- **`generalization_paper_suite.py`**: Runs the paper-facing CIFAR/MVTec suites, including CE, Focal Loss, Logit Adjustment, Class-Balanced CE, Balanced Softmax, LDAM-DRW, cRT-style retraining, Dense EDL, Static 2:4, RigL-style 2:4, and GUDS-EDL.

### Group B: Industrial Defect / Anomaly Detection
- **`mvtec_ad_runner.py`**: Runs real MVTec AD image-level binary rare-defect classification. It now fails fast if a real category is not found; dummy tensors are available only with `--allow_dummy_data` for dry-runs.
- **`mvtec_patchcore_reference.py`**: Runs a normal-only PatchCore-lite ResNet-18 feature baseline as an anomaly-detection reference for MVTec AD.

### Group C: High-Stakes Rare-Event Case Study
- **ISIC 2024**: Evaluated via the main core file `../guds_edl_core.py`. Requires the real ISIC 2024 Kaggle competition input; dummy tensors are available only with `--allow_dummy_data` for dry-runs.

## 3. How to Run Experiments

### Kaggle From GitHub
After this repository has been pushed to GitHub, you do not need to upload the
whole codebase as a Kaggle Dataset. In a Kaggle notebook, clone the repository
into `/kaggle/working` and run the suite from there:

```bash
cd /kaggle/working
git clone https://github.com/minhduc110207/MDEP-Microglial-Driven-Evidential-Pruning.git
cd MDEP-Microglial-Driven-Evidential-Pruning
python experiments/run_kaggle_paper_suite.py --smoke
```

For the full paper-facing run:

```bash
python experiments/run_kaggle_paper_suite.py --isic_suite all --no_save_model --keep_going
```

If the GitHub repository is private, add a Kaggle secret named `GITHUB_TOKEN`
with read access to the repository, then clone with the token inside the
notebook without printing it:

```python
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
!git clone https://{token}@github.com/minhduc110207/MDEP-Microglial-Driven-Evidential-Pruning.git
```

See `experiments/KAGGLE_GITHUB_SETUP.md` for the complete Kaggle setup order.

### Kaggle Paper Suite (Recommended)
After cloning or copying the repository to `/kaggle/working`, run:

```bash
python experiments/run_kaggle_paper_suite.py --isic_suite all --no_save_model --keep_going
```

Before spending GPU hours, run a one-epoch smoke test:

```bash
python experiments/run_kaggle_paper_suite.py --smoke
```

The suite writes logs, checkpoints, and metrics under:

```text
paper_experiment_outputs/
```

By default, a full run uses seeds `42 43 44` for reproducibility. A smoke test
uses only seed `42`. To run a cheaper first full pass, explicitly pass one seed:

```bash
python experiments/run_kaggle_paper_suite.py --isic_suite all --seeds 42 --no_save_model --keep_going
```

For ISIC-only table reproduction:

```bash
python experiments/isic_paper_experiments.py --suite main_tables
```

For all ISIC baselines and ablations:

```bash
python experiments/isic_paper_experiments.py --suite all
```

The complete planned CIFAR/MVTec baseline suites are run through:

```bash
python experiments/generalization_paper_suite.py --benchmark cifar --ratio 100 --epochs 100 --seeds 42 43 44
python experiments/generalization_paper_suite.py --benchmark mvtec --category hazelnut --epochs 20 --seeds 42 43 44
python experiments/mvtec_patchcore_reference.py --category hazelnut --seeds 42 43 44
```

Hardware profiling and summary aggregation:

```bash
python experiments/hardware_profile.py
python experiments/summarize_results.py
```

### Automated Batch Mode (Recommended)
You can automatically execute the full suite of runners across datasets using the provided batch script:
1. Open PowerShell or Command Prompt.
2. Execute: `.\run_benchmarks.bat`

### Component Ablation Study
To investigate the contribution of individual GUDS-EDL components, run the
paper-facing ablation suite:

```bash
python experiments/isic_paper_experiments.py --suite ablations --seeds 42 43 44 --no_save_model
```

The notebook `experiments/ablation_experiments.ipynb` is only a thin wrapper
around this command; it does not define separate metrics or simulated results.

### Manual CLI Execution
You can also run any of the benchmarks manually from the terminal. The core and runners support argparse flags to toggle ablations.
```bash
# Full GUDS-EDL on CIFAR-100-LT
python cifar_lt_runner.py --imbalance_ratio 100 --epochs 100

# Ablation: Disable Astrocyte Regrowing (Topology Freezing)
python cifar_lt_runner.py --imbalance_ratio 100 --disable_regrower

# Ablation: Magnitude Pruning & Random Growth Baseline
python cifar_lt_runner.py --imbalance_ratio 100 --pruner_type magnitude --regrower_type random
```

## 4. Expected Outputs
During training and evaluation, the scripts will generate:
- `paper_experiment_outputs/**/metrics.json`: Per-run metric records.
- `paper_experiment_outputs/summary_metrics.csv`: Mean/std aggregation across seeds after `summarize_results.py`.
- Optional model checkpoints unless `--no_save_model` is passed.
- Logs for each all-in-one suite command under `paper_experiment_outputs/logs/`.

The metric records now include paper-facing long-tail, calibration, failure-detection, uncertainty-separation, clinical utility, MVTec image-level, and hardware structural/profiling metrics where applicable.

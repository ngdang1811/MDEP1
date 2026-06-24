# Kaggle Setup From GitHub

This guide is the recommended way to run the paper experiments on Kaggle once
the repository is available on GitHub. The code should come from GitHub; the
datasets should be attached through Kaggle inputs.

## 1. Create the Kaggle Notebook

1. Create a new Kaggle notebook.
2. Set accelerator to GPU.
3. Add the required datasets under **Add Input**:
   - ISIC 2024 training metadata and images for the main paper experiments.
   - MVTec AD only if you want to run the planned industrial anomaly protocol.
   - CIFAR-100 is downloaded automatically by `torchvision`, so it usually does
     not need a Kaggle input dataset.

## 2. Clone the Repository

For a public GitHub repository, run this in the first notebook cell:

```bash
%cd /kaggle/working
!git clone https://github.com/minhduc110207/MDEP-Microglial-Driven-Evidential-Pruning.git
%cd MDEP-Microglial-Driven-Evidential-Pruning
```

For a private repository, create a Kaggle secret named `GITHUB_TOKEN` with read
access to the repository, then run:

```python
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
!git clone https://{token}@github.com/minhduc110207/MDEP-Microglial-Driven-Evidential-Pruning.git
%cd MDEP-Microglial-Driven-Evidential-Pruning
```

Do not print the token in notebook output.

## 3. Install Runtime Dependencies

Kaggle normally includes PyTorch, torchvision, numpy, pandas, scikit-learn, and
matplotlib. If a package is missing, install only the missing package:

```bash
!pip install -q scikit-learn matplotlib pandas
```

## 4. Verify the Code Path

Run a smoke test before launching the full experiment suite:

```bash
!python experiments/run_kaggle_paper_suite.py --smoke
```

The smoke test is the fastest check that imports, dataset discovery, output
folders, and the core training loop are wired correctly.

## 5. Run the Paper-Facing Suite

Run all ISIC baselines and ablations described by the paper-facing experiment
map:

```bash
!python experiments/run_kaggle_paper_suite.py --isic_suite all --no_save_model --keep_going
```

The outputs are written to:

```text
paper_experiment_outputs/
```

## 6. Optional Protocols

Run the complete planned CIFAR-100-LT baseline suite:

```bash
!python experiments/generalization_paper_suite.py --benchmark cifar --ratio 10 --epochs 100 --seeds 42 43 44
!python experiments/generalization_paper_suite.py --benchmark cifar --ratio 50 --epochs 100 --seeds 42 43 44
!python experiments/generalization_paper_suite.py --benchmark cifar --ratio 100 --epochs 100 --seeds 42 43 44
```

Run the complete MVTec AD baseline suite after attaching a real MVTec dataset:

```bash
!python experiments/generalization_paper_suite.py --benchmark mvtec --category hazelnut --epochs 20 --seeds 42 43 44
!python experiments/generalization_paper_suite.py --benchmark mvtec --category bottle --epochs 20 --seeds 42 43 44
!python experiments/mvtec_patchcore_reference.py --category hazelnut --seeds 42 43 44
!python experiments/mvtec_patchcore_reference.py --category bottle --seeds 42 43 44
```

If no real MVTec category is found, the runner fails fast by default. This is
intentional for paper experiments: all reported runs should use the real MVTec
AD folders. Use `--allow_dummy_data` only for local dry-runs of the classifier
runner; the PatchCore-lite reference always requires real MVTec images.

Run hardware profiling and aggregate all seed results:

```bash
!python experiments/hardware_profile.py
!python experiments/summarize_results.py
```

The optional additional-backbone protocol is heavy and should be run in a
separate notebook/session:

```bash
!python experiments/backbone_generalization_runner.py --backbones resnet18 convnext_tiny swin_t --epochs 40 --seeds 42 43 44
```

## 7. Updating Code During a Kaggle Run

If you push fixes to GitHub while the Kaggle notebook is open, update the local
copy with:

```bash
%cd /kaggle/working/MDEP-Microglial-Driven-Evidential-Pruning
!git pull
```

Then rerun the smoke test before the full run.

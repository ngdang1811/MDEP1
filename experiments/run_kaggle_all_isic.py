"""
Run the ISIC 2024 GUDS-EDL experiment suite sequentially on Kaggle.

Usage from a Kaggle notebook after copying the repo to /kaggle/working:

    %cd /kaggle/working
    !python experiments/run_kaggle_all_isic.py

Each run writes its stdout/stderr log and checkpoints into:

    /kaggle/working/experiment_outputs/<experiment_name>/

The script removes top-level checkpoints before each run so experiments do not
resume from or overwrite each other by accident.

Scope note:
    This is a legacy single-core wrapper around guds_edl_core.py. For the
    complete paper-facing suite, including Fisher/Flexible/R-EDL proxies,
    topology-cache and calibration ablations, CIFAR-100-LT, MVTec AD,
    multi-seed execution, and hardware profiling, use:

        python experiments/run_kaggle_paper_suite.py --isic_suite all
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Experiment:
    name: str
    args: tuple[str, ...]


EXPERIMENTS: tuple[Experiment, ...] = (
    # Main ISIC result row for main_text.tex Tables 1--2.
    Experiment(
        "01_full_guds_edl_class_conditioned",
        ("--regrower_type", "class_conditioned"),
    ),

    # Component ablations corresponding to Appendix C.
    Experiment(
        "02_without_uncertainty_pruner",
        ("--disable_pruner", "--regrower_type", "class_conditioned"),
    ),
    Experiment(
        "03_without_regrower",
        ("--disable_regrower",),
    ),
    Experiment(
        "04_symmetric_kl",
        ("--kl_scaling", "symmetric", "--regrower_type", "class_conditioned"),
    ),
    Experiment(
        "05_without_efl",
        ("--disable_efl", "--regrower_type", "class_conditioned"),
    ),
    Experiment(
        "06_without_anti_crystallization",
        ("--disable_anticryst", "--regrower_type", "class_conditioned"),
    ),

    # Mathematical ablations corresponding to Appendix C.
    Experiment(
        "07_absolute_pruner_class_conditioned_regrower",
        ("--pruner_type", "absolute_grad", "--regrower_type", "class_conditioned"),
    ),
    Experiment(
        "08_signed_pruner_kl_uniform_regrower",
        ("--regrower_type", "kl_uniform"),
    ),

    # Topology baselines exposed by the same CLI surface.
    Experiment(
        "09_magnitude_pruner_random_regrower",
        ("--pruner_type", "magnitude", "--regrower_type", "random"),
    ),
    Experiment(
        "10_random_topology_baseline",
        ("--pruner_type", "random", "--regrower_type", "random"),
    ),
)

PAPER_COVERAGE = (
    "Covered by this legacy script:",
    "- main_text.tex Tables 1--2: GUDS-EDL (Ours) row via 01_full_guds_edl_class_conditioned.",
    "- Appendix C: w/o pruner, w/o regrower, symmetric KL, w/o EFL, w/o anti-crystallization, absolute pruner, KL-uniform regrower, and random/magnitude topology baselines.",
    "",
    "Use run_kaggle_paper_suite.py for complete paper coverage:",
    "- ISIC long-tailed, evidential, dynamic sparse, GUDS, topology-cache, and calibration ablations.",
    "- CIFAR-100-LT planned baseline suite for 1:10, 1:50, and 1:100.",
    "- MVTec AD planned baseline suite for selected categories.",
    "- Multi-seed execution, hardware profiling, and summary CSV generation.",
)

TOP_LEVEL_OUTPUTS = (
    "latest_checkpoint.pth",
    "best_checkpoint.pth",
    "resnet_calibrated_adaptive.pth",
    "mdep_model.pth",
)


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent.parent, Path.cwd()):
        if (candidate / "guds_edl_core.py").exists():
            return candidate
    raise FileNotFoundError("Could not locate guds_edl_core.py.")


def working_dir(repo_root: Path) -> Path:
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        return kaggle_working
    return repo_root


def clean_previous_outputs(out_dir: Path, repo_root: Path) -> None:
    for name in TOP_LEVEL_OUTPUTS:
        path = out_dir / name
        if path.exists():
            path.unlink()

    artifacts = repo_root / "artifacts"
    if artifacts.exists():
        shutil.rmtree(artifacts)


def collect_outputs(run_dir: Path, out_dir: Path, repo_root: Path) -> None:
    for name in TOP_LEVEL_OUTPUTS:
        src = out_dir / name
        if src.exists():
            shutil.move(str(src), str(run_dir / name))

    artifacts = repo_root / "artifacts"
    if artifacts.exists():
        dst = run_dir / "artifacts"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(artifacts), str(dst))


def run_experiment(exp: Experiment, repo_root: Path, out_root: Path, out_dir: Path) -> int:
    run_dir = out_root / exp.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    clean_previous_outputs(out_dir, repo_root)

    cmd = [sys.executable, str(repo_root / "guds_edl_core.py"), *exp.args]
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("WANDB_SILENT", "true")
    env.setdefault("PYTHONUNBUFFERED", "1")

    print("\n" + "=" * 88)
    print(f"Running {exp.name}")
    print("Command:", " ".join(cmd))
    print("Log:", log_path)
    print("=" * 88)

    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

    collect_outputs(run_dir, out_dir, repo_root)
    return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all Kaggle ISIC GUDS-EDL experiments.")
    parser.add_argument(
        "--start_at",
        type=int,
        default=1,
        help="1-based experiment index to start from, useful after a Kaggle timeout.",
    )
    parser.add_argument(
        "--stop_after",
        type=int,
        default=len(EXPERIMENTS),
        help="1-based experiment index to stop after.",
    )
    args = parser.parse_args()

    repo_root = find_repo_root()
    out_dir = working_dir(repo_root)
    out_root = out_dir / "experiment_outputs"
    out_root.mkdir(parents=True, exist_ok=True)

    selected = EXPERIMENTS[args.start_at - 1 : args.stop_after]
    if not selected:
        raise ValueError("No experiments selected. Check --start_at and --stop_after.")

    failures: list[tuple[str, int]] = []
    for exp in selected:
        code = run_experiment(exp, repo_root, out_root, out_dir)
        if code != 0:
            failures.append((exp.name, code))
            print(f"[ERROR] {exp.name} failed with exit code {code}. Stopping.")
            break

    summary_path = out_root / "SUMMARY.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("GUDS-EDL Kaggle ISIC experiment suite\n")
        f.write(f"Repo root: {repo_root}\n")
        f.write(f"Output root: {out_root}\n\n")
        for idx, exp in enumerate(EXPERIMENTS, start=1):
            status = "selected" if exp in selected else "not selected"
            f.write(f"{idx:02d}. {exp.name}: {status}; args={' '.join(exp.args)}\n")
        f.write("\nPaper coverage:\n")
        for line in PAPER_COVERAGE:
            f.write(line + "\n")
        if failures:
            f.write("\nFailures:\n")
            for name, code in failures:
                f.write(f"- {name}: exit code {code}\n")
        else:
            f.write("\nFailures: none in selected range\n")

    print("\nSuite summary:", summary_path)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

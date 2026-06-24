"""
Kaggle all-in-one paper experiment launcher.

This script runs the experiment groups referenced by main_text.tex:

1. ISIC 2024 main-table baselines, GUDS-EDL ablations, calibration ablations,
   topology-cache ablation, and quality-gated reports.
2. CIFAR-100-LT planned generalization baselines for ratios 1:10, 1:50, 1:100.
3. MVTec AD image-level planned generalization baselines for selected categories.
4. MVTec AD normal-only PatchCore-lite reference baseline.
5. Hardware profiling for dense/static-2:4/GUDS structural efficiency.

Recommended Kaggle usage:

    %cd /kaggle/working
    !python experiments/run_kaggle_paper_suite.py --isic_suite all

For a quick smoke test:

    !python experiments/run_kaggle_paper_suite.py --smoke
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO_ROOT) / "paper_experiment_outputs"


@dataclass(frozen=True)
class CommandSpec:
    name: str
    command: list[str]


def run_command(spec: CommandSpec) -> int:
    run_dir = OUTPUT_ROOT / "logs" / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("WANDB_SILENT", "true")
    env.setdefault("PYTHONUNBUFFERED", "1")

    start = time.time()
    print("\n" + "=" * 96)
    print(f"[START] {spec.name}")
    print("Command:", " ".join(spec.command))
    print("Log:", log_path)
    print("=" * 96)

    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            spec.command,
            cwd=str(REPO_ROOT),
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
    code = process.wait()
    elapsed = time.time() - start
    status = "OK" if code == 0 else f"FAILED exit={code}"
    print("-" * 96)
    print(f"[END] {spec.name} | {status} | elapsed={elapsed / 60:.1f} min | log={log_path}")
    print("-" * 96)
    return code


def build_commands(args: argparse.Namespace) -> list[CommandSpec]:
    epochs = 1 if args.smoke else args.epochs
    batch_size = min(args.batch_size, 8) if args.smoke else args.batch_size

    commands: list[CommandSpec] = []
    if not args.skip_isic:
        isic_command = [
            sys.executable,
            str(REPO_ROOT / "experiments" / "isic_paper_experiments.py"),
            "--suite",
            args.isic_suite,
            "--epochs",
            str(epochs),
            "--batch_size",
            str(batch_size),
            "--seeds",
            *[str(seed) for seed in args.seeds],
        ]
        if args.no_save_model:
            isic_command.append("--no_save_model")
        commands.append(
            CommandSpec(
                name=f"isic_{args.isic_suite}",
                command=isic_command,
            )
        )

    if not args.skip_cifar:
        for ratio in args.cifar_ratios:
            commands.append(
                CommandSpec(
                    name=f"cifar100lt_ir{ratio}",
                    command=[
                        sys.executable,
                        str(REPO_ROOT / "experiments" / "generalization_paper_suite.py"),
                        "--benchmark",
                        "cifar",
                        "--ratio",
                        str(ratio),
                        "--epochs",
                        str(args.cifar_epochs if not args.smoke else 1),
                        "--batch_size",
                        str(batch_size),
                        "--seeds",
                        *[str(seed) for seed in args.seeds],
                    ],
                )
            )

    if not args.skip_mvtec:
        for category in args.mvtec_categories:
            commands.append(
                CommandSpec(
                    name=f"mvtec_{category}",
                    command=[
                        sys.executable,
                        str(REPO_ROOT / "experiments" / "generalization_paper_suite.py"),
                        "--benchmark",
                        "mvtec",
                        "--category",
                        category,
                        "--epochs",
                        str(args.mvtec_epochs if not args.smoke else 1),
                        "--batch_size",
                        str(min(batch_size, 16)),
                        "--seeds",
                        *[str(seed) for seed in args.seeds],
                    ],
                )
            )
            if not args.skip_mvtec_reference:
                commands.append(
                    CommandSpec(
                        name=f"mvtec_patchcore_{category}",
                        command=[
                            sys.executable,
                            str(REPO_ROOT / "experiments" / "mvtec_patchcore_reference.py"),
                            "--category",
                            category,
                            "--batch_size",
                            str(min(batch_size, 16)),
                            "--seeds",
                            *[str(seed) for seed in args.seeds],
                        ],
                    )
                )
    if not args.skip_hardware:
        commands.append(
            CommandSpec(
                name="hardware_profile",
                command=[
                    sys.executable,
                    str(REPO_ROOT / "experiments" / "hardware_profile.py"),
                    "--batch_size",
                    str(min(batch_size, 16) if args.smoke else batch_size),
                    "--iters",
                    str(5 if args.smoke else args.hardware_iters),
                    "--warmup",
                    str(2 if args.smoke else args.hardware_warmup),
                ],
            )
        )
    if args.include_backbones:
        commands.append(
            CommandSpec(
                name="backbone_generalization",
                command=[
                    sys.executable,
                    str(REPO_ROOT / "experiments" / "backbone_generalization_runner.py"),
                    "--epochs",
                    str(1 if args.smoke else args.backbone_epochs),
                    "--batch_size",
                    str(min(batch_size, 8)),
                    "--seeds",
                    *[str(seed) for seed in args.seeds],
                ],
            )
        )
    if not args.skip_summary:
        commands.append(
            CommandSpec(
                name="summarize_results",
                command=[
                    sys.executable,
                    str(REPO_ROOT / "experiments" / "summarize_results.py"),
                ],
            )
        )
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all paper experiments on Kaggle.")
    parser.add_argument("--isic_suite", choices=["main_tables", "baselines", "ablations", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=40, help="Epochs for ISIC experiments.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cifar_epochs", type=int, default=100)
    parser.add_argument("--mvtec_epochs", type=int, default=20)
    parser.add_argument("--backbone_epochs", type=int, default=40)
    parser.add_argument("--hardware_iters", type=int, default=50)
    parser.add_argument("--hardware_warmup", type=int, default=10)
    parser.add_argument("--cifar_ratios", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--mvtec_categories", nargs="+", default=["hazelnut", "bottle"])
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--skip_isic", action="store_true")
    parser.add_argument("--skip_cifar", action="store_true")
    parser.add_argument("--skip_mvtec", action="store_true")
    parser.add_argument("--skip_mvtec_reference", action="store_true", help="Skip PatchCore-lite MVTec reference runs.")
    parser.add_argument("--skip_hardware", action="store_true")
    parser.add_argument("--skip_summary", action="store_true")
    parser.add_argument("--include_backbones", action="store_true", help="Also run the heavyweight ConvNeXt/Swin backbone protocol.")
    parser.add_argument("--no_save_model", action="store_true", help="Avoid saving every model checkpoint in large multi-seed sweeps.")
    parser.add_argument("--smoke", action="store_true", help="Run a 1-epoch smoke test with small batches.")
    parser.add_argument("--keep_going", action="store_true", help="Continue after a failed command.")
    args = parser.parse_args()
    if args.seeds is None:
        args.seeds = [42] if args.smoke else [42, 43, 44]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    commands = build_commands(args)
    if not commands:
        raise ValueError("No commands selected.")

    failures: list[tuple[str, int]] = []
    for spec in commands:
        code = run_command(spec)
        if code != 0:
            failures.append((spec.name, code))
            print(f"[ERROR] {spec.name} failed with exit code {code}.")
            if not args.keep_going:
                break

    summary_path = OUTPUT_ROOT / "PAPER_SUITE_SUMMARY.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Paper experiment suite summary\n\n")
        for spec in commands:
            f.write(f"- {spec.name}: {' '.join(spec.command)}\n")
        if failures:
            f.write("\nFailures:\n")
            for name, code in failures:
                f.write(f"- {name}: exit code {code}\n")
        else:
            f.write("\nFailures: none\n")

    print(f"\nSuite summary: {summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

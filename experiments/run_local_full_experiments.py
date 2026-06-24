"""
Local full experiment launcher for the GUDS-EDL paper suite.

This runner is the local-machine counterpart of run_kaggle_paper_suite.py. It
does not require Kaggle paths. Point it to local datasets with ISIC_ROOT and
MVTEC_ROOT, or pass --isic_root / --mvtec_root.

Typical smoke test:

    python experiments/run_local_full_experiments.py --smoke --isic_root D:\\datasets\\isic-2024-challenge

Typical full run:

    python experiments/run_local_full_experiments.py ^
      --isic_root D:\\datasets\\isic-2024-challenge ^
      --mvtec_root D:\\datasets\\mvtec_anomaly_detection ^
      --isic_suite all ^
      --seeds 42 43 44 ^
      --no_save_model ^
      --keep_going

The terminal prints compact progress. Full stdout/stderr for every sub-run is
saved under paper_experiment_outputs/local_logs/<timestamp>/.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "paper_experiment_outputs"
KNOWN_MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


@dataclass(frozen=True)
class CommandSpec:
    name: str
    command: list[str]


def has_file_tree(root: Path, filename: str, max_depth: int = 3) -> bool:
    if not root.exists():
        return False
    root_depth = len(root.parts)
    for path in root.rglob(filename):
        if len(path.parts) - root_depth <= max_depth:
            return True
    return False


def detect_isic_root(path_arg: str | None) -> Path | None:
    candidates: list[Path] = []
    for value in [path_arg, os.environ.get("ISIC_ROOT"), "data/isic-2024-challenge", "data/isic2024"]:
        if value:
            candidates.append(Path(value).expanduser().resolve())
    for candidate in candidates:
        if (candidate / "train-metadata.csv").exists():
            return candidate
        if has_file_tree(candidate, "train-metadata.csv"):
            return candidate
    return None


def detect_mvtec_root(path_arg: str | None) -> Path | None:
    candidates: list[Path] = []
    for value in [path_arg, os.environ.get("MVTEC_ROOT"), "data/mvtec_ad", "data/mvtec"]:
        if value:
            candidates.append(Path(value).expanduser().resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def detect_mvtec_categories(root: Path | None) -> list[str]:
    if root is None or not root.exists():
        return []
    found = []
    for category in KNOWN_MVTEC_CATEGORIES:
        for path in [root / category, *root.rglob(category)]:
            if path.is_dir() and (path / "train").exists() and (path / "test").exists():
                found.append(category)
                break
    return sorted(set(found))


def compact_line(line: str) -> bool:
    markers = [
        "[START]",
        "[END]",
        "[RUN]",
        "[TRAIN]",
        "[CAL]",
        "[DONE]",
        "[ERROR]",
        "[WARN]",
        "Traceback",
        "RuntimeError",
        "ValueError",
        "FileNotFoundError",
        "Completed ",
        "Saved summary",
        "Saved hardware profile",
        "All selected",
    ]
    return any(marker in line for marker in markers)


def run_command(spec: CommandSpec, env: dict[str, str], log_dir: Path, stream_mode: str) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{spec.name}.log"
    start = time.time()
    print("\n" + "=" * 100)
    print(f"[START] {spec.name}")
    print("Command:", " ".join(spec.command))
    print("Log:", log_path)
    print("=" * 100)

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
            log_file.write(line)
            if stream_mode == "full" or (stream_mode == "compact" and compact_line(line)):
                print(line, end="")
    code = process.wait()
    elapsed = time.time() - start
    status = "OK" if code == 0 else f"FAILED exit={code}"
    print("-" * 100)
    print(f"[END] {spec.name} | {status} | elapsed={elapsed / 60:.1f} min")
    print("-" * 100)
    return code


def python_cmd(args: list[str]) -> list[str]:
    return [sys.executable, *args]


def build_commands(args: argparse.Namespace, mvtec_categories: list[str]) -> list[CommandSpec]:
    isic_epochs = 1 if args.smoke else args.epochs
    cifar_epochs = 1 if args.smoke else args.cifar_epochs
    mvtec_epochs = 1 if args.smoke else args.mvtec_epochs
    batch_size = min(args.batch_size, 8) if args.smoke else args.batch_size
    seeds = [42] if args.smoke else args.seeds
    commands: list[CommandSpec] = []

    if not args.skip_isic:
        cmd = python_cmd([
            "experiments/isic_paper_experiments.py",
            "--suite",
            args.isic_suite,
            "--epochs",
            str(isic_epochs),
            "--batch_size",
            str(batch_size),
            "--seeds",
            *map(str, seeds),
            "--log_every",
            str(args.log_every),
        ])
        if args.no_save_model:
            cmd.append("--no_save_model")
        if args.allow_dummy_data:
            cmd.append("--allow_dummy_data")
        if args.cpu:
            cmd.append("--cpu")
        if args.verbose_structural_logs:
            cmd.append("--verbose_structural_logs")
        commands.append(CommandSpec(f"isic_{args.isic_suite}", cmd))

    if not args.skip_cifar:
        for ratio in args.cifar_ratios:
            cmd = python_cmd([
                "experiments/generalization_paper_suite.py",
                "--benchmark",
                "cifar",
                "--ratio",
                str(ratio),
                "--epochs",
                str(cifar_epochs),
                "--batch_size",
                str(batch_size),
                "--seeds",
                *map(str, seeds),
                "--log_every",
                str(args.log_every),
            ])
            if args.allow_dummy_data:
                cmd.append("--allow_dummy_data")
            if args.cpu:
                cmd.append("--cpu")
            if args.verbose_structural_logs:
                cmd.append("--verbose_structural_logs")
            commands.append(CommandSpec(f"cifar100lt_ir{ratio}", cmd))

    if not args.skip_mvtec:
        for category in mvtec_categories:
            cmd = python_cmd([
                "experiments/generalization_paper_suite.py",
                "--benchmark",
                "mvtec",
                "--category",
                category,
                "--epochs",
                str(mvtec_epochs),
                "--batch_size",
                str(min(batch_size, 16)),
                "--seeds",
                *map(str, seeds),
                "--log_every",
                str(args.log_every),
            ])
            if args.allow_dummy_data:
                cmd.append("--allow_dummy_data")
            if args.cpu:
                cmd.append("--cpu")
            if args.verbose_structural_logs:
                cmd.append("--verbose_structural_logs")
            commands.append(CommandSpec(f"mvtec_{category}", cmd))

            if not args.skip_mvtec_reference:
                ref_cmd = python_cmd([
                    "experiments/mvtec_patchcore_reference.py",
                    "--category",
                    category,
                    "--batch_size",
                    str(min(batch_size, 16)),
                    "--seeds",
                    *map(str, seeds),
                ])
                if args.cpu:
                    ref_cmd.append("--cpu")
                commands.append(CommandSpec(f"mvtec_patchcore_{category}", ref_cmd))

    if not args.skip_hardware:
        cmd = python_cmd([
            "experiments/hardware_profile.py",
            "--batch_size",
            str(min(batch_size, 16)),
            "--iters",
            str(5 if args.smoke else args.hardware_iters),
            "--warmup",
            str(2 if args.smoke else args.hardware_warmup),
        ])
        if args.cpu:
            cmd.append("--cpu")
        commands.append(CommandSpec("hardware_profile", cmd))

    if args.include_backbones:
        cmd = python_cmd([
            "experiments/backbone_generalization_runner.py",
            "--epochs",
            str(1 if args.smoke else args.backbone_epochs),
            "--batch_size",
            str(min(batch_size, 8)),
            "--seeds",
            *map(str, seeds),
        ])
        if args.no_save_model:
            pass
        if args.allow_dummy_data:
            cmd.append("--allow_dummy_data")
        if args.cpu:
            cmd.append("--cpu")
        commands.append(CommandSpec("backbone_generalization", cmd))

    if not args.skip_summary:
        commands.append(CommandSpec("summarize_results", python_cmd([
            "experiments/summarize_results.py",
            "--root",
            str(OUTPUT_ROOT),
        ])))
    return commands


def print_environment_summary(args: argparse.Namespace, isic_root: Path | None, mvtec_root: Path | None, categories: list[str]) -> None:
    print("\nLocal Experiment Configuration")
    print("-" * 80)
    print(f"Repo root      : {REPO_ROOT}")
    print(f"Python         : {sys.executable}")
    print(f"Output root    : {OUTPUT_ROOT}")
    print(f"ISIC_ROOT      : {isic_root if isic_root else 'not found'}")
    print(f"MVTEC_ROOT     : {mvtec_root if mvtec_root else 'not found'}")
    print(f"MVTec cats     : {categories if categories else 'none'}")
    print(f"Seeds          : {[42] if args.smoke else args.seeds}")
    print(f"Mode           : {'smoke' if args.smoke else 'full'}")
    print(f"Stream mode    : {args.stream}")
    print("-" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full GUDS-EDL paper suite locally.")
    parser.add_argument("--isic_root", type=str, help="Local ISIC 2024 root containing train-metadata.csv.")
    parser.add_argument("--mvtec_root", type=str, help="Local MVTec AD root containing category folders.")
    parser.add_argument("--isic_suite", choices=["main_tables", "baselines", "ablations", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=40, help="Epochs for ISIC experiments.")
    parser.add_argument("--cifar_epochs", type=int, default=100)
    parser.add_argument("--mvtec_epochs", type=int, default=20)
    parser.add_argument("--backbone_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--cifar_ratios", type=int, nargs="+", default=[10, 50, 100], choices=[10, 50, 100])
    parser.add_argument("--mvtec_categories", nargs="+", help="MVTec categories to run; defaults to auto-detected categories.")
    parser.add_argument("--hardware_iters", type=int, default=50)
    parser.add_argument("--hardware_warmup", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--stream", choices=["compact", "full", "none"], default="compact")
    parser.add_argument("--skip_isic", action="store_true")
    parser.add_argument("--skip_cifar", action="store_true")
    parser.add_argument("--skip_mvtec", action="store_true")
    parser.add_argument("--skip_mvtec_reference", action="store_true")
    parser.add_argument("--skip_hardware", action="store_true")
    parser.add_argument("--skip_summary", action="store_true")
    parser.add_argument("--include_backbones", action="store_true")
    parser.add_argument("--no_save_model", action="store_true", help="Do not save model checkpoints for large sweeps.")
    parser.add_argument("--allow_dummy_data", action="store_true", help="Only for dry-runs; never use for paper results.")
    parser.add_argument("--smoke", action="store_true", help="Run 1 epoch and seed 42 only.")
    parser.add_argument("--keep_going", action="store_true", help="Continue after failed sub-runs.")
    parser.add_argument("--verbose_structural_logs", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if shutil.which(sys.executable) is None and not Path(sys.executable).exists():
        print(f"[WARN] Could not verify Python executable: {sys.executable}")

    isic_root = detect_isic_root(args.isic_root)
    mvtec_root = detect_mvtec_root(args.mvtec_root)
    categories = args.mvtec_categories or detect_mvtec_categories(mvtec_root)

    if isic_root is not None:
        os.environ["ISIC_ROOT"] = str(isic_root)
    elif not args.skip_isic and not args.allow_dummy_data:
        print("[WARN] ISIC dataset not found. Pass --isic_root or set ISIC_ROOT, or use --skip_isic.")

    if mvtec_root is not None:
        os.environ["MVTEC_ROOT"] = str(mvtec_root)
    elif not args.skip_mvtec and not args.allow_dummy_data:
        print("[WARN] MVTec root not found. Pass --mvtec_root or set MVTEC_ROOT, or use --skip_mvtec.")

    if not categories and not args.skip_mvtec:
        if args.allow_dummy_data:
            categories = ["hazelnut"]
        else:
            print("[WARN] No MVTec categories detected; MVTec suite will be skipped.")
            args.skip_mvtec = True

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = OUTPUT_ROOT / "local_logs" / stamp
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("WANDB_SILENT", "true")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    print_environment_summary(args, isic_root, mvtec_root, categories)
    commands = build_commands(args, categories)
    print("Planned sub-runs:")
    for idx, spec in enumerate(commands, start=1):
        print(f"  {idx:02d}. {spec.name}")

    failed: list[tuple[str, int]] = []
    for spec in commands:
        code = run_command(spec, env=env, log_dir=log_dir, stream_mode=args.stream)
        if code != 0:
            failed.append((spec.name, code))
            if not args.keep_going:
                break

    print("\nLocal suite finished.")
    print(f"Logs   : {log_dir}")
    print(f"Output : {OUTPUT_ROOT}")
    if failed:
        print("Failed sub-runs:")
        for name, code in failed:
            print(f"  - {name}: exit {code}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

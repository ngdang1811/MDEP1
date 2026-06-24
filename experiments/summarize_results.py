"""
Aggregate paper_experiment_outputs metrics into a compact CSV table.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def run_key(path: Path, record: dict) -> tuple[str, str, str]:
    benchmark = record.get("benchmark")
    run_name = record.get("run_name")
    if benchmark is None:
        parts = path.parts
        benchmark = "isic" if "isic" in parts else ("hardware" if "hardware" in parts else "unknown")
    if run_name is None:
        if benchmark == "isic":
            run_name = "isic2024"
        elif "backbones" in path.parts:
            run_name = record.get("backbone", "unknown_backbone")
        else:
            run_name = "default"
    experiment = record.get("experiment", {})
    if isinstance(experiment, dict):
        experiment_name = experiment.get("name", record.get("mode", "unknown"))
    else:
        experiment_name = str(experiment)
    return str(benchmark), str(run_name), str(experiment_name)


def load_rows(root: Path):
    grouped = defaultdict(lambda: defaultdict(list))
    for metrics_path in root.rglob("metrics.json"):
        try:
            record = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] Could not read {metrics_path}: {exc}")
            continue
        key = run_key(metrics_path, record)
        metrics = record.get("metrics", {})
        quality_gate = record.get("quality_gate", {})
        combined = {}
        if isinstance(metrics, dict):
            combined.update(metrics)
        if isinstance(quality_gate, dict):
            combined.update(quality_gate)
        for metric_name, value in combined.items():
            if is_number(value):
                grouped[key][metric_name].append(float(value))
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize paper experiment metrics across seeds.")
    parser.add_argument("--root", type=Path, default=Path("/kaggle/working/paper_experiment_outputs"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root
    if not root.exists():
        root = Path("paper_experiment_outputs")
    output = args.output or (root / "summary_metrics.csv")

    grouped = load_rows(root)
    rows = []
    for (benchmark, run_name, experiment), metrics in sorted(grouped.items()):
        for metric_name, values in sorted(metrics.items()):
            arr = np.asarray(values, dtype=float)
            rows.append({
                "benchmark": benchmark,
                "run_name": run_name,
                "experiment": experiment,
                "metric": metric_name,
                "n": len(values),
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(values) > 1 else 0.0,
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["benchmark", "run_name", "experiment", "metric", "n", "mean", "std"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary CSV: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

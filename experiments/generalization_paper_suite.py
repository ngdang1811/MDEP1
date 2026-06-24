"""
Run the planned non-ISIC generalization protocols from main_text.tex.

The ISIC case study has its own richer runner. This file covers the planned
CIFAR-100-LT and MVTec AD protocols with the baseline families named in the
paper: CE, Focal Loss, Logit Adjustment, Dense EDL, Static 2:4 EDL,
RigL-style 2:4, and full GUDS-EDL.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from guds_edl_core import (  # noqa: E402
    EvidenceLayer,
    compute_adaptive_ece,
    compute_aurc,
    compute_ece,
    compute_uncertainties,
    evaluate,
    print_sparsity_report,
    replace_conv2d_with_mdep,
)
from experiments.cifar_lt_runner import get_cifar100_lt_dataloaders  # noqa: E402
from experiments.isic_paper_experiments import (  # noqa: E402
    EXPERIMENTS,
    collect_softmax_outputs,
    json_safe,
    prior_logit_delta,
    run_calibration,
    run_softmax_calibration,
    seed_everything,
    train_guds,
    train_standard,
    uses_softmax_evaluation,
)
from experiments.metrics_ext import (  # noqa: E402
    binary_extended_metrics,
    collect_evidential_outputs,
    multiclass_extended_metrics,
    uncertainty_separation_metrics,
)
from experiments.mvtec_ad_runner import get_mvtec_ad_classification_dataloaders  # noqa: E402


PLANNED_EXPERIMENTS = [
    "standard_ce",
    "focal_loss",
    "logit_adjustment",
    "class_balanced_ce",
    "balanced_softmax",
    "ldam_drw",
    "decoupled_crt",
    "dense_edl",
    "static_24_edl",
    "rigl_style_24",
    "full_guds",
]


class EvidenceResNet(nn.Module):
    def __init__(self, num_classes: int, dataset: str, pretrained: bool):
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
            except Exception:
                weights = None
        try:
            self.backbone = models.resnet18(weights=weights)
        except Exception as exc:
            print(f"[WARN] Could not load pretrained ResNet-18 weights ({exc}); using random init.")
            self.backbone = models.resnet18(weights=None)

        if dataset == "cifar":
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.backbone.maxpool = nn.Identity()

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fc = nn.Sequential(nn.Linear(in_features, num_classes), EvidenceLayer(activation="softplus"))
        nn.init.normal_(self.fc[0].weight, mean=0.0, std=0.001)
        nn.init.constant_(self.fc[0].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.backbone(x))


def output_root(benchmark: str) -> Path:
    root = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO_ROOT
    return root / "paper_experiment_outputs" / benchmark


@torch.no_grad()
def collect_multiclass_logits(model: nn.Module, loader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_list = []
    labels_list = []
    linear = model.fc[0]
    for inputs, targets in loader:
        inputs = inputs.to(device)
        logits_list.append(linear(model.backbone(inputs)).detach())
        labels_list.append(targets.to(device))
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def calibrate_multiclass(
    model: nn.Module,
    cal_loader,
    device: torch.device,
    mode: str,
    p_true: list[float],
    p_train: list[float],
    probability_family: str = "evidential",
) -> tuple[float, torch.Tensor | None, dict[str, float]]:
    logits, labels = collect_multiclass_logits(model, cal_loader, device)
    prior_delta = prior_logit_delta(
        p_true,
        p_train,
        logits.shape[1],
        device=logits.device,
        dtype=logits.dtype,
    )
    logits_for_calibration = logits + prior_delta
    evidence_layer = model.fc[1]

    def calibration_loss(scaled_logits: torch.Tensor) -> torch.Tensor:
        if probability_family == "softmax":
            return F.cross_entropy(scaled_logits, labels)
        evidence = evidence_layer(scaled_logits)
        unc = compute_uncertainties(evidence)
        probs = (unc["alpha"] / unc["S"]).clamp_min(1e-8)
        return F.nll_loss(torch.log(probs), labels)

    if mode == "none":
        return 1.0, None, {}
    if mode == "temperature_only":
        temp_param = nn.Parameter(torch.ones(1, device=device) * 1.5)
        optimizer = torch.optim.LBFGS([temp_param], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            model.zero_grad(set_to_none=True)
            loss = calibration_loss(logits_for_calibration / temp_param.clamp_min(0.1))
            loss.backward()
            return loss

        optimizer.step(closure)
        return max(0.1, float(temp_param.detach().item())), None, {}
    if mode == "bias_temperature":
        temp_param = nn.Parameter(torch.ones(1, device=device) * 1.5)
        bias_param = nn.Parameter(torch.zeros(logits.shape[1], device=device))
        optimizer = torch.optim.LBFGS([temp_param, bias_param], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            model.zero_grad(set_to_none=True)
            loss = calibration_loss(logits_for_calibration / temp_param.clamp_min(0.1) + bias_param)
            loss.backward()
            return loss

        optimizer.step(closure)
        return max(0.1, float(temp_param.detach().item())), bias_param.detach(), {}
    raise ValueError(f"Unknown calibration mode: {mode}")


def cifar_class_counts(imbalance_ratio: int) -> list[int]:
    return [
        max(1, int(500 * (1.0 / imbalance_ratio) ** (cls_idx / 99.0)))
        for cls_idx in range(100)
    ]


@torch.no_grad()
def evaluate_multiclass(
    model: nn.Module,
    test_loader,
    device: torch.device,
    num_classes: int,
    temperature: float,
    bias: torch.Tensor | None,
    class_counts: list[int] | None = None,
    probability_family: str = "evidential",
) -> dict[str, float]:
    model.eval()
    linear = model.fc[0]
    evidence_layer = model.fc[1]
    targets_all = []
    probs_all = []
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        logits = linear(model.backbone(inputs)) / temperature
        if bias is not None:
            logits = logits + bias
        if probability_family == "softmax":
            probs = F.softmax(logits, dim=1)
        else:
            evidence = evidence_layer(logits)
            unc = compute_uncertainties(evidence)
            probs = unc["alpha"] / unc["S"]
        targets_all.append(targets.numpy())
        probs_all.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(targets_all)
    probs = np.concatenate(probs_all, axis=0)
    y_pred = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    correct = (y_pred == y_true).astype(float)
    one_hot = np.eye(num_classes)[y_true]

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "aurc": float(compute_aurc(y_true, y_pred, confidences)),
    }
    ece_adaptive, _, _, _ = compute_adaptive_ece(confidences, correct)
    ece_eq_width, _, _, _ = compute_ece(confidences, correct)
    metrics["ece_adaptive"] = float(ece_adaptive)
    metrics["ece_eq_width"] = float(ece_eq_width)

    try:
        metrics["macro_auroc"] = float(roc_auc_score(one_hot, probs, average="macro", multi_class="ovr"))
    except ValueError as exc:
        print(f"[WARN] macro_auroc unavailable: {exc}")
        metrics["macro_auroc"] = float("nan")
    try:
        metrics["macro_pr_auc"] = float(average_precision_score(one_hot, probs, average="macro"))
    except ValueError as exc:
        print(f"[WARN] macro_pr_auc unavailable: {exc}")
        metrics["macro_pr_auc"] = float("nan")

    if class_counts is not None:
        class_counts_arr = np.asarray(class_counts)
        few_shot_classes = np.where(class_counts_arr <= 20)[0]
        if len(few_shot_classes):
            per_class_acc = []
            for cls in few_shot_classes:
                mask = y_true == cls
                if mask.any():
                    per_class_acc.append(float((y_pred[mask] == y_true[mask]).mean()))
            metrics["few_shot_accuracy"] = float(np.mean(per_class_acc)) if per_class_acc else float("nan")
            metrics["few_shot_class_count"] = int(len(few_shot_classes))
    metrics.update(multiclass_extended_metrics(y_true, probs, class_counts=class_counts))
    return metrics


def make_loaders(benchmark: str, args: argparse.Namespace, seed: int):
    if benchmark == "cifar":
        return get_cifar100_lt_dataloaders(args.ratio, args.batch_size, seed=seed)
    if benchmark == "mvtec":
        return get_mvtec_ad_classification_dataloaders(
            args.category,
            args.batch_size,
            seed=seed,
            allow_dummy_data=args.allow_dummy_data,
        )
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def run_one(benchmark: str, experiment_name: str, args: argparse.Namespace, seed: int) -> dict:
    spec = EXPERIMENTS[experiment_name]
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loaders = make_loaders(benchmark, args, seed)
    train_loader, val_loader, cal_loader, test_loader, class_weights, p_true, p_train = loaders
    num_classes = 100 if benchmark == "cifar" else 2
    class_counts = cifar_class_counts(args.ratio) if benchmark == "cifar" else None
    softmax_eval = uses_softmax_evaluation(spec)

    pretrained = benchmark != "cifar" and not args.no_pretrained
    model = EvidenceResNet(num_classes=num_classes, dataset=benchmark, pretrained=pretrained)
    if spec.sparse:
        replace_conv2d_with_mdep(model)
    model = model.to(device)

    lr = args.lr
    if lr is None:
        lr = 1e-3 if benchmark == "cifar" else 1e-4

    run_name = f"ir{args.ratio}" if benchmark == "cifar" else args.category
    run_dir = output_root(benchmark) / run_name / experiment_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[RUN] benchmark={benchmark} case={run_name} experiment={experiment_name} seed={seed} epochs={args.epochs} output={run_dir}")

    if spec.use_mdep_trainer:
        history = train_guds(
            model,
            train_loader,
            device,
            spec,
            class_weights,
            args.epochs,
            lr,
            log_every=args.log_every,
            verbose_structural_logs=args.verbose_structural_logs,
        )
    else:
        history = train_standard(
            model,
            train_loader,
            device,
            spec,
            class_weights,
            p_true,
            p_train,
            args.epochs,
            lr,
            log_every=args.log_every,
        )

    if benchmark == "cifar":
        temperature, bias, thresholds = calibrate_multiclass(
            model,
            cal_loader,
            device,
            spec.calibration_mode,
            p_true,
            p_train,
            probability_family="softmax" if softmax_eval else "evidential",
        )
    else:
        if softmax_eval:
            temperature, bias, thresholds, _ = run_softmax_calibration(
                model,
                cal_loader,
                val_loader,
                device,
                spec.calibration_mode,
                p_true,
                p_train,
            )
        else:
            temperature, bias, thresholds = run_calibration(
                model,
                cal_loader,
                val_loader,
                device,
                spec.calibration_mode,
                p_true,
                p_train,
            )
    prior_delta = prior_logit_delta(
        p_true,
        p_train,
        num_classes,
        device=device,
        dtype=torch.float32,
    )
    eval_bias = prior_delta / max(temperature, 1e-8)
    if bias is not None:
        eval_bias = eval_bias + bias.to(device=device, dtype=eval_bias.dtype)
    if hasattr(model.fc[1], "logit_adjustment"):
        model.fc[1].logit_adjustment = torch.zeros(1, dtype=torch.float32, device=device)

    if benchmark == "cifar":
        metrics = evaluate_multiclass(
            model,
            test_loader,
            device,
            num_classes=num_classes,
            temperature=temperature,
            bias=eval_bias,
            class_counts=class_counts,
            probability_family="softmax" if softmax_eval else "evidential",
        )
    else:
        if softmax_eval:
            outputs = collect_softmax_outputs(model, test_loader, device, temperature=temperature, bias=eval_bias)
            y_true = outputs["y_true"]
            probs = outputs["probs"]
            y_pred = (probs[:, 1] >= 0.5).astype(int)
            confidences = probs.max(axis=1)
            correct = (y_pred == y_true).astype(float)
            ece_adaptive, _, _, _ = compute_adaptive_ece(confidences, correct)
            ece_eq_width, _, _, _ = compute_ece(confidences, correct)
            metrics = {
                "macro_auroc": float(roc_auc_score(y_true, probs[:, 1])),
                "pr_auc": float(average_precision_score(y_true, probs[:, 1])),
                "ece_adaptive": float(ece_adaptive),
                "ece_eq_width": float(ece_eq_width),
            }
        else:
            _, metrics = evaluate(
                model,
                val_loader,
                test_loader,
                device,
                num_classes=num_classes,
                temperature=temperature,
                bias=eval_bias,
                plot=False,
            )
            outputs = collect_evidential_outputs(model, test_loader, device, temperature=temperature, bias=eval_bias)
        metrics.update(binary_extended_metrics(
            outputs["y_true"],
            outputs["probs"],
            thresholds={
                "balanced": float(thresholds.get("balanced", 0.5)),
                "high_recall": float(thresholds.get("high_recall", thresholds.get("rule_out", 0.5))),
            },
        ))
        metrics.update(uncertainty_separation_metrics(
            outputs["y_true"],
            outputs["y_pred"],
            outputs["u_e"],
            outputs["u_a"],
        ))

    if spec.sparse:
        print_sparsity_report(model)

    result = {
        "benchmark": benchmark,
        "run_name": run_name,
        "experiment": asdict(spec),
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "temperature": temperature,
        "bias": bias,
        "evaluation_bias": eval_bias,
        "prior_delta": prior_delta,
        "thresholds": thresholds,
        "p_true": p_true,
        "p_train": p_train,
        "history": history,
        "metrics": metrics,
        "evaluator": "softmax" if softmax_eval else "evidential",
    }
    (run_dir / "metrics.json").write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")
    if args.save_model:
        torch.save(model.state_dict(), run_dir / "model_state.pth")
    return result


def selected_experiments(args: argparse.Namespace) -> list[str]:
    if args.experiment:
        return args.experiment
    return PLANNED_EXPERIMENTS


def main() -> int:
    parser = argparse.ArgumentParser(description="Run planned CIFAR-100-LT and MVTec AD paper protocols.")
    parser.add_argument("--benchmark", choices=["cifar", "mvtec"], required=True)
    parser.add_argument("--experiment", action="append", choices=PLANNED_EXPERIMENTS)
    parser.add_argument("--ratio", type=int, default=100, choices=[10, 50, 100])
    parser.add_argument("--category", type=str, default="hazelnut")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--allow_dummy_data", action="store_true", help="Permit synthetic dummy data for dry-runs only.")
    parser.add_argument("--log_every", type=int, default=5, help="Print training progress every N epochs.")
    parser.add_argument("--verbose_structural_logs", action="store_true", help="Print detailed per-layer structural update diagnostics.")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.benchmark == "mvtec" and args.batch_size == 128:
        args.batch_size = 32
    if args.benchmark == "mvtec" and args.epochs == 100:
        args.epochs = 20

    all_results = []
    seeds = args.seeds if args.seeds else [args.seed]
    for seed in seeds:
        for experiment_name in selected_experiments(args):
            all_results.append(run_one(args.benchmark, experiment_name, args, seed))

    suffix = f"ir{args.ratio}" if args.benchmark == "cifar" else args.category
    summary_path = output_root(args.benchmark) / f"{suffix}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(json_safe(all_results), indent=2), encoding="utf-8")
    print(f"\nCompleted {args.benchmark} planned protocol. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

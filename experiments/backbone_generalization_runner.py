"""
Optional additional-backbone protocol for GUDS-EDL.

The main paper focuses on ResNet-18. This runner makes the planned backbone
extension executable for ResNet-18, ConvNeXt-Tiny, and Swin-Tiny on the ISIC
case-study split.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from guds_edl_core import (  # noqa: E402
    AdaptiveThresholdDecisionSupport,
    EvidenceLayer,
    evaluate,
    get_imbalanced_dataloaders,
    print_sparsity_report,
    replace_conv2d_with_mdep,
)
from experiments.generalization_paper_suite import EvidenceResNet  # noqa: E402
from experiments.isic_paper_experiments import (  # noqa: E402
    EXPERIMENTS,
    json_safe,
    prior_logit_delta,
    quality_gate_report,
    run_calibration,
    seed_everything,
    train_guds,
)


class EvidenceTorchvisionBackbone(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, pretrained: bool):
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name == "resnet18":
            wrapped = EvidenceResNet(num_classes=num_classes, dataset="mvtec", pretrained=pretrained)
            self.backbone = wrapped.backbone
            self.fc = wrapped.fc
            return

        if backbone_name == "convnext_tiny":
            weights = None
            if pretrained:
                try:
                    weights = models.ConvNeXt_Tiny_Weights.DEFAULT
                except Exception:
                    weights = None
            try:
                self.backbone = models.convnext_tiny(weights=weights)
            except Exception as exc:
                print(f"[WARN] Could not load ConvNeXt-T weights ({exc}); using random init.")
                self.backbone = models.convnext_tiny(weights=None)
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Identity()
        elif backbone_name == "swin_t":
            weights = None
            if pretrained:
                try:
                    weights = models.Swin_T_Weights.DEFAULT
                except Exception:
                    weights = None
            try:
                self.backbone = models.swin_t(weights=weights)
            except Exception as exc:
                print(f"[WARN] Could not load Swin-T weights ({exc}); using random init.")
                self.backbone = models.swin_t(weights=None)
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.fc = nn.Sequential(nn.Linear(in_features, num_classes), EvidenceLayer(activation="softplus"))
        nn.init.normal_(self.fc[0].weight, mean=0.0, std=0.001)
        nn.init.constant_(self.fc[0].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.backbone(x))


def output_root() -> Path:
    root = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO_ROOT
    return root / "paper_experiment_outputs" / "backbones"


def run_one(backbone: str, args: argparse.Namespace, seed: int) -> dict:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loaders = get_imbalanced_dataloaders(
        batch_size=args.batch_size,
        test_ratio=args.test_ratio,
        subsample_ratio=args.subsample_ratio,
        seed=seed,
        allow_dummy_data=args.allow_dummy_data,
    )
    train_loader, val_loader, cal_loader, test_loader, num_classes, class_weights, p_true, p_train = loaders
    spec = EXPERIMENTS["full_guds"]
    model = EvidenceTorchvisionBackbone(backbone, num_classes=num_classes, pretrained=not args.no_pretrained)
    replace_conv2d_with_mdep(model)
    model = model.to(device)

    run_dir = output_root() / backbone / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 90}\nBackbone protocol | {backbone} | seed={seed}\nOutput: {run_dir}\n{'=' * 90}")

    history = train_guds(model, train_loader, device, spec, class_weights, args.epochs, args.lr)
    temperature, bias, thresholds = run_calibration(
        model,
        cal_loader,
        val_loader,
        device,
        spec.calibration_mode,
        p_true,
        p_train,
    )
    decision_support = AdaptiveThresholdDecisionSupport(
        model,
        is_resnet=True,
        thresholds=thresholds,
        temperature=temperature,
        bias=bias,
        true_class_prior=p_true,
        train_class_prior=p_train,
    )
    quality_metrics = quality_gate_report(decision_support, test_loader, device)
    decision_support.restore_model()

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

    _, metrics = evaluate(model, val_loader, test_loader, device, num_classes, temperature=temperature, bias=eval_bias, plot=False)
    print_sparsity_report(model)

    result = {
        "backbone": backbone,
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
        "quality_gate": quality_metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")
    if args.save_model:
        torch.save(model.state_dict(), run_dir / "model_state.pth")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run optional ISIC additional-backbone GUDS-EDL experiments.")
    parser.add_argument("--backbones", nargs="+", default=["resnet18", "convnext_tiny", "swin_t"], choices=["resnet18", "convnext_tiny", "swin_t"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--test_ratio", type=float, default=0.20)
    parser.add_argument("--subsample_ratio", type=int, default=20)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--allow_dummy_data", action="store_true", help="Permit synthetic dummy data for dry-runs only.")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    output_root().mkdir(parents=True, exist_ok=True)
    all_results = []
    seeds = args.seeds if args.seeds else [args.seed]
    for seed in seeds:
        for backbone in args.backbones:
            all_results.append(run_one(backbone, args, seed))

    summary_path = output_root() / "backbone_summary.json"
    summary_path.write_text(json.dumps(json_safe(all_results), indent=2), encoding="utf-8")
    print(f"Completed backbone protocol. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

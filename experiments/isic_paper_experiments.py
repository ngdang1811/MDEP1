"""
Train/evaluate ISIC 2024 experiments referenced by main_text.tex.

This runner is designed for Kaggle after the repo has been copied to
/kaggle/working. It reuses the dataset split, calibration, and metrics from
guds_edl_core.py, then adds paper-facing baseline variants that can be trained
from the same command-line surface.

Examples:

    python experiments/isic_paper_experiments.py --experiment full_guds
    python experiments/isic_paper_experiments.py --suite main_tables
    python experiments/isic_paper_experiments.py --suite all

Outputs:

    /kaggle/working/paper_experiment_outputs/isic/<experiment_name>/
        run_config.json
        metrics.json
        model_state.pth
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from guds_edl_core import (  # noqa: E402
    AdaptiveThresholdDecisionSupport,
    EvidenceLayer,
    EvidentialFocalLoss,
    MDEPTrainer,
    MDEPConv2d,
    MDEPLinear,
    compute_adaptive_ece,
    compute_aurc,
    compute_class_conditional_ece,
    compute_ece,
    compute_isic_pauc,
    compute_patient_level_se_top15,
    compute_uncertainties,
    evaluate,
    evaluate_adaptive_modes,
    get_imbalanced_dataloaders,
    print_sparsity_report,
    replace_conv2d_with_mdep,
)
from experiments.metrics_ext import (  # noqa: E402
    binary_extended_metrics,
    collect_evidential_outputs,
    uncertainty_separation_metrics,
)


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    family: str
    description: str
    sparse: bool = False
    static_sparse: bool = False
    use_mdep_trainer: bool = False
    loss_name: str = "edl"
    pruner_type: str = "signed_first_order"
    regrower_type: str = "class_conditioned"
    disable_pruner: bool = False
    disable_regrower: bool = False
    kl_scaling: str = "asymmetric"
    disable_efl: bool = False
    disable_anticryst: bool = False
    logit_adjustment_train: bool = False
    disable_topology_cache: bool = False
    calibration_mode: str = "bias_temperature"
    classifier_retrain: bool = False


EXPERIMENTS: dict[str, ExperimentSpec] = {
    # Main result rows in main_text.tex Tables 1--2.
    "standard_ce": ExperimentSpec(
        name="standard_ce",
        family="long_tailed_baseline",
        description="Dense ResNet-18 trained with standard cross-entropy.",
        loss_name="ce",
    ),
    "focal_loss": ExperimentSpec(
        name="focal_loss",
        family="long_tailed_baseline",
        description="Dense ResNet-18 trained with focal loss.",
        loss_name="focal",
    ),
    "logit_adjustment": ExperimentSpec(
        name="logit_adjustment",
        family="long_tailed_baseline",
        description="Dense ResNet-18 trained with logit-adjusted cross-entropy.",
        loss_name="ce",
        logit_adjustment_train=True,
    ),
    "class_balanced_ce": ExperimentSpec(
        name="class_balanced_ce",
        family="long_tailed_baseline",
        description="Class-Balanced Loss baseline using effective-number reweighting.",
        loss_name="class_balanced_ce",
    ),
    "balanced_softmax": ExperimentSpec(
        name="balanced_softmax",
        family="long_tailed_baseline",
        description="Balanced Softmax baseline using train-prior logits inside the CE objective.",
        loss_name="balanced_softmax",
    ),
    "ldam_drw": ExperimentSpec(
        name="ldam_drw",
        family="long_tailed_baseline",
        description="LDAM with deferred effective-number reweighting.",
        loss_name="ldam_drw",
    ),
    "decoupled_crt": ExperimentSpec(
        name="decoupled_crt",
        family="long_tailed_baseline",
        description="cRT-style baseline: dense CE representation learning followed by classifier retraining.",
        loss_name="ce",
        classifier_retrain=True,
    ),
    "dense_edl": ExperimentSpec(
        name="dense_edl",
        family="evidential_baseline",
        description="Dense EDL baseline with symmetric KL.",
        loss_name="edl",
        kl_scaling="symmetric",
        disable_efl=True,
    ),
    "fisher_edl": ExperimentSpec(
        name="fisher_edl",
        family="evidential_baseline",
        description="Dense EDL with an additional Fisher-information penalty.",
        loss_name="fisher_edl",
        kl_scaling="symmetric",
        disable_efl=True,
    ),
    "flexible_edl": ExperimentSpec(
        name="flexible_edl",
        family="evidential_baseline",
        description="Dense EDL with a learnable positive evidence scale.",
        loss_name="edl",
        kl_scaling="symmetric",
        disable_efl=True,
    ),
    "r_edl": ExperimentSpec(
        name="r_edl",
        family="evidential_baseline",
        description="Relaxed EDL proxy: reduced KL pressure and no focal modulation.",
        loss_name="r_edl",
        kl_scaling="symmetric",
        disable_efl=True,
    ),
    "static_24_edl": ExperimentSpec(
        name="static_24_edl",
        family="dynamic_sparse_baseline",
        description="Static 2:4 sparse EDL with fixed magnitude-derived masks.",
        sparse=True,
        static_sparse=True,
        loss_name="edl",
        kl_scaling="symmetric",
        disable_efl=True,
    ),
    "rigl_style_24": ExperimentSpec(
        name="rigl_style_24",
        family="dynamic_sparse_baseline",
        description="RigL-style 2:4 proxy using absolute-gradient pruning and gradient regrowth.",
        sparse=True,
        use_mdep_trainer=True,
        loss_name="edl",
        pruner_type="absolute_grad",
        regrower_type="gradient",
        kl_scaling="symmetric",
        disable_efl=True,
    ),
    "full_guds": ExperimentSpec(
        name="full_guds",
        family="proposed",
        description="Full GUDS-EDL with signed pruner, class-conditioned regrower, EFL, and asymmetric KL.",
        sparse=True,
        use_mdep_trainer=True,
        loss_name="edl",
        pruner_type="signed_first_order",
        regrower_type="class_conditioned",
        kl_scaling="asymmetric",
    ),
    # Appendix C ablations.
    "guds_without_pruner": ExperimentSpec(
        name="guds_without_pruner",
        family="ablation",
        description="GUDS-EDL without uncertainty-guided pruning.",
        sparse=True,
        use_mdep_trainer=True,
        disable_pruner=True,
    ),
    "guds_without_regrower": ExperimentSpec(
        name="guds_without_regrower",
        family="ablation",
        description="GUDS-EDL without evidence-seeking regrowth.",
        sparse=True,
        use_mdep_trainer=True,
        disable_regrower=True,
    ),
    "guds_symmetric_kl": ExperimentSpec(
        name="guds_symmetric_kl",
        family="ablation",
        description="GUDS-EDL with symmetric KL instead of asymmetric KL.",
        sparse=True,
        use_mdep_trainer=True,
        kl_scaling="symmetric",
    ),
    "guds_without_efl": ExperimentSpec(
        name="guds_without_efl",
        family="ablation",
        description="GUDS-EDL without Evidential Focal Loss modulation.",
        sparse=True,
        use_mdep_trainer=True,
        disable_efl=True,
    ),
    "guds_without_anticryst": ExperimentSpec(
        name="guds_without_anticryst",
        family="ablation",
        description="GUDS-EDL without anti-crystallization noise.",
        sparse=True,
        use_mdep_trainer=True,
        disable_anticryst=True,
    ),
    "guds_absolute_pruner": ExperimentSpec(
        name="guds_absolute_pruner",
        family="ablation",
        description="GUDS-EDL with absolute-gradient pruning instead of signed pruning.",
        sparse=True,
        use_mdep_trainer=True,
        pruner_type="absolute_grad",
    ),
    "guds_kl_uniform_regrower": ExperimentSpec(
        name="guds_kl_uniform_regrower",
        family="ablation",
        description="GUDS-EDL with KL-to-uniform regrowth instead of class-conditioned regrowth.",
        sparse=True,
        use_mdep_trainer=True,
        regrower_type="kl_uniform",
    ),
    "guds_without_topology_cache": ExperimentSpec(
        name="guds_without_topology_cache",
        family="ablation",
        description="GUDS-EDL without amortized topology caching; structural signals are recomputed per batch.",
        sparse=True,
        use_mdep_trainer=True,
        disable_topology_cache=True,
    ),
    "guds_temperature_only": ExperimentSpec(
        name="guds_temperature_only",
        family="ablation",
        description="GUDS-EDL calibrated with scalar temperature only, without bias correction.",
        sparse=True,
        use_mdep_trainer=True,
        calibration_mode="temperature_only",
    ),
    "guds_no_posthoc_calibration": ExperimentSpec(
        name="guds_no_posthoc_calibration",
        family="ablation",
        description="GUDS-EDL evaluated without post-hoc temperature or bias calibration.",
        sparse=True,
        use_mdep_trainer=True,
        calibration_mode="none",
    ),
}


SUITES: dict[str, list[str]] = {
    "main_tables": [
        "fisher_edl",
        "flexible_edl",
        "r_edl",
        "full_guds",
    ],
    "baselines": [
        "standard_ce",
        "focal_loss",
        "logit_adjustment",
        "class_balanced_ce",
        "balanced_softmax",
        "ldam_drw",
        "decoupled_crt",
        "dense_edl",
        "fisher_edl",
        "flexible_edl",
        "r_edl",
        "static_24_edl",
        "rigl_style_24",
    ],
    "ablations": [
        "full_guds",
        "guds_without_pruner",
        "guds_without_regrower",
        "guds_symmetric_kl",
        "guds_without_efl",
        "guds_without_anticryst",
        "guds_absolute_pruner",
        "guds_kl_uniform_regrower",
        "guds_without_topology_cache",
        "guds_temperature_only",
        "guds_no_posthoc_calibration",
    ],
}
SUITES["all"] = list(dict.fromkeys(SUITES["baselines"] + SUITES["ablations"]))


DISCRIMINATIVE_LOSS_NAMES = {"ce", "focal", "class_balanced_ce", "balanced_softmax", "ldam_drw"}


def uses_softmax_evaluation(spec: ExperimentSpec) -> bool:
    return spec.loss_name in DISCRIMINATIVE_LOSS_NAMES or spec.classifier_retrain


class FlexibleEvidenceLayer(EvidenceLayer):
    """Softplus evidence with a learnable positive logit scale."""

    def __init__(self, max_evidence: float = 20.0):
        super().__init__(activation="softplus", max_evidence=max_evidence)
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x * torch.exp(self.log_scale))


class ResNetEvidenceModel(nn.Module):
    """ResNet-18 backbone with a paper-compatible `fc=[linear,evidence]` head."""

    def __init__(self, num_classes: int, flexible: bool = False, pretrained: bool = True):
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
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        evidence = FlexibleEvidenceLayer() if flexible else EvidenceLayer(activation="softplus")
        self.fc = nn.Sequential(nn.Linear(in_features, num_classes), evidence)
        nn.init.normal_(self.fc[0].weight, mean=0.0, std=0.001)
        nn.init.constant_(self.fc[0].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.backbone(x))


class FisherEDLLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, fisher_lambda: float = 1e-3):
        super().__init__()
        self.base_loss = base_loss
        self.fisher_lambda = fisher_lambda

    def forward(self, evidence: torch.Tensor, targets: torch.Tensor, epoch: int | None = None) -> torch.Tensor:
        base = self.base_loss(evidence, targets, epoch=epoch)
        alpha = evidence + 1.0
        strength = alpha.sum(dim=1, keepdim=True)
        fisher = torch.mean(torch.clamp(torch.polygamma(1, alpha) - torch.polygamma(1, strength), min=0.0))
        return base + self.fisher_lambda * fisher


class RelaxedEDLLoss(nn.Module):
    def __init__(self, num_classes: int, class_weights: torch.Tensor | None, total_epochs: int):
        super().__init__()
        self.loss = EvidentialFocalLoss(
            gamma=0.0,
            num_classes=num_classes,
            kl_lambda=0.01,
            class_weights=class_weights,
            warmup_epochs=max(1, int(0.30 * total_epochs)),
            total_epochs=total_epochs,
            disable_efl=True,
            kl_scaling="symmetric",
        )

    def forward(self, evidence: torch.Tensor, targets: torch.Tensor, epoch: int | None = None) -> torch.Tensor:
        return self.loss(evidence, targets, epoch=epoch)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def output_root() -> Path:
    root = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO_ROOT
    return root / "paper_experiment_outputs" / "isic"


def set_static_sparse_mode(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (MDEPLinear, MDEPConv2d)):
            module.warmup = False
            module.gamma = 0.15
            module.static_24_baseline = True


def model_head(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    head = model.fc if hasattr(model, "fc") else model.head
    return head[0], head[1]


def prior_logit_delta(
    p_true: list[float],
    p_train: list[float],
    num_classes: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    values = [math.log(p_true[c] + 1e-8) - math.log(p_train[c] + 1e-8) for c in range(num_classes)]
    return torch.tensor(values, dtype=dtype, device=device)


@torch.no_grad()
def collect_logits_labels(model: nn.Module, loader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    linear, _ = model_head(model)
    logits_list = []
    labels_list = []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        if hasattr(model, "backbone"):
            features = model.backbone(inputs)
        else:
            original_head = model.fc if hasattr(model, "fc") else model.head
            if hasattr(model, "fc"):
                model.fc = nn.Identity()
                features = model(inputs)
                model.fc = original_head
            else:
                model.head = nn.Identity()
                features = model(inputs)
                model.head = original_head
        logits_list.append(linear(features).detach())
        labels_list.append(targets.to(device))
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def optimize_thresholds(
    model: nn.Module,
    val_loader,
    device: torch.device,
    temperature: float,
    bias: torch.Tensor | None,
    prior_delta: torch.Tensor | None = None,
) -> dict[str, float]:
    linear, evidence_layer = model_head(model)
    logits, labels = collect_logits_labels(model, val_loader, device)
    with torch.no_grad():
        if prior_delta is not None:
            logits = logits + prior_delta.to(device=logits.device, dtype=logits.dtype)
        scaled_logits = logits / temperature
        if bias is not None:
            scaled_logits = scaled_logits + bias
        evidence = evidence_layer(scaled_logits)
        unc = compute_uncertainties(evidence)
        probs = (unc["alpha"] / unc["S"]).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()

    best_t_bal_acc = 0.5
    best_bal_acc = 0.0
    best_t_clinical = 0.5
    best_spec_at_sens80 = 0.0
    found_sens80 = False
    for threshold in np.linspace(0.01, 0.99, 199):
        y_pred = (probs[:, 1] >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        bal_acc = 0.5 * (sens + spec)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_t_bal_acc = float(threshold)
        if sens >= 0.80 and (spec > best_spec_at_sens80 or not found_sens80):
            best_spec_at_sens80 = spec
            best_t_clinical = float(threshold)
            found_sens80 = True

    return {
        "rule_out": best_t_clinical,
        "high_recall": best_t_clinical,
        "double_read": best_t_clinical,
        "balanced": best_t_bal_acc,
        "rule_in": 0.5,
    }


def run_calibration(
    model: nn.Module,
    cal_loader,
    val_loader,
    device: torch.device,
    mode: str,
    p_true: list[float],
    p_train: list[float],
) -> tuple[float, torch.Tensor | None, dict[str, float]]:
    linear, evidence_layer = model_head(model)
    logits, labels = collect_logits_labels(model, cal_loader, device)
    prior_delta = prior_logit_delta(
        p_true,
        p_train,
        linear.out_features,
        device=logits.device,
        dtype=logits.dtype,
    )
    logits_for_calibration = logits + prior_delta

    def evidential_nll(scaled_logits: torch.Tensor) -> torch.Tensor:
        evidence = evidence_layer(scaled_logits)
        unc = compute_uncertainties(evidence)
        probs = (unc["alpha"] / unc["S"]).clamp_min(1e-8)
        return F.nll_loss(torch.log(probs), labels)

    if mode == "none":
        temperature = 1.0
        bias = None
    elif mode == "temperature_only":
        temp_param = nn.Parameter(torch.ones(1, device=device) * 1.5)
        optimizer = optim.LBFGS([temp_param], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            model.zero_grad(set_to_none=True)
            loss = evidential_nll(logits_for_calibration / temp_param.clamp_min(0.1))
            loss.backward()
            return loss

        optimizer.step(closure)
        temperature = max(0.1, float(temp_param.detach().item()))
        bias = None
    elif mode == "bias_temperature":
        temp_param = nn.Parameter(torch.ones(1, device=device) * 1.5)
        bias_param = nn.Parameter(torch.zeros(linear.out_features, device=device))
        optimizer = optim.LBFGS([temp_param, bias_param], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            model.zero_grad(set_to_none=True)
            loss = evidential_nll(logits_for_calibration / temp_param.clamp_min(0.1) + bias_param)
            loss.backward()
            return loss

        optimizer.step(closure)
        temperature = max(0.1, float(temp_param.detach().item()))
        bias = bias_param.detach()
    else:
        raise ValueError(f"Unknown calibration_mode: {mode}")

    thresholds = optimize_thresholds(model, val_loader, device, temperature, bias, prior_delta=prior_delta)
    print(
        f"[CAL] mode={mode} | T={temperature:.4f} | "
        f"prior_delta={prior_delta.detach().cpu().numpy()} | "
        f"bias={None if bias is None else bias.detach().cpu().numpy()} | thresholds={thresholds}"
    )
    return temperature, bias, thresholds


@torch.no_grad()
def collect_softmax_outputs(
    model: nn.Module,
    loader,
    device: torch.device,
    temperature: float,
    bias: torch.Tensor | None,
) -> dict[str, np.ndarray]:
    logits, labels = collect_logits_labels(model, loader, device)
    scaled_logits = logits / temperature
    if bias is not None:
        scaled_logits = scaled_logits + bias.to(device=scaled_logits.device, dtype=scaled_logits.dtype)
    probs = F.softmax(scaled_logits, dim=1).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy().astype(int)
    return {
        "y_true": y_true,
        "probs": probs,
        "y_pred": probs.argmax(axis=1),
        "confidences": probs.max(axis=1),
    }


def thresholds_from_probabilities(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    best_t_bal_acc = 0.5
    best_bal_acc = 0.0
    best_t_clinical = 0.5
    best_spec_at_sens80 = 0.0
    found_sens80 = False
    for threshold in np.linspace(0.01, 0.99, 199):
        y_pred = (probs[:, 1] >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        bal_acc = 0.5 * (sens + spec)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_t_bal_acc = float(threshold)
        if sens >= 0.80 and (spec > best_spec_at_sens80 or not found_sens80):
            best_spec_at_sens80 = spec
            best_t_clinical = float(threshold)
            found_sens80 = True
    return {
        "rule_out": best_t_clinical,
        "high_recall": best_t_clinical,
        "double_read": best_t_clinical,
        "balanced": best_t_bal_acc,
        "rule_in": 0.5,
    }


def run_softmax_calibration(
    model: nn.Module,
    cal_loader,
    val_loader,
    device: torch.device,
    mode: str,
    p_true: list[float],
    p_train: list[float],
) -> tuple[float, torch.Tensor | None, dict[str, float], torch.Tensor]:
    logits, labels = collect_logits_labels(model, cal_loader, device)
    prior_delta = prior_logit_delta(p_true, p_train, logits.shape[1], device=logits.device, dtype=logits.dtype)
    logits_for_calibration = logits + prior_delta

    if mode == "none":
        temperature = 1.0
        bias = None
    elif mode == "temperature_only":
        temp_param = nn.Parameter(torch.ones(1, device=device) * 1.5)
        optimizer = optim.LBFGS([temp_param], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits_for_calibration / temp_param.clamp_min(0.1), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        temperature = max(0.1, float(temp_param.detach().item()))
        bias = None
    elif mode == "bias_temperature":
        temp_param = nn.Parameter(torch.ones(1, device=device) * 1.5)
        bias_param = nn.Parameter(torch.zeros(logits.shape[1], device=device))
        optimizer = optim.LBFGS([temp_param, bias_param], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits_for_calibration / temp_param.clamp_min(0.1) + bias_param, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        temperature = max(0.1, float(temp_param.detach().item()))
        bias = bias_param.detach()
    else:
        raise ValueError(f"Unknown calibration_mode: {mode}")

    eval_bias = prior_delta / max(temperature, 1e-8)
    if bias is not None:
        eval_bias = eval_bias + bias.to(device=device, dtype=eval_bias.dtype)
    val_outputs = collect_softmax_outputs(model, val_loader, device, temperature, eval_bias)
    thresholds = thresholds_from_probabilities(val_outputs["y_true"], val_outputs["probs"])
    print(f"[CAL] mode={mode} | evaluator=softmax | T={temperature:.4f} | thresholds={thresholds}")
    return temperature, bias, thresholds, prior_delta


def test_frame_from_loader(test_loader) -> pd.DataFrame:
    if isinstance(test_loader.dataset, torch.utils.data.Subset) and hasattr(test_loader.dataset.dataset, "data_frame"):
        return test_loader.dataset.dataset.data_frame.iloc[test_loader.dataset.indices]
    if hasattr(test_loader.dataset, "data_frame"):
        return test_loader.dataset.data_frame
    return pd.DataFrame({
        "target": [test_loader.dataset[i][1].item() for i in range(len(test_loader.dataset))],
        "patient_id": [f"patient_{i // 5}" for i in range(len(test_loader.dataset))],
    })


def evaluate_softmax_baseline(
    model: nn.Module,
    test_loader,
    device: torch.device,
    temperature: float,
    eval_bias: torch.Tensor,
    thresholds: dict[str, float],
    deployment_prevalence: float,
) -> dict[str, float]:
    outputs = collect_softmax_outputs(model, test_loader, device, temperature, eval_bias)
    y_true = outputs["y_true"]
    probs = outputs["probs"]
    y_pred = (probs[:, 1] >= 0.5).astype(int)
    confidences = probs.max(axis=1)
    correct = (y_pred == y_true).astype(float)
    ece_adaptive, _, _, _ = compute_adaptive_ece(confidences, correct)
    ece_eq_width, _, _, _ = compute_ece(confidences, correct)
    class_eces = compute_class_conditional_ece(probs, y_true)
    threshold_report = {
        "balanced": float(thresholds.get("balanced", 0.5)),
        "high_recall": float(thresholds.get("high_recall", thresholds.get("rule_out", 0.5))),
    }
    metrics = {
        "pauc": compute_isic_pauc(y_true, probs[:, 1], min_tpr=0.80),
        "se_top15": compute_patient_level_se_top15(test_frame_from_loader(test_loader), probs),
        "pr_auc": average_precision_score(y_true, probs[:, 1]),
        "macro_auroc": roc_auc_score(y_true, probs[:, 1]),
        "aurc": compute_aurc(y_true, y_pred, confidences),
        "ece_adaptive": float(ece_adaptive),
        "ece_eq_width": float(ece_eq_width),
        "class_ece_0": float(class_eces[0]),
        "class_ece_1": float(class_eces[1]),
        "balanced_accuracy_default": float(balanced_accuracy_score(y_true, y_pred)),
    }
    metrics.update(binary_extended_metrics(
        y_true,
        probs,
        thresholds=threshold_report,
        deployment_prevalence=deployment_prevalence,
    ))
    return metrics


@torch.no_grad()
def quality_gate_report(decision_support: AdaptiveThresholdDecisionSupport, test_loader, device: torch.device) -> dict[str, float]:
    targets_all = []
    decisions_all = []
    ua_all = []
    for inputs, targets in test_loader:
        final_decision, _, _, u_a = decision_support(inputs.to(device), mode="balanced", quality_gated=True)
        targets_all.append(targets.numpy())
        decisions_all.append(final_decision.cpu().numpy())
        ua_all.append(u_a.squeeze(-1).cpu().numpy())

    y_true = np.concatenate(targets_all)
    decisions = np.concatenate(decisions_all)
    u_a = np.concatenate(ua_all)
    accepted = decisions != 3
    discarded = decisions == 3
    report = {
        "quality_gate_accepted_coverage": float(accepted.mean()) if len(accepted) else 0.0,
        "quality_gate_discard_rate": float(discarded.mean()) if len(discarded) else 0.0,
        "quality_gate_mean_ua_accepted": float(u_a[accepted].mean()) if accepted.any() else 0.0,
        "quality_gate_mean_ua_discarded": float(u_a[discarded].mean()) if discarded.any() else 0.0,
    }
    if accepted.any():
        valid_pred = decisions[accepted]
        valid_true = y_true[accepted]
        tn, fp, fn, tp = confusion_matrix(valid_true, valid_pred, labels=[0, 1]).ravel()
        report.update({
            "quality_gate_sensitivity": float(tp / (tp + fn + 1e-8)),
            "quality_gate_specificity": float(tn / (tn + fp + 1e-8)),
            "quality_gate_error_rate": float((valid_pred != valid_true).mean()),
        })
    return report


def make_loss(spec: ExperimentSpec, num_classes: int, class_weights: torch.Tensor, total_epochs: int, device: torch.device) -> nn.Module:
    if spec.loss_name == "r_edl":
        return RelaxedEDLLoss(num_classes, class_weights.to(device), total_epochs)

    base = EvidentialFocalLoss(
        gamma=1.2,
        num_classes=num_classes,
        kl_lambda=0.1,
        class_weights=class_weights.to(device),
        warmup_epochs=max(1, int(0.30 * total_epochs)),
        total_epochs=total_epochs,
        disable_efl=spec.disable_efl,
        kl_scaling=spec.kl_scaling,
    )
    if spec.loss_name == "fisher_edl":
        return FisherEDLLoss(base)
    return base


def logits_from_model(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    return model.fc[0](model.backbone(inputs))


def effective_number_weights(p_train: list[float], beta: float = 0.9999, device: torch.device | None = None) -> torch.Tensor:
    """Class-Balanced Loss weights from relative training frequencies."""
    counts = torch.tensor(p_train, dtype=torch.float32, device=device)
    counts = counts / counts[counts > 0].min().clamp_min(1e-8)
    weights = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta, device=counts.device), counts).clamp_max(1.0 - 1e-8))
    weights = weights / weights.mean().clamp_min(1e-8)
    return weights


def ldam_margins(p_train: list[float], max_margin: float = 0.5, device: torch.device | None = None) -> torch.Tensor:
    counts = torch.tensor(p_train, dtype=torch.float32, device=device)
    counts = counts / counts[counts > 0].min().clamp_min(1e-8)
    margins = 1.0 / torch.sqrt(torch.sqrt(counts.clamp_min(1.0)))
    margins = margins * (max_margin / margins.max().clamp_min(1e-8))
    return margins


def ce_or_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    spec: ExperimentSpec,
    class_weights: torch.Tensor,
    p_true: list[float],
    p_train: list[float],
    epoch: int,
    total_epochs: int,
) -> torch.Tensor:
    adjusted = logits
    if spec.logit_adjustment_train:
        delta = torch.tensor(
            [math.log(p_true[c] + 1e-8) - math.log(p_train[c] + 1e-8) for c in range(logits.shape[1])],
            dtype=logits.dtype,
            device=logits.device,
        )
        adjusted = logits + delta

    if spec.loss_name == "ce":
        return F.cross_entropy(adjusted, targets, weight=class_weights.to(logits.device))

    if spec.loss_name == "class_balanced_ce":
        cb_weights = effective_number_weights(p_train, device=logits.device)
        return F.cross_entropy(adjusted, targets, weight=cb_weights)

    if spec.loss_name == "balanced_softmax":
        log_prior = torch.tensor(
            [math.log(max(p, 1e-8)) for p in p_train],
            dtype=logits.dtype,
            device=logits.device,
        )
        return F.cross_entropy(adjusted + log_prior, targets)

    if spec.loss_name == "ldam_drw":
        margins = ldam_margins(p_train, device=logits.device)
        one_hot = F.one_hot(targets, num_classes=logits.shape[1]).to(logits.dtype)
        logits_m = adjusted - one_hot * margins.unsqueeze(0)
        weight = None
        if epoch >= int(0.75 * total_epochs):
            weight = effective_number_weights(p_train, device=logits.device)
        return F.cross_entropy(30.0 * logits_m, targets, weight=weight)

    probs = F.softmax(adjusted, dim=1)
    pt = probs.gather(1, targets.view(-1, 1)).clamp_min(1e-8)
    ce = F.cross_entropy(adjusted, targets, weight=class_weights.to(logits.device), reduction="none").view(-1, 1)
    return torch.mean(((1.0 - pt) ** 2.0) * ce)


def retrain_classifier_crt(
    model: nn.Module,
    train_loader,
    device: torch.device,
    class_weights: torch.Tensor,
    total_epochs: int,
    lr: float,
) -> list[dict[str, float]]:
    """Classifier re-training baseline inspired by cRT/decoupled classifiers."""
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.fc[0].parameters():
        param.requires_grad = True

    epochs = max(1, int(0.10 * total_epochs))
    optimizer = optim.AdamW(model.fc[0].parameters(), lr=lr, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    weights = class_weights.to(device)
    for epoch in range(epochs):
        model.train()
        losses = []
        start = time.time()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = logits_from_model(model, inputs)
            loss = F.cross_entropy(logits, targets, weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.fc[0].parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        avg_loss = float(np.mean(losses)) if losses else 0.0
        elapsed = time.time() - start
        history.append({"epoch": epoch + 1, "loss": avg_loss, "seconds": elapsed, "stage": "crt"})
        print(f"cRT [{epoch + 1:>2}/{epochs}] | loss={avg_loss:.4f} | {elapsed:.1f}s")

    for param in model.backbone.parameters():
        param.requires_grad = True
    return history


def train_standard(
    model: nn.Module,
    train_loader,
    device: torch.device,
    spec: ExperimentSpec,
    class_weights: torch.Tensor,
    p_true: list[float],
    p_train: list[float],
    total_epochs: int,
    lr: float,
    log_every: int = 5,
) -> list[dict[str, float]]:
    params = [p for name, p in model.named_parameters() if "scores" not in name]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    history: list[dict[str, float]] = []

    if spec.static_sparse:
        set_static_sparse_mode(model)

    criterion = None
    if spec.loss_name not in DISCRIMINATIVE_LOSS_NAMES:
        criterion = make_loss(spec, model.fc[0].out_features, class_weights, total_epochs, device)

    for epoch in range(total_epochs):
        model.train()
        if spec.static_sparse:
            set_static_sparse_mode(model)
        losses = []
        start = time.time()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                if spec.loss_name in DISCRIMINATIVE_LOSS_NAMES:
                    logits = logits_from_model(model, inputs)
                    loss = ce_or_focal_loss(logits, targets, spec, class_weights, p_true, p_train, epoch, total_epochs)
                else:
                    evidence = model(inputs)
                    loss = criterion(evidence, targets, epoch=epoch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        avg_loss = float(np.mean(losses)) if losses else 0.0
        elapsed = time.time() - start
        history.append({"epoch": epoch + 1, "loss": avg_loss, "seconds": elapsed})
        if epoch == 0 or (epoch + 1) % max(log_every, 1) == 0 or (epoch + 1) == total_epochs:
            print(f"[TRAIN] epoch={epoch + 1:03d}/{total_epochs:03d} loss={avg_loss:.4f} time={elapsed:.1f}s")
    if spec.classifier_retrain:
        history.extend(retrain_classifier_crt(model, train_loader, device, class_weights, total_epochs, lr))
    return history


def train_guds(
    model: nn.Module,
    train_loader,
    device: torch.device,
    spec: ExperimentSpec,
    class_weights: torch.Tensor,
    total_epochs: int,
    lr: float,
    log_every: int = 5,
    verbose_structural_logs: bool = False,
) -> list[dict[str, float]]:
    warmup_epochs = max(1, int(0.30 * total_epochs))
    criterion = make_loss(spec, model.fc[0].out_features, class_weights, total_epochs, device)
    params = [p for name, p in model.named_parameters() if "scores" not in name]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
    trainer_args = SimpleNamespace(
        disable_pruner=spec.disable_pruner,
        disable_regrower=spec.disable_regrower,
        pruner_type=spec.pruner_type,
        regrower_type=spec.regrower_type,
        kl_scaling=spec.kl_scaling,
        disable_efl=spec.disable_efl,
        disable_anticryst=spec.disable_anticryst,
        use_anticryst=not spec.disable_anticryst,
        disable_topology_cache=spec.disable_topology_cache,
        verbose_structural_logs=verbose_structural_logs,
    )
    trainer = MDEPTrainer(model, optimizer, criterion, total_epochs, warmup_epochs, args=trainer_args)
    history = []
    for epoch in range(total_epochs):
        loss = trainer.train_epoch(epoch, train_loader, device, print_interval=200)
        history.append({"epoch": epoch + 1, "loss": float(loss), "gamma": float(trainer.step_gamma(epoch))})
        if epoch == 0 or (epoch + 1) % max(log_every, 1) == 0 or (epoch + 1) == total_epochs:
            print(f"[TRAIN] epoch={epoch + 1:03d}/{total_epochs:03d} loss={loss:.4f} gamma={trainer.step_gamma(epoch):.4f}")
    return history


def run_one(spec: ExperimentSpec, args: argparse.Namespace, seed: int) -> dict:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    run_dir = output_root() / spec.name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"\n[RUN] dataset=isic experiment={spec.name} family={spec.family} "
        f"evaluator={'softmax' if uses_softmax_evaluation(spec) else 'evidential'} "
        f"seed={seed} epochs={args.epochs} device={device} output={run_dir}"
    )

    loaders = get_imbalanced_dataloaders(
        batch_size=args.batch_size,
        test_ratio=args.test_ratio,
        subsample_ratio=args.subsample_ratio,
        seed=seed,
        allow_dummy_data=args.allow_dummy_data,
    )
    train_loader, val_loader, cal_loader, test_loader, num_classes, class_weights, p_true, p_train = loaders

    model = ResNetEvidenceModel(
        num_classes=num_classes,
        flexible=(spec.name == "flexible_edl"),
        pretrained=not args.no_pretrained,
    )
    if spec.sparse:
        replace_conv2d_with_mdep(model)
    model = model.to(device)

    if spec.use_mdep_trainer:
        history = train_guds(
            model,
            train_loader,
            device,
            spec,
            class_weights,
            args.epochs,
            args.lr,
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
            args.lr,
            log_every=args.log_every,
        )

    quality_metrics = {}
    if uses_softmax_evaluation(spec):
        temperature, bias, thresholds, prior_delta = run_softmax_calibration(
            model,
            cal_loader,
            val_loader,
            device,
            spec.calibration_mode,
            p_true,
            p_train,
        )
        eval_bias = prior_delta / max(temperature, 1e-8)
        if bias is not None:
            eval_bias = eval_bias + bias.to(device=device, dtype=eval_bias.dtype)
        if hasattr(model.fc[1], "logit_adjustment"):
            model.fc[1].logit_adjustment = torch.zeros(1, dtype=torch.float32, device=device)
        metrics = evaluate_softmax_baseline(
            model,
            test_loader,
            device,
            temperature,
            eval_bias,
            thresholds,
            args.deployment_prevalence,
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
        decision_support = AdaptiveThresholdDecisionSupport(
            model,
            is_resnet=True,
            thresholds=thresholds,
            temperature=temperature,
            bias=bias,
            true_class_prior=p_true,
            train_class_prior=p_train,
        )
        evaluate_adaptive_modes(decision_support, test_loader, device)
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
        threshold_report = {
            "balanced": float(thresholds.get("balanced", 0.5)),
            "high_recall": float(thresholds.get("high_recall", thresholds.get("rule_out", 0.5))),
        }
        metrics.update(binary_extended_metrics(
            outputs["y_true"],
            outputs["probs"],
            thresholds=threshold_report,
            deployment_prevalence=args.deployment_prevalence,
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
        "experiment": asdict(spec),
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "temperature": temperature,
        "bias": bias,
        "calibration_bias": bias,
        "evaluation_bias": eval_bias,
        "prior_delta": prior_delta,
        "thresholds": thresholds,
        "p_true": p_true,
        "p_train": p_train,
        "history": history,
        "metrics": metrics,
        "quality_gate": quality_metrics,
        "evaluator": "softmax" if uses_softmax_evaluation(spec) else "evidential",
    }
    (run_dir / "run_config.json").write_text(json.dumps(json_safe(asdict(spec)), indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")
    if not args.no_save_model:
        torch.save(model.state_dict(), run_dir / "model_state.pth")
    print(f"[DONE] Saved metrics and model state to {run_dir}")
    return result


def selected_experiments(args: argparse.Namespace) -> list[ExperimentSpec]:
    if args.experiment:
        return [EXPERIMENTS[name] for name in args.experiment]
    names = SUITES[args.suite]
    return [EXPERIMENTS[name] for name in names]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ISIC 2024 paper experiments.")
    parser.add_argument("--experiment", action="append", choices=sorted(EXPERIMENTS), help="Run one experiment; can be repeated.")
    parser.add_argument("--suite", choices=sorted(SUITES), default="main_tables")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", help="Run all selected experiments for these seeds.")
    parser.add_argument("--test_ratio", type=float, default=0.20)
    parser.add_argument("--subsample_ratio", type=int, default=20)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--no_save_model", action="store_true")
    parser.add_argument("--allow_dummy_data", action="store_true", help="Permit synthetic dummy data for dry-runs only.")
    parser.add_argument("--deployment_prevalence", type=float, default=0.0015, help="Prevalence used for PPV/NPV/NNB reporting.")
    parser.add_argument("--log_every", type=int, default=5, help="Print training progress every N epochs.")
    parser.add_argument("--verbose_structural_logs", action="store_true", help="Print detailed per-layer structural update diagnostics.")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    output_root().mkdir(parents=True, exist_ok=True)
    all_results = []
    seeds = args.seeds if args.seeds else [args.seed]
    for seed in seeds:
        for spec in selected_experiments(args):
            all_results.append(run_one(spec, args, seed))

    summary_path = output_root() / "isic_summary.json"
    summary_path.write_text(json.dumps(json_safe(all_results), indent=2), encoding="utf-8")
    print(f"\nAll selected ISIC experiments completed. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

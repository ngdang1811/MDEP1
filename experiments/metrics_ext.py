"""
Paper-facing evaluation metrics used by the experiment runners.

These helpers intentionally live under experiments/ so the core training file
does not become coupled to every benchmark-specific reporting convention.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    top_k_accuracy_score,
)

from guds_edl_core import compute_uncertainties


def _safe_float(value: float) -> float:
    try:
        if np.isfinite(value):
            return float(value)
    except Exception:
        pass
    return float("nan")


def _safe_metric(fn, default: float = float("nan")) -> float:
    try:
        return _safe_float(fn())
    except Exception:
        return default


def _integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = getattr(np, "trapz")
    return float(integrate(y, x))


def _ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    total = max(len(confidences), 1)
    ece = 0.0
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidences > left) & (confidences <= right)
        if not mask.any():
            continue
        ece += (mask.sum() / total) * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def _one_hot(y_true: np.ndarray, num_classes: int) -> np.ndarray:
    eye = np.eye(num_classes, dtype=float)
    return eye[y_true.astype(int)]


@torch.no_grad()
def collect_evidential_outputs(
    model: nn.Module,
    loader,
    device: torch.device,
    temperature: float = 1.0,
    bias: torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    """Collect calibrated probabilities and evidential uncertainties."""
    model.eval()
    is_wrapped = hasattr(model, "backbone") and hasattr(model, "fc") and isinstance(model.fc, nn.Sequential)

    if is_wrapped:
        linear = model.fc[0]
        evidence_layer = model.fc[1]
    else:
        head = model.fc if hasattr(model, "fc") else model.head
        linear = head[0]
        evidence_layer = head[1]
        if hasattr(model, "fc"):
            original_head = model.fc
            model.fc = nn.Identity()
        else:
            original_head = model.head
            model.head = nn.Identity()

    targets_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    ue_all: list[np.ndarray] = []
    ua_all: list[np.ndarray] = []

    try:
        for inputs, targets in loader:
            inputs = inputs.to(device)
            if is_wrapped:
                features = model.backbone(inputs)
            else:
                features = model(inputs)
            logits = linear(features) / temperature
            if bias is not None:
                logits = logits + bias
            evidence = evidence_layer(logits)
            unc = compute_uncertainties(evidence)
            probs = unc["alpha"] / unc["S"]
            targets_all.append(targets.detach().cpu().numpy())
            probs_all.append(probs.detach().cpu().numpy())
            ue_all.append(unc["epistemic"].detach().cpu().numpy().reshape(-1))
            ua_all.append(unc["aleatoric"].detach().cpu().numpy().reshape(-1))
    finally:
        if not is_wrapped:
            if hasattr(model, "fc"):
                model.fc = original_head
            else:
                model.head = original_head

    y_true = np.concatenate(targets_all).astype(int)
    probs = np.concatenate(probs_all, axis=0)
    y_pred = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    return {
        "y_true": y_true,
        "probs": probs,
        "y_pred": y_pred,
        "confidences": confidences,
        "u_e": np.concatenate(ue_all),
        "u_a": np.concatenate(ua_all),
    }


def failure_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    errors = (y_true != y_pred).astype(int)
    uncertainty_score = 1.0 - confidences
    metrics: dict[str, float] = {
        f"{prefix}error_rate": float(errors.mean()) if len(errors) else float("nan"),
    }
    if len(np.unique(errors)) == 2:
        metrics[f"{prefix}failure_auroc"] = _safe_metric(lambda: roc_auc_score(errors, uncertainty_score))
        metrics[f"{prefix}failure_aupr"] = _safe_metric(lambda: average_precision_score(errors, uncertainty_score))
    else:
        metrics[f"{prefix}failure_auroc"] = float("nan")
        metrics[f"{prefix}failure_aupr"] = float("nan")

    order = np.argsort(-confidences)
    sorted_errors = errors[order]
    for coverage in (0.80, 0.90, 0.95):
        k = max(1, int(math.ceil(coverage * len(sorted_errors))))
        metrics[f"{prefix}risk_at_{int(coverage * 100)}pct_coverage"] = float(sorted_errors[:k].mean())

    cumulative_errors = np.cumsum(sorted_errors)
    counts = np.arange(1, len(sorted_errors) + 1)
    risks = cumulative_errors / counts
    for target_risk in (0.01, 0.05, 0.10):
        valid = np.where(risks <= target_risk)[0]
        coverage = float((valid.max() + 1) / len(sorted_errors)) if len(valid) else 0.0
        metrics[f"{prefix}coverage_at_{int(target_risk * 100)}pct_risk"] = coverage

    oracle_errors = np.sort(errors)
    oracle_risks = np.cumsum(oracle_errors) / counts
    coverages = counts / max(len(errors), 1)
    aurc = _integrate_trapezoid(risks, coverages)
    oracle_aurc = _integrate_trapezoid(oracle_risks, coverages)
    metrics[f"{prefix}e_aurc"] = max(0.0, aurc - oracle_aurc)
    return metrics


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denom = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return (centre - margin) / denom, (centre + margin) / denom


def _prevalence_adjusted_values(sens: float, spec: float, prevalence: float) -> tuple[float, float]:
    ppv_den = sens * prevalence + (1.0 - spec) * (1.0 - prevalence)
    npv_den = spec * (1.0 - prevalence) + (1.0 - sens) * prevalence
    ppv = sens * prevalence / ppv_den if ppv_den > 0 else float("nan")
    npv = spec * (1.0 - prevalence) / npv_den if npv_den > 0 else float("nan")
    return ppv, npv


def binary_extended_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: dict[str, float] | None = None,
    deployment_prevalence: float | None = None,
    prefix: str = "",
) -> dict[str, float]:
    pos = probs[:, 1]
    y_pred_default = (pos >= 0.5).astype(int)
    metrics: dict[str, float] = {
        f"{prefix}nll": _safe_metric(lambda: log_loss(y_true, probs, labels=[0, 1])),
        f"{prefix}brier_pos": _safe_metric(lambda: brier_score_loss(y_true, pos)),
        f"{prefix}image_auroc": _safe_metric(lambda: roc_auc_score(y_true, pos)),
        f"{prefix}image_ap": _safe_metric(lambda: average_precision_score(y_true, pos)),
    }

    precision, recall, pr_thresholds = precision_recall_curve(y_true, pos)
    f1_values = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_idx = int(np.nanargmax(f1_values)) if len(f1_values) else 0
    metrics[f"{prefix}f1_max"] = float(f1_values[best_idx]) if len(f1_values) else float("nan")
    if len(pr_thresholds):
        threshold_idx = min(best_idx, len(pr_thresholds) - 1)
        metrics[f"{prefix}threshold_f1_max"] = float(pr_thresholds[threshold_idx])

    metrics.update(failure_detection_metrics(y_true, y_pred_default, probs.max(axis=1), prefix=prefix))

    if thresholds is None:
        thresholds = {"default": 0.5}
    elif "default" not in thresholds:
        thresholds = {"default": 0.5, **thresholds}

    for name, threshold in thresholds.items():
        pred = (pos >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)
        key = f"{prefix}{name}"
        metrics[f"{key}_positive_prediction_rate"] = float(pred.mean())
        metrics[f"{key}_ppv_empirical"] = float(ppv)
        metrics[f"{key}_npv_empirical"] = float(npv)
        metrics[f"{key}_f1"] = _safe_metric(lambda pred=pred: f1_score(y_true, pred, zero_division=0))
        sens_low, sens_high = _wilson_interval(int(tp), int(tp + fn))
        spec_low, spec_high = _wilson_interval(int(tn), int(tn + fp))
        metrics[f"{key}_sens_wilson_low"] = float(sens_low)
        metrics[f"{key}_sens_wilson_high"] = float(sens_high)
        metrics[f"{key}_spec_wilson_low"] = float(spec_low)
        metrics[f"{key}_spec_wilson_high"] = float(spec_high)
        if deployment_prevalence is not None:
            ppv_adj, npv_adj = _prevalence_adjusted_values(sens, spec, deployment_prevalence)
            metrics[f"{key}_ppv_at_deployment_prevalence"] = float(ppv_adj)
            metrics[f"{key}_npv_at_deployment_prevalence"] = float(npv_adj)
            metrics[f"{key}_number_needed_to_biopsy"] = float(1.0 / ppv_adj) if ppv_adj > 0 else float("nan")

    for pt in (0.01, 0.05, 0.10):
        pred = (pos >= pt).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        n = max(len(y_true), 1)
        net_benefit = (tp / n) - (fp / n) * (pt / max(1.0 - pt, 1e-12))
        metrics[f"{prefix}decision_curve_net_benefit_pt{int(pt * 100):02d}"] = float(net_benefit)
    return metrics


def uncertainty_separation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    u_e: np.ndarray,
    u_a: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    errors = (y_true != y_pred).astype(int)
    correct = errors == 0
    metrics = {
        f"{prefix}mean_ue_correct": float(u_e[correct].mean()) if correct.any() else float("nan"),
        f"{prefix}mean_ue_incorrect": float(u_e[~correct].mean()) if (~correct).any() else float("nan"),
        f"{prefix}mean_ua_correct": float(u_a[correct].mean()) if correct.any() else float("nan"),
        f"{prefix}mean_ua_incorrect": float(u_a[~correct].mean()) if (~correct).any() else float("nan"),
    }
    if len(np.unique(errors)) == 2:
        metrics[f"{prefix}error_detection_auroc_ue"] = _safe_metric(lambda: roc_auc_score(errors, u_e))
        metrics[f"{prefix}error_detection_auroc_ua"] = _safe_metric(lambda: roc_auc_score(errors, u_a))
    else:
        metrics[f"{prefix}error_detection_auroc_ue"] = float("nan")
        metrics[f"{prefix}error_detection_auroc_ua"] = float("nan")
    return metrics


def multiclass_extended_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_counts: Iterable[int] | None = None,
    prefix: str = "",
) -> dict[str, float]:
    num_classes = probs.shape[1]
    y_pred = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    correct = (y_pred == y_true).astype(float)
    one_hot = _one_hot(y_true, num_classes)
    metrics: dict[str, float] = {
        f"{prefix}nll": _safe_metric(lambda: log_loss(y_true, probs, labels=list(range(num_classes)))),
        f"{prefix}brier_multiclass": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        f"{prefix}top5_accuracy": _safe_metric(lambda: top_k_accuracy_score(y_true, probs, k=min(5, num_classes), labels=list(range(num_classes)))),
        f"{prefix}macro_f1_strict": _safe_metric(lambda: f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    metrics.update(failure_detection_metrics(y_true, y_pred, confidences, prefix=prefix))

    class_eces = []
    for cls in range(num_classes):
        class_targets = (y_true == cls).astype(float)
        if class_targets.any():
            class_eces.append(_ece(probs[:, cls], class_targets))
    metrics[f"{prefix}classwise_ece_mean"] = float(np.mean(class_eces)) if class_eces else float("nan")
    metrics[f"{prefix}classwise_ece_max"] = float(np.max(class_eces)) if class_eces else float("nan")

    if class_counts is not None:
        counts = np.asarray(list(class_counts))
        groups = {
            "many": np.where(counts > 100)[0],
            "medium": np.where((counts > 20) & (counts <= 100))[0],
            "few": np.where(counts <= 20)[0],
        }
        group_accs = []
        for group_name, classes in groups.items():
            per_class_acc = []
            sample_mask = np.isin(y_true, classes)
            for cls in classes:
                cls_mask = y_true == cls
                if cls_mask.any():
                    per_class_acc.append(float((y_pred[cls_mask] == y_true[cls_mask]).mean()))
            metrics[f"{prefix}{group_name}_shot_accuracy"] = float(np.mean(per_class_acc)) if per_class_acc else float("nan")
            metrics[f"{prefix}{group_name}_shot_class_count"] = int(len(classes))
            if sample_mask.any():
                metrics[f"{prefix}{group_name}_shot_ece"] = _ece(confidences[sample_mask], correct[sample_mask])
            else:
                metrics[f"{prefix}{group_name}_shot_ece"] = float("nan")
            if per_class_acc:
                group_accs.append(float(np.mean(per_class_acc)))
        metrics[f"{prefix}worst_group_accuracy"] = float(np.nanmin(group_accs)) if group_accs else float("nan")
    return metrics

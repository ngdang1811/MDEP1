"""
PatchCore-lite reference baseline for MVTec AD image-level anomaly detection.

This is not part of the GUDS-EDL classifier family. It is a benchmark-reference
runner so the paper can compare its image-level MVTec protocol against a
standard normal-only anomaly-detection style baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.isic_paper_experiments import json_safe  # noqa: E402
from experiments.metrics_ext import binary_extended_metrics  # noqa: E402
from experiments.mvtec_ad_runner import MVTecImageLevelDataset, _find_mvtec_category_dir  # noqa: E402


def output_root() -> Path:
    root = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO_ROOT
    return root / "paper_experiment_outputs" / "mvtec_patchcore"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_category_samples(category_dir: str) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    train_good = []
    test_samples = []
    train_good_dir = Path(category_dir) / "train" / "good"
    test_dir = Path(category_dir) / "test"
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    for path in train_good_dir.rglob("*"):
        if path.suffix.lower() in image_exts:
            train_good.append((str(path), 0))

    for defect_dir in test_dir.iterdir():
        if not defect_dir.is_dir():
            continue
        label = 0 if defect_dir.name.lower() == "good" else 1
        for path in defect_dir.rglob("*"):
            if path.suffix.lower() in image_exts:
                test_samples.append((str(path), label))

    if not train_good or len({label for _, label in test_samples}) < 2:
        raise ValueError(f"MVTec category at {category_dir} does not contain train/good and mixed test labels.")
    return train_good, test_samples


class ResNetPatchFeatures(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        try:
            backbone = models.resnet18(weights=weights)
        except Exception as exc:
            print(f"[WARN] Could not load pretrained ResNet-18 weights ({exc}); using random init.")
            backbone = models.resnet18(weights=None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x.flatten(2).transpose(1, 2)


@torch.no_grad()
def collect_memory_bank(
    model: nn.Module,
    loader,
    device: torch.device,
    max_patches: int,
) -> torch.Tensor:
    patches = []
    for inputs, _ in loader:
        feats = model(inputs.to(device))
        feats = torch.nn.functional.normalize(feats.reshape(-1, feats.shape[-1]), dim=1)
        patches.append(feats.cpu())
    memory = torch.cat(patches, dim=0)
    if memory.shape[0] > max_patches:
        indices = torch.randperm(memory.shape[0])[:max_patches]
        memory = memory[indices]
    return memory.to(device)


@torch.no_grad()
def score_images(
    model: nn.Module,
    loader,
    memory: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = []
    scores = []
    for inputs, targets in loader:
        feats = model(inputs.to(device))
        bsz, num_patches, dim = feats.shape
        feats = torch.nn.functional.normalize(feats.reshape(-1, dim), dim=1)
        nearest = []
        for start in range(0, feats.shape[0], chunk_size):
            dist = torch.cdist(feats[start:start + chunk_size], memory)
            nearest.append(dist.min(dim=1).values)
        patch_scores = torch.cat(nearest, dim=0).view(bsz, num_patches)
        image_scores = patch_scores.max(dim=1).values
        labels.append(targets.numpy())
        scores.append(image_scores.detach().cpu().numpy())
    return np.concatenate(labels).astype(int), np.concatenate(scores)


def run_one(args: argparse.Namespace, seed: int) -> dict:
    seed_everything(seed)
    category_dir = _find_mvtec_category_dir(args.category)
    if category_dir is None:
        raise FileNotFoundError(
            f"Real MVTec category '{args.category}' not found. Add MVTec AD under /kaggle/input or set MVTEC_ROOT."
        )

    train_good, test_samples = collect_category_samples(category_dir)
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    workers = 0 if os.name == "nt" else 2
    train_loader = DataLoader(
        MVTecImageLevelDataset(train_good, transform=transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
    )
    test_loader = DataLoader(
        MVTecImageLevelDataset(test_samples, transform=transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = ResNetPatchFeatures(pretrained=not args.no_pretrained).to(device).eval()
    memory = collect_memory_bank(model, train_loader, device, args.max_memory_patches)
    y_true, anomaly_scores = score_images(model, test_loader, memory, device, args.chunk_size)
    if anomaly_scores.max() > anomaly_scores.min():
        pos_probs = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
    else:
        pos_probs = np.zeros_like(anomaly_scores)
    probs = np.stack([1.0 - pos_probs, pos_probs], axis=1)

    metrics = {
        "image_auroc": float(roc_auc_score(y_true, anomaly_scores)),
        "image_ap": float(average_precision_score(y_true, anomaly_scores)),
        "memory_patches": float(memory.shape[0]),
        "train_good_images": float(len(train_good)),
        "test_images": float(len(test_samples)),
    }
    metrics.update(binary_extended_metrics(y_true, probs, thresholds={"default": 0.5}, prefix="patchcore_"))

    run_dir = output_root() / args.category / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "benchmark": "mvtec_patchcore",
        "run_name": args.category,
        "experiment": {
            "name": "patchcore_lite_resnet18",
            "family": "anomaly_detection_reference",
            "description": "Normal-only PatchCore-lite image-level reference using pretrained ResNet-18 patch features.",
        },
        "seed": seed,
        "category_dir": category_dir,
        "metrics": metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")
    print(json.dumps(json_safe(result), indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PatchCore-lite MVTec AD reference baseline.")
    parser.add_argument("--category", type=str, default="hazelnut")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_memory_patches", type=int, default=20000)
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else [args.seed]
    for seed in seeds:
        run_one(args, seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

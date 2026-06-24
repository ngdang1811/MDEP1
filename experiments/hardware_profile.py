"""
Hardware-oriented profiling for the planned efficiency protocol.

This script measures structural sparsity, active parameters, peak CUDA memory,
and forward throughput. It does not claim real 2:4 Tensor Core acceleration;
that requires specialized sparse kernels such as cuSPARSELt or TensorRT.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from guds_edl_core import MDEPConv2d, MDEPLinear, generate_2_4_mask, replace_conv2d_with_mdep  # noqa: E402
from experiments.generalization_paper_suite import EvidenceResNet  # noqa: E402
from experiments.isic_paper_experiments import json_safe  # noqa: E402


def output_root() -> Path:
    root = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO_ROOT
    return root / "paper_experiment_outputs" / "hardware"


def activate_sparse_masks(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (MDEPLinear, MDEPConv2d)):
            module.warmup = False
            module.mask.copy_(generate_2_4_mask(module.scores.data))


def structural_stats(model: nn.Module) -> dict[str, float]:
    total_params = 0
    active_params = 0
    sparse_params = 0
    sparse_active = 0
    valid_24_blocks = 0
    total_24_blocks = 0

    for module in model.modules():
        if isinstance(module, (MDEPLinear, MDEPConv2d)):
            mask = module.mask.detach()
            n = mask.numel()
            active = int(mask.sum().item())
            sparse_params += n
            sparse_active += active
            total_params += n
            active_params += active
            if n % 4 == 0:
                blocks = mask.view(-1, 4)
                total_24_blocks += blocks.shape[0]
                valid_24_blocks += int((blocks.sum(dim=1) == 2).sum().item())
        else:
            for param in module.parameters(recurse=False):
                n = param.numel()
                total_params += n
                active_params += n

    sparse_density = float(sparse_active / max(sparse_params, 1))
    active_density = float(active_params / max(total_params, 1))
    tensor_core_upper_bound = 1.0 / max(sparse_density, 1e-8) if sparse_params > 0 else 1.0
    return {
        "total_params": float(total_params),
        "active_params": float(active_params),
        "active_density": active_density,
        "sparse_wrapped_params": float(sparse_params),
        "sparse_wrapped_active_params": float(sparse_active),
        "sparse_wrapped_density": sparse_density,
        "theoretical_sparse_wrapped_param_reduction": float(1.0 - sparse_density),
        "theoretical_2to4_tensor_core_speedup_upper_bound": float(min(tensor_core_upper_bound, 2.0)),
        "valid_24_block_fraction": float(valid_24_blocks / max(total_24_blocks, 1)),
    }


@torch.no_grad()
def profile_forward(model: nn.Module, device: torch.device, batch_size: int, image_size: int, warmup: int, iters: int) -> dict[str, float]:
    model.eval().to(device)
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    result = {
        "batch_size": float(batch_size),
        "image_size": float(image_size),
        "iterations": float(iters),
        "seconds": float(elapsed),
        "images_per_second": float(batch_size * iters / max(elapsed, 1e-8)),
        "milliseconds_per_batch": float(1000.0 * elapsed / max(iters, 1)),
    }
    if device.type == "cuda":
        result["peak_cuda_memory_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024**2))
    return result


def build_model(mode: str, num_classes: int, pretrained: bool) -> nn.Module:
    model = EvidenceResNet(num_classes=num_classes, dataset="mvtec", pretrained=pretrained)
    if mode in {"static_24", "guds"}:
        replace_conv2d_with_mdep(model)
        activate_sparse_masks(model)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile dense/static-2:4/GUDS structural efficiency.")
    parser.add_argument("--modes", nargs="+", default=["dense", "static_24", "guds"], choices=["dense", "static_24", "guds"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_root().mkdir(parents=True, exist_ok=True)
    results = []
    for mode in args.modes:
        model = build_model(mode, args.num_classes, pretrained=not args.no_pretrained)
        stats = structural_stats(model)
        timing = profile_forward(model, device, args.batch_size, args.image_size, args.warmup, args.iters)
        result = {
            "benchmark": "hardware",
            "run_name": "resnet18_224",
            "experiment": {"name": mode, "family": "hardware_profile"},
            "mode": mode,
            "device": str(device),
            "structural_stats": stats,
            "forward_profile": timing,
            "metrics": {
                "active_density": stats["active_density"],
                "sparse_wrapped_density": stats["sparse_wrapped_density"],
                "valid_24_block_fraction": stats["valid_24_block_fraction"],
                "theoretical_2to4_tensor_core_speedup_upper_bound": stats["theoretical_2to4_tensor_core_speedup_upper_bound"],
                "images_per_second": timing["images_per_second"],
                "milliseconds_per_batch": timing["milliseconds_per_batch"],
                "peak_cuda_memory_mb": timing.get("peak_cuda_memory_mb", float("nan")),
            },
            "kernel_note": "Standard PyTorch masked execution; not a cuSPARSELt/TensorRT sparse Tensor Core benchmark.",
            "reporting_scope": (
                "Use active density, valid_24_block_fraction, and theoretical upper bound as structural-feasibility metrics. "
                "Use images_per_second only as masked-PyTorch throughput unless the model is exported to a real 2:4 sparse kernel."
            ),
        }
        results.append(result)
        mode_dir = output_root() / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        (mode_dir / "metrics.json").write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")
        print(json.dumps(json_safe(result), indent=2))

    summary_path = output_root() / "hardware_profile.json"
    summary_path.write_text(json.dumps(json_safe(results), indent=2), encoding="utf-8")
    print(f"Saved hardware profile: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

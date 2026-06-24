import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset, random_split
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split

# Add parent directory to path to import core framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from guds_edl_core import (
    EvidenceLayer, replace_conv2d_with_mdep, EvidentialFocalLoss, 
    MDEPTrainer, evaluate
)

class MVTecImageLevelDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def _find_mvtec_category_dir(category):
    candidates = []
    if os.environ.get("MVTEC_ROOT"):
        candidates.append(os.environ["MVTEC_ROOT"])
    candidates.extend([
        "./data/mvtec_ad",
        "./data/mvtec",
        "/kaggle/input",
    ])

    for base in candidates:
        if not os.path.isdir(base):
            continue
        direct = os.path.join(base, category)
        if os.path.isdir(direct) and os.path.isdir(os.path.join(direct, "test")):
            return direct
        for root, dirs, _ in os.walk(base):
            if os.path.basename(root).lower() == category.lower() and os.path.isdir(os.path.join(root, "test")):
                return root
            if root.replace(base, "").count(os.sep) > 4:
                dirs[:] = []
    return None


def _collect_mvtec_samples(category_dir):
    samples = []
    for split in ["train", "test"]:
        split_dir = os.path.join(category_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for defect_type in os.listdir(split_dir):
            defect_dir = os.path.join(split_dir, defect_type)
            if not os.path.isdir(defect_dir):
                continue
            label = 0 if defect_type.lower() == "good" else 1
            for root, _, files in os.walk(defect_dir):
                for file_name in files:
                    if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                        samples.append((os.path.join(root, file_name), label))
    return samples


def _stratified_subsets(dataset, labels, seed=42):
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, stratify=labels, random_state=seed
    )
    train_idx, valcal_idx = train_test_split(
        train_idx, test_size=0.20, stratify=np.asarray(labels)[train_idx], random_state=seed
    )
    val_idx, cal_idx = train_test_split(
        valcal_idx, test_size=0.50, stratify=np.asarray(labels)[valcal_idx], random_state=seed
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, cal_idx), Subset(dataset, test_idx)


def get_mvtec_ad_classification_dataloaders(category="hazelnut", batch_size=32, seed=42, allow_dummy_data=False):
    """
    Simulates MVTec AD dataset loading for Image-Level Classification.
    Normal images = Class 0 (Majority), Defective images = Class 1 (Minority)
    Requires a real MVTec category by default. Dummy tensors are available only
    for explicit dry-runs with allow_dummy_data=True.
    """
    print(f"Loading MVTec AD ({category}) for Image-Level Classification...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    category_dir = _find_mvtec_category_dir(category)
    if category_dir is not None:
        samples = _collect_mvtec_samples(category_dir)
        labels = [label for _, label in samples]
        if len(set(labels)) == 2 and min(np.bincount(labels)) >= 4:
            print(f"✅ Found real MVTec category at: {category_dir}")
            print(f"📊 Samples: {len(samples)} | normal={labels.count(0)} | defect={labels.count(1)}")
            dataset = MVTecImageLevelDataset(samples, transform=transform)
            train_ds, val_ds, cal_ds, test_ds = _stratified_subsets(dataset, labels, seed=seed)
            workers = 2 if os.name != 'nt' else 0
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
            cal_loader = DataLoader(cal_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
            train_labels = [labels[i] for i in train_ds.indices]
            counts = np.bincount(train_labels, minlength=2)
            cw = torch.tensor([1.0, max(1.0, counts[0] / max(counts[1], 1))], dtype=torch.float32)
            p_true = [0.5, 0.5]
            p_train = [counts[0] / max(counts.sum(), 1), counts[1] / max(counts.sum(), 1)]
            return train_loader, val_loader, cal_loader, test_loader, cw, p_true, p_train

    if not allow_dummy_data:
        raise FileNotFoundError(
            f"Real MVTec category '{category}' not found. Add the MVTec AD Kaggle "
            "dataset so category folders such as bottle/ and hazelnut/ are visible "
            "under /kaggle/input, or set MVTEC_ROOT. Use --allow_dummy_data only for dry-runs."
        )

    print("⚠ Real MVTec category not found. Falling back to dummy tensors because allow_dummy_data=True.")
    # Represents 500 normal samples and 20 anomalies (1:25 extreme imbalance)
    X_normal = torch.randn(500, 3, 224, 224)
    y_normal = torch.zeros(500, dtype=torch.long)
    X_anomaly = torch.randn(20, 3, 224, 224) * 1.5 + 0.5
    y_anomaly = torch.ones(20, dtype=torch.long)
    
    X = torch.cat([X_normal, X_anomaly])
    Y = torch.cat([y_normal, y_anomaly])
    dataset = TensorDataset(X, Y)
    
    train_len = 350
    val_len = 50
    cal_len = 50
    test_len = len(dataset) - train_len - val_len - cal_len
    
    train_ds, val_ds, cal_ds, test_ds = random_split(
        dataset, [train_len, val_len, cal_len, test_len],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    cal_loader = DataLoader(cal_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    cw = torch.tensor([1.0, 500.0/20.0], dtype=torch.float32)
    p_true = [0.5, 0.5] # Assume uninformative prior for test
    p_train = [500.0/520.0, 20.0/520.0]
    
    return train_loader, val_loader, cal_loader, test_loader, cw, p_true, p_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MVTec AD Image-Level Benchmark for GUDS-EDL")
    parser.add_argument("--category", type=str, default="hazelnut")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_dummy_data", action="store_true", help="Permit synthetic dummy data for dry-runs only.")
    
    # GUDS-EDL Ablation Flags
    parser.add_argument('--disable_pruner', action='store_true')
    parser.add_argument('--disable_regrower', action='store_true')
    parser.add_argument('--pruner_type', type=str, default='signed_first_order', choices=['signed_first_order', 'absolute_grad', 'magnitude', 'random'])
    parser.add_argument('--regrower_type', type=str, default='class_conditioned', choices=['kl_uniform', 'class_conditioned', 'gradient', 'random'])
    parser.add_argument('--kl_scaling', type=str, default='asymmetric', choices=['asymmetric', 'symmetric'])
    parser.add_argument('--disable_efl', action='store_true')
    parser.add_argument('--disable_anticryst', action='store_true')
    
    args = parser.parse_args()
    args.use_anticryst = not args.disable_anticryst

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥  Device: {device}")
    print(f"Starting GUDS-EDL on MVTec AD ({args.category})")
    print(f"⚙️ Ablations: {vars(args)}")
    
    train_loader, val_loader, cal_loader, test_loader, cw, p_true, p_train = get_mvtec_ad_classification_dataloaders(
        args.category,
        args.batch_size,
        seed=args.seed,
        allow_dummy_data=args.allow_dummy_data,
    )
    
    # 1. Initialize Binary Classification Model (ResNet-18)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 2),
        EvidenceLayer(activation='softplus')
    )
    nn.init.normal_(model.fc[0].weight, mean=0, std=0.001)
    nn.init.constant_(model.fc[0].bias, 0)
    
    # 2. Convert to Sparse
    replace_conv2d_with_mdep(model)
    model = model.to(device)
    
    # 3. Setup Loss and Trainer
    warmup_epochs = max(1, int(0.2 * args.epochs))
    criterion = EvidentialFocalLoss(
        gamma=1.2, num_classes=2, kl_lambda=0.1,
        class_weights=cw.to(device),
        warmup_epochs=warmup_epochs, total_epochs=args.epochs,
        disable_efl=args.disable_efl, kl_scaling=args.kl_scaling
    )
    
    trainable_params = [p for name, p in model.named_parameters() if 'scores' not in name]
    optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
    trainer = MDEPTrainer(model, optimizer, criterion, args.epochs, warmup_epochs, args=args)
    
    # 4. Train Loop
    start_time = time.time()
    for epoch in range(args.epochs):
        loss = trainer.train_epoch(epoch, train_loader, device, print_interval=10)
        phase = "Warm-up" if epoch < warmup_epochs else "Dynamic 2:4"
        gamma = trainer.step_gamma(epoch)
        print(f"Epoch [{epoch+1}/{args.epochs}] | {phase} | loss: {loss:.4f} | gamma: {gamma:.4f}")
        
    print(f"Training finished in {(time.time()-start_time)/60:.1f} minutes.")
    
    # 5. Calibration & Evaluation
    print("\n--- Running Bias-Corrected Temperature Calibration ---")
    from experiments.isic_paper_experiments import prior_logit_delta, run_calibration

    temperature, bias, thresholds = run_calibration(
        model,
        cal_loader,
        val_loader,
        device,
        "bias_temperature",
        p_true,
        p_train,
    )
    
    print("\n--- Final Test Evaluation ---")
    prior_delta = prior_logit_delta(p_true, p_train, 2, device=device, dtype=torch.float32)
    eval_bias = prior_delta / max(temperature, 1e-8)
    if bias is not None:
        eval_bias = eval_bias + bias.to(device=device, dtype=eval_bias.dtype)
    model.fc[1].logit_adjustment = torch.zeros(1, dtype=torch.float32, device=device)
    
    _, metrics = evaluate(model, val_loader, test_loader, device, num_classes=2, temperature=temperature, bias=eval_bias, plot=False)
    
    print("\n✅ MVTec AD Summary Results:")
    print(f"  Macro-AUROC: {metrics.get('macro_auroc', 0):.4f}")
    print(f"  AURC:        {metrics.get('aurc', 0):.4f}")
    print(f"  PR-AUC:      {metrics.get('pr_auc', 0):.4f}")
    print(f"  ECE (Adp):   {metrics.get('ece_adaptive', 0):.4f}")
    
    print("\nRun completed. Update Table 2 in main_text.tex with the results.")

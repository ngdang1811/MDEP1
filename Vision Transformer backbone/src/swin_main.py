import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
import os
import math
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss, f1_score, precision_recall_curve, auc
)

# Import local MDEP components
from swin_agents import MDEPLinear, MDEPConv2d, update_scores_agents
from swin_mdep import EvidenceLayer, compute_uncertainties, replace_swin_linear_with_mdep, LogPriorCorrection
from swin_trainer import SwinMDEPTrainer

# --- Evidential Focal Loss ---
def kl_divergence(alpha, num_classes):
    """KL divergence between Dirichlet(alpha) and uniform Dirichlet(1,...,1)."""
    beta = torch.ones(1, num_classes, dtype=torch.float32, device=alpha.device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

class EvidentialFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, num_classes=2, kl_lambda=0.1, class_weights=None, annealing_epochs=10, warmup_epochs=5, total_epochs=30):
        super(EvidentialFocalLoss, self).__init__()
        self.base_gamma = gamma
        self.gamma = gamma
        self.num_classes = num_classes
        self.kl_lambda = kl_lambda
        self.annealing_epochs = annealing_epochs
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(self, evidence, targets, epoch=None):
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes=self.num_classes).float()
            
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        p_hat = alpha / S
        
        # Cross entropy term under Evidential Deep Learning
        loss_ce = torch.sum(targets * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
        
        # Dynamic gamma scheduling
        if epoch is not None:
            if epoch < self.warmup_epochs:
                gamma_val = 0.0
            elif epoch < self.warmup_epochs + 5:
                gamma_val = self.base_gamma * (epoch - self.warmup_epochs) / 5.0
            else:
                gamma_val = self.base_gamma
        else:
            gamma_val = self.gamma
            
        self.gamma = gamma_val
        
        # Focal weight modulation
        p_target = torch.sum(targets * p_hat, dim=1, keepdim=True)
        focal_weight = (1.0 - p_target.detach()) ** gamma_val
        
        if self.class_weights is not None:
            sample_weight = torch.sum(targets * self.class_weights.unsqueeze(0), dim=1, keepdim=True)
        else:
            sample_weight = 1.0
            
        # KL regularizer to shrink evidence for incorrect classes
        alpha_tilde = targets + (1 - targets) * alpha
        loss_kl = kl_divergence(alpha_tilde, self.num_classes)
        
        if epoch is not None and self.annealing_epochs > 0:
            annealing_coef = min(1.0, epoch / self.annealing_epochs)
        else:
            annealing_coef = 1.0
            
        loss = sample_weight * (focal_weight * loss_ce + self.kl_lambda * annealing_coef * loss_kl)
        return torch.mean(loss)

# --- ISIC Dataset Helper ---
class ISICDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.data_frame = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        isic_id = self.data_frame.iloc[idx]['isic_id']
        image = None
        if self.image_dir:
            img_path = os.path.join(self.image_dir, f"{isic_id}.jpg")
            if os.path.exists(img_path):
                try:
                    image = Image.open(img_path).convert('RGB')
                except Exception:
                    image = None
                    
        if image is None:
            image = Image.new('RGB', (224, 224), color='black')

        target = self.data_frame.iloc[idx]['target']
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(target, dtype=torch.long)

def get_isic_dataloaders(batch_size=32, debug=False, subsample_ratio=20):
    num_classes = 2
    
    # Check common Kaggle paths or fallbacks
    kaggle_input = '/kaggle/input'
    csv_path = None
    image_dir = None
    
    if os.path.isdir(kaggle_input) and not debug:
        for root, dirs, files in os.walk(kaggle_input):
            if 'train-metadata.csv' in files:
                csv_path = os.path.join(root, 'train-metadata.csv')
                for img_sub in ['train-image/image', 'train-image', 'train-images']:
                    candidate_img = os.path.join(root, img_sub)
                    if os.path.isdir(candidate_img):
                        image_dir = candidate_img
                        break
                break
                
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),  # Slightly larger for random crop
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])
    
    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if debug or csv_path is None or not os.path.exists(csv_path):
        print("[INFO] Dataset CSV not found. Building dummy dataloaders for verify/diagnose.")
        # 160 train samples, 40 test samples
        X = torch.randn(200, 3, 224, 224)
        Y = torch.randint(0, 2, (200,))
        full = TensorDataset(X, Y)
        tr = Subset(full, range(140))
        va = Subset(full, range(140, 160))
        te = Subset(full, range(160, 200))
        p_true = [0.5, 0.5]
        p_train = [0.5, 0.5]
        return (DataLoader(tr, batch_size=batch_size, shuffle=True),
                DataLoader(va, batch_size=batch_size),
                DataLoader(te, batch_size=batch_size),
                num_classes,
                torch.ones(num_classes), p_true, p_train)

    df = pd.read_csv(csv_path)
    # Ensure patient_id exists and has no NaNs
    df = df.dropna(subset=['patient_id']).reset_index(drop=True)
    
    # Group by patient_id and get the max target for each patient to preserve class balance
    patient_df = df.groupby('patient_id')['target'].max().reset_index()
    
    # Split patients into train/test (80/20) stratified by patient-level target
    train_patients, test_patients = train_test_split(
        patient_df, test_size=0.2, stratify=patient_df['target'], random_state=42
    )
    
    # Split train patients into train/val (70/10 of total, which is 12.5% of train)
    train_patients, val_patients = train_test_split(
        train_patients, test_size=0.125, stratify=train_patients['target'], random_state=42
    )
    
    # Map patients back to the original dataframe
    train_df = df[df['patient_id'].isin(train_patients['patient_id'])].reset_index(drop=True)
    val_df = df[df['patient_id'].isin(val_patients['patient_id'])].reset_index(drop=True)
    test_df = df[df['patient_id'].isin(test_patients['patient_id'])].reset_index(drop=True)
    
    # Calculate true prior probabilities before subsampling
    class_counts_true = train_df['target'].value_counts().sort_index()
    total_true = len(train_df)
    p_true = [class_counts_true.get(c, 0) / total_true for c in range(num_classes)]

    if subsample_ratio is not None and subsample_ratio > 0:
        train_malignant = train_df[train_df['target'] == 1]
        train_benign = train_df[train_df['target'] == 0]
        num_train_malignant = len(train_malignant)
        if num_train_malignant > 0:
            num_benign_to_sample = min(len(train_benign), num_train_malignant * subsample_ratio)
            train_benign_sampled = train_benign.sample(n=num_benign_to_sample, random_state=42)
            train_df = pd.concat([train_malignant, train_benign_sampled]).reset_index(drop=True)

    # Calculate train prior probabilities after subsampling
    class_counts_train = train_df['target'].value_counts().sort_index()
    total_train = len(train_df)
    p_train = [class_counts_train.get(c, 0) / total_train for c in range(num_classes)]

    train_ds = ISICDataset(train_df, image_dir, transform=train_tf)
    val_ds = ISICDataset(val_df, image_dir, transform=test_tf)
    test_ds = ISICDataset(test_df, image_dir, transform=test_tf)
    
    import platform
    workers = 0 if platform.system() == 'Windows' else 2
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    
    beta = 0.999
    cw_raw = [(1.0 - beta) / (1.0 - beta ** class_counts_train.get(c, 1)) for c in range(num_classes)]
    cw = torch.tensor([w / cw_raw[0] for w in cw_raw], dtype=torch.float32)
    print(f"Class weights: {dict(enumerate(cw.tolist()))}")
    return train_loader, val_loader, test_loader, num_classes, cw, p_true, p_train

# --- Helper to load ResNet Teacher ---
def replace_resnet_with_evidential(model):
    """Helper to convert standard ResNet to Evidential Conv/Linear modules matching checkpoint structures."""
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            new = MDEPConv2d(
                module.in_channels, module.out_channels, module.kernel_size,
                stride=module.stride, padding=module.padding,
                dilation=module.dilation, groups=module.groups, bias=(module.bias is not None)
            )
            new.weight.data.copy_(module.weight.data)
            new.scores.data.copy_(torch.abs(module.weight.data))
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)
            setattr(model, name, new)
        elif isinstance(module, nn.Linear):
            new = MDEPLinear(
                module.in_features, module.out_features, bias=(module.bias is not None)
            )
            new.weight.data.copy_(module.weight.data)
            new.scores.data.copy_(torch.abs(module.weight.data))
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)
            setattr(model, name, new)
        else:
            replace_resnet_with_evidential(module)

def load_resnet_teacher(device, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"[WARNING] Teacher checkpoint file '{checkpoint_path}' not found. Distillation will be disabled.")
        return None
        
    print(f"[INFO] Rebuilding ResNet-18 Teacher from checkpoint: {checkpoint_path}")
    try:
        model = models.resnet18(weights=None)
        # Match training head format
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 2),
            EvidenceLayer(activation='softplus')
        )
        replace_resnet_with_evidential(model)
        
        # Load weights
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        # Freeze all teacher weights
        for p in model.parameters():
            p.requires_grad = False
            
        print("ResNet Teacher loaded successfully.")
        return model
    except Exception as e:
        print(f"Failed to load ResNet Teacher: {e}")
        return None

# --- ECE Calculation helper ---
def compute_ece(confidences, accuracies, n_bins=15):
    """Expected Calibration Error with equal-width bins."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_accs, bin_confs, bin_sizes = [], [], []
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            bin_sizes.append(0)
            continue
        b_acc  = accuracies[mask].mean()
        b_conf = confidences[mask].mean()
        b_size = mask.sum()
        ece += (b_size / len(confidences)) * abs(b_acc - b_conf)
        bin_accs.append(b_acc)
        bin_confs.append(b_conf)
        bin_sizes.append(b_size)
    return ece, bin_accs, bin_confs, bin_sizes


def plot_reliability_diagram(bin_accs, bin_confs, bin_sizes, n_bins=15):
    """Reliability diagram: accuracy vs confidence per bin."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    x = np.arange(n_bins)
    width = 0.8
    ax.bar(x, bin_accs, width, label='Accuracy', color='#4e79a7', alpha=0.85)
    ax.bar(x, bin_confs, width, label='Confidence', color='#e15759', alpha=0.4)
    ax.plot([-0.5, n_bins - 0.5], [0, 1], 'k--', linewidth=1, label='Perfect')
    ax.set_xlabel('Bin')
    ax.set_ylabel('Value')
    ax.set_title('Reliability Diagram')
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    os.makedirs('artifacts', exist_ok=True)
    plt.savefig('artifacts/reliability_diagram.png')
    plt.close()


def plot_uncertainty_histogram(u_e_correct, u_e_incorrect):
    """Overlaid histograms of epistemic uncertainty for correct vs wrong."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    if len(u_e_correct) > 0:
        ax.hist(u_e_correct, bins=40, alpha=0.6, label='Correct', color='#59a14f')
    if len(u_e_incorrect) > 0:
        ax.hist(u_e_incorrect, bins=40, alpha=0.6, label='Incorrect', color='#e15759')
    ax.set_xlabel('Epistemic Uncertainty (u_e)')
    ax.set_ylabel('Count')
    ax.set_title('Uncertainty Distribution')
    ax.legend()
    plt.tight_layout()
    os.makedirs('artifacts', exist_ok=True)
    plt.savefig('artifacts/uncertainty_histogram.png')
    plt.close()


def plot_pr_curve(y_true, probs):
    """Precision-Recall Curve with PR-AUC."""
    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(recall, precision, color='#86bcB6', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs('artifacts', exist_ok=True)
    plt.savefig('artifacts/pr_curve.png')
    plt.close()


def plot_risk_coverage_curve(y_true, y_pred, confidences):
    """Risk-Coverage curve and AURC (Area Under Risk-Coverage)."""
    sorted_indices = np.argsort(-confidences)
    sorted_true = y_true[sorted_indices]
    sorted_pred = y_pred[sorted_indices]
    
    n_samples = len(y_true)
    errors = (sorted_true != sorted_pred).astype(float)
    cumulative_errors = np.cumsum(errors)
    
    coverages = np.arange(1, n_samples + 1) / n_samples
    risks = cumulative_errors / np.arange(1, n_samples + 1)
        
    aurc = auc(coverages, risks)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(coverages, risks, color='#f28e2b', lw=2, label=f'Risk-Coverage (AURC = {aurc:.4f})')
    ax.set_xlabel('Coverage')
    ax.set_ylabel('Risk (Error Rate)')
    ax.set_title('Risk-Coverage Curve')
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs('artifacts', exist_ok=True)
    plt.savefig('artifacts/risk_coverage_curve.png')
    plt.close()


def check_representational_collapse(model):
    """
    Diagnoses Representational Collapse in 2:4 structured sparsity.
    """
    print("\n[DIAG] Representational Collapse Diagnostics")
    print("-" * 105)
    print(f"  {'Layer':30s} | {'Score Std':12s} | {'Score Mean':12s} | {'Grad Norm':12s} | {'Status'}")
    print("-" * 105)
    
    all_pass = True
    for name, module in model.named_modules():
        if isinstance(module, (MDEPLinear, MDEPConv2d)):
            scores = module.scores.data
            std = scores.std().item() if scores.numel() > 1 else 0.0
            mean = scores.mean().item()
            
            grad_norm = 0.0
            grad_L = getattr(module, 'grad_L_w', None)
            if grad_L is not None:
                grad_norm = grad_L.norm().item()
                
            status = "[OK] PASS"
            issues = []
            if std <= 1e-4:
                issues.append("Zero Variance")
            if mean <= -10.0:
                issues.append("Negative Drift")
            if grad_norm <= 1e-6:
                issues.append("Dead Gradient")
                
            if issues:
                status = "[FAIL] (" + ", ".join(issues) + ")"
                all_pass = False
                
            print(f"  {name:30s} | {std:12.4e} | {mean:12.4f} | {grad_norm:12.4e} | {status}")
            
    print("-" * 105)
    if all_pass:
        print("  >> OVERALL STATUS: HEALTHY (No Representational Collapse Detected)")
    else:
        print("  >> OVERALL STATUS: WARNING (Representational Collapse Detected in some layers)")
    print()


def print_sparsity_report(model):
    """Per-layer and total sparsity stats + 2:4 pattern check + MACs estimation."""
    print("\n[SPARSITY] Sparsity & Hardware Metrics Report")
    print("-" * 75)
    total_params = 0
    total_zeros  = 0
    total_macs_dense = 0
    total_macs_sparse = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (MDEPLinear, MDEPConv2d)):
            mask = module.mask
            n = mask.numel()
            z = (mask == 0).sum().item()
            total_params += n
            total_zeros  += z
            sparsity = z / n * 100 if n > 0 else 0.0
            
            macs_dense = n
            macs_sparse = n - z
            total_macs_dense += macs_dense
            total_macs_sparse += macs_sparse
            
            if n % 4 == 0:
                blocks = mask.view(-1, 4)
                valid = (blocks.sum(dim=1) == 2).all().item()
                pattern = "[OK] 2:4 (TensorCore Ready)" if valid else "[FAIL] Not 2:4"
            else:
                pattern = "[SKIP] (size%4!=0)"
            print(f"  {name:30s} | {sparsity:5.1f}% sparse | {pattern}")
            
    overall = total_zeros / total_params * 100 if total_params > 0 else 0.0
    macs_saved = (total_macs_dense - total_macs_sparse) / total_macs_dense * 100 if total_macs_dense > 0 else 0.0
    print("-" * 75)
    print(f"  {'TOTAL PARAMS':30s} | {overall:5.1f}% sparse")
    print(f"  {'THEORETICAL MACs SAVED':30s} | {macs_saved:5.1f}% reduction in MDEP layers")
    print("  *(Note: Ampere GPU Tensor Cores provide 2x speedup for strict 2:4 sparsity)*")
    print()
    
    check_representational_collapse(model)


def compute_isic_pauc(y_true, y_pred_prob, min_tpr=0.80):
    """
    2024 ISIC Challenge metric: pAUC above a given true positive rate (TPR).
    """
    from sklearn.metrics import roc_curve, auc
    v_gt = abs(np.asarray(y_true) - 1)
    v_pred = -1.0 * np.asarray(y_pred_prob)
    max_fpr = abs(1 - min_tpr)
    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected min_tpr in (0, 1]")
    fpr_threshold = fpr <= max_fpr
    return auc(fpr[fpr_threshold], tpr[fpr_threshold])


@torch.no_grad()
def evaluate(model, val_loader, test_loader, device, num_classes):
    """Full evaluation: metrics, plots, and uncertainty analysis."""
    model.eval()

    best_t = 0.5
    if num_classes == 2:
        val_targets = []
        val_probs = []
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            evidence = model(inputs)
            unc = compute_uncertainties(evidence)
            p_hat = (unc['alpha'] / unc['S']).cpu().numpy()
            val_targets.append(targets.numpy())
            val_probs.append(p_hat)
            
        val_y_true = np.concatenate(val_targets)
        val_probs = np.concatenate(val_probs, axis=0)
        
        best_bal_acc = 0.0
        for t in np.linspace(0.01, 0.99, 199):
            y_pred_t = (val_probs[:, 1] >= t).astype(int)
            bal_acc_t = balanced_accuracy_score(val_y_true, y_pred_t)
            if bal_acc_t > best_bal_acc:
                best_bal_acc = bal_acc_t
                best_t = t
        print(f"[Validation] Optimized decision threshold: {best_t:.4f}")

    all_targets  = []
    all_preds    = []
    all_confs    = []
    all_probs    = []
    all_u_e      = []
    all_u_a      = []

    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        evidence = model(inputs)
        unc = compute_uncertainties(evidence)

        alpha = unc['alpha']
        S     = unc['S']
        p_hat = (alpha / S).cpu().numpy()
        preds = p_hat.argmax(axis=1) if num_classes > 2 else (p_hat[:, 1] >= best_t).astype(int)
        confs = p_hat.max(axis=1)

        all_targets.append(targets.numpy())
        all_preds.append(preds)
        all_confs.append(confs)
        all_probs.append(p_hat)
        all_u_e.append(unc['epistemic'].cpu().numpy()[:, 0])
        all_u_a.append(unc['aleatoric'].cpu().numpy()[:, 0])

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    confs  = np.concatenate(all_confs)
    probs  = np.concatenate(all_probs, axis=0)
    u_e    = np.concatenate(all_u_e)
    u_a    = np.concatenate(all_u_a)
    correct = (y_pred == y_true).astype(float)

    # ── Scalar Metrics ───────────────────
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    if num_classes == 2:
        macro_auroc = roc_auc_score(y_true, probs[:, 1], average='macro')
        pr_auc = average_precision_score(y_true, probs[:, 1])
        brier = brier_score_loss(y_true, probs[:, 1])
        try:
            pauc = roc_auc_score(y_true, probs[:, 1], max_fpr=0.2)
        except ValueError:
            pauc = float('nan')
        try:
            pauc_isic = compute_isic_pauc(y_true, probs[:, 1], min_tpr=0.80)
        except Exception:
            pauc_isic = float('nan')
            
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        
        y_pred_opt = y_pred
        bal_acc_opt = bal_acc
        macro_f1_opt = macro_f1
        tn_opt, fp_opt, fn_opt, tp_opt = tn, fp, fn, tp
        sensitivity_opt = sensitivity
        specificity_opt = specificity
    else:
        macro_auroc = roc_auc_score(y_true, probs, multi_class='ovr', average='macro')
        pauc = float('nan')
        pauc_isic = float('nan')
        pr_auc = float('nan')
        brier = float('nan')
        sensitivity = float('nan')
        specificity = float('nan')
        best_t = 0.5
        bal_acc_opt = bal_acc
        macro_f1_opt = macro_f1
        sensitivity_opt = float('nan')
        specificity_opt = float('nan')
        y_pred_opt = y_pred

    ece_val, bin_accs, bin_confs, bin_sizes = compute_ece(confs, correct)

    # Minority-ECE (class 1 = malignant)
    minority_mask = (y_true == 1)
    if minority_mask.sum() > 0:
        m_ece, _, _, _ = compute_ece(confs[minority_mask], correct[minority_mask])
    else:
        m_ece = float('nan')

    # ── Print Results ──────────────────────────────────────────────
    print("\n[EVAL] Evaluation Results (Threshold = 0.5)")
    print("=" * 50)
    print(f"  Balanced Accuracy     : {bal_acc:.4f}")
    print(f"  Macro F1-Score        : {macro_f1:.4f}")
    if num_classes == 2:
        print(f"  Sensitivity (Recall)  : {sensitivity:.4f}")
        print(f"  Specificity           : {specificity:.4f}")
        print(f"  PR-AUC                : {pr_auc:.4f}")
        print(f"  Brier Score           : {brier:.4f}")
    print(f"  Macro-AUROC           : {macro_auroc:.4f}")
    print(f"  pAUC (@ 20% FPR)      : {pauc:.4f}")
    print(f"  pAUC (ISIC 2024 metric): {pauc_isic:.4f}")
    print(f"  ECE (15 bins)         : {ece_val:.4f}")
    print(f"  Minority-ECE (cls 1)  : {m_ece:.4f}")
    print(f"  Mean Epistemic u_e    : {u_e.mean():.4f}")
    print(f"  Mean Aleatoric u_a    : {u_a.mean():.4f}")
    print("=" * 50)

    if num_classes == 2:
        print(f"\n[EVAL] Evaluation Results (Optimized Threshold = {best_t:.4f})")
        print("=" * 50)
        print(f"  Balanced Accuracy     : {bal_acc_opt:.4f}")
        print(f"  Macro F1-Score        : {macro_f1_opt:.4f}")
        print(f"  Sensitivity (Recall)  : {sensitivity_opt:.4f}")
        print(f"  Specificity           : {specificity_opt:.4f}")
        print("=" * 50)

    # ── Plots ──────────────────────────────────────────────────────
    plot_reliability_diagram(bin_accs, bin_confs, bin_sizes)
    plot_uncertainty_histogram(
        u_e[correct.astype(bool)],
        u_e[~correct.astype(bool)],
    )
    if num_classes == 2:
        plot_pr_curve(y_true, probs[:, 1])
    plot_risk_coverage_curve(y_true, y_pred_opt, confs)

    return macro_auroc

# --- main() entrypoint ---
def main():
    parser = argparse.ArgumentParser(description="MDEP Swin-T Training Entrypoint")
    parser.add_argument('--debug', action='store_true', help="Run on synthetic verify data")
    parser.add_argument('--epochs', type=int, default=30, help="Total training epochs")
    parser.add_argument('--warmup-epochs', type=int, default=5, help="Warm-up dense epochs")
    parser.add_argument('--batch-size', type=int, default=32, help="DataLoader batch size")
    parser.add_argument('--teacher-path', type=str, default='../RESNET50 backbone/model_checkpoint.pth', help="Path to Teacher checkpoint")
    parser.add_argument('--subsample-ratio', type=int, default=20, help="Ratio of benign to malignant samples in training set (set to 0 or None to disable)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Target Device: {device}")

    # Load datasets
    train_loader, val_loader, test_loader, num_classes, class_weights, p_true, p_train = get_isic_dataloaders(
        batch_size=args.batch_size, debug=args.debug, subsample_ratio=args.subsample_ratio
    )
    print(f"[INFO] Dataloader parsed: {len(train_loader)} train batches, {len(test_loader)} test batches.")

    # Initialize Swin-T Student
    print("[INFO] Initializing Swin-T Student model...")
    try:
        student = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
    except Exception:
        print("[WARNING] Could not load Swin_T ImageNet weights. Initializing randomly.")
        student = models.swin_t(weights=None)
        
    # Replace Head with Evidential Deep Learning interface
    in_features = student.head.in_features
    student.head = nn.Sequential(
        nn.Linear(in_features, num_classes),
        LogPriorCorrection(p_true, p_train),
        EvidenceLayer(activation='softplus')
    )
    nn.init.normal_(student.head[0].weight, mean=0, std=0.001)
    nn.init.constant_(student.head[0].bias, 0)
    
    # Replace intermediate Linear / Conv2d layers in features with MDEP modules
    replace_swin_linear_with_mdep(student.features)
    student.to(device)

    # Load ResNet Teacher if available for distillation
    teacher = None
    if not args.debug:
        candidate_paths = [
            args.teacher_path,
            'model_checkpoint.pth',
            '../RESNET50 backbone/model_checkpoint.pth',
            './RESNET50 backbone/model_checkpoint.pth',
            '/kaggle/input/mdep-resnet50-backbone/model_checkpoint.pth',
        ]
        teacher_path = args.teacher_path
        for path in candidate_paths:
            if path and os.path.exists(path):
                teacher_path = path
                break
        teacher = load_resnet_teacher(device, teacher_path)
    else:
        print("[INFO] Running in debug mode. Distillation is disabled.")

    # Set up Loss Criterion
    criterion = EvidentialFocalLoss(
        gamma=2.0, num_classes=num_classes, kl_lambda=0.1,
        class_weights=class_weights.to(device),
        warmup_epochs=args.warmup_epochs, total_epochs=args.epochs
    )

    # Prevent optimizer hijacking by keeping structural scores parameters out of weight decay and Adam updates
    trainable_params = [p for name, p in student.named_parameters() if 'scores' not in name]
    optimizer = optim.AdamW(trainable_params, lr=2.0e-5, weight_decay=0.01)

    # Cosine annealing scheduler (warmup is handled in train_epoch for epoch < 1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - 1, eta_min=1.0e-6
    )

    trainer = SwinMDEPTrainer(
        model=student,
        optimizer=optimizer,
        criterion=criterion,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        teacher_model=teacher
    )

    print("\nStarting Swin-T MDEP Training Run")
    print("=" * 80)
    for epoch in range(args.epochs):
        loss = trainer.train_epoch(epoch, train_loader, device, scheduler=scheduler)
        if epoch >= 1:  # LR warmup period is 1 epoch, after that scheduler takes over
            scheduler.step()
        phase = "Warm-up (Dense)" if epoch < args.warmup_epochs else "MDEP 2:4 Active Sparsity"
        gamma = trainer.step_gamma(epoch)
        alpha_d = trainer.step_alpha_d(epoch)
        
        print(
            f"  Epoch [{epoch+1:>2}/{args.epochs:>2}] "
            f"| Loss: {loss:.4f} "
            f"| Phase: {phase:<24} "
            f"| gamma: {gamma:.3f} | alpha_d: {alpha_d:.3f}"
        )
    print("=" * 80)
    print("Training sequence completed.")

    # Perform evaluation
    evaluate(student, val_loader, test_loader, device, num_classes)
    print_sparsity_report(student)

    # Save Student checkpoint
    save_path = 'swin_mdep_checkpoint.pth'
    torch.save(student.state_dict(), save_path)
    print(f"Checkpoint saved to '{save_path}'")

if __name__ == '__main__':
    main()

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import os
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset, Dataset

from edl_core import EvidenceLayer
from losses import EvidentialFocalLoss
from trainer import MDEPTrainer
from mdep_agents import MDEPConv2d, MDEPLinear

def replace_conv2d_with_mdep(model):
    """
    Recursively replaces nn.Conv2d and nn.Linear with MDEP dynamic sparse equivalents.
    In a real scenario, you might want to skip the first and last layers.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            mdep_conv = MDEPConv2d(
                module.in_channels, module.out_channels, module.kernel_size,
                stride=module.stride, padding=module.padding, bias=(module.bias is not None)
            )
            # Copy pretrained weights if needed
            mdep_conv.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                mdep_conv.bias.data.copy_(module.bias.data)
            setattr(model, name, mdep_conv)
        elif isinstance(module, nn.Linear):
            mdep_lin = MDEPLinear(module.in_features, module.out_features, bias=(module.bias is not None))
            mdep_lin.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                mdep_lin.bias.data.copy_(module.bias.data)
            setattr(model, name, mdep_lin)
        else:
            replace_conv2d_with_mdep(module)

class ISICDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # ISIC usually uses 'isic_id' for filenames and 'target' for labels
        img_name = os.path.join(self.image_dir, f"{self.data_frame.iloc[idx]['isic_id']}.jpg")
        
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            # Fallback for missing images
            image = Image.new('RGB', (224, 224), color='black')
            
        target = self.data_frame.iloc[idx]['target']
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(target, dtype=torch.long)

def get_isic_dataloader(batch_size=32):
    """
    Data loader for the ISIC 2024 dataset on Kaggle.
    Adjust paths if the dataset is mounted differently.
    """
    csv_path = '/kaggle/input/isic-2024-challenge/train-metadata.csv'
    image_dir = '/kaggle/input/isic-2024-challenge/train-image/image/'
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if not os.path.exists(csv_path):
        print("ISIC dataset not found at expected Kaggle path. Falling back to dummy data.")
        X = torch.randn(100, 3, 224, 224)
        Y = torch.randint(0, 2, (100,)) # Binary classification for ISIC
        dataset = TensorDataset(X, Y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True), 2, torch.ones(2)
        
    dataset = ISICDataset(csv_file=csv_path, image_dir=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
    
    # ISIC is binary (benign vs malignant)
    num_classes = 2 
    df = pd.read_csv(csv_path)
    class_counts = df['target'].value_counts().sort_index()
    total = len(df)
    import math
    cw_raw = [math.sqrt(total / class_counts.get(c, 1)) for c in range(num_classes)]
    majority_weight = cw_raw[0]
    cw = torch.tensor([w / majority_weight for w in cw_raw], dtype=torch.float32)
    
    return dataloader, num_classes, cw

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize dataloader first to get the correct number of classes
    dataloader, num_classes, class_weights = get_isic_dataloader(batch_size=32)
    
    # 1. Load off-the-shelf ResNet from torchvision
    # ResNet18 is used here for demonstration
    model = models.resnet18(weights=None)
    
    # Modify the head for Evidential Deep Learning
    in_features = model.fc.in_features
    # Remove standard classification head and append EvidenceLayer to ensure non-negative output
    model.fc = nn.Sequential(
        nn.Linear(in_features, num_classes),
        EvidenceLayer(activation='softplus')
    )
    # Initialize evidence output to be small to prevent KL explosion
    nn.init.normal_(model.fc[0].weight, mean=0, std=0.001)
    nn.init.constant_(model.fc[0].bias, 0)
    
    # Convert standard dense layers to MDEP Multi-Agent Sparse layers
    replace_conv2d_with_mdep(model)
    model = model.to(device)
    
    total_epochs = 20
    warmup_epochs = 6
    
    # 2. Setup Loss, Optimizer, Trainer
    criterion = EvidentialFocalLoss(
        gamma=1.2, num_classes=num_classes, kl_lambda=0.1,
        class_weights=class_weights.to(device),
        warmup_epochs=warmup_epochs, total_epochs=total_epochs
    )
    # Khắc phục lỗi Optimizer Hijacking: chặn 'scores' khỏi AdamW
    trainable_params = [p for name, p in model.named_parameters() if 'scores' not in name]
    optimizer = optim.Adam(trainable_params, lr=4.0e-05)
    
    trainer = MDEPTrainer(model, optimizer, criterion, total_epochs, warmup_epochs)
    
    # 3. Training Loop
    print("Starting Training (MDEP Framework)...")
    for epoch in range(total_epochs):
        loss = trainer.train_epoch(epoch, dataloader, device)
        is_warmup = epoch < warmup_epochs
        phase = "Warm-up (Dense)" if is_warmup else "Dynamic 2:4 Sparsity"
        print(f"Epoch [{epoch+1}/{total_epochs}] | Phase: {phase} | Loss: {loss:.4f}")
        
    print("Training complete! Model is structured with 2:4 sparsity in target layers.")

if __name__ == "__main__":
    main()

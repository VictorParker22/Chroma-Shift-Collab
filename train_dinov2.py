#!/usr/bin/env python3
"""
train_dinov2.py — Chroma-Shift DINOv2 Fine-Tuning for Vegetation Classification
=================================================================================
Trains a linear classification head on top of a frozen DINOv2 ViT-Small backbone
to classify multispectral captures as healthy_vegetation vs stressed_soil.

Optimized for NVIDIA Jetson Orin Nano (8 GB LPDDR5):
  - Frozen backbone (only ~768 trainable params in linear head)
  - AMP (mixed precision) for Ampere Tensor Cores
  - Batch size 16, 2 data workers

Usage:
    python train_dinov2.py                    # train with defaults
    python train_dinov2.py --epochs 50        # more epochs
    python train_dinov2.py --lr 5e-4          # custom learning rate
"""

import os
import sys
import time
import argparse
import json
from datetime import datetime

import torch
import gc
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "dino_v2_classify")

CLASSES = ["healthy_vegetation", "stressed_soil"]
NUM_CLASSES = 2

# ImageNet normalization (DINOv2 pretrained)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ═══════════════════════════════════════════════════════════════════════════
# Model: DINOv2 ViT-Small + Linear Head
# ═══════════════════════════════════════════════════════════════════════════

class DINOv2Classifier(nn.Module):
    """
    Frozen DINOv2 backbone + trainable linear classification head.
    Using ViT-Small/14 (21M params, 384-dim embeddings).
    """
    def __init__(self, num_classes=2, dropout=0.1):
        super().__init__()
        # Load DINOv2 ViT-Small from torch hub
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14",
            pretrained=True
        )
        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # Classification head
        embed_dim = self.backbone.embed_dim  # 384 for ViT-S
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)  # [B, 384]
        return self.head(features)


# ═══════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════

def get_transforms(train=True, image_size=168):
    """Augmentation pipeline matching our TAO spec."""
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.1, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def load_datasets(data_dir, batch_size=4, num_workers=2, image_size=168):
    """Load train/val datasets from folder-per-class structure."""
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = datasets.ImageFolder(train_dir, transform=get_transforms(train=True, image_size=image_size))
    val_dataset = datasets.ImageFolder(val_dir, transform=get_transforms(train=False, image_size=image_size))

    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")
    print(f"  Classes: {train_dataset.classes}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    return train_loader, val_loader, train_dataset.classes


# ═══════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    # Keep backbone frozen/eval
    model.backbone.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DINOv2 vegetation classifier training.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=168)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Chroma-Shift — DINOv2 Vegetation Classifier Training     ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║   Backbone: DINOv2 ViT-Small/14 (frozen)                   ║")
    print(f"║   Epochs: {args.epochs:3d}  Batch: {args.batch_size:2d}  LR: {args.lr}                   ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}", end="")
    if torch.cuda.is_available():
        print(f" ({torch.cuda.get_device_name(0)})")
        # Memory optimizations for shared-memory Jetson
        torch.cuda.empty_cache()
        gc.collect()
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print(" (WARNING: No GPU — training will be slow)")

    # --- Data ---
    print("\n  Loading datasets …")
    train_loader, val_loader, classes = load_datasets(
        args.data_dir, args.batch_size, args.workers, args.image_size
    )

    # --- Model ---
    print("\n  Loading DINOv2 backbone …")
    model = DINOv2Classifier(num_classes=NUM_CLASSES).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable:    {trainable:,} (linear head only)")

    # --- Optimizer & Scheduler ---
    optimizer = optim.AdamW(
        model.head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Cosine annealing with warmup
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=args.warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs]
    )

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # --- Training ---
    print(f"\n  Starting training …\n")
    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(f"  Epoch {epoch:3d}/{args.epochs} │ "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:6.2f}% │ "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:6.2f}% │ "
              f"LR: {lr:.2e} │ {elapsed:.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": lr,
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(args.results_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "classes": classes,
            }, best_path)
            print(f"           ★ New best: {val_acc:.2f}% → {best_path}")

        # Periodic checkpoint
        if epoch % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(args.results_dir, f"checkpoint_epoch{epoch:03d}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "classes": classes,
            }, ckpt_path)

    # --- Save final model + training log ---
    final_path = os.path.join(args.results_dir, "final_model.pth")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "best_val_acc": best_val_acc,
    }, final_path)

    log_path = os.path.join(args.results_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'═' * 62}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best val accuracy: {best_val_acc:.2f}%")
    print(f"  Best model: {os.path.join(args.results_dir, 'best_model.pth')}")
    print(f"  Final model: {final_path}")
    print(f"  Training log: {log_path}")
    print(f"{'═' * 62}\n")


if __name__ == "__main__":
    main()

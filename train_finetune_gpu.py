#!/usr/bin/env python3
"""
train_finetune_gpu.py — Full DINOv2 Fine-Tuning for Desktop GPU
================================================================
Designed for NVIDIA RTX 4070 (12GB VRAM) / i9-13900K (32 threads).

Unlike train_dinov2_v2.py (which freezes the backbone and trains only
a 770-param linear head), this script UNFREEZES the backbone and
fine-tunes the entire 22M-param model on our field data.

Supports 2-class (healthy/stressed) or 4-class (healthy_vegetation,
stressed_vegetation, bare_soil, non_vegetation) training.

Key differences from Jetson training:
  - Full backbone fine-tuning (not just linear probe)
  - Mixed precision (FP16) for 2× speedup on RTX 4070
  - Differential learning rates (backbone lr = head lr / 10)
  - Stronger augmentation (RandAugment, CutMix)
  - Cosine annealing with warm restarts
  - Gradient accumulation for effective batch size 64

Setup (one-time, on the i9):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    git clone https://github.com/VictorParker22/Chroma-Shift.git
    cd Chroma-Shift

Usage:
    python train_finetune_gpu.py                          # full pipeline
    python train_finetune_gpu.py --skip-phase1            # reuse cached features for eval only
    python train_finetune_gpu.py --epochs 30 --lr 2e-5    # custom
    python train_finetune_gpu.py --backbone dinov2_vitb14  # try ViT-Base (86M params)

After training:
    git add results/dino_v2_classify/best_model.pth
    git commit -m "Full fine-tune on RTX 4070: XX.XX% accuracy"
    git push
    # Then on Jetson: git pull && python convert_tensorrt.py
"""

import os
import sys
import time
import json
import argparse
import math
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from collections import Counter
from PIL import Image
from torchvision import transforms, datasets

# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results" / "dino_v2_classify"

CLASSES_2 = ["healthy_vegetation", "stressed_soil"]
CLASSES_4 = ["healthy_vegetation", "stressed_vegetation", "bare_soil", "non_vegetation"]
NUM_CLASSES = None  # Set from --num-classes or auto-detected from data
IMAGE_SIZE = 224  # DINOv2 native resolution (not 168 like linear probe)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ═══════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════

class DINOv2Classifier(nn.Module):
    """DINOv2 backbone + classification head (fully trainable)."""

    def __init__(self, backbone_name="dinov2_vits14", num_classes=2, dropout=0.2):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", backbone_name, pretrained=True
        )
        embed_dim = self.backbone.embed_dim  # 384 for ViT-S, 768 for ViT-B
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )
        self.embed_dim = embed_dim
        self.backbone_name = backbone_name

    def forward(self, x):
        features = self.backbone(x)  # [B, embed_dim]
        return self.head(features)

    def get_param_groups(self, backbone_lr, head_lr, weight_decay=0.01):
        """Differential learning rates: backbone gets lower lr."""
        # Backbone params (frozen layers get even lower lr)
        backbone_params = []
        for name, param in self.backbone.named_parameters():
            backbone_params.append(param)

        head_params = list(self.head.parameters())

        return [
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": head_params, "lr": head_lr, "weight_decay": 0.0},
        ]


# ═══════════════════════════════════════════════════════════════════════════
# Data Augmentation
# ═══════════════════════════════════════════════════════════════════════════

def get_train_transforms():
    """Strong augmentation for fine-tuning."""
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.15)),
    ])


def get_val_transforms():
    """Deterministic transforms for validation."""
    return transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE * 1.14)),  # 256
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
        print(f"  VRAM: {vram / 1e9:.1f} GB")

    # Data
    train_dir = Path(args.data_dir) / "train"
    val_dir = Path(args.data_dir) / "val"

    train_dataset = datasets.ImageFolder(str(train_dir), transform=get_train_transforms())
    val_dataset = datasets.ImageFolder(str(val_dir), transform=get_val_transforms())

    # Auto-detect number of classes from data
    num_classes = args.num_classes or len(train_dataset.classes)
    classes = train_dataset.classes
    print(f"  Detected {num_classes} classes: {classes}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # Compute class weights for imbalanced data
    class_counts = Counter(train_dataset.targets)
    total = len(train_dataset)
    class_weights = torch.tensor(
        [total / (num_classes * class_counts.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float32
    ).to(device)
    print(f"  Class weights: {', '.join(f'{w:.2f}' for w in class_weights)}")

    print(f"  Train: {len(train_dataset):,} images ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_dataset):,} images")
    print(f"  Classes: {classes}")

    # Model
    print(f"\n  Loading {args.backbone} backbone …")
    model = DINOv2Classifier(
        backbone_name=args.backbone,
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable:    {trainable:,}")

    # Optimizer with differential learning rates
    backbone_lr = args.lr / 10  # backbone gets 10× lower lr
    head_lr = args.lr
    param_groups = model.get_param_groups(backbone_lr, head_lr, args.weight_decay)
    optimizer = optim.AdamW(param_groups)

    print(f"  Backbone LR: {backbone_lr:.2e}")
    print(f"  Head LR:     {head_lr:.2e}")

    # Cosine scheduler with warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    scaler = GradScaler(enabled=args.fp16)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # Gradient accumulation
    accum_steps = max(1, args.effective_batch // args.batch_size)
    print(f"  Effective batch: {args.batch_size * accum_steps}")
    print(f"  FP16: {args.fp16}")
    print(f"  Label smoothing: 0.1")

    # Training
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    history = []

    print(f"\n  Training for {args.epochs} epochs …\n")
    print(f"  {'Epoch':>5s} │ {'Train Loss':>10s} {'Train Acc':>10s} │ "
          f"{'Val Loss':>10s} {'Val Acc':>10s} │ {'LR':>10s} │ {'Time':>6s}")
    print(f"  {'─' * 78}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        optimizer.zero_grad()

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            with autocast(enabled=args.fp16):
                outputs = model(images)
                loss = criterion(outputs, labels) / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item() * accum_steps * images.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = 100.0 * correct / total

        # ── Validate ──
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast(enabled=args.fp16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * images.size(0)
                preds = outputs.argmax(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss_sum / val_total
        val_acc = 100.0 * val_correct / val_total
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[1]["lr"]  # head lr

        print(f"  {epoch:5d} │ {train_loss:10.4f} {train_acc:9.2f}% │ "
              f"{val_loss:10.4f} {val_acc:9.2f}% │ {lr_now:10.2e} │ {elapsed:5.1f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "lr": lr_now,
        })

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, epoch, val_acc, RESULTS_DIR / "best_model.pth")
            print(f"         ★ New best: {val_acc:.2f}%")

            # Also save confusion matrix for best model
            save_confusion_matrix(all_preds, all_labels, RESULTS_DIR)

        # Periodic checkpoints
        if epoch % 10 == 0:
            save_checkpoint(model, epoch, val_acc,
                          RESULTS_DIR / f"finetune_epoch{epoch:03d}.pth")

    # Final save
    save_checkpoint(model, args.epochs, val_acc, RESULTS_DIR / "final_model.pth")

    with open(RESULTS_DIR / "training_log.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'═' * 62}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best val accuracy: {best_val_acc:.2f}%")
    print(f"  Model saved to: {RESULTS_DIR / 'best_model.pth'}")
    print(f"{'═' * 62}")
    print(f"\n  Next steps:")
    print(f"    git add results/dino_v2_classify/best_model.pth")
    print(f"    git commit -m 'Full fine-tune: {best_val_acc:.2f}% accuracy'")
    print(f"    git push")
    print(f"    # On Jetson: git pull && python convert_tensorrt.py\n")

    return best_val_acc


def save_checkpoint(model, epoch, val_acc, path):
    """Save in the same format as train_dinov2_v2.py for compatibility."""
    state_dict = {}
    for k, v in model.backbone.state_dict().items():
        state_dict[f"backbone.{k}"] = v
    for k, v in model.head.state_dict().items():
        state_dict[f"head.{k}"] = v

    torch.save({
        "epoch": epoch,
        "model_state_dict": state_dict,
        "val_acc": val_acc,
        "classes": classes if 'classes' in dir() else (CLASSES_4 if num_classes == 4 else CLASSES_2),
        "backbone": model.backbone_name,
        "embed_dim": model.embed_dim,
        "training": "full_finetune",
        "timestamp": datetime.now().isoformat(),
    }, path)


def save_confusion_matrix(predictions, labels, results_dir):
    """Save confusion matrix for best model."""
    predictions = np.array(predictions)
    labels = np.array(labels)

    n = len(set(labels))
    cm = np.zeros((n, n), dtype=int)
    for true, pred in zip(labels, predictions):
        cm[int(true)][int(pred)] += 1

    metrics = {}
    cls_names = CLASSES_4 if n == 4 else CLASSES_2
    for i in range(n):
        cls = cls_names[i] if i < len(cls_names) else f"class_{i}"
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(n)) - tp
        fn = sum(cm[i][j] for j in range(n)) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        metrics[cls] = {"precision": round(precision, 4), "recall": round(recall, 4),
                       "f1": round(f1, 4), "support": int(tp + fn)}

    accuracy = np.trace(cm) / cm.sum()
    report = {
        "accuracy": round(float(accuracy), 4),
        "confusion_matrix": cm.tolist(),
        "classes": cls_names[:n],
        "per_class": metrics,
        "total_samples": int(cm.sum()),
        "training_type": "full_finetune",
    }

    with open(results_dir / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Full DINOv2 fine-tuning (Desktop GPU)")
    parser.add_argument("--backbone", default="dinov2_vits14",
                       choices=["dinov2_vits14", "dinov2_vitb14"],
                       help="DINOv2 backbone (vits14=22M, vitb14=86M)")
    parser.add_argument("--num-classes", type=int, default=None,
                       help="Number of classes (auto-detected from data if not set)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Per-GPU batch size (32 fits in 12GB)")
    parser.add_argument("--effective-batch", type=int, default=64,
                       help="Effective batch size via gradient accumulation")
    parser.add_argument("--lr", type=float, default=2e-5,
                       help="Head learning rate (backbone gets lr/10)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8,
                       help="DataLoader workers (i9 can handle 8+)")
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Mixed precision training (default: on)")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Chroma-Shift — DINOv2 Full Fine-Tuning (Desktop GPU)     ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║   Backbone: {args.backbone:<47s}║")
    print(f"║   Epochs:   {args.epochs:<47d}║")
    print(f"║   LR:       {args.lr:<47.1e}║")
    print("╚══════════════════════════════════════════════════════════════╝")

    train(args)


if __name__ == "__main__":
    main()

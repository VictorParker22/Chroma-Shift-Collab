#!/usr/bin/env python3
"""
train_dinov2_v2.py — Memory-efficient DINOv2 training for Jetson (8GB)
=======================================================================
Two-phase approach:
  Phase 1: Extract DINOv2 features on CPU and cache to disk (one-time)
  Phase 2: Train a linear head on cached features (fast, low memory)

This avoids loading the 22M-param backbone into GPU during training.

Usage:
    python train_dinov2_v2.py                         # full pipeline
    python train_dinov2_v2.py --skip-extract           # reuse cached features
    python train_dinov2_v2.py --epochs 20 --lr 5e-4   # custom training
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from PIL import Image
from torchvision import transforms, datasets

# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results" / "dino_v2_classify"
FEATURES_DIR = SCRIPT_DIR / "results" / "cached_features"

CLASSES = ["healthy_vegetation", "stressed_soil"]
NUM_CLASSES = 2
IMAGE_SIZE = 168

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Feature Extraction (CPU, one-time)
# ═══════════════════════════════════════════════════════════════════════════

def extract_features(data_dir, features_dir, batch_size=8, num_workers=2):
    """Extract DINOv2 features from all images and cache to disk."""
    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)

    print("\n  Phase 1: Extracting DINOv2 features (CPU) …")
    print("  Loading DINOv2 ViT-Small/14 backbone …")

    backbone = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14", pretrained=True
    )
    backbone.eval()
    device = torch.device("cpu")
    backbone.to(device)

    embed_dim = backbone.embed_dim  # 384 for ViT-S
    print(f"  Backbone loaded: embed_dim={embed_dim}")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    for split in ["train", "val"]:
        split_dir = Path(data_dir) / split
        if not split_dir.exists():
            print(f"  ⚠️  {split_dir} not found, skipping")
            continue

        dataset = datasets.ImageFolder(str(split_dir), transform=transform)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False,
        )

        all_features = []
        all_labels = []
        total = len(dataset)

        print(f"\n  Extracting {split}: {total:,} images …")
        t0 = time.time()

        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):
                features = backbone(images)  # [B, 384]
                all_features.append(features.numpy())
                all_labels.append(labels.numpy())

                if (i + 1) % 500 == 0:
                    done = min((i + 1) * batch_size, total)
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    eta = (total - done) / rate if rate > 0 else 0
                    print(f"    {done:,}/{total:,} ({done/total*100:.1f}%) "
                          f"| {rate:.0f} img/s | ETA: {eta/60:.1f}min")

        features_np = np.concatenate(all_features, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)

        feat_path = features_dir / f"{split}_features.npy"
        label_path = features_dir / f"{split}_labels.npy"
        np.save(feat_path, features_np)
        np.save(label_path, labels_np)

        elapsed = time.time() - t0
        print(f"  ✅ {split}: {features_np.shape[0]:,} features ({features_np.shape[1]}d) "
              f"saved in {elapsed:.0f}s")

    # Clean up backbone from memory
    del backbone
    import gc; gc.collect()

    return embed_dim


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Train Linear Head (fast, tiny memory)
# ═══════════════════════════════════════════════════════════════════════════

class LinearClassifier(nn.Module):
    def __init__(self, embed_dim=384, num_classes=2, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(self.dropout(x))


def train_head(features_dir, results_dir, epochs=15, batch_size=256, lr=1e-3,
               weight_decay=0.01, warmup_epochs=2):
    """Train linear head on pre-extracted features."""
    features_dir = Path(features_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n  Phase 2: Training linear classifier …")

    # Load cached features
    train_features = torch.from_numpy(np.load(features_dir / "train_features.npy"))
    train_labels = torch.from_numpy(np.load(features_dir / "train_labels.npy"))
    val_features = torch.from_numpy(np.load(features_dir / "val_features.npy"))
    val_labels = torch.from_numpy(np.load(features_dir / "val_labels.npy"))

    embed_dim = train_features.shape[1]
    print(f"  Train: {train_features.shape[0]:,} × {embed_dim}d")
    print(f"  Val:   {val_features.shape[0]:,} × {embed_dim}d")

    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model (tiny — just 770 params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearClassifier(embed_dim, NUM_CLASSES).to(device)
    trainable = sum(p.numel() for p in model.parameters())
    print(f"  Device: {device}  Trainable params: {trainable:,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine scheduler with warmup
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])

    criterion = nn.CrossEntropyLoss()

    print(f"\n  Training for {epochs} epochs (batch={batch_size}, lr={lr}) …\n")
    best_val_acc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * features.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = 100.0 * correct / total

        # Validate
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * features.size(0)
                val_correct += outputs.argmax(1).eq(labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = 100.0 * val_correct / val_total
        scheduler.step()
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"  Epoch {epoch:2d}/{epochs} │ "
              f"Train: {train_loss:.4f} / {train_acc:6.2f}% │ "
              f"Val: {val_loss:.4f} / {val_acc:6.2f}% │ "
              f"LR: {lr_now:.2e} │ {elapsed:.1f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "lr": lr_now,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save as full model (with backbone) for compatibility
            _save_full_model(model, embed_dim, epoch, val_acc,
                             results_dir / "best_model.pth")
            print(f"         ★ New best: {val_acc:.2f}%")

        if epoch % 5 == 0:
            _save_full_model(model, embed_dim, epoch, val_acc,
                             results_dir / f"checkpoint_epoch{epoch:03d}.pth")

    # Final save
    _save_full_model(model, embed_dim, epochs, val_acc,
                     results_dir / "final_model.pth")

    with open(results_dir / "training_log.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'═' * 62}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best val accuracy: {best_val_acc:.2f}%")
    print(f"  Results: {results_dir}")
    print(f"{'═' * 62}\n")
    return best_val_acc


def _save_full_model(head_model, embed_dim, epoch, val_acc, path):
    """
    Save checkpoint in the same format as train_dinov2.py (full model)
    so export_and_infer.py and jetson_main.py can load it directly.
    """
    # Build full model state dict (backbone + head)
    # We load the backbone state dict and merge with the trained head
    full_state = {}

    # Get backbone state dict from hub cache
    backbone = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14", pretrained=True
    )
    for k, v in backbone.state_dict().items():
        full_state[f"backbone.{k}"] = v

    # Add head state dict
    for k, v in head_model.state_dict().items():
        # Map LinearClassifier params → DINOv2Classifier.head params
        if k == "fc.weight":
            full_state["head.1.weight"] = v
        elif k == "fc.bias":
            full_state["head.1.bias"] = v
        # dropout has no state

    torch.save({
        "epoch": epoch,
        "model_state_dict": full_state,
        "val_acc": val_acc,
        "classes": CLASSES,
    }, path)

    del backbone


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip feature extraction (reuse cached)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--extract-batch", type=int, default=8,
                        help="Batch size for feature extraction (CPU)")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--features-dir", type=str, default=str(FEATURES_DIR))
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Chroma-Shift — DINOv2 Training v2 (Feature Caching)      ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║   Phase 1: Extract features (CPU, one-time)                ║")
    print(f"║   Phase 2: Train linear head ({args.epochs} epochs, fast)              ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    if not args.skip_extract:
        extract_features(args.data_dir, args.features_dir,
                         batch_size=args.extract_batch)
    else:
        print("\n  Skipping feature extraction (using cached)")

    train_head(args.features_dir, args.results_dir,
               epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()

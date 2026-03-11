#!/usr/bin/env python3
"""
build_dataset.py — Large-scale dataset builder for Chroma-Shift
================================================================
Downloads PlantVillage + PlantDoc, curates into healthy/stressed binary classes,
applies 5x augmentation, merges with field captures, and builds the final
train/val dataset for DINOv2 training.

Usage:
    python build_dataset.py                     # full pipeline
    python build_dataset.py --skip-download     # skip download (already have data)
    python build_dataset.py --augment-factor 10 # 10x augmentation
"""

import os
import sys
import shutil
import random
import argparse
import subprocess
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATASETS_DIR = SCRIPT_DIR / "datasets"         # raw downloaded datasets
CURATED_DIR = SCRIPT_DIR / "datasets" / "curated"  # sorted healthy/stressed
DATA_DIR = SCRIPT_DIR / "data"                  # final train/val for training
CAPTURES_DIR = SCRIPT_DIR / "captures"          # our field captures

CLASSES = ["healthy_vegetation", "stressed_soil"]
TRAIN_RATIO = 0.85
IMAGE_SIZE = (224, 224)  # resize target

# ---------------------------------------------------------------------------
# PlantVillage class → binary mapping
# Anything with "healthy" in the folder name → healthy_vegetation
# Everything else (diseased) → stressed_soil
# ---------------------------------------------------------------------------
PLANTVILLAGE_HEALTHY_KEYWORDS = ["healthy"]
PLANTVILLAGE_STRESSED_KEYWORDS = [
    "blight", "rot", "spot", "rust", "scab", "mold", "virus",
    "wilt", "mosaic", "curl", "miner", "bacteria", "fungal",
    "powdery", "downy", "septoria", "target", "common_rust",
    "cercospora", "esca", "black_rot", "cedar_apple", "haunglongbing",
    "greening", "citrus", "bacterial", "leaf_scorch", "tungro",
]

# PlantDoc has folders like "Apple Healthy", "Tomato Early Blight", etc.
PLANTDOC_HEALTHY_KEYWORDS = ["healthy", "Healthy"]
# ---------------------------------------------------------------------------


def download_plantvillage():
    """Download PlantVillage dataset from GitHub."""
    dest = DATASETS_DIR / "PlantVillage-Dataset"
    if dest.exists():
        print(f"  ✓ PlantVillage already downloaded at {dest}")
        return dest

    print("  📥 Downloading PlantVillage (~1.2 GB) …")
    cmd = [
        "git", "clone", "--depth", "1",
        "https://github.com/spMohanty/PlantVillage-Dataset.git",
        str(dest),
    ]
    subprocess.run(cmd, cwd=str(DATASETS_DIR), check=True)
    print(f"  ✓ PlantVillage downloaded to {dest}")
    return dest


def download_plantdoc():
    """Download PlantDoc dataset from GitHub."""
    dest = DATASETS_DIR / "PlantDoc-Dataset"
    if dest.exists():
        print(f"  ✓ PlantDoc already downloaded at {dest}")
        return dest

    print("  📥 Downloading PlantDoc (~300 MB) …")
    cmd = [
        "git", "clone", "--depth", "1",
        "https://github.com/pratikkayal/PlantDoc-Dataset.git",
        str(dest),
    ]
    subprocess.run(cmd, cwd=str(DATASETS_DIR), check=True)
    print(f"  ✓ PlantDoc downloaded to {dest}")
    return dest


def is_image(path):
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def classify_plantvillage_folder(folder_name):
    """Map PlantVillage folder name → binary class."""
    name_lower = folder_name.lower()
    for kw in PLANTVILLAGE_HEALTHY_KEYWORDS:
        if kw in name_lower:
            return "healthy_vegetation"
    return "stressed_soil"  # diseased = stressed


def classify_plantdoc_folder(folder_name):
    """Map PlantDoc folder name → binary class."""
    for kw in PLANTDOC_HEALTHY_KEYWORDS:
        if kw in folder_name:
            return "healthy_vegetation"
    return "stressed_soil"


def curate_plantvillage(pv_dir):
    """Sort PlantVillage images into healthy/stressed."""
    # PlantVillage has: raw/color/ and raw/grayscale/ and raw/segmented/
    # Use color images (most useful for DINOv2)
    color_dir = pv_dir / "raw" / "color"
    if not color_dir.exists():
        print(f"  ⚠️  PlantVillage color dir not found at {color_dir}")
        return 0, 0

    healthy_count = 0
    stressed_count = 0

    for class_dir in sorted(color_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = classify_plantvillage_folder(class_dir.name)
        dest_dir = CURATED_DIR / label
        dest_dir.mkdir(parents=True, exist_ok=True)

        for img_path in class_dir.iterdir():
            if is_image(img_path):
                dest = dest_dir / f"pv_{class_dir.name}_{img_path.name}"
                if not dest.exists():
                    shutil.copy2(img_path, dest)
                if label == "healthy_vegetation":
                    healthy_count += 1
                else:
                    stressed_count += 1

    return healthy_count, stressed_count


def curate_plantdoc(pd_dir):
    """Sort PlantDoc images into healthy/stressed."""
    healthy_count = 0
    stressed_count = 0

    # PlantDoc has train/ and test/ directories with class subfolders
    for split in ["train", "test"]:
        split_dir = pd_dir / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = classify_plantdoc_folder(class_dir.name)
            dest_dir = CURATED_DIR / label
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img_path in class_dir.iterdir():
                if is_image(img_path):
                    dest = dest_dir / f"pd_{class_dir.name}_{img_path.name}"
                    if not dest.exists():
                        shutil.copy2(img_path, dest)
                    if label == "healthy_vegetation":
                        healthy_count += 1
                    else:
                        stressed_count += 1

    return healthy_count, stressed_count


def curate_field_captures():
    """Add our field-captured images (from data/ train/val dirs)."""
    healthy_count = 0
    stressed_count = 0

    for split in ["train", "val"]:
        for label in CLASSES:
            src_dir = DATA_DIR / split / label
            if not src_dir.exists():
                continue
            dest_dir = CURATED_DIR / label
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img_path in src_dir.iterdir():
                if is_image(img_path) and "_red" in img_path.name:
                    # Use only Red images (most info for visual classification)
                    dest = dest_dir / f"field_{img_path.name}"
                    if not dest.exists():
                        shutil.copy2(img_path, dest)
                    if label == "healthy_vegetation":
                        healthy_count += 1
                    else:
                        stressed_count += 1

    return healthy_count, stressed_count


# ═══════════════════════════════════════════════════════════════════════════
# Augmentation — 5x (or Nx) data multiplication
# ═══════════════════════════════════════════════════════════════════════════

def augment_image(img, aug_id):
    """
    Apply a unique augmentation based on aug_id.
    Returns a new PIL Image.
    """
    w, h = img.size

    if aug_id == 0:
        # Horizontal flip + brightness shift
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))

    elif aug_id == 1:
        # Vertical flip + contrast shift
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.7, 1.4))

    elif aug_id == 2:
        # Random rotation (15-45°) + saturation shift
        angle = random.uniform(15, 45) * random.choice([-1, 1])
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(0.6, 1.5))

    elif aug_id == 3:
        # Random crop (80-95% of image) + slight blur
        crop_scale = random.uniform(0.80, 0.95)
        cw, ch = int(w * crop_scale), int(h * crop_scale)
        x = random.randint(0, w - cw)
        y = random.randint(0, h - ch)
        img = img.crop((x, y, x + cw, y + ch))
        if random.random() > 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    elif aug_id == 4:
        # Both flips + color jitter + slight rotation
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        angle = random.uniform(-10, 10)
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))

    else:
        # Extra augmentations for factor > 5: aggressive transforms
        # Random combination of multiple effects
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        angle = random.uniform(-30, 30)
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.5, 1.5))
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(0.3, 1.7))
        # Random crop
        crop_scale = random.uniform(0.70, 0.95)
        cw, ch = int(w * crop_scale), int(h * crop_scale)
        x = random.randint(0, max(1, w - cw))
        y = random.randint(0, max(1, h - ch))
        img = img.crop((x, y, x + cw, y + ch))

    # Always resize to standard size
    img = img.resize(IMAGE_SIZE, Image.BILINEAR)
    return img


def augment_dataset(curated_dir, factor=5):
    """
    Apply augmentation to all curated images.
    Creates augmented copies alongside originals.
    """
    print(f"\n  🔄 Augmenting dataset ({factor}x) …")

    for label in CLASSES:
        label_dir = curated_dir / label
        if not label_dir.exists():
            continue

        # Get only original (non-augmented) images
        originals = [f for f in label_dir.iterdir()
                     if is_image(f) and "_aug" not in f.name]
        count = 0

        for img_path in originals:
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            for aug_id in range(factor):
                aug_name = f"{img_path.stem}_aug{aug_id}{img_path.suffix}"
                aug_path = label_dir / aug_name
                if aug_path.exists():
                    continue
                try:
                    aug_img = augment_image(img, aug_id)
                    aug_img.save(aug_path, quality=90)
                    count += 1
                except Exception:
                    continue

        print(f"    {label}: +{count} augmented images")


# ═══════════════════════════════════════════════════════════════════════════
# Final Dataset Builder
# ═══════════════════════════════════════════════════════════════════════════

def balance_classes(curated_dir):
    """Balance classes by undersampling the larger class."""
    counts = {}
    for label in CLASSES:
        label_dir = curated_dir / label
        if label_dir.exists():
            files = [f for f in label_dir.iterdir() if is_image(f)]
            counts[label] = len(files)
        else:
            counts[label] = 0

    min_count = min(counts.values())
    print(f"\n  ⚖️  Balancing classes (min={min_count:,})")

    balanced = {}
    for label in CLASSES:
        label_dir = curated_dir / label
        files = sorted([f for f in label_dir.iterdir() if is_image(f)])
        random.shuffle(files)
        balanced[label] = files[:min_count]
        print(f"    {label}: {counts[label]:,} → {min_count:,}")

    return balanced


def build_train_val(balanced_files, output_dir, train_ratio=TRAIN_RATIO):
    """Split balanced files into train/val and copy to output."""
    # Clean output
    for split in ["train", "val"]:
        for label in CLASSES:
            d = output_dir / split / label
            d.mkdir(parents=True, exist_ok=True)

    print(f"\n  📁 Building train/val split ({train_ratio:.0%}/{1-train_ratio:.0%}) …")

    for label, files in balanced_files.items():
        random.shuffle(files)
        split_idx = int(len(files) * train_ratio)
        train_files = files[:split_idx]
        val_files = files[split_idx:]

        for f in train_files:
            dest = output_dir / "train" / label / f.name
            shutil.copy2(f, dest)

        for f in val_files:
            dest = output_dir / "val" / label / f.name
            shutil.copy2(f, dest)

        print(f"    {label}: train={len(train_files):,}  val={len(val_files):,}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Build large-scale training dataset")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading datasets (use existing)")
    parser.add_argument("--augment-factor", type=int, default=5,
                        help="Augmentation multiplier (default: 5x)")
    parser.add_argument("--no-augment", action="store_true",
                        help="Skip augmentation")
    parser.add_argument("--no-balance", action="store_true",
                        help="Don't balance classes")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Chroma-Shift — Large-Scale Dataset Builder               ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║   Sources: PlantVillage + PlantDoc + Field Captures        ║")
    print(f"║   Augmentation: {args.augment_factor}x                                        ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    CURATED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download datasets
    if not args.skip_download:
        print("\n  Step 1: Downloading public datasets …")
        pv_dir = download_plantvillage()
        pd_dir = download_plantdoc()
    else:
        pv_dir = DATASETS_DIR / "PlantVillage-Dataset"
        pd_dir = DATASETS_DIR / "PlantDoc-Dataset"
        print("\n  Step 1: Using existing downloads")

    # 2. Curate into binary classes
    print("\n  Step 2: Curating into healthy/stressed classes …")

    pv_h, pv_s = curate_plantvillage(pv_dir)
    print(f"    PlantVillage → healthy={pv_h:,}  stressed={pv_s:,}")

    pd_h, pd_s = curate_plantdoc(pd_dir)
    print(f"    PlantDoc     → healthy={pd_h:,}  stressed={pd_s:,}")

    fc_h, fc_s = curate_field_captures()
    print(f"    Field caps   → healthy={fc_h:,}  stressed={fc_s:,}")

    total_h = pv_h + pd_h + fc_h
    total_s = pv_s + pd_s + fc_s
    print(f"\n    TOTAL: healthy={total_h:,}  stressed={total_s:,}  combined={total_h+total_s:,}")

    # 3. Augment
    if not args.no_augment:
        augment_dataset(CURATED_DIR, factor=args.augment_factor)

    # 4. Balance classes
    if not args.no_balance:
        balanced = balance_classes(CURATED_DIR)
    else:
        balanced = {}
        for label in CLASSES:
            label_dir = CURATED_DIR / label
            balanced[label] = [f for f in label_dir.iterdir() if is_image(f)]

    # 5. Build train/val split
    # Back up old data dir
    old_data = DATA_DIR.parent / "data_field_only"
    if DATA_DIR.exists() and not old_data.exists():
        print(f"\n  💾 Backing up field-only dataset → {old_data}")
        shutil.copytree(DATA_DIR, old_data)

    # Clear and rebuild
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    build_train_val(balanced, DATA_DIR)

    # Final stats
    train_h = len(list((DATA_DIR / "train" / "healthy_vegetation").iterdir()))
    train_s = len(list((DATA_DIR / "train" / "stressed_soil").iterdir()))
    val_h = len(list((DATA_DIR / "val" / "healthy_vegetation").iterdir()))
    val_s = len(list((DATA_DIR / "val" / "stressed_soil").iterdir()))

    print(f"\n{'═' * 62}")
    print(f"  DATASET READY")
    print(f"  Train:  healthy={train_h:,}  stressed={train_s:,}  total={train_h+train_s:,}")
    print(f"  Val:    healthy={val_h:,}  stressed={val_s:,}  total={val_h+val_s:,}")
    print(f"  Grand total: {train_h+train_s+val_h+val_s:,} images")
    print(f"{'═' * 62}\n")


if __name__ == "__main__":
    main()

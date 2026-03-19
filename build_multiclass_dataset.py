#!/usr/bin/env python3
"""
build_multiclass_dataset.py — Expand 2-class dataset to 4 classes
==================================================================
Auto-splits existing Chroma-Shift training data using NDVI values
embedded in filenames, and downloads EuroSAT for non-vegetation class.

Classes:
    0: healthy_vegetation  — NDVI > 0.35, green active plants
    1: stressed_vegetation — NDVI 0.15-0.35, wilting/yellow plants
    2: bare_soil           — NDVI 0.05-0.20, exposed dirt
    3: non_vegetation      — concrete, asphalt, water (from EuroSAT)

Run on the desktop (RTX 4070 machine):
    cd Chroma-Shift
    python build_multiclass_dataset.py
    python train_finetune_gpu.py --data-dir data_4class --num-classes 4

Author: Chroma-Shift Team
"""

import os
import re
import sys
import shutil
import random
import hashlib
import argparse
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from collections import Counter

# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
OLD_DATA_DIR = SCRIPT_DIR / "data"
NEW_DATA_DIR = SCRIPT_DIR / "data_4class"
EUROSAT_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"
EUROSAT_DIR = SCRIPT_DIR / "eurosat_raw"

CLASSES_4 = [
    "healthy_vegetation",
    "stressed_vegetation",
    "bare_soil",
    "non_vegetation",
]

# EuroSAT classes to use for non_vegetation
EUROSAT_NON_VEG_CLASSES = [
    "Industrial",
    "Residential",
    "Highway",
    "River",
    "SeaLake",
    "AnnualCrop",  # bare fields between seasons
]

# NDVI thresholds for auto-splitting
NDVI_HEALTHY_THRESHOLD = 0.30     # Above = healthy_vegetation
NDVI_STRESSED_LOW = 0.15          # 0.15 - 0.30 = stressed_vegetation
NDVI_BARE_SOIL_CUTOFF = 0.15     # Below = bare_soil

# Target samples per class (for balance)
TARGET_PER_CLASS = 3000
VAL_RATIO = 0.15

random.seed(42)


# ═══════════════════════════════════════════════════════════════════════════
# NDVI extraction from filename
# ═══════════════════════════════════════════════════════════════════════════

def extract_ndvi(filename):
    """Extract NDVI from filename like 'ndvi0.2038_red_aug02.jpg'."""
    match = re.search(r'ndvi([\-]?[\d]+\.[\d]+)', filename)
    if match:
        return float(match.group(1))
    return None


def classify_by_ndvi(ndvi_value, current_class):
    """Determine the 4-class label based on NDVI and original class."""
    if current_class == "healthy_vegetation":
        if ndvi_value is not None and ndvi_value < NDVI_STRESSED_LOW:
            return "bare_soil"
        elif ndvi_value is not None and ndvi_value < NDVI_HEALTHY_THRESHOLD:
            return "stressed_vegetation"
        return "healthy_vegetation"
    else:  # stressed_soil
        if ndvi_value is not None and ndvi_value < NDVI_BARE_SOIL_CUTOFF:
            return "bare_soil"
        elif ndvi_value is not None and ndvi_value >= NDVI_HEALTHY_THRESHOLD:
            return "stressed_vegetation"  # moderate NDVI but visually stressed
        return "stressed_vegetation"


# ═══════════════════════════════════════════════════════════════════════════
# Dataset operations
# ═══════════════════════════════════════════════════════════════════════════

def split_existing_data():
    """Reclassify existing 2-class data into 4 classes using NDVI values."""
    print("\n[1/4] Reclassifying existing data using NDVI thresholds …")
    
    classified = {cls: [] for cls in CLASSES_4}
    stats = Counter()
    no_ndvi_count = 0
    
    for split in ["train", "val"]:
        split_dir = OLD_DATA_DIR / split
        if not split_dir.exists():
            print(f"  ⚠️  {split_dir} not found, skipping")
            continue
            
        for old_class in ["healthy_vegetation", "stressed_soil"]:
            class_dir = split_dir / old_class
            if not class_dir.exists():
                continue
                
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                    continue
                    
                ndvi = extract_ndvi(img_path.name)
                if ndvi is None:
                    # No NDVI in filename — keep in original class
                    new_class = "healthy_vegetation" if old_class == "healthy_vegetation" else "stressed_vegetation"
                    no_ndvi_count += 1
                else:
                    new_class = classify_by_ndvi(ndvi, old_class)
                
                classified[new_class].append(img_path)
                stats[f"{split}/{old_class} → {new_class}"] += 1
    
    print(f"\n  Reclassification results:")
    for transition, count in sorted(stats.items()):
        print(f"    {transition}: {count}")
    if no_ndvi_count > 0:
        print(f"    (no NDVI in filename: {no_ndvi_count} images, kept in original class)")
    
    for cls, files in classified.items():
        print(f"  {cls}: {len(files)} images")
    
    return classified


def download_eurosat():
    """Download EuroSAT RGB dataset for non-vegetation images."""
    print("\n[2/4] Downloading EuroSAT for non-vegetation class …")
    
    zip_path = SCRIPT_DIR / "EuroSAT_RGB.zip"
    
    if EUROSAT_DIR.exists() and any(EUROSAT_DIR.iterdir()):
        print(f"  EuroSAT already at {EUROSAT_DIR}, skipping download")
    elif zip_path.exists():
        print(f"  Extracting {zip_path} …")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(EUROSAT_DIR)
        print(f"  ✅ Extracted")
    else:
        print(f"  Downloading from Zenodo ({EUROSAT_URL}) …")
        print(f"  This is ~90MB, one-time download.")
        try:
            urlretrieve(EUROSAT_URL, zip_path)
            print(f"  ✅ Downloaded. Extracting …")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(EUROSAT_DIR)
            print(f"  ✅ Extracted to {EUROSAT_DIR}")
        except Exception as e:
            print(f"  ⚠️  Download failed: {e}")
            print(f"  You can manually download from: {EUROSAT_URL}")
            print(f"  Place the zip at: {zip_path}")
            return []
    
    # Find the actual data directory (may be nested)
    eurosat_data = None
    for root, dirs, files in os.walk(EUROSAT_DIR):
        if any(d in dirs for d in EUROSAT_NON_VEG_CLASSES):
            eurosat_data = Path(root)
            break
    
    if not eurosat_data:
        print(f"  ⚠️  Could not find EuroSAT class directories")
        return []
    
    # Collect non-vegetation images
    non_veg_images = []
    for cls in EUROSAT_NON_VEG_CLASSES:
        cls_dir = eurosat_data / cls
        if cls_dir.exists():
            imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.tif"))
            non_veg_images.extend(imgs)
            print(f"  {cls}: {len(imgs)} images")
    
    random.shuffle(non_veg_images)
    print(f"  Total non-vegetation: {len(non_veg_images)} images")
    return non_veg_images


def build_dataset(classified, non_veg_images, target_per_class=TARGET_PER_CLASS):
    """Build balanced 4-class dataset with train/val split."""
    print(f"\n[3/4] Building balanced dataset (target: {target_per_class}/class) …")
    
    # Add non-vegetation
    classified["non_vegetation"].extend(non_veg_images)
    
    # Create output directories
    for split in ["train", "val"]:
        for cls in CLASSES_4:
            (NEW_DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    
    total_train = 0
    total_val = 0
    
    for cls in CLASSES_4:
        images = classified[cls]
        random.shuffle(images)
        
        # Cap at target_per_class
        if len(images) > target_per_class:
            images = images[:target_per_class]
        
        val_count = max(1, int(len(images) * VAL_RATIO))
        val_images = images[:val_count]
        train_images = images[val_count:]
        
        # Copy to output (Windows long-path safe)
        def safe_copy(src, dst):
            """Copy with Windows \\\\?\\ prefix for paths > 260 chars."""
            s, d = str(src.resolve()), str(dst.resolve())
            if os.name == 'nt':
                if not s.startswith('\\\\?\\'):
                    s = '\\\\?\\' + s
                if not d.startswith('\\\\?\\'):
                    d = '\\\\?\\' + d
            shutil.copy2(s, d)

        for img in train_images:
            dst = NEW_DATA_DIR / "train" / cls / img.name
            if not dst.exists():
                safe_copy(img, dst)
        
        for img in val_images:
            dst = NEW_DATA_DIR / "val" / cls / img.name
            if not dst.exists():
                safe_copy(img, dst)
        
        print(f"  {cls}: {len(train_images)} train + {len(val_images)} val = {len(images)}")
        total_train += len(train_images)
        total_val += len(val_images)
    
    print(f"\n  Total: {total_train} train + {total_val} val = {total_train + total_val}")
    return total_train, total_val


def verify_dataset():
    """Verify the built dataset."""
    print("\n[4/4] Verifying dataset …")
    
    for split in ["train", "val"]:
        split_dir = NEW_DATA_DIR / split
        print(f"\n  {split}/")
        for cls in CLASSES_4:
            cls_dir = split_dir / cls
            if cls_dir.exists():
                count = len(list(cls_dir.iterdir()))
                print(f"    {cls}: {count}")
            else:
                print(f"    {cls}: ⚠️ MISSING")
    
    print(f"\n  ✅ Dataset ready at: {NEW_DATA_DIR}")
    print(f"\n  Next step:")
    print(f"    python train_finetune_gpu.py --data-dir data_4class --num-classes 4")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Build 4-class dataset from existing data")
    parser.add_argument("--target", type=int, default=TARGET_PER_CLASS,
                       help=f"Target samples per class (default: {TARGET_PER_CLASS})")
    parser.add_argument("--skip-eurosat", action="store_true",
                       help="Skip EuroSAT download (use only existing data)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show reclassification stats without copying files")
    args = parser.parse_args()
    
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Chroma-Shift — Build 4-Class Dataset                     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    # Step 1: Split existing data by NDVI
    classified = split_existing_data()
    
    if args.dry_run:
        print("\n  Dry run complete. No files copied.")
        return
    
    # Step 2: Download EuroSAT for non-vegetation
    non_veg_images = []
    if not args.skip_eurosat:
        non_veg_images = download_eurosat()
    
    # Step 3: Build balanced dataset
    build_dataset(classified, non_veg_images, args.target)
    
    # Step 4: Verify
    verify_dataset()


if __name__ == "__main__":
    main()

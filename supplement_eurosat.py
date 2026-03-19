#!/usr/bin/env python3
"""Supplement bare_soil + stressed_vegetation with EuroSAT images."""
import shutil, os, random
from pathlib import Path

random.seed(42)
d4 = Path("data_4class")

# Find EuroSAT root
eurosat_root = None
for root, dirs, files in os.walk("eurosat_raw"):
    if "Forest" in dirs:
        eurosat_root = Path(root)
        break
assert eurosat_root, "EuroSAT root not found"
print(f"EuroSAT root: {eurosat_root}")

# EuroSAT -> our class mapping (only classes that need supplementing)
eurosat_map = {
    "PermanentCrop": "stressed_vegetation",
    "AnnualCrop": "bare_soil",
}

TARGET = 3000
VAL_RATIO = 0.15


def win_path(p):
    s = str(Path(p).resolve())
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + s
    return s


for eurosat_cls, our_cls in eurosat_map.items():
    src_dir = eurosat_root / eurosat_cls
    if not src_dir.exists():
        print(f"  SKIP {eurosat_cls}: not found")
        continue

    imgs = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.tif"))
    random.shuffle(imgs)

    train_dir = d4 / "train" / our_cls
    val_dir = d4 / "val" / our_cls
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    existing_train = len(list(train_dir.iterdir()))
    existing_val = len(list(val_dir.iterdir()))
    existing = existing_train + existing_val

    needed = max(0, TARGET - existing)
    if needed == 0:
        print(f"  {eurosat_cls} -> {our_cls}: already at {existing}, skip")
        continue

    imgs = imgs[:needed]
    val_count = max(1, int(len(imgs) * VAL_RATIO))
    val_imgs = imgs[:val_count]
    train_imgs = imgs[val_count:]

    copied = 0
    for img in train_imgs:
        dst = train_dir / img.name
        if not dst.exists():
            shutil.copy2(win_path(img), win_path(dst))
            copied += 1
    for img in val_imgs:
        dst = val_dir / img.name
        if not dst.exists():
            shutil.copy2(win_path(img), win_path(dst))
            copied += 1

    print(f"  {eurosat_cls} -> {our_cls}: added {copied} (was {existing}, now {existing + copied})")

# Final tally
print("\n=== Final Dataset ===")
for split in ["train", "val"]:
    for cls in ["healthy_vegetation", "stressed_vegetation", "bare_soil", "non_vegetation"]:
        p = d4 / split / cls
        c = len(list(p.iterdir())) if p.exists() else 0
        print(f"  {split}/{cls}: {c}")

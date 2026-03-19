#!/usr/bin/env python3
"""Remove all images with paths > 250 chars from data_4class."""
import os
from pathlib import Path

count = 0
for split in ["train", "val"]:
    split_dir = Path("data_4class") / split
    if not split_dir.exists():
        continue
    for cls_dir in split_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        for img in list(cls_dir.iterdir()):
            full = str(img.resolve())
            if len(full) > 250:
                win_path = "\\\\?\\" + full
                try:
                    os.remove(win_path)
                    count += 1
                    print(f"Removed: {img.name[:80]}...")
                except Exception as e:
                    print(f"Failed: {img.name[:80]}... — {e}")

print(f"\nRemoved {count} long-path files")

# Quick verification
from PIL import Image
bad = 0
for split in ["train", "val"]:
    split_dir = Path("data_4class") / split
    if not split_dir.exists():
        continue
    for cls_dir in split_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        for img in cls_dir.iterdir():
            try:
                im = Image.open(img)
                im.verify()
            except Exception as e:
                bad += 1
                try:
                    os.remove("\\\\?\\" + str(img.resolve()))
                    print(f"Removed bad image: {img.name[:80]}...")
                except:
                    print(f"Bad but can't remove: {img.name[:80]}...")

print(f"Removed {bad} corrupt images")

# Final tally
print("\n=== Final Dataset ===")
for split in ["train", "val"]:
    for cls in ["healthy_vegetation", "stressed_vegetation", "bare_soil", "non_vegetation"]:
        p = Path("data_4class") / split / cls
        c = len(list(p.iterdir())) if p.exists() else 0
        print(f"  {split}/{cls}: {c}")

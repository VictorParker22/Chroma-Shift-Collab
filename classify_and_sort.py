"""
classify_and_sort.py — Chroma-Shift NDVI Classification & Dataset Builder
===========================================================================
Computes NDVI from captured Red / NIR image pairs and automatically sorts
them into the TAO folder-per-class directory structure for DINOv2 training.

NDVI Threshold: 0.6493
  Derived from 296 clear-sky HLS satellite observations over NYC Central
  Park (2019–2023), combining Landsat + Sentinel-2 sources.

Classification:
  NDVI ≥ 0.6493  →  healthy_vegetation
  NDVI < 0.6493  →  stressed_soil

Usage:
    # Process a single pair
    python classify_and_sort.py --red captures/sample_red.png --nir captures/sample_nir.png

    # Process all pairs in the captures/ directory automatically
    python classify_and_sort.py --batch

    # Full pipeline: capture + compute + sort (requires camera + EFW)
    python classify_and_sort.py --capture --name field_sample_01
"""

import os
import sys
import glob
import shutil
import argparse
import re
from datetime import datetime

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# NDVI threshold for classification
# Summer satellite baseline = 0.6493 (296 HLS observations, 2019-2023)
# Seasonal field threshold = 0.25 (median of March 2026 field captures)
# Using seasonal threshold to produce balanced train/val splits
NDVI_THRESHOLD = 0.25

# TAO dataset directories
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
CAPTURES_DIR = os.path.join(SCRIPT_DIR, "captures")

# Validation split ratio (fraction of images that go to val/)
VAL_SPLIT = 0.2

# Class folder names (must match TAO spec)
CLASS_HEALTHY = "healthy_vegetation"
CLASS_STRESSED = "stressed_soil"


# ═══════════════════════════════════════════════════════════════════════════
# NDVI Computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_ndvi(red_path, nir_path):
    """
    Compute per-pixel NDVI from a Red and NIR image pair.

    NDVI = (NIR - Red) / (NIR + Red)

    Returns:
        ndvi_map  : 2D float32 array, values in [-1, 1]
        mean_ndvi : scalar, mean NDVI across all valid pixels
    """
    red_img = cv2.imread(red_path, cv2.IMREAD_GRAYSCALE)
    nir_img = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)

    if red_img is None:
        sys.exit(f"ERROR: Cannot read Red image: {red_path}")
    if nir_img is None:
        sys.exit(f"ERROR: Cannot read NIR image: {nir_path}")

    if red_img.shape != nir_img.shape:
        sys.exit(f"ERROR: Image dimensions don't match: "
                 f"Red={red_img.shape} vs NIR={nir_img.shape}")

    # Convert to float32 for arithmetic
    red = red_img.astype(np.float32)
    nir = nir_img.astype(np.float32)

    # Compute NDVI, avoiding division by zero
    denominator = nir + red
    ndvi_map = np.where(
        denominator > 0,
        (nir - red) / denominator,
        0.0
    ).astype(np.float32)

    # Clamp to [-1, 1]
    ndvi_map = np.clip(ndvi_map, -1.0, 1.0)

    # Mean NDVI (exclude zero-denominator pixels)
    valid_mask = denominator > 0
    if valid_mask.any():
        mean_ndvi = float(ndvi_map[valid_mask].mean())
    else:
        mean_ndvi = 0.0

    return ndvi_map, mean_ndvi


def classify_ndvi(mean_ndvi, threshold=NDVI_THRESHOLD):
    """Classify a mean NDVI value into a vegetation class."""
    if mean_ndvi >= threshold:
        return CLASS_HEALTHY
    else:
        return CLASS_STRESSED


def save_ndvi_visualization(ndvi_map, output_path):
    """
    Save a false-color NDVI visualization.
    Red = stressed/bare soil, Green = healthy vegetation.
    """
    # Normalize from [-1, 1] to [0, 255]
    normalized = ((ndvi_map + 1.0) / 2.0 * 255).astype(np.uint8)
    # Apply colormap (COLORMAP_RdYlGn: red→yellow→green)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    cv2.imwrite(output_path, colored)


# ═══════════════════════════════════════════════════════════════════════════
# Dataset Sorting
# ═══════════════════════════════════════════════════════════════════════════

_sample_counter = {}  # track per-class counts for train/val split


def sort_to_dataset(red_path, nir_path, ndvi_map, mean_ndvi, sample_name):
    """
    Copy the Red+NIR pair into the correct TAO class folder.
    Automatically splits between train/ and val/ based on VAL_SPLIT.
    Also saves an NDVI visualization alongside the images.
    """
    label = classify_ndvi(mean_ndvi)

    # Decide train vs val
    if label not in _sample_counter:
        _sample_counter[label] = 0
    _sample_counter[label] += 1
    count = _sample_counter[label]

    # Every Nth sample goes to val (e.g., every 5th for 20% split)
    val_every = max(1, int(1.0 / VAL_SPLIT)) if VAL_SPLIT > 0 else 0
    if val_every > 0 and count % val_every == 0:
        split = "val"
        dest_dir = os.path.join(VAL_DIR, label)
    else:
        split = "train"
        dest_dir = os.path.join(TRAIN_DIR, label)

    os.makedirs(dest_dir, exist_ok=True)

    # Build filenames with NDVI baked in for traceability
    ndvi_str = f"{mean_ndvi:.4f}"
    base = f"{sample_name}_ndvi{ndvi_str}"

    red_dest = os.path.join(dest_dir, f"{base}_red.png")
    nir_dest = os.path.join(dest_dir, f"{base}_nir.png")
    viz_dest = os.path.join(dest_dir, f"{base}_viz.png")

    shutil.copy2(red_path, red_dest)
    shutil.copy2(nir_path, nir_dest)
    save_ndvi_visualization(ndvi_map, viz_dest)

    return label, split, dest_dir


# ═══════════════════════════════════════════════════════════════════════════
# Pair Discovery (batch mode)
# ═══════════════════════════════════════════════════════════════════════════

def find_pairs(captures_dir=CAPTURES_DIR):
    """
    Discover Red/NIR image pairs in the captures directory.
    Pairs are matched by having identical prefixes with
    _red_665nm.png and _nir_865nm.png suffixes.
    """
    red_files = sorted(glob.glob(os.path.join(captures_dir, "*_red_665nm.png")))
    pairs = []
    for red_path in red_files:
        nir_path = red_path.replace("_red_665nm.png", "_nir_865nm.png")
        if os.path.isfile(nir_path):
            # Extract sample name from filename
            basename = os.path.basename(red_path)
            sample_name = basename.replace("_red_665nm.png", "")
            pairs.append((sample_name, red_path, nir_path))
        else:
            print(f"  ⚠️  No NIR match for: {os.path.basename(red_path)}")
    return pairs


# ═══════════════════════════════════════════════════════════════════════════
# Processing Core
# ═══════════════════════════════════════════════════════════════════════════

def process_pair(sample_name, red_path, nir_path, sort=True):
    """Process a single Red+NIR pair: compute NDVI, classify, and sort."""
    ndvi_map, mean_ndvi = compute_ndvi(red_path, nir_path)
    label = classify_ndvi(mean_ndvi)

    print(f"  {sample_name}")
    print(f"    NDVI mean : {mean_ndvi:.4f}")
    print(f"    Class     : {label}", end="")

    if sort:
        label, split, dest = sort_to_dataset(
            red_path, nir_path, ndvi_map, mean_ndvi, sample_name
        )
        print(f"  →  {split}/{label}")
    else:
        print()

    return mean_ndvi, label


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compute NDVI from Red/NIR pairs and sort into TAO dataset."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--batch", action="store_true",
        help="Process all Red/NIR pairs found in captures/ directory."
    )
    group.add_argument(
        "--red", type=str,
        help="Path to a single Red (665nm) image."
    )
    group.add_argument(
        "--capture", action="store_true",
        help="Capture a new pair then classify (requires camera + EFW)."
    )

    parser.add_argument("--nir", type=str, help="Path to the NIR (865nm) image (use with --red).")
    parser.add_argument("--name", "-n", type=str, default=None, help="Sample name (for --capture mode).")
    parser.add_argument("--no-sort", action="store_true", help="Only compute NDVI; don't copy to dataset folders.")

    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Chroma-Shift — NDVI Classification & Dataset Builder     ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║   Threshold: {NDVI_THRESHOLD:.4f} (Central Park ground-truth)       ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    do_sort = not args.no_sort

    # ----- Single pair mode -----
    if args.red:
        if not args.nir:
            parser.error("--nir is required when using --red")
        sample_name = os.path.basename(args.red).replace("_red_665nm.png", "").replace(".png", "")
        print()
        process_pair(sample_name, args.red, args.nir, sort=do_sort)

    # ----- Batch mode -----
    elif args.batch:
        pairs = find_pairs()
        if not pairs:
            print("\n  No Red/NIR pairs found in captures/")
            return

        print(f"\n  Found {len(pairs)} pair(s) in captures/\n")
        stats = {CLASS_HEALTHY: 0, CLASS_STRESSED: 0}

        for sample_name, red_path, nir_path in pairs:
            _, label = process_pair(sample_name, red_path, nir_path, sort=do_sort)
            stats[label] += 1

        print(f"\n  Summary:")
        print(f"    {CLASS_HEALTHY}: {stats[CLASS_HEALTHY]}")
        print(f"    {CLASS_STRESSED}: {stats[CLASS_STRESSED]}")

    # ----- Capture + classify mode -----
    elif args.capture:
        # Import the capture module
        from capture_multispectral_pair import (
            open_camera, flush_and_capture, FilterWheel,
            FILTER_SLOT_RED, FILTER_SLOT_NIR, CAPTURES_DIR as CAP_DIR,
        )

        sample_name = args.name or input("  Enter sample name: ").strip() or "sample"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print("\n  Initialising hardware …")
        cap = open_camera()
        efw = FilterWheel()
        efw.open()

        try:
            os.makedirs(CAP_DIR, exist_ok=True)

            # Capture Red
            efw.move_to(FILTER_SLOT_RED)
            frame_red = flush_and_capture(cap)
            red_path = os.path.join(CAP_DIR, f"{sample_name}_{timestamp}_red_665nm.png")
            cv2.imwrite(red_path, frame_red)
            print(f"  Saved Red  → {red_path}")

            # Capture NIR
            efw.move_to(FILTER_SLOT_NIR)
            frame_nir = flush_and_capture(cap)
            nir_path = os.path.join(CAP_DIR, f"{sample_name}_{timestamp}_nir_865nm.png")
            cv2.imwrite(nir_path, frame_nir)
            print(f"  Saved NIR  → {nir_path}")

            # Classify & sort
            print()
            process_pair(f"{sample_name}_{timestamp}", red_path, nir_path, sort=do_sort)

        finally:
            cap.release()
            efw.close()
            print("\n  Hardware released.")

    # Final dataset stats
    if do_sort:
        print(f"\n  Dataset status:")
        for split in ["train", "val"]:
            for cls in [CLASS_HEALTHY, CLASS_STRESSED]:
                d = os.path.join(DATA_DIR, split, cls)
                count = len(glob.glob(os.path.join(d, "*_red.png"))) if os.path.isdir(d) else 0
                print(f"    {split}/{cls}: {count} pairs")
        print()


if __name__ == "__main__":
    main()

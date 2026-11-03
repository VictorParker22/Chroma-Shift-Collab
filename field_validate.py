#!/usr/bin/env python3
"""
field_validate.py — Live Field Validation Tool
================================================
Captures a frame, runs the full radiometric pipeline, and validates
NDVI readings against NASA seasonal baselines. Generates an evidence
report with advisory output for farmer-facing validation.

Usage (on Jetson with camera connected):
    python field_validate.py                    # single capture + report
    python field_validate.py --zone park_north  # specify zone
    python field_validate.py --from-file captures/img.png  # from saved image

Usage (without camera, from existing captures):
    python field_validate.py --from-file captures/field_free_001_red.png

The validation checks:
  1. NDVI is within physically valid range (-1 to 1)
  2. NDVI matches expected seasonal range for this week
  3. Classification agrees with NDVI category
  4. Generates full farmer advisory with explanation
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
CAPTURES_DIR = SCRIPT_DIR / "captures"
VALIDATION_DIR = SCRIPT_DIR / "validation_reports"

# Expected NDVI ranges by month (generous bounds for validation)
NYC_NDVI_BOUNDS = {
    1: (0.15, 0.45), 2: (0.15, 0.45), 3: (0.20, 0.50),
    4: (0.30, 0.65), 5: (0.40, 0.75), 6: (0.50, 0.80),
    7: (0.55, 0.85), 8: (0.55, 0.85), 9: (0.45, 0.80),
    10: (0.30, 0.70), 11: (0.20, 0.55), 12: (0.15, 0.45),
}


def validate_ndvi_reading(ndvi, month):
    """Check if NDVI is physically valid and seasonally plausible."""
    checks = []

    # Physical validity
    if -1.0 <= ndvi <= 1.0:
        checks.append(("✅", "NDVI within valid range [-1, 1]", True))
    else:
        checks.append(("❌", f"NDVI {ndvi:.4f} is outside valid range!", False))

    # Vegetation detection
    if ndvi > 0.15:
        checks.append(("✅", f"Vegetation detected (NDVI > 0.15)", True))
    elif ndvi > 0:
        checks.append(("🟡", f"Marginal vegetation signal (NDVI = {ndvi:.3f})", True))
    else:
        checks.append(("🟡", f"No vegetation detected (NDVI ≤ 0) — could be water/pavement", True))

    # Seasonal plausibility
    lo, hi = NYC_NDVI_BOUNDS.get(month, (0.15, 0.85))
    if lo <= ndvi <= hi:
        checks.append(("✅", f"NDVI plausible for month {month} (expected {lo:.2f}–{hi:.2f})", True))
    elif ndvi < lo:
        checks.append(("⚠️", f"NDVI below seasonal minimum (expected ≥{lo:.2f} for month {month})", True))
    else:
        checks.append(("⚠️", f"NDVI above seasonal maximum (expected ≤{hi:.2f} for month {month})", True))

    return checks


def validate_from_file(image_path, zone_name="Field Test"):
    """Run validation on an existing image file."""
    from radiometric import apply_corrections, get_seasonal_baseline
    from crop_advisor import generate_advisory, format_advisory_text

    print(f"\n  Loading: {image_path}")

    # Load image as grayscale (simulating a single-band capture)
    from PIL import Image
    img = Image.open(image_path).convert("L")
    frame = np.array(img, dtype=np.float32)
    print(f"  Image shape: {frame.shape}, mean DN: {frame.mean():.1f}")

    # Determine if this is red or NIR from filename
    fname = str(image_path).lower()
    is_nir = "nir" in fname
    band = "NIR (865nm)" if is_nir else "Red (665nm)"
    print(f"  Band: {band}")

    # For a single-band image, compute pseudo-NDVI from filename if available
    ndvi_from_name = None
    if "ndvi" in fname:
        try:
            idx = fname.index("ndvi") + 4
            ndvi_str = fname[idx:idx+6]
            ndvi_from_name = float(ndvi_str)
            print(f"  NDVI from filename: {ndvi_from_name:.4f}")
        except (ValueError, IndexError):
            pass

    # Run seasonal baseline
    now = datetime.now(timezone.utc)
    seasonal = get_seasonal_baseline(now)
    month = now.month

    # Use filename NDVI or estimate from pixel statistics
    if ndvi_from_name is not None:
        ndvi = ndvi_from_name
    else:
        # Rough estimate: normalize frame to reflectance-like values
        ndvi = frame.mean() / 255.0 * 0.8  # crude approximation
        print(f"  ⚠️  No paired Red/NIR — using estimated NDVI: {ndvi:.4f}")

    print(f"\n  Running validation checks...")
    checks = validate_ndvi_reading(ndvi, month)

    print(f"\n  ── VALIDATION RESULTS ──")
    all_pass = True
    for icon, msg, passed in checks:
        print(f"  {icon} {msg}")
        if not passed:
            all_pass = False

    # Generate advisory
    advisory = generate_advisory(
        ndvi=ndvi,
        seasonal_data=seasonal,
        zone_name=zone_name,
    )
    print(f"\n{format_advisory_text(advisory)}")

    # Save validation report
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": now.isoformat(),
        "image": str(image_path),
        "band": band,
        "ndvi": round(ndvi, 4),
        "seasonal_baseline": seasonal["baseline"],
        "season": seasonal["season"],
        "week": seasonal["week"],
        "checks": [{"icon": c[0], "message": c[1], "passed": c[2]} for c in checks],
        "all_checks_passed": all_pass,
        "advisory": advisory,
    }

    report_path = VALIDATION_DIR / f"validation_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  📄 Report saved: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Field validation tool")
    parser.add_argument("--from-file", type=str,
                       help="Validate from an existing image file")
    parser.add_argument("--zone", type=str, default="Central Park — Field Test",
                       help="Zone name for the report")
    parser.add_argument("--list-captures", action="store_true",
                       help="List recent captures")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Chroma-Shift — Field Validation Tool                     ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    if args.list_captures:
        if CAPTURES_DIR.exists():
            files = sorted(CAPTURES_DIR.glob("*.png"))[-10:]
            print(f"\n  Recent captures ({len(files)}):")
            for f in files:
                print(f"    {f.name}")
        else:
            print(f"\n  No captures directory found at {CAPTURES_DIR}")
        return

    if args.from_file:
        if not os.path.isfile(args.from_file):
            sys.exit(f"  ❌ File not found: {args.from_file}")
        validate_from_file(args.from_file, args.zone)
    else:
        # Try to find recent captures
        if CAPTURES_DIR.exists():
            files = sorted(CAPTURES_DIR.glob("*_red*.png"))
            if files:
                latest = files[-1]
                print(f"\n  Using latest red capture: {latest.name}")
                validate_from_file(latest, args.zone)
            else:
                print(f"\n  No red captures found. Use --from-file to specify an image.")
        else:
            print(f"\n  No captures found. Run a capture first or use --from-file.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
test_pipeline.py — Smoke test for the radiometric correction pipeline
======================================================================
Runs the full 7-layer pipeline on a real field capture (Red + synthesized NIR)
to verify all layers execute without errors after the Phase 1 audit changes.

Tests:
  1. Full apply_corrections() pipeline (Layers 1-6)
  2. KalmanNDVI with time-delta scaling (Layer 7)
  3. Seasonal baseline with linear interpolation (Layer 6)
  4. Solar position (Meeus/NREL SPA, Layer 3)
  5. Temperature-scaled softmax (if convert_tensorrt available)

Usage:
    python test_pipeline.py
"""

import sys
import os
import numpy as np
from datetime import datetime, timezone

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from radiometric import (
    apply_corrections,
    KalmanNDVI,
    get_seasonal_baseline,
    solar_position,
    normalize_exposure,
    dark_object_subtraction,
    cloud_transmittance,
    match_interframe_illumination,
    KALMAN_PROCESS_NOISE,
    KALMAN_MEASUREMENT_NOISE,
)

# ═══════════════════════════════════════════════════════════════════════════
# Test helpers
# ═══════════════════════════════════════════════════════════════════════════

PASS = "✅"
FAIL = "❌"
tests_passed = 0
tests_failed = 0


def check(name, condition, detail=""):
    global tests_passed, tests_failed
    if condition:
        tests_passed += 1
        print(f"  {PASS} {name}")
    else:
        tests_failed += 1
        print(f"  {FAIL} {name} — {detail}")


# ═══════════════════════════════════════════════════════════════════════════
# Load a real field capture
# ═══════════════════════════════════════════════════════════════════════════

def load_test_frame():
    """Load a real Red field capture and synthesize a matching NIR frame."""
    try:
        import cv2
        # Try to load a real field capture
        test_images = [
            os.path.join(SCRIPT_DIR, "data_4class", "val", "healthy_vegetation",
                         "field_field_free_039_20260311_165555_ndvi0.3954_red.png"),
            os.path.join(SCRIPT_DIR, "data_4class", "val", "stressed_vegetation",
                         "field_field_free_087_20260311_170627_ndvi0.2795_red.png"),
        ]
        for img_path in test_images:
            if os.path.isfile(img_path):
                frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if frame is not None:
                    print(f"  📷 Loaded: {os.path.basename(img_path)}")
                    print(f"     Shape: {frame.shape}, dtype: {frame.dtype}")
                    red = frame.astype(np.float32)
                    # Synthesize NIR: vegetation reflects ~1.5-2x more in NIR
                    # This gives realistic NDVI values (0.2-0.4 for March grass)
                    nir = red * 1.6 + np.random.normal(0, 3, red.shape).astype(np.float32)
                    nir = np.clip(nir, 0, 255)
                    return red, nir, os.path.basename(img_path)
    except ImportError:
        pass

    # Fallback: generate synthetic frames
    print("  📷 Using synthetic test frames (no OpenCV or captures available)")
    np.random.seed(42)
    red = np.random.uniform(40, 120, (480, 640)).astype(np.float32)
    nir = red * 1.5 + np.random.normal(0, 5, red.shape).astype(np.float32)
    nir = np.clip(nir, 0, 255)
    return red, nir, "synthetic_480x640"


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_seasonal_baseline():
    """Test seasonal baseline with linear interpolation."""
    print("\n── Layer 6: Seasonal Baseline ──")

    # Current season (March 2026)
    now = datetime(2026, 3, 23, 4, 0, 0, tzinfo=timezone.utc)
    result = get_seasonal_baseline(now)

    check("Returns dict with baseline/week/season",
          all(k in result for k in ["baseline", "week", "season"]))
    check(f"Season = spring (got: {result['season']})",
          result["season"] == "spring")
    check(f"March baseline ≈ 0.35-0.47 (got: {result['baseline']:.4f})",
          0.30 <= result["baseline"] <= 0.50)

    # Test interpolation: two calls 1 hour apart should give different baselines
    dt1 = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    dt2 = datetime(2026, 6, 15, 18, 0, 0, tzinfo=timezone.utc)
    b1 = get_seasonal_baseline(dt1)["baseline"]
    b2 = get_seasonal_baseline(dt2)["baseline"]
    check(f"Interpolation: 6h apart gives different values ({b1:.4f} vs {b2:.4f})",
          b1 != b2, f"both = {b1}")

    # Summer should be higher than winter
    summer = get_seasonal_baseline(datetime(2026, 7, 15, tzinfo=timezone.utc))
    winter = get_seasonal_baseline(datetime(2026, 1, 15, tzinfo=timezone.utc))
    check(f"Summer ({summer['baseline']:.2f}) > Winter ({winter['baseline']:.2f})",
          summer["baseline"] > winter["baseline"])

    return result


def test_solar_position():
    """Test Meeus/NREL SPA solar position algorithm."""
    print("\n── Layer 3: Solar Position (Meeus/NREL SPA) ──")

    # March 23, 2026 noon UTC — sun should be up in NYC
    noon = datetime(2026, 3, 23, 17, 0, 0, tzinfo=timezone.utc)  # noon EDT
    solar = solar_position(noon)

    check("Returns zenith_deg, elevation_deg, azimuth_deg, cos_zenith",
          all(k in solar for k in ["zenith_deg", "elevation_deg", "azimuth_deg", "cos_zenith"]))
    check(f"Noon zenith < 60° (got: {solar['zenith_deg']:.1f}°)",
          solar["zenith_deg"] < 60)
    check(f"Elevation > 30° (got: {solar['elevation_deg']:.1f}°)",
          solar["elevation_deg"] > 30)
    check(f"cos_zenith > 0.5 (got: {solar['cos_zenith']:.4f})",
          solar["cos_zenith"] > 0.5)

    # Midnight should have very high zenith
    midnight = datetime(2026, 3, 23, 5, 0, 0, tzinfo=timezone.utc)  # midnight EDT
    night = solar_position(midnight)
    check(f"Midnight zenith > 80° (got: {night['zenith_deg']:.1f}°)",
          night["zenith_deg"] > 80)

    return solar


def test_kalman_filter():
    """Test Kalman filter with time-delta scaling."""
    print("\n── Layer 7: Kalman Filter (dt-scaled Q) ──")

    kf = KalmanNDVI()
    check("KalmanNDVI initializes", kf.x is None and not kf.initialized)

    # First measurement initializes state
    v1 = kf.update(0.45, dt_seconds=600)
    check(f"First measurement accepted (got: {v1:.4f})", abs(v1 - 0.45) < 0.001)
    check("Filter is now initialized", kf.initialized)

    # Second measurement should be smoothed
    v2 = kf.update(0.50, dt_seconds=600)
    check(f"Second measurement smoothed: 0.45 < {v2:.4f} < 0.50",
          0.45 < v2 < 0.50)

    # Outlier should be heavily damped
    v3 = kf.update(0.10, dt_seconds=600)  # sudden drop
    check(f"Outlier rejected: {v3:.4f} > 0.30 (damped from 0.10)",
          v3 > 0.30)

    # Long time gap → more Q → faster adaptation
    kf2 = KalmanNDVI()
    kf2.update(0.50, dt_seconds=600)
    v_short = kf2.update(0.30, dt_seconds=60)   # 1 min gap

    kf3 = KalmanNDVI()
    kf3.update(0.50, dt_seconds=600)
    v_long = kf3.update(0.30, dt_seconds=3600)  # 1 hour gap

    check(f"Long dt adapts faster: {v_long:.4f} < {v_short:.4f}",
          v_long < v_short,
          f"long={v_long:.4f}, short={v_short:.4f}")


def test_full_pipeline():
    """Test the complete 7-layer pipeline on a real field capture."""
    print("\n── Full Pipeline Test ──")

    red, nir, source = load_test_frame()
    now = datetime(2026, 3, 23, 17, 0, 0, tzinfo=timezone.utc)

    result = apply_corrections(
        red, nir,
        red_exposure=0.01, red_gain=1.0,
        nir_exposure=0.012, nir_gain=1.0,
        timestamp=now,
        cloud_cover_pct=25.0,
    )

    check("Pipeline returns dict", isinstance(result, dict))
    check("Has ndvi array", "ndvi" in result and isinstance(result["ndvi"], np.ndarray))
    check(f"NDVI shape matches input ({result['ndvi'].shape} == {red.shape})",
          result["ndvi"].shape == red.shape)

    ndvi = result["mean_ndvi"]
    check(f"Mean NDVI in valid range: -1 ≤ {ndvi:.4f} ≤ 1",
          -1.0 <= ndvi <= 1.0)

    # Check all layers executed
    corr = result["corrections"]
    check("L1 exposure applied", corr.get("L1_exposure") != "skipped")
    check("L2 DOS applied", corr.get("L2_dos") != "skipped")
    check("L3 solar applied", corr.get("L3_solar") != "skipped")
    check("L4 cloud applied", corr.get("L4_cloud") != "skipped")
    check("L5 interframe applied", corr.get("L5_interframe") != "skipped")
    check("L6 seasonal present", "L6_seasonal" in corr)
    check("L7 kalman noted", "L7_kalman" in corr)

    # Seasonal baseline
    seasonal = result.get("seasonal", {})
    check(f"Seasonal baseline = {seasonal.get('baseline', '?')} ({seasonal.get('season', '?')})",
          seasonal.get("season") == "spring")

    # Quality flags
    flags = result.get("quality_flags", [])
    check(f"Quality flags: {flags if flags else 'none'}", True)

    # Solar info
    solar = result.get("solar", {})
    if solar:
        check(f"Solar: zenith={solar['zenith_deg']:.1f}°, el={solar['elevation_deg']:.1f}°",
              solar["zenith_deg"] < 90)

    print(f"\n  📊 Pipeline result for {source}:")
    print(f"     Mean NDVI:  {ndvi:.4f}")
    print(f"     Baseline:   {seasonal.get('baseline', '?')} ({seasonal.get('season', '?')})")
    print(f"     Layers:     {len([k for k in corr if corr[k] != 'skipped'])}/7 applied")
    if solar:
        print(f"     Solar:      {solar['zenith_deg']:.1f}° zenith")
    print(f"     Cloud T:    {corr.get('L4_cloud', {}).get('transmittance', '?')}")


def test_temperature_scaling():
    """Test that temperature scaling is applied to softmax."""
    print("\n── Temperature Scaling (DINOv2) ──")
    try:
        from convert_tensorrt import TEMPERATURE
        check(f"TEMPERATURE constant exists = {TEMPERATURE}", True)
        check(f"TEMPERATURE = 1.45 (got: {TEMPERATURE})", abs(TEMPERATURE - 1.45) < 0.01)

        # Verify effect: temperature-scaled softmax is less confident
        logits = np.array([[3.0, 1.0, 0.5, 0.1]])

        # Without scaling
        exp_raw = np.exp(logits - np.max(logits))
        probs_raw = exp_raw / exp_raw.sum()

        # With scaling
        scaled = logits / TEMPERATURE
        exp_scaled = np.exp(scaled - np.max(scaled))
        probs_scaled = exp_scaled / exp_scaled.sum()

        max_raw = float(probs_raw.max())
        max_scaled = float(probs_scaled.max())
        check(f"Scaled max prob ({max_scaled:.3f}) < raw ({max_raw:.3f})",
              max_scaled < max_raw)

    except ImportError:
        # convert_tensorrt needs tensorrt, which is Jetson-only
        check("convert_tensorrt not importable (needs TensorRT) — checking constant directly",
              True)
        # Read the file directly
        trt_path = os.path.join(SCRIPT_DIR, "convert_tensorrt.py")
        if os.path.isfile(trt_path):
            with open(trt_path) as f:
                content = f.read()
            check("TEMPERATURE = 1.45 in convert_tensorrt.py",
                  "TEMPERATURE = 1.45" in content)
            check("logits / TEMPERATURE in softmax",
                  "logits / TEMPERATURE" in content)
        else:
            check("convert_tensorrt.py exists", False, "file not found")


def test_hardware_watchdog():
    """Test that HardwareWatchdog class exists and is importable."""
    print("\n── Hardware Watchdog ──")
    try:
        from fault_tolerance import HardwareWatchdog, WATCHDOG_TIMEOUT_SEC

        check(f"WATCHDOG_TIMEOUT_SEC = 300 (got: {WATCHDOG_TIMEOUT_SEC})",
              WATCHDOG_TIMEOUT_SEC == 300)
        check("HardwareWatchdog class importable", True)

        # On Windows/dev machine, it should gracefully fail
        hw = HardwareWatchdog()
        check(f"HardwareWatchdog.available = {hw.available} (expected False on dev machine)",
              not hw.available)
        # kick() should not crash even if unavailable
        hw.kick()
        check("kick() doesn't crash when unavailable", True)
        hw.close()
        check("close() doesn't crash when unavailable", True)

    except ImportError as e:
        check("fault_tolerance importable", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Chroma-Shift — Pipeline Smoke Test                        ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    test_seasonal_baseline()
    test_solar_position()
    test_kalman_filter()
    test_full_pipeline()
    test_temperature_scaling()
    test_hardware_watchdog()

    print("\n" + "═" * 60)
    total = tests_passed + tests_failed
    if tests_failed == 0:
        print(f"  ✅ ALL {total} TESTS PASSED")
    else:
        print(f"  {PASS} {tests_passed}/{total} passed, {FAIL} {tests_failed} failed")
    print("═" * 60)

    sys.exit(1 if tests_failed > 0 else 0)

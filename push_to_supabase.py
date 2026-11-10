"""
push_to_supabase.py — Chroma-Shift Jetson → Supabase Push Service
===================================================================
Pushes NDVI readings, stress alerts, system status, and field zone
updates to Supabase. The farmer app reads these in real-time.

    Jetson → Supabase REST API → Farmer App (realtime subscription)

Tables:
    alerts         — Crop stress detections
    ndvi_readings  — Per-scan NDVI values
    system_status  — Jetson heartbeat
    field_zones    — Field zone NDVI updates

Usage:
    # Push a single NDVI reading + alert if stress detected
    python push_to_supabase.py --scan --zone B3 --ndvi 0.52 --band NIR

    # Push system heartbeat
    python push_to_supabase.py --heartbeat

    # Push satellite ground-truth data
    python push_to_supabase.py --satellite

    # Full auto-scan: capture + compute NDVI + push + alert
    python push_to_supabase.py --auto --zone B3
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime, timezone

import numpy as np
from dotenv import load_dotenv
from supabase import create_client

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    sys.exit("ERROR: Set SUPABASE_URL and SUPABASE_KEY in .env")

SATELLITE_CSV = os.path.join(SCRIPT_DIR, "ndvi_central_park.csv")


# ---------------------------------------------------------------------------
# Field Zone Baselines
# ---------------------------------------------------------------------------
FIELD_ZONES = {
    "A1": {"name": "North Orchard",  "baseline_ndvi": 0.72},
    "A2": {"name": "North Meadow",   "baseline_ndvi": 0.68},
    "B1": {"name": "East Ridge",     "baseline_ndvi": 0.65},
    "B2": {"name": "East Valley",    "baseline_ndvi": 0.71},
    "B3": {"name": "South Field",    "baseline_ndvi": 0.74},
    "C1": {"name": "West Terrace",   "baseline_ndvi": 0.69},
}

# Filter wheel slot mapping.
# The field unit carries a Red bandpass and a NIR filter. Slots 3 and 4 are
# empty: Red-Edge is a future concept, and SWIR is outside what the silicon
# sensor can see. Earlier versions listed Sentinel-2 bands here.
FILTER_BANDS = {
    0: {"name": "Clear",     "wavelength": "400-700 nm"},
    1: {"name": "Red",       "wavelength": "640-670 nm"},
    2: {"name": "NIR",       "wavelength": "850-880 nm"},
    3: {"name": "Empty",     "wavelength": "-"},
    4: {"name": "Empty",     "wavelength": "-"},
}

# Stress thresholds (NDVI drop from baseline)
STRESS_THRESHOLDS = [
    (0.15, "critical", "stress"),
    (0.10, "high",     "stress"),
    (0.05, "medium",   "watch"),
    (0.00, "low",      "healthy"),
]

# Stress type descriptions
STRESS_TYPES = {
    "water_stress":          "💧 Water Stress",
    "nutrient_deficiency":   "🌿 Nutrient Deficiency",
    "disease_detected":      "🦠 Disease Detected",
    "pest_damage":           "🐛 Pest Damage",
    "heat_stress":           "🌡️ Heat Stress",
}


# ---------------------------------------------------------------------------
# Supabase client
# ---------------------------------------------------------------------------
def get_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# ═══════════════════════════════════════════════════════════════════════════
# Stress Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_stress(ndvi, baseline):
    """
    Compare measured NDVI against the field zone baseline.
    Returns (severity, status, drop).
    """
    drop = baseline - ndvi
    for threshold, severity, status in STRESS_THRESHOLDS:
        if drop > threshold:
            return severity, status, drop
    return "low", "healthy", drop


def infer_stress_type(ndvi, drop):
    """
    Infer the most likely stress type from NDVI characteristics.
    A real system would use the DINOv2 model output; this is a heuristic.
    """
    if drop > 0.15:
        return "water_stress"
    elif drop > 0.10:
        return "nutrient_deficiency"
    elif drop > 0.05:
        return "heat_stress"
    return "water_stress"


# ═══════════════════════════════════════════════════════════════════════════
# 1. Push Alert
# ═══════════════════════════════════════════════════════════════════════════

def push_alert(client, zone_id, ndvi, band="NIR", confidence=None):
    """Push a stress alert to the alerts table."""
    zone = FIELD_ZONES.get(zone_id)
    if not zone:
        print(f"  ⚠️  Unknown zone: {zone_id}")
        return None

    baseline = zone["baseline_ndvi"]
    severity, status, drop = detect_stress(ndvi, baseline)

    # Only push alerts for actual stress
    if status == "healthy":
        print(f"  Zone {zone_id}: NDVI {ndvi:.2f} is healthy (baseline {baseline}). No alert.")
        return None

    stress_type = infer_stress_type(ndvi, drop)
    zone_name = zone["name"]

    # Build alert
    alert = {
        "severity": severity,
        "stress_type": stress_type,
        "field_zone": zone_id,
        "confidence": confidence or round(min(0.95, 0.5 + drop), 2),
        "ndvi_value": round(ndvi, 4),
        "ndvi_baseline": baseline,
        "title": f"{STRESS_TYPES.get(stress_type, stress_type)} — {zone_name}",
        "message": (
            f"NDVI dropped to {ndvi:.2f} (baseline {baseline}). "
            f"Deficit of {drop:.2f} detected in {zone_name}."
        ),
        "recommendation": _generate_recommendation(stress_type, drop, zone_name),
        "filter_band": band,
    }

    response = client.table("alerts").insert(alert).execute()
    print(f"  🚨 Alert pushed: [{severity.upper()}] {alert['title']}")
    return response


def _generate_recommendation(stress_type, drop, zone_name):
    """Generate actionable recommendation based on stress type."""
    pct = int(drop * 100)
    recommendations = {
        "water_stress": f"Increase irrigation in {zone_name} by {pct}% for the next 48 hours.",
        "nutrient_deficiency": f"Apply nitrogen-rich fertilizer to {zone_name}. Recheck in 7 days.",
        "disease_detected": f"Inspect {zone_name} for fungal/bacterial signs. Consider targeted fungicide.",
        "pest_damage": f"Deploy pest traps in {zone_name}. Inspect leaf undersides for damage.",
        "heat_stress": f"Increase shading or irrigation frequency in {zone_name} during peak hours.",
    }
    return recommendations.get(stress_type, f"Investigate {zone_name} for stress factors.")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Push NDVI Reading
# ═══════════════════════════════════════════════════════════════════════════

def push_ndvi_reading(client, ndvi, zone_id, band="NIR", source="jetson"):
    """Push an NDVI reading to the ndvi_readings table."""
    reading = {
        "value": round(ndvi, 4),
        "field_zone": zone_id,
        "source": source,
        "band_used": band,
    }
    response = client.table("ndvi_readings").insert(reading).execute()
    print(f"  📊 NDVI reading pushed: {ndvi:.4f} (zone {zone_id}, {band})")
    return response


# ═══════════════════════════════════════════════════════════════════════════
# 3. Push System Status (Heartbeat)
# ═══════════════════════════════════════════════════════════════════════════

def get_jetson_stats():
    """Read system stats from the Jetson."""
    stats = {
        "online": True,
        "mode": "scanning",
        "uptime_seconds": 0,
        "cpu_temp_c": 0.0,
        "gpu_usage_pct": 0.0,
        "storage_used_gb": 0.0,
        "storage_total_gb": 0.0,
        "alerts_today": 0,
        "scans_today": 0,
        "filter_position": 0,
        "filter_band_name": "Clear",
        "filter_wavelength_nm": "400-700",
    }

    # Uptime
    try:
        with open("/proc/uptime") as f:
            stats["uptime_seconds"] = int(float(f.read().split()[0]))
    except Exception:
        pass

    # CPU temperature
    try:
        for zone in sorted(os.listdir("/sys/class/thermal/")):
            temp_file = f"/sys/class/thermal/{zone}/temp"
            if os.path.isfile(temp_file):
                with open(temp_file) as f:
                    stats["cpu_temp_c"] = round(int(f.read().strip()) / 1000.0, 1)
                break
    except Exception:
        pass

    # Storage
    try:
        st = os.statvfs("/")
        total = st.f_blocks * st.f_frsize
        free = st.f_bfree * st.f_frsize
        used = total - free
        stats["storage_total_gb"] = round(total / (1024**3), 1)
        stats["storage_used_gb"] = round(used / (1024**3), 1)
    except Exception:
        pass

    return stats


def push_heartbeat(client, mode="scanning", filter_pos=0):
    """Push a system status heartbeat."""
    stats = get_jetson_stats()
    stats["mode"] = mode

    if filter_pos in FILTER_BANDS:
        band = FILTER_BANDS[filter_pos]
        stats["filter_position"] = filter_pos
        stats["filter_band_name"] = band["name"]
        stats["filter_wavelength_nm"] = band["wavelength"]

    response = client.table("system_status").insert(stats).execute()
    print(f"  💓 Heartbeat pushed: uptime={stats['uptime_seconds']}s, "
          f"cpu={stats['cpu_temp_c']}°C, "
          f"storage={stats['storage_used_gb']}/{stats['storage_total_gb']}GB")
    return response


# ═══════════════════════════════════════════════════════════════════════════
# 4. Update Field Zone
# ═══════════════════════════════════════════════════════════════════════════

def update_field_zone(client, zone_id, ndvi):
    """Update a field zone's current NDVI and status."""
    zone = FIELD_ZONES.get(zone_id)
    if not zone:
        print(f"  ⚠️  Unknown zone: {zone_id}")
        return None

    _, status, _ = detect_stress(ndvi, zone["baseline_ndvi"])
    now = datetime.now(timezone.utc).isoformat()

    response = (
        client.table("field_zones")
        .update({
            "ndvi_current": round(ndvi, 4),
            "status": status,
            "last_scan": now,
        })
        .eq("id", zone_id)
        .execute()
    )
    print(f"  🗺️  Zone {zone_id} updated: NDVI={ndvi:.4f}, status={status}")
    return response


def push_scan_result(zone_id, ndvi, classification, confidence, advisory, weather=None):
    """
    Push a complete scan result to Supabase (NDVI + zone update + alert if needed).

    This is the single-call interface for the full pipeline to push results.
    """
    client = get_client()

    # 1. Push NDVI reading
    push_ndvi_reading(client, ndvi, zone_id, band="NIR", source="jetson")

    # 2. Update field zone
    update_field_zone(client, zone_id, ndvi)

    # 3. Push alert if stress detected
    zone = FIELD_ZONES.get(zone_id)
    baseline = zone["baseline_ndvi"] if zone else 0.4
    severity, status, _ = detect_stress(ndvi, baseline)
    if status in ("watch", "alert", "critical"):
        push_alert(client, zone_id, ndvi, band="NIR", confidence=confidence)

    # 4. Push heartbeat with classification info
    push_heartbeat(client, mode=f"scan_{classification}", filter_pos=0)

    print(f"  ✅ Full scan result pushed to Supabase (zone {zone_id})")


# ═══════════════════════════════════════════════════════════════════════════
# Satellite ground-truth push
# ═══════════════════════════════════════════════════════════════════════════

def push_satellite(client):
    """Push processed satellite NDVI CSV to ndvi_readings as source=satellite."""
    import pandas as pd

    if not os.path.isfile(SATELLITE_CSV):
        print(f"  Satellite CSV not found: {SATELLITE_CSV}")
        return 0

    df = pd.read_csv(SATELLITE_CSV)
    print(f"  Loaded {len(df)} satellite observations")

    records = []
    for _, row in df.iterrows():
        records.append({
            "value": round(float(row["NDVI"]), 4),
            "field_zone": "Central_Park",
            "source": row.get("Source", "satellite").lower(),
            "band_used": "NIR",
        })

    batch_size = 500
    total = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        response = client.table("ndvi_readings").insert(batch).execute()
        pushed = len(response.data) if response.data else 0
        total += pushed

    print(f"  ✅ Pushed {total} satellite records to ndvi_readings")
    return total


# ═══════════════════════════════════════════════════════════════════════════
# Full Scan Pipeline: capture + compute + push + alert
# ═══════════════════════════════════════════════════════════════════════════

def auto_scan(client, zone_id, band="NIR"):
    """
    Full pipeline: capture Red+NIR pair → compute NDVI → push reading →
    push alert if stressed → update field zone.
    """
    from capture_multispectral_pair import (
        open_camera, flush_and_capture, FilterWheel,
        FILTER_SLOT_RED, FILTER_SLOT_NIR,
    )
    import cv2

    print(f"\n  Auto-scan: zone={zone_id}, band={band}")

    # Capture
    print("  Capturing Red+NIR pair …")
    cap = open_camera()
    efw = FilterWheel()
    efw.open()

    try:
        efw.move_to(FILTER_SLOT_RED)
        red_frame = flush_and_capture(cap)

        efw.move_to(FILTER_SLOT_NIR)
        nir_frame = flush_and_capture(cap)
    finally:
        cap.release()
        efw.close()

    # Compute NDVI
    red = cv2.cvtColor(red_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    nir = cv2.cvtColor(nir_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    denom = nir + red
    ndvi_map = np.where(denom > 0, (nir - red) / denom, 0.0)
    mean_ndvi = float(ndvi_map[denom > 0].mean()) if (denom > 0).any() else 0.0

    print(f"  Computed NDVI: {mean_ndvi:.4f}")

    # Push everything
    push_ndvi_reading(client, mean_ndvi, zone_id, band)
    push_alert(client, zone_id, mean_ndvi, band)
    update_field_zone(client, zone_id, mean_ndvi)
    push_heartbeat(client, mode="scanning", filter_pos=2)

    return mean_ndvi


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Chroma-Shift — Push data to Supabase."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scan", action="store_true",
                       help="Push a manual NDVI reading (use --zone and --ndvi).")
    group.add_argument("--heartbeat", action="store_true",
                       help="Push a system status heartbeat.")
    group.add_argument("--satellite", action="store_true",
                       help="Push satellite ground-truth NDVI data.")
    group.add_argument("--auto", action="store_true",
                       help="Full auto-scan: capture + NDVI + push + alert.")

    parser.add_argument("--zone", "-z", type=str, default="B3",
                        help="Field zone ID (default: B3).")
    parser.add_argument("--ndvi", type=float, default=None,
                        help="Manual NDVI value (for --scan mode).")
    parser.add_argument("--band", type=str, default="NIR",
                        help="Filter band name (default: NIR).")

    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Chroma-Shift — Supabase Push Service                     ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    client = get_client()
    print(f"  Connected: {SUPABASE_URL}\n")

    if args.scan:
        if args.ndvi is None:
            parser.error("--ndvi is required with --scan")
        push_ndvi_reading(client, args.ndvi, args.zone, args.band)
        push_alert(client, args.zone, args.ndvi, args.band)
        update_field_zone(client, args.zone, args.ndvi)

    elif args.heartbeat:
        push_heartbeat(client)

    elif args.satellite:
        push_satellite(client)

    elif args.auto:
        auto_scan(client, args.zone, args.band)

    print("\n  Done.")


if __name__ == "__main__":
    main()

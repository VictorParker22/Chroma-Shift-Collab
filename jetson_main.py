#!/usr/bin/env python3
"""
jetson_main.py — Chroma-Shift Continuous Scan Loop
====================================================
Single entry-point that runs continuously on the Jetson Orin Nano.

Loop:
  1. Fetch live weather from NWS (Central Park / KNYC)
  2. Capture Red + NIR image pair (filter wheel rotation)
  3. Compute per-pixel NDVI
  4. Compare to field zone baseline (weather-adjusted)
  5. Push alert to Supabase if stressed (with weather context)
  6. Log NDVI reading
  7. Update field zone status
  8. Push weather data to Supabase
  9. Push system heartbeat
  10. Sleep → repeat

Usage:
    python jetson_main.py                          # scan all zones, default interval
    python jetson_main.py --zone B3                # single zone only
    python jetson_main.py --interval 300           # every 5 minutes
    python jetson_main.py --interval 900 --zone B3 # zone B3 every 15 min
"""

import os
import sys
import time
import signal
import argparse
import ctypes
import traceback
from datetime import datetime, timezone

import cv2
import numpy as np
from dotenv import load_dotenv
from supabase import create_client
from weather import get_current_weather, assess_weather_stress, correlate_ndvi_weather, push_weather
from radiometric import (
    apply_corrections, solar_position, get_seasonal_baseline,
    KalmanNDVI, cloud_transmittance
)

# AI Classifier — lazy-loaded inside Docker container
# When running natively (without container), set AI_MODEL_PATH to skip
AI_MODEL_PATH = os.environ.get(
    "AI_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "results", "dino_v2_classify", "best_model.pth")
)

# 4-class labels (must match training order)
AI_CLASSES_4 = ["bare_soil", "healthy_vegetation", "non_vegetation", "stressed_vegetation"]
AI_IMAGE_SIZE = 224

# ---------------------------------------------------------------------------
# Paths & env
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # service_role key

if not SUPABASE_URL or not SUPABASE_KEY:
    sys.exit("ERROR: Set SUPABASE_URL and SUPABASE_KEY in .env")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
FILTER_SLOT_RED = 1          # 665 nm filter on order; slot empty, frame is unfiltered
FILTER_SLOT_NIR = 2          # 865 nm bandpass, installed
SETTLE_TIME_SEC = 1.5
FLUSH_FRAMES = 5
EFW_POLL_INTERVAL = 0.3
EFW_POLL_TIMEOUT = 30

DEFAULT_SCAN_INTERVAL = 600  # 10 minutes
HEARTBEAT_INTERVAL = 60      # 1 minute

CAPTURES_DIR = os.path.join(SCRIPT_DIR, "captures")
STORAGE_BUCKET = "camera_captures"

# Field zones and baselines
FIELD_ZONES = {
    "A1": {"name": "North Orchard",  "baseline": 0.72},
    "A2": {"name": "North Meadow",   "baseline": 0.68},
    "B1": {"name": "East Ridge",     "baseline": 0.65},
    "B2": {"name": "East Valley",    "baseline": 0.71},
    "B3": {"name": "South Field",    "baseline": 0.74},
    "C1": {"name": "West Terrace",   "baseline": 0.69},
    "T1": {"name": "USS Maine Monument", "baseline": 0.6493},
}

# Filter band lookup
# Installed hardware only. Earlier versions listed Red-Edge and SWIR here: those
# are Sentinel-2/Landsat bands from the calibration research, copied into this
# table as if filters were installed. They were not (and the IMX477 cannot see
# SWIR). Removed before the NASA submission. See README "Hardware reality".
FILTER_BANDS = {
    0: ("Clear", "400-700"),
    1: ("Red slot, no filter", "broadband"),
    2: ("NIR",   "865"),
    3: ("Empty", "-"),
    4: ("Empty", "-"),
}

# Stress thresholds
STRESS_LEVELS = [
    (0.15, "critical", "stress"),
    (0.10, "high",     "stress"),
    (0.05, "medium",   "watch"),
    (0.00, "low",      "healthy"),
]

STRESS_TYPES = {
    "critical": "water_stress",
    "high":     "water_stress",
    "medium":   "heat_stress",
    "low":      "water_stress",
}

STRESS_ICONS = {
    "water_stress":        "💧",
    "nutrient_deficiency": "🌿",
    "disease_detected":    "🦠",
    "pest_damage":         "🐛",
    "heat_stress":         "🌡️",
}

# Graceful shutdown
_running = True
def _signal_handler(sig, frame):
    global _running
    print("\n  ⏹  Shutdown signal received. Finishing current cycle…")
    _running = False

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ═══════════════════════════════════════════════════════════════════════════
# EFW — ctypes wrapper (same as capture_multispectral_pair.py)
# ═══════════════════════════════════════════════════════════════════════════


class EFW_INFO(ctypes.Structure):
    """Mirrors the EFW_INFO struct from the ZWO EFW SDK."""
    _fields_ = [
        ("ID", ctypes.c_int),
        ("Name", ctypes.c_char * 64),
        ("slotNum", ctypes.c_int),
    ]


class FilterWheel:
    def __init__(self):
        lib_path = os.path.join(SCRIPT_DIR, "lib", "libEFWFilter.so")
        ctypes.CDLL("libudev.so", mode=ctypes.RTLD_GLOBAL)
        self._lib = ctypes.CDLL(lib_path)
        self._lib.EFWOpen.restype = ctypes.c_int
        self._lib.EFWSetPosition.restype = ctypes.c_int
        self._lib.EFWGetPosition.restype = ctypes.c_int
        self._lib.EFWGetProperty.restype = ctypes.c_int
        self._lib.EFWClose.restype = ctypes.c_int
        self._id = None

    def open(self):
        if self._lib.EFWGetNum() == 0:
            raise RuntimeError("No EFW detected")
        eid = ctypes.c_int()
        self._lib.EFWGetID(0, ctypes.byref(eid))
        self._id = eid.value
        rc = self._lib.EFWOpen(self._id)
        if rc != 0:
            raise RuntimeError(f"EFWOpen failed (rc={rc})")
        # REQUIRED: EFWGetProperty must be called after Open, before SetPosition
        info = EFW_INFO()
        self._lib.EFWGetProperty(self._id, ctypes.byref(info))
        # Wait for calibration
        self._wait_ready()

    def close(self):
        if self._id is not None:
            self._lib.EFWClose(self._id)
            self._id = None

    def move_to(self, slot):
        self._lib.EFWSetPosition(self._id, slot)
        deadline = time.time() + EFW_POLL_TIMEOUT
        pos = ctypes.c_int()
        while time.time() < deadline:
            self._lib.EFWGetPosition(self._id, ctypes.byref(pos))
            if pos.value == slot:
                break
            time.sleep(EFW_POLL_INTERVAL)
        time.sleep(SETTLE_TIME_SEC)

    def _wait_ready(self):
        pos = ctypes.c_int()
        deadline = time.time() + EFW_POLL_TIMEOUT
        while time.time() < deadline:
            self._lib.EFWGetPosition(self._id, ctypes.byref(pos))
            if pos.value >= 0:
                return
            time.sleep(EFW_POLL_INTERVAL)


# ═══════════════════════════════════════════════════════════════════════════
# Camera
# ═══════════════════════════════════════════════════════════════════════════

def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera /dev/video{CAMERA_INDEX}")
    return cap

def flush_and_capture(cap):
    for _ in range(FLUSH_FRAMES):
        cap.read()
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Frame capture failed")
    return frame


# ═══════════════════════════════════════════════════════════════════════════
# NDVI
# ═══════════════════════════════════════════════════════════════════════════

def compute_ndvi(red_frame, nir_frame):
    red = cv2.cvtColor(red_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    nir = cv2.cvtColor(nir_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    denom = nir + red
    ndvi = np.where(denom > 0, (nir - red) / denom, 0.0)
    valid = denom > 0
    mean = float(ndvi[valid].mean()) if valid.any() else 0.0
    return ndvi, mean


# ═══════════════════════════════════════════════════════════════════════════
# Stress Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_stress(ndvi, baseline):
    drop = baseline - ndvi
    for threshold, severity, status in STRESS_LEVELS:
        if drop > threshold:
            return severity, status, drop
    return "low", "healthy", drop


def build_recommendation(stress_type, drop, zone_name):
    pct = int(drop * 100)
    recs = {
        "water_stress":   f"Increase irrigation in {zone_name} by {pct}% for the next 48 hours.",
        "heat_stress":    f"Increase shading or irrigation frequency in {zone_name} during peak hours.",
        "nutrient_deficiency": f"Apply nitrogen-rich fertilizer to {zone_name}. Recheck in 7 days.",
        "disease_detected": f"Inspect {zone_name} for fungal/bacterial signs. Consider targeted fungicide.",
        "pest_damage":    f"Deploy pest traps in {zone_name}. Inspect leaf undersides for damage.",
    }
    return recs.get(stress_type, f"Investigate {zone_name}.")


# ═══════════════════════════════════════════════════════════════════════════
# Jetson System Stats
# ═══════════════════════════════════════════════════════════════════════════

def get_system_stats():
    stats = {
        "online": True,
        "uptime_seconds": 0,
        "cpu_temp_c": 0.0,
        "gpu_usage_pct": 0.0,
        "storage_used_gb": 0.0,
        "storage_total_gb": 0.0,
    }
    try:
        with open("/proc/uptime") as f:
            stats["uptime_seconds"] = int(float(f.read().split()[0]))
    except Exception:
        pass
    try:
        for zone in sorted(os.listdir("/sys/class/thermal/")):
            tf = f"/sys/class/thermal/{zone}/temp"
            if os.path.isfile(tf):
                with open(tf) as f:
                    stats["cpu_temp_c"] = round(int(f.read().strip()) / 1000.0, 1)
                break
    except Exception:
        pass
    try:
        st = os.statvfs("/")
        total = st.f_blocks * st.f_frsize
        used = total - st.f_bfree * st.f_frsize
        stats["storage_total_gb"] = round(total / (1024**3), 1)
        stats["storage_used_gb"] = round(used / (1024**3), 1)
    except Exception:
        pass
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# Supabase Push Functions
# ═══════════════════════════════════════════════════════════════════════════

def push_ndvi_reading(sb, ndvi, zone_id, band="NIR"):
    sb.table("ndvi_readings").insert({
        "value":      round(ndvi, 4),
        "field_zone": zone_id,
        "source":     "jetson",
        "band_used":  band,
    }).execute()


def upload_capture(sb, frame, zone_id, timestamp, label="stress"):
    """
    Save frame as .jpg, upload to camera_captures bucket, return public URL.
    """
    filename = f"{zone_id}_{timestamp}_{label}.jpg"
    local_path = os.path.join(CAPTURES_DIR, filename)
    os.makedirs(CAPTURES_DIR, exist_ok=True)
    cv2.imwrite(local_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # Upload to Supabase Storage
    storage_path = f"{zone_id}/{filename}"
    with open(local_path, "rb") as f:
        sb.storage.from_(STORAGE_BUCKET).upload(
            path=storage_path,
            file=f,
            file_options={"content-type": "image/jpeg", "upsert": "true"},
        )

    # Get public URL
    public_url = sb.storage.from_(STORAGE_BUCKET).get_public_url(storage_path)
    return public_url


def push_alert(sb, zone_id, ndvi, severity, stress_type, drop, band="NIR", weather=None, image_url=None):
    zone = FIELD_ZONES[zone_id]
    icon = STRESS_ICONS.get(stress_type, "⚠️")
    zone_name = zone["name"]

    # Build message with weather context
    msg = f"NDVI dropped to {ndvi:.2f} (baseline {zone['baseline']}). Deficit of {drop:.2f} detected in {zone_name}."
    if weather:
        temp = weather.get('temp_c')
        desc = weather.get('description', '')
        precip = weather.get('precip_1h_mm', 0)
        weather_ctx = []
        if temp is not None:
            weather_ctx.append(f"Temp: {temp}°C")
        if desc:
            weather_ctx.append(desc)
        if precip and precip > 0:
            weather_ctx.append(f"Precip: {precip}mm/hr")
        if weather_ctx:
            msg += f" Weather: {', '.join(weather_ctx)}."

    payload = {
        "severity":      severity,
        "stress_type":   stress_type,
        "field_zone":    zone_id,
        "confidence":    round(min(0.95, 0.5 + drop), 2),
        "ndvi_value":    round(ndvi, 4),
        "ndvi_baseline": zone["baseline"],
        "title":         f"{icon} {stress_type.replace('_', ' ').title()} — {zone_name}",
        "message":       msg,
        "recommendation": build_recommendation(stress_type, drop, zone_name),
        "filter_band":   band,
    }
    if image_url:
        payload["image_url"] = image_url

    sb.table("alerts").insert(payload).execute()


def update_field_zone(sb, zone_id, ndvi, status):
    now = datetime.now(timezone.utc).isoformat()
    sb.table("field_zones").update({
        "ndvi_current": round(ndvi, 4),
        "status":       status,
        "last_scan":    now,
    }).eq("id", zone_id).execute()


def push_heartbeat(sb, mode, filter_pos, scans, alerts_count):
    stats = get_system_stats()
    band_name, wavelength = FILTER_BANDS.get(filter_pos, ("Unknown", "?"))
    stats.update({
        "mode":               mode,
        "filter_position":    filter_pos,
        "filter_band_name":   band_name,
        "filter_wavelength_nm": wavelength,
        "scans_today":        scans,
        "alerts_today":       alerts_count,
    })
    sb.table("system_status").insert(stats).execute()


# ═══════════════════════════════════════════════════════════════════════════
# Single Scan Cycle
# ═══════════════════════════════════════════════════════════════════════════

def scan_zone(cap, efw, sb, zone_id, scan_count, alert_count,
              weather=None, ai_classify=None,
              scene_analyzer=None, kalman_filters=None):
    """
    Full pipeline for one zone:
      capture → radiometric correction → VILA scene analysis → corrected NDVI
      → AI classify → compare (seasonal + weather-adjusted) → alert → log

    Returns:
        (new_alerts, scan_result) where scan_result is a dict with
        ndvi, baseline, vila_result, and solar for adaptive scheduling.
    """
    zone = FIELD_ZONES[zone_id]
    zone_name = zone["name"]
    now_str = datetime.now().strftime("%H:%M:%S")
    now_utc = datetime.now(timezone.utc)

    print(f"  [{now_str}] Scanning {zone_id} ({zone_name}) …", end=" ", flush=True)

    # 1. Capture Red (with camera metadata)
    efw.move_to(FILTER_SLOT_RED)
    red_frame = flush_and_capture(cap)
    red_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    red_gain = cap.get(cv2.CAP_PROP_GAIN)

    # 2. Capture NIR (with camera metadata)
    efw.move_to(FILTER_SLOT_NIR)
    nir_frame = flush_and_capture(cap)
    nir_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    nir_gain = cap.get(cv2.CAP_PROP_GAIN)

    # 3. Solar position (pure math, no API)
    solar = solar_position(now_utc)

    # 4. VILA Scene Analysis (VLM or heuristic fallback)
    vila_result = None
    vila_overrides = None
    if scene_analyzer is not None:
        try:
            vila_result = scene_analyzer.analyze(
                red_frame, nir_frame, weather, solar, now_utc
            )
            vila_overrides = vila_result.get("corrections")
            vila_expl = vila_result.get("explanation", "")
            vila_conf = vila_result.get("confidence", 0.5)
            print(f"🧠VILA({vila_conf:.0%})", end=" ")
        except Exception as e:
            print(f"⚠️VILA:{e}", end=" ")

    # 5. Radiometric Correction Pipeline (7 layers)
    cloud_cover = weather.get("cloud_cover_pct") if weather else None
    corrected = apply_corrections(
        cv2.cvtColor(red_frame, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(nir_frame, cv2.COLOR_BGR2GRAY),
        red_exposure=red_exposure, red_gain=red_gain,
        nir_exposure=nir_exposure, nir_gain=nir_gain,
        timestamp=now_utc,
        cloud_cover_pct=cloud_cover,
        vila_overrides=vila_overrides,
    )
    mean_ndvi = corrected["mean_ndvi"]
    quality_flags = corrected.get("quality_flags", [])

    # 6. Kalman temporal filter (Layer 7 — per-zone)
    raw_ndvi = mean_ndvi
    if kalman_filters and zone_id in kalman_filters:
        mean_ndvi = round(kalman_filters[zone_id].update(mean_ndvi), 4)

    # 7. Seasonal baseline (Layer 6)
    seasonal = corrected.get("seasonal", get_seasonal_baseline(now_utc))
    baseline = seasonal["baseline"]
    season = seasonal["season"]

    # 8. Weather-NDVI correlation
    if weather:
        corr = correlate_ndvi_weather(mean_ndvi, baseline, weather)
        weather_summary = corr["weather_summary"]
        weather_mod = corr["stress_modifier"]
    else:
        weather_summary = "N/A"
        weather_mod = 0.0

    # 9. Compare to seasonal baseline → stress detection
    severity, status, drop = detect_stress(mean_ndvi, baseline)
    stress_type = STRESS_TYPES.get(severity, "water_stress")

    # 10. AI Classification (DINOv2 via TensorRT)
    ai_result = None
    if ai_classify is not None:
        try:
            ai_result = ai_classify(red_frame)
            ai_class = ai_result["class"]
            ai_conf = ai_result["confidence"]
            ai_icons = {
                "healthy_vegetation": "🌿",
                "stressed_vegetation": "🟡",
                "bare_soil": "🟤",
                "non_vegetation": "⬛",
            }
            ai_icon = ai_icons.get(ai_class, "🤖")
            print(f"{ai_icon}AI:{ai_class}({ai_conf:.0f}%)", end=" ")
        except Exception as e:
            print(f"⚠️AI:{e}", end=" ")

    # 11. Upload capture + push alert if stressed
    new_alerts = 0
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if status != "healthy":
        image_url = None
        try:
            combined = np.hstack([red_frame, nir_frame])
            image_url = upload_capture(sb, combined, zone_id, ts, "stress")
            print(f"📸", end=" ")
        except Exception as e:
            print(f"⚠️upload:{e}", end=" ")

        push_alert(sb, zone_id, mean_ndvi, severity, stress_type, drop,
                   weather=weather, image_url=image_url)
        new_alerts = 1
        print(f"🚨 {severity.upper()}", end=" ")

    # 12. Log NDVI reading (with correction metadata + VILA explanation)
    ndvi_payload = {
        "value":      round(mean_ndvi, 4),
        "field_zone": zone_id,
        "source":     "jetson",
        "band_used":  "NIR",
    }
    if ai_result:
        ndvi_payload["ai_class"] = ai_result["class"]
        ndvi_payload["ai_confidence"] = round(ai_result["confidence"] / 100, 4)
    if vila_result:
        ndvi_payload["vila_explanation"] = vila_result.get("explanation", "")[:500]
    if quality_flags:
        ndvi_payload["quality_flags"] = ", ".join(quality_flags)
    try:
        sb.table("ndvi_readings").insert(ndvi_payload).execute()
    except Exception as e:
        # Supabase may not have vila_explanation/quality_flags columns yet
        # Fall back to minimal payload
        sb.table("ndvi_readings").insert({
            "value": round(mean_ndvi, 4), "field_zone": zone_id,
            "source": "jetson", "band_used": "NIR",
        }).execute()

    # 13. Update field zone
    update_field_zone(sb, zone_id, mean_ndvi, status)

    # 14. Push weather data
    if weather:
        try:
            push_weather(sb, weather, ndvi=mean_ndvi, field_zone=zone_id)
        except Exception as e:
            print(f"⚠️weather_push:{e}", end=" ")

    # 15. Heartbeat
    push_heartbeat(sb, "scanning", FILTER_SLOT_NIR, scan_count, alert_count + new_alerts)

    # 15b. Crop Advisory (with confidence thresholds)
    advisory = None
    try:
        from crop_advisor import generate_advisory, format_advisory_text
        advisory = generate_advisory(
            ndvi=mean_ndvi,
            seasonal_data=seasonal,
            weather=weather,
            classification=ai_result,
            zone_name=zone_name,
        )
        # Print confidence flags if present
        if advisory.get("confidence_flags"):
            for flag in advisory["confidence_flags"]:
                print(f"           {flag}")
    except Exception as e:
        pass  # crop_advisor is optional

    # Summary
    marker = "✅" if status == "healthy" else "⚠️"
    print(f"NDVI={mean_ndvi:.4f} (raw={raw_ndvi:.4f}, baseline={baseline:.2f}/{season}) → {status} {marker}")
    if ai_result:
        print(f"           🤖 AI: {ai_result['class']} ({ai_result['confidence']:.1f}%)")
    if vila_result:
        print(f"           🧠 VILA: {vila_result.get('explanation', '')[:80]}")
    if quality_flags:
        print(f"           ⚠️  Flags: {', '.join(quality_flags)}")
    if weather:
        print(f"           🌤️  {weather_summary} (modifier: {weather_mod:+.3f})")

    # Save images locally
    os.makedirs(CAPTURES_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(CAPTURES_DIR, f"{zone_id}_{ts}_red.png"), red_frame)
    cv2.imwrite(os.path.join(CAPTURES_DIR, f"{zone_id}_{ts}_nir.png"), nir_frame)

    # Return scan result for adaptive scheduling (with classification)
    scan_result = {
        "ndvi": mean_ndvi,
        "baseline": baseline,
        "vila_result": vila_result,
        "solar": solar,
        "classification": ai_result,
        "advisory": advisory,
    }
    return new_alerts, scan_result


# ═══════════════════════════════════════════════════════════════════════════
# Main Loop
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Chroma-Shift Jetson main loop.")
    parser.add_argument("--zone", "-z", type=str, default=None,
                        help="Scan a single zone (e.g. B3). Default: scan all zones.")
    parser.add_argument("--interval", "-i", type=int, default=DEFAULT_SCAN_INTERVAL,
                        help=f"Seconds between scan cycles (default: {DEFAULT_SCAN_INTERVAL}).")
    parser.add_argument("--once", action="store_true",
                        help="Run a single scan cycle then exit.")
    args = parser.parse_args()

    # Determine zones to scan
    if args.zone:
        if args.zone not in FIELD_ZONES:
            sys.exit(f"Unknown zone: {args.zone}. Available: {', '.join(FIELD_ZONES)}")
        zones = [args.zone]
    else:
        zones = list(FIELD_ZONES.keys())

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     Chroma-Shift — Jetson Main Loop                        ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Zones: {', '.join(zones):51s}║")
    print(f"║  Interval: {args.interval}s{' (single run)' if args.once else '':44s}║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # --- Connect Supabase ---
    print("\n  Connecting to Supabase …", end=" ", flush=True)
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("connected ✅")

    # --- Init hardware ---
    print("  Opening camera …", end=" ", flush=True)
    cap = open_camera()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"/dev/video{CAMERA_INDEX} ({w}×{h}) ✅")

    print("  Opening filter wheel …", end=" ", flush=True)
    efw = FilterWheel()
    efw.open()
    print("EFW ready ✅")

    # --- Push initial heartbeat ---
    try:
        push_heartbeat(sb, "scanning", 0, 0, 0)
        print("  Initial heartbeat pushed 💓")
    except Exception as e:
        print(f"  ⚠️  Heartbeat failed (network?): {e}")

    # --- Init Radiometric Correction (per-zone Kalman filters) ---
    kalman_filters = {zone_id: KalmanNDVI() for zone_id in zones}
    print(f"  Kalman filters initialized for {len(zones)} zones 📊")

    # --- Init Adaptive Sensing Scheduler ---
    adaptive_scheduler = None
    try:
        from adaptive_sensing import AdaptiveSensingScheduler
        adaptive_scheduler = AdaptiveSensingScheduler(base_interval=args.interval)
        print(f"  Adaptive scheduler initialized (interval: {args.interval}s base, "
              f"range: 120-1800s) 🎯")
    except Exception as e:
        print(f"  ⚠️  Adaptive scheduler not available: {e}")
        print(f"      (using fixed {args.interval}s interval)")

    # --- Init Fault Tolerance (SPACE-GRADE) ---
    from fault_tolerance import (
        FaultManager, SystemSelfTest, RemoteConfig,
        AnomalyDetector, Watchdog
    )

    # Self-test on boot — verify all components
    selftest = SystemSelfTest()
    starting_mode, test_results = selftest.run_all()

    # Fault manager — error tracking + mode transitions
    fault_mgr = FaultManager(supabase_client=sb)
    fault_mgr.mode = starting_mode

    # Anomaly detector — sensor health monitoring
    anomaly_detector = AnomalyDetector()

    # Remote config — ground control can change parameters
    remote_config = RemoteConfig(supabase_client=sb)

    # Watchdog — auto-restart if main loop hangs
    watchdog = Watchdog(timeout_sec=300)
    print("  🐕 Watchdog armed (300s timeout)")
    print(f"  🛡️  Fault tolerance: mode={fault_mgr.mode}, "
          f"anomaly detector active")

    # --- Init VILA Scene Analyzer (VLM or heuristic fallback) ---
    scene_analyzer = None
    try:
        from vila_analyzer import VilaSceneAnalyzer
        scene_analyzer = VilaSceneAnalyzer()
    except Exception as e:
        print(f"  ⚠️  VILA scene analyzer not available: {e}")
        print("      (radiometric corrections will use heuristic fallback)")
        try:
            from vila_analyzer import VilaSceneAnalyzer
            scene_analyzer = VilaSceneAnalyzer.__new__(VilaSceneAnalyzer)
            scene_analyzer.model = None
            scene_analyzer.model_name = ""
            scene_analyzer._fallback = True
            print("      Heuristic fallback loaded ✅")
        except Exception:
            pass

    # --- Init AI Classifier (TensorRT > PyTorch > NDVI-only) ---
    ai_classify = None
    AI_ENGINE_PATH = os.path.join(
        SCRIPT_DIR, "results", "dino_v2_classify", "chroma_shift_fp16.engine"
    )

    # ── Try TensorRT first (FP16, ~20ms/frame on GPU) ──
    if os.path.isfile(AI_ENGINE_PATH):
        try:
            print("  Loading TensorRT FP16 engine …", end=" ", flush=True)
            from convert_tensorrt import TRTClassifier, CLASSES as AI_CLASSES

            _trt_cls = TRTClassifier(AI_ENGINE_PATH)

            # Warmup with correct 224px input size
            _dummy = np.random.randn(1, 3, AI_IMAGE_SIZE, AI_IMAGE_SIZE).astype(np.float32)
            for _ in range(3):
                _trt_cls.predict(_dummy)

            def _classify_frame_trt(frame):
                """Classify a raw OpenCV BGR frame using TensorRT."""
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                from PIL import Image as PILImage
                img = PILImage.fromarray(rgb)
                return _trt_cls.classify_image_pil(img)

            # Add a PIL classify method for 4-class model
            def _classify_pil(self, pil_img):
                pil_img = pil_img.resize(
                    (AI_IMAGE_SIZE, AI_IMAGE_SIZE), PILImage.BILINEAR
                )
                arr = np.array(pil_img, dtype=np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                arr = (arr - mean) / std
                arr = arr.transpose(2, 0, 1)[np.newaxis, ...]
                probs = self.predict(arr)
                idx = int(np.argmax(probs))
                classes = AI_CLASSES_4
                result = {
                    "class": classes[idx],
                    "confidence": float(probs[idx]) * 100,
                }
                # Include per-class probabilities for confidence threshold system
                for i, cls_name in enumerate(classes):
                    result[f"{cls_name}_prob"] = float(probs[i])
                return result

            from PIL import Image as PILImage
            TRTClassifier.classify_image_pil = _classify_pil

            def _classify_frame(frame):
                """Classify a raw OpenCV BGR frame via TensorRT."""
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = PILImage.fromarray(rgb)
                return _trt_cls.classify_image_pil(img)

            ai_classify = _classify_frame
            print(f"TensorRT FP16 loaded (4-class, {AI_IMAGE_SIZE}px, ~6.7ms/frame) ✅")
        except Exception as e:
            print(f"\n  ⚠️  TensorRT load failed: {e}")
            print("      Falling back to PyTorch …")

    # ── Fall back to PyTorch (CPU, ~1000ms/frame) ──
    if ai_classify is None and os.path.isfile(AI_MODEL_PATH):
        try:
            print("  Loading DINOv2 classifier (PyTorch) …", end=" ", flush=True)
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from PIL import Image
            from torchvision import transforms as T

            AI_CLASSES = AI_CLASSES_4
            AI_IMG_SIZE = AI_IMAGE_SIZE
            AI_MEAN = [0.485, 0.456, 0.406]
            AI_STD = [0.229, 0.224, 0.225]

            num_classes = len(AI_CLASSES)

            class _DINOv2Cls(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.backbone = torch.hub.load(
                        "facebookresearch/dinov2", "dinov2_vits14", pretrained=True
                    )
                    for p in self.backbone.parameters():
                        p.requires_grad = False
                    self.backbone.eval()
                    self.head = nn.Sequential(
                        nn.Dropout(0.1), nn.Linear(self.backbone.embed_dim, num_classes)
                    )
                def forward(self, x):
                    return self.head(self.backbone(x))

            _ai_device = torch.device("cpu")  # CPU to preserve GPU RAM for capture
            _ai_model = _DINOv2Cls()
            _ckpt = torch.load(AI_MODEL_PATH, map_location=_ai_device, weights_only=False)
            _ai_model.load_state_dict(_ckpt["model_state_dict"])
            _ai_model.to(_ai_device)
            _ai_model.eval()

            _ai_transform = T.Compose([
                T.Resize((AI_IMG_SIZE, AI_IMG_SIZE)),
                T.ToTensor(),
                T.Normalize(AI_MEAN, AI_STD),
            ])

            @torch.no_grad()
            def _classify_frame(frame):
                """Classify a raw OpenCV BGR frame."""
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                tensor = _ai_transform(img).unsqueeze(0).to(_ai_device)
                logits = _ai_model(tensor)
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()
                idx = int(probs.argmax())
                result = {
                    "class": AI_CLASSES[idx],
                    "confidence": float(probs[idx]) * 100,
                }
                for i, cls_name in enumerate(AI_CLASSES):
                    result[f"{cls_name}_prob"] = float(probs[i])
                return result

            ai_classify = _classify_frame
            val_acc = _ckpt.get("val_acc", "?")
            print(f"DINOv2 ViT-S/14 loaded ({num_classes}-class, val_acc={val_acc}%, ~1s/frame CPU) ✅")
        except Exception as e:
            print(f"\n  ⚠️  AI classifier not available: {e}")
            print("      (NDVI-only mode — scans will continue without AI)")

    if ai_classify is None:
        print(f"  🤖 No AI model found — NDVI-only mode")
        print(f"      (run convert_tensorrt.py to build TensorRT engine)")

    print()  # blank line before scan loop

    # --- Scan loop ---
    scan_count = 0
    alert_count = 0
    cycle = 0

    try:
        while _running:
            cycle += 1
            cycle_start = time.time()

            print(f"═══ Cycle {cycle} ═══════════════════════════════════════════")

            # Fetch weather once per cycle (shared across all zones)
            weather = None
            try:
                weather = get_current_weather()
                if weather:
                    print(f"  🌤️  Weather: {weather.get('temp_c')}°C, "
                          f"{weather.get('humidity_pct', 0):.0f}% RH, "
                          f"{weather.get('description', 'N/A')}")
            except Exception as e:
                print(f"  ⚠️  Weather fetch failed: {e}")

            # Adaptive zone ordering — scan highest-priority zones first
            scan_zones = zones
            if adaptive_scheduler:
                scan_zones = adaptive_scheduler.rank_zones(zones)

            # Check remote config for ground control overrides
            force_safe = remote_config.get("force_safe_mode", False, cast=bool)
            if force_safe and not fault_mgr.is_safe_mode:
                print("  🛰️  Ground control forced SAFE MODE")
                fault_mgr.mode = "safe"

            remote_interval = remote_config.get("scan_interval", None, cast=int)
            if remote_interval:
                next_interval = remote_interval
                print(f"  🛰️  Ground control override: interval={remote_interval}s")
            else:
                next_interval = args.interval

            for zone_id in scan_zones:
                if not _running:
                    break
                try:
                    new_alerts, scan_result = scan_zone(
                        cap, efw, sb, zone_id, scan_count, alert_count,
                        weather,
                        ai_classify=None if fault_mgr.skip_ai else ai_classify,
                        scene_analyzer=None if fault_mgr.skip_vila else scene_analyzer,
                        kalman_filters=kalman_filters,
                    )
                    scan_count += 1
                    alert_count += new_alerts

                    # Anomaly check on NDVI
                    if scan_result:
                        ndvi_anomalies = anomaly_detector.check_ndvi(
                            scan_result["ndvi"], zone_id
                        )
                        for a in ndvi_anomalies:
                            print(f"           🔴 ANOMALY: {a['message']}")
                            fault_mgr.record_error(
                                "anomaly_detector", a["message"],
                                {"zone": zone_id, "type": a["type"]}
                            )

                    # --- Adaptive Scheduling Decision ---
                    if adaptive_scheduler and scan_result:
                        decision = adaptive_scheduler.compute_next_interval(
                            zone_id,
                            ndvi=scan_result["ndvi"],
                            seasonal_baseline=scan_result["baseline"],
                            vila_result=scan_result.get("vila_result"),
                            weather=weather,
                            solar=scan_result.get("solar"),
                            classification=scan_result.get("classification"),
                        )
                        # Use the shortest interval across all zones
                        next_interval = min(next_interval, decision["interval_sec"])
                        print(f"           🎯 Adaptive: {decision['interval_sec']}s, "
                              f"priority={decision['priority']}, "
                              f"mode={decision['power_mode']}")
                        if decision["reason"] != "nominal operations":
                            print(f"              → {decision['reason']}")

                    # Record success — may upgrade operating mode
                    fault_mgr.record_success()
                    watchdog.kick()

                except Exception as e:
                    fault_mgr.record_error("scan_zone", e,
                                           {"zone": zone_id})
                    traceback.print_exc()

            # Cycle summary
            if adaptive_scheduler:
                bw = adaptive_scheduler.get_bandwidth_estimate()
                print(f"  Scans: {scan_count}  Alerts: {alert_count}  "
                      f"Next: {next_interval}s  BW: {bw['total_kb']}KB  "
                      f"Mode: {fault_mgr.mode}  "
                      f"Health: {anomaly_detector.get_health_score()}%")
            else:
                print(f"  Scans: {scan_count}  Alerts: {alert_count}  "
                      f"Mode: {fault_mgr.mode}")

            if args.once:
                print("\n  --once flag set. Exiting.")
                break

            # Sleep until next cycle (adaptive or fixed)
            elapsed = time.time() - cycle_start
            sleep_time = max(0, next_interval - elapsed)
            if sleep_time > 0 and _running:
                if adaptive_scheduler and next_interval != args.interval:
                    print(f"  ⏱️  Adaptive interval: {int(next_interval)}s "
                          f"(base: {args.interval}s) … (Ctrl+C to stop)\n")
                else:
                    print(f"  Next cycle in {int(sleep_time)}s … (Ctrl+C to stop)\n")
                # Sleep in small increments so SIGINT is responsive
                wake_at = time.time() + sleep_time
                while time.time() < wake_at and _running:
                    watchdog.kick()
                    time.sleep(1)

    finally:
        # --- Cleanup ---
        print("\n  Shutting down …")
        cap.release()
        efw.close()
        try:
            push_heartbeat(sb, "idle", 0, scan_count, alert_count)
            print("  Final heartbeat pushed (mode=idle)")
        except Exception:
            pass
        print("  Hardware released. Goodbye.\n")


if __name__ == "__main__":
    main()

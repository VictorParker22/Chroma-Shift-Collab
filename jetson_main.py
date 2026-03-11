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

# AI Classifier — lazy-loaded inside Docker container
# When running natively (without container), set AI_MODEL_PATH to skip
AI_MODEL_PATH = os.environ.get(
    "AI_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "results", "dino_v2_classify", "best_model.pth")
)

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
FILTER_SLOT_RED = 1          # 640-670 nm
FILTER_SLOT_NIR = 2          # 850-880 nm
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
FILTER_BANDS = {
    0: ("Clear",     "400-700"),
    1: ("Red",       "640-670"),
    2: ("NIR",       "850-880"),
    3: ("Red-Edge",  "700-740"),
    4: ("SWIR",      "1570-1650"),
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
    _fields_ = [("ID", ctypes.c_int), ("Name", ctypes.c_char * 64), ("slotNum", ctypes.c_int)]

class FilterWheel:
    def __init__(self):
        lib_path = os.path.join(SCRIPT_DIR, "lib", "libEFWFilter.so")
        ctypes.CDLL("libudev.so", mode=ctypes.RTLD_GLOBAL)
        self._lib = ctypes.CDLL(lib_path)
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

def scan_zone(cap, efw, sb, zone_id, scan_count, alert_count, weather=None, ai_classify=None):
    """
    Full pipeline for one zone:
      capture → NDVI → AI classify → compare (weather-adjusted) → alert → log → update → heartbeat
    """
    zone = FIELD_ZONES[zone_id]
    zone_name = zone["name"]
    baseline = zone["baseline"]
    now_str = datetime.now().strftime("%H:%M:%S")

    print(f"  [{now_str}] Scanning {zone_id} ({zone_name}) …", end=" ", flush=True)

    # 1. Capture Red
    efw.move_to(FILTER_SLOT_RED)
    red_frame = flush_and_capture(cap)

    # 2. Capture NIR
    efw.move_to(FILTER_SLOT_NIR)
    nir_frame = flush_and_capture(cap)

    # 3. Compute NDVI
    ndvi_map, mean_ndvi = compute_ndvi(red_frame, nir_frame)

    # 4. Weather-NDVI correlation
    if weather:
        corr = correlate_ndvi_weather(mean_ndvi, baseline, weather)
        weather_summary = corr["weather_summary"]
        weather_mod = corr["stress_modifier"]
    else:
        weather_summary = "N/A"
        weather_mod = 0.0

    # 5. Compare to baseline → stress detection
    severity, status, drop = detect_stress(mean_ndvi, baseline)
    stress_type = STRESS_TYPES.get(severity, "water_stress")

    # 5b. AI Classification (DINOv2)
    ai_result = None
    if ai_classify is not None:
        try:
            # Classify the Red frame
            ai_result = ai_classify(red_frame)
            ai_class = ai_result["class"]
            ai_conf = ai_result["confidence"]
            ai_icon = "🌿" if ai_class == "healthy_vegetation" else "🟤"
            print(f"{ai_icon}AI:{ai_class}({ai_conf:.0f}%)", end=" ")
        except Exception as e:
            print(f"⚠️AI:{e}", end=" ")

    # 6. Upload capture + push alert if stressed (with weather context)
    new_alerts = 0
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if status != "healthy":
        # Upload stress capture to Supabase Storage
        image_url = None
        try:
            # Create a combined Red+NIR side-by-side image for the alert
            combined = np.hstack([red_frame, nir_frame])
            image_url = upload_capture(sb, combined, zone_id, ts, "stress")
            print(f"📸", end=" ")
        except Exception as e:
            print(f"⚠️upload:{e}", end=" ")

        push_alert(sb, zone_id, mean_ndvi, severity, stress_type, drop,
                   weather=weather, image_url=image_url)
        new_alerts = 1
        print(f"🚨 {severity.upper()}", end=" ")

    # 7. Log NDVI reading (include AI result)
    ndvi_payload = {
        "value":      round(mean_ndvi, 4),
        "field_zone": zone_id,
        "source":     "jetson",
        "band_used":  "NIR",
    }
    if ai_result:
        ndvi_payload["ai_class"] = ai_result["class"]
        ndvi_payload["ai_confidence"] = round(ai_result["confidence"] / 100, 4)
    sb.table("ndvi_readings").insert(ndvi_payload).execute()

    # 8. Update field zone
    update_field_zone(sb, zone_id, mean_ndvi, status)

    # 9. Push weather data (correlated with this NDVI reading)
    if weather:
        try:
            push_weather(sb, weather, ndvi=mean_ndvi, field_zone=zone_id)
        except Exception as e:
            print(f"⚠️weather_push:{e}", end=" ")

    # 10. Heartbeat
    push_heartbeat(sb, "scanning", FILTER_SLOT_NIR, scan_count, alert_count + new_alerts)

    # Summary
    marker = "✅" if status == "healthy" else "⚠️"
    print(f"NDVI={mean_ndvi:.4f} (baseline {baseline}) → {status} {marker}")
    if ai_result:
        print(f"           🤖 AI: {ai_result['class']} ({ai_result['confidence']:.1f}% confidence)")
    if weather:
        print(f"           🌤️  {weather_summary} (modifier: {weather_mod:+.3f})")

    # Save images locally
    os.makedirs(CAPTURES_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(CAPTURES_DIR, f"{zone_id}_{ts}_red.png"), red_frame)
    cv2.imwrite(os.path.join(CAPTURES_DIR, f"{zone_id}_{ts}_nir.png"), nir_frame)

    return new_alerts


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
    push_heartbeat(sb, "scanning", 0, 0, 0)
    print("  Initial heartbeat pushed 💓")

    # --- Init AI Classifier (optional) ---
    ai_classify = None
    if os.path.isfile(AI_MODEL_PATH):
        try:
            print("  Loading DINOv2 classifier …", end=" ", flush=True)
            # Import here to avoid hard dependency on torch for non-AI runs
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from PIL import Image
            from torchvision import transforms as T

            AI_CLASSES = ["healthy_vegetation", "stressed_soil"]
            AI_IMAGE_SIZE = 168
            AI_MEAN = [0.485, 0.456, 0.406]
            AI_STD = [0.229, 0.224, 0.225]

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
                        nn.Dropout(0.1), nn.Linear(self.backbone.embed_dim, 2)
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
                T.Resize((AI_IMAGE_SIZE, AI_IMAGE_SIZE)),
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
                return {
                    "class": AI_CLASSES[idx],
                    "confidence": float(probs[idx]) * 100,
                    "healthy_prob": float(probs[0]),
                    "stressed_prob": float(probs[1]),
                }

            ai_classify = _classify_frame
            val_acc = _ckpt.get("val_acc", "?")
            print(f"DINOv2 ViT-S/14 loaded (val_acc={val_acc}%) ✅")
        except Exception as e:
            print(f"\n  ⚠️  AI classifier not available: {e}")
            print("      (NDVI-only mode — scans will continue without AI)")
    else:
        print(f"  🤖 No AI model at {AI_MODEL_PATH} — NDVI-only mode")

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

            for zone_id in zones:
                if not _running:
                    break
                try:
                    new_alerts = scan_zone(cap, efw, sb, zone_id, scan_count, alert_count,
                                           weather, ai_classify=ai_classify)
                    scan_count += 1
                    alert_count += new_alerts
                except Exception as e:
                    print(f"  ⚠️  Error scanning {zone_id}: {e}")
                    traceback.print_exc()

            print(f"  Scans: {scan_count}  Alerts: {alert_count}")

            if args.once:
                print("\n  --once flag set. Exiting.")
                break

            # Sleep until next cycle
            elapsed = time.time() - cycle_start
            sleep_time = max(0, args.interval - elapsed)
            if sleep_time > 0 and _running:
                print(f"  Next cycle in {int(sleep_time)}s … (Ctrl+C to stop)\n")
                # Sleep in small increments so SIGINT is responsive
                wake_at = time.time() + sleep_time
                while time.time() < wake_at and _running:
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

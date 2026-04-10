#!/usr/bin/env python3
"""
generate_sim_data.py — Generate OnAIR Simulation Telemetry
===========================================================
Converts ndvi_central_park.csv (296 satellite observations) into
chroma_shift_sim.csv for OnAIR CSV adapter testing.

Each row in the output represents one telemetry frame as it would
arrive from the cFS Software Bus, containing:
  - Camera telemetry (Red/NIR DN values from satellite reflectance)
  - Weather observations (synthetic seasonal model for NYC)
  - System health (nominal Jetson Orin Nano values)

Usage:
    python generate_sim_data.py
"""

import csv
import math
import os
import random
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

INPUT_CSV = os.path.join(PROJECT_ROOT, "ndvi_central_park.csv")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "chroma_shift_sim.csv")

HEADERS = [
    "timestamp_utc",
    "zone_id",
    "red_mean_dn",
    "red_exposure",
    "red_gain",
    "nir_mean_dn",
    "nir_exposure",
    "nir_gain",
    "frame_width",
    "frame_height",
    "cloud_cover_pct",
    "temp_c",
    "humidity_pct",
    "wind_speed_ms",
    "precip_1h_mm",
    "cpu_temp_c",
    "storage_used_gb",
    "uptime_seconds",
    "operating_mode",
]

# Zone rotation for simulation
ZONES = ["T1", "A1", "A2", "B1", "B2", "B3", "C1"]


def synthetic_weather(date_str):
    """Generate plausible NYC weather from date using seasonal model."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        dt = datetime(2023, 6, 15)

    day_of_year = dt.timetuple().tm_yday
    # Seasonal temperature model: ~0C in Jan, ~26C in July
    temp = 13.0 + 13.0 * math.sin(2 * math.pi * (day_of_year - 80) / 365)
    temp += random.gauss(0, 3)

    # Humidity: higher in summer
    humidity = 55 + 15 * math.sin(2 * math.pi * (day_of_year - 80) / 365)
    humidity += random.gauss(0, 10)
    humidity = max(20, min(100, humidity))

    # Cloud cover: random with slight seasonal bias
    cloud = random.uniform(0, 80) + 10 * math.cos(2 * math.pi * day_of_year / 365)
    cloud = max(0, min(100, cloud))

    # Wind: 2-8 m/s average
    wind = random.uniform(2, 8)

    # Precipitation: higher probability in spring/fall
    precip = 0.0
    if random.random() < 0.15:
        precip = random.uniform(0.5, 5.0)

    return {
        "cloud_cover_pct": round(cloud, 1),
        "temp_c": round(temp, 1),
        "humidity_pct": round(humidity, 1),
        "wind_speed_ms": round(wind, 1),
        "precip_1h_mm": round(precip, 1),
    }


def reflectance_to_dn(reflectance, max_dn=255):
    """Convert satellite reflectance [0-1] to simulated camera DN [0-255]."""
    return round(min(max_dn, max(0, reflectance * max_dn * 2)), 1)


def main():
    if not os.path.isfile(INPUT_CSV):
        print(f"ERROR: {INPUT_CSV} not found")
        return

    rows = []
    with open(INPUT_CSV, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            zone = ZONES[i % len(ZONES)]
            date = row.get("Date", "2023-06-15")
            weather = synthetic_weather(date)

            # Convert satellite reflectance to simulated camera DN
            red_refl = float(row.get("Red", 0.05))
            nir_refl = float(row.get("NIR", 0.15))

            sim_row = {
                "timestamp_utc": f"{date}T12:00:00Z",
                "zone_id": zone,
                "red_mean_dn": reflectance_to_dn(red_refl),
                "red_exposure": 0.01,
                "red_gain": 1.0,
                "nir_mean_dn": reflectance_to_dn(nir_refl),
                "nir_exposure": 0.01,
                "nir_gain": 1.0,
                "frame_width": 10,
                "frame_height": 10,
                **weather,
                "cpu_temp_c": round(42.0 + random.gauss(0, 3), 1),
                "storage_used_gb": round(4.2 + i * 0.01, 2),
                "uptime_seconds": 3600 + i * 600,
                "operating_mode": "full",
            }
            rows.append(sim_row)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} simulation frames -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

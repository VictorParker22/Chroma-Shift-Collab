"""
weather.py — Chroma-Shift NWS Weather Integration
====================================================
Pulls live weather data from the National Weather Service API
for NYC Central Park (station KNYC) and correlates with NDVI.

NWS API is free, no API key required.
Endpoint: https://api.weather.gov/stations/KNYC/observations/latest

Correlation logic:
  - High temp + low rainfall → potential drought stress → expect NDVI drop
  - Extended rainfall → healthy conditions → expect stable/high NDVI
  - Frost events → cold stress → expect NDVI drop
"""

import os
import sys
import json
import urllib.request
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# NWS Configuration
# ---------------------------------------------------------------------------
# Central Park, NYC
NWS_STATION = "KNYC"
NWS_GRID = "OKX/34,38"
LATITUDE = 40.7851
LONGITUDE = -73.9683

OBSERVATIONS_URL = f"https://api.weather.gov/stations/{NWS_STATION}/observations/latest"
FORECAST_URL = f"https://api.weather.gov/gridpoints/{NWS_GRID}/forecast"
HOURLY_FORECAST_URL = f"https://api.weather.gov/gridpoints/{NWS_GRID}/forecast/hourly"

# User-Agent required by NWS API
HEADERS = {
    "User-Agent": "(Chroma-Shift Jetson, chromashift@example.com)",
    "Accept": "application/geo+json",
}

# Stress correlation thresholds
HIGH_TEMP_C = 35.0         # above this → heat stress risk
FROST_TEMP_C = 0.0         # below this → frost damage risk
DROUGHT_DAYS = 7           # consecutive dry days → drought stress
LOW_HUMIDITY_PCT = 30.0    # below this → evapotranspiration stress


# ═══════════════════════════════════════════════════════════════════════════
# API Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _nws_get(url):
    """Fetch JSON from the NWS API with proper headers."""
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  ⚠️  NWS API error: {e}")
        return None


def _safe_val(obj):
    """Extract value from a NWS measurement object, return None if missing."""
    if obj is None:
        return None
    val = obj.get("value")
    return val


# ═══════════════════════════════════════════════════════════════════════════
# Current Conditions
# ═══════════════════════════════════════════════════════════════════════════

def get_current_weather():
    """
    Fetch latest observation from KNYC (Central Park).

    Returns dict with:
      temp_c, temp_f, humidity, wind_speed_ms, wind_direction_deg,
      description, precip_1h_mm, precip_6h_mm, pressure_pa,
      visibility_m, dewpoint_c, timestamp
    """
    data = _nws_get(OBSERVATIONS_URL)
    if not data:
        return None

    props = data.get("properties", {})

    temp_c = _safe_val(props.get("temperature"))
    humidity = _safe_val(props.get("relativeHumidity"))
    wind = _safe_val(props.get("windSpeed"))
    wind_dir = _safe_val(props.get("windDirection"))
    precip_1h = _safe_val(props.get("precipitationLastHour"))
    precip_6h = _safe_val(props.get("precipitationLast6Hours"))
    pressure = _safe_val(props.get("barometricPressure"))
    visibility = _safe_val(props.get("visibility"))
    dewpoint = _safe_val(props.get("dewpoint"))
    desc = props.get("textDescription", "")
    ts = props.get("timestamp", "")

    weather = {
        "temp_c": round(temp_c, 1) if temp_c is not None else None,
        "temp_f": round(temp_c * 9 / 5 + 32, 1) if temp_c is not None else None,
        "humidity_pct": round(humidity, 1) if humidity is not None else None,
        "wind_speed_ms": round(wind, 1) if wind is not None else None,
        "wind_direction_deg": round(wind_dir) if wind_dir is not None else None,
        "description": desc,
        "precip_1h_mm": round(precip_1h, 2) if precip_1h is not None else 0.0,
        "precip_6h_mm": round(precip_6h, 2) if precip_6h is not None else 0.0,
        "pressure_pa": round(pressure) if pressure is not None else None,
        "visibility_m": round(visibility) if visibility is not None else None,
        "dewpoint_c": round(dewpoint, 1) if dewpoint is not None else None,
        "timestamp": ts,
        "station": NWS_STATION,
    }
    return weather


# ═══════════════════════════════════════════════════════════════════════════
# Weather-NDVI Correlation
# ═══════════════════════════════════════════════════════════════════════════

def assess_weather_stress(weather):
    """
    Analyze current weather for conditions that correlate with NDVI changes.

    Returns dict with:
      risk_factors: list of active risk strings
      stress_modifier: float adjustment to expected NDVI (-0.1 to 0.05)
      summary: human-readable summary
    """
    if not weather:
        return {"risk_factors": [], "stress_modifier": 0.0, "summary": "Weather data unavailable"}

    risks = []
    modifier = 0.0

    temp = weather.get("temp_c")
    humidity = weather.get("humidity_pct")
    precip = weather.get("precip_1h_mm") or 0
    precip_6h = weather.get("precip_6h_mm") or 0

    # Heat stress
    if temp is not None and temp > HIGH_TEMP_C:
        risks.append(f"🌡️ High temp ({temp}°C) — heat stress risk")
        modifier -= 0.05

    # Frost
    if temp is not None and temp <= FROST_TEMP_C:
        risks.append(f"🥶 Frost ({temp}°C) — cold damage risk")
        modifier -= 0.08

    # Low humidity → high evapotranspiration
    if humidity is not None and humidity < LOW_HUMIDITY_PCT:
        risks.append(f"💨 Low humidity ({humidity}%) — moisture stress")
        modifier -= 0.03

    # Recent rainfall (positive for vegetation)
    if precip_6h > 5.0:
        risks.append(f"🌧️ Good rainfall ({precip_6h}mm/6h) — beneficial")
        modifier += 0.03
    elif precip_6h == 0 and precip == 0:
        risks.append("☀️ No recent precipitation — monitor for dryness")
        modifier -= 0.02

    # No risks found
    if not risks:
        risks.append("✅ Conditions normal")

    summary_parts = []
    if temp is not None:
        summary_parts.append(f"{temp}°C")
    if humidity is not None:
        summary_parts.append(f"{humidity:.0f}% RH")
    if precip_6h > 0:
        summary_parts.append(f"{precip_6h}mm rain/6h")
    desc = weather.get("description", "")
    if desc:
        summary_parts.append(desc)

    return {
        "risk_factors": risks,
        "stress_modifier": round(modifier, 3),
        "summary": " | ".join(summary_parts) if summary_parts else "N/A",
    }


def correlate_ndvi_weather(ndvi, baseline, weather):
    """
    Combine NDVI reading with weather assessment to provide
    a more informed stress evaluation.

    Returns enhanced analysis dict.
    """
    weather_assessment = assess_weather_stress(weather)
    drop = baseline - ndvi
    adjusted_baseline = baseline + weather_assessment["stress_modifier"]

    # If weather accounts for the NDVI drop, lower severity
    weather_explains_drop = False
    if drop > 0.05 and weather_assessment["stress_modifier"] < -0.03:
        weather_explains_drop = True

    return {
        "ndvi": round(ndvi, 4),
        "baseline": baseline,
        "adjusted_baseline": round(adjusted_baseline, 4),
        "drop": round(drop, 4),
        "weather": weather,
        "weather_risks": weather_assessment["risk_factors"],
        "weather_summary": weather_assessment["summary"],
        "stress_modifier": weather_assessment["stress_modifier"],
        "weather_explains_drop": weather_explains_drop,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Supabase Push
# ═══════════════════════════════════════════════════════════════════════════

def push_weather(sb, weather, ndvi=None, field_zone=None):
    """Push a weather reading to the weather_data table."""
    if not weather:
        return

    record = {
        "station": weather.get("station", NWS_STATION),
        "temp_c": weather.get("temp_c"),
        "temp_f": weather.get("temp_f"),
        "humidity_pct": weather.get("humidity_pct"),
        "wind_speed_ms": weather.get("wind_speed_ms"),
        "wind_direction_deg": weather.get("wind_direction_deg"),
        "description": weather.get("description"),
        "precip_1h_mm": weather.get("precip_1h_mm"),
        "precip_6h_mm": weather.get("precip_6h_mm"),
        "pressure_pa": weather.get("pressure_pa"),
        "visibility_m": weather.get("visibility_m"),
        "dewpoint_c": weather.get("dewpoint_c"),
        "observation_time": weather.get("timestamp"),
    }

    if ndvi is not None:
        record["ndvi_at_reading"] = round(ndvi, 4)
    if field_zone:
        record["field_zone"] = field_zone

    sb.table("weather_data").insert(record).execute()


# ═══════════════════════════════════════════════════════════════════════════
# CLI test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Chroma-Shift — NWS Weather (Central Park)               ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    w = get_current_weather()
    if w:
        print(f"  Station:     {w['station']}")
        print(f"  Temp:        {w['temp_c']}°C ({w['temp_f']}°F)")
        print(f"  Humidity:    {w['humidity_pct']}%")
        print(f"  Wind:        {w['wind_speed_ms']} m/s")
        print(f"  Conditions:  {w['description']}")
        print(f"  Precip 1h:   {w['precip_1h_mm']} mm")
        print(f"  Precip 6h:   {w['precip_6h_mm']} mm")
        print(f"  Timestamp:   {w['timestamp']}")

        print("\n  --- Stress Assessment ---")
        assessment = assess_weather_stress(w)
        for r in assessment["risk_factors"]:
            print(f"  {r}")
        print(f"  NDVI modifier: {assessment['stress_modifier']:+.3f}")

        print(f"\n  --- Correlation Example (NDVI=0.60, baseline=0.74) ---")
        corr = correlate_ndvi_weather(0.60, 0.74, w)
        print(f"  Drop:              {corr['drop']}")
        print(f"  Adjusted baseline: {corr['adjusted_baseline']}")
        print(f"  Weather explains:  {corr['weather_explains_drop']}")
    else:
        print("  Failed to fetch weather data.")

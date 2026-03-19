#!/usr/bin/env python3
"""
crop_advisor.py — Context-Aware Farmer Advisory System
=======================================================
Fuses NDVI readings, seasonal baseline, weather data, classification
results, and historical trends to generate actionable farmer advisories.

Instead of just "healthy" or "stressed", the advisor explains:
  - WHAT is happening (NDVI context in plain English)
  - WHY it's happening (seasonal, weather, or anomaly-driven)
  - WHAT TO DO (specific, actionable recommendations)
  - CONFIDENCE LEVEL (how trustworthy this reading is)

Used by:
  - jetson_main.py → push to Supabase dashboard for farmer view
  - field_validate.py → live field validation reports
"""

from datetime import datetime, timezone


# ═══════════════════════════════════════════════════════════════════════════
# NDVI Interpretation Ranges
# ═══════════════════════════════════════════════════════════════════════════

NDVI_RANGES = [
    (-1.0, 0.00, "water_or_bare",   "Water body or bare surface (no vegetation detected)"),
    (0.00, 0.15, "bare_soil",       "Bare soil or dead/dormant vegetation"),
    (0.15, 0.30, "sparse",          "Sparse or stressed vegetation — low chlorophyll activity"),
    (0.30, 0.45, "moderate",        "Moderate vegetation — early growth or late-season decline"),
    (0.45, 0.60, "active",          "Active, growing vegetation — healthy chlorophyll levels"),
    (0.60, 0.75, "dense_healthy",   "Dense, healthy vegetation — peak growing season"),
    (0.75, 1.00, "very_dense",      "Very dense canopy — strong photosynthetic activity"),
]


def interpret_ndvi(ndvi):
    """Return the vegetation category and description for a given NDVI."""
    for low, high, category, description in NDVI_RANGES:
        if low <= ndvi < high:
            return category, description
    return "very_dense", NDVI_RANGES[-1][3]


# ═══════════════════════════════════════════════════════════════════════════
# Seasonal Context
# ═══════════════════════════════════════════════════════════════════════════

SEASON_CONTEXT = {
    "winter": {
        "expected": "Low NDVI is normal — most vegetation is dormant in NYC winters.",
        "healthy_note": "Vegetation appears green despite winter — this may be evergreen species or a mild spell.",
        "stressed_note": "Stress in winter is unusual — could indicate freeze damage or salt exposure from road treatment.",
    },
    "spring": {
        "expected": "NDVI should be rising as plants break dormancy (green-up phase).",
        "healthy_note": "Vegetation is greening up on schedule — healthy spring emergence.",
        "stressed_note": "Low NDVI during spring green-up may indicate delayed emergence, late frost damage, or insufficient moisture.",
    },
    "summer": {
        "expected": "Peak growing season — NDVI should be at its highest (0.65–0.75 for NYC).",
        "healthy_note": "Vegetation at peak health — strong photosynthesis and canopy density.",
        "stressed_note": "Low NDVI during peak season is a strong stress indicator — likely water stress, heat stress, disease, or pest damage.",
    },
    "autumn": {
        "expected": "NDVI declining as leaves senesce — yellowing and leaf drop expected.",
        "healthy_note": "Normal autumn senescence — vegetation naturally winding down for winter.",
        "stressed_note": "Premature decline may indicate drought carry-over, early frost, or fungal infection.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Stress Diagnostics
# ═══════════════════════════════════════════════════════════════════════════

def diagnose_stress(ndvi, baseline, weather=None):
    """
    Attempt to diagnose the cause of stress based on available data.

    Returns a list of (cause, confidence, explanation, recommendation) tuples.
    """
    drop = baseline - ndvi
    diagnoses = []

    if weather:
        temp = weather.get("temp_c")
        humidity = weather.get("humidity_pct")
        precip = weather.get("precip_1h_mm", 0)
        wind = weather.get("wind_speed_ms")
        description = weather.get("description", "").lower()

        # Water stress: high temp, low humidity, no rain
        if temp and temp > 30 and precip < 0.5:
            diagnoses.append((
                "water_stress", 0.85,
                f"High temperature ({temp}°C) with no recent precipitation suggests water deficit.",
                "Increase irrigation by 20-30% for the next 48 hours. Water deeply in early morning to reduce evaporation."
            ))
        elif temp and temp > 25 and humidity and humidity < 40:
            diagnoses.append((
                "water_stress", 0.70,
                f"Warm conditions ({temp}°C) with low humidity ({humidity}%) indicate potential water stress.",
                "Monitor soil moisture. Consider irrigation if conditions persist for >24 hours."
            ))

        # Heat stress: very high temp
        if temp and temp > 35:
            diagnoses.append((
                "heat_stress", 0.80,
                f"Extreme heat ({temp}°C) can damage cellular processes and reduce photosynthesis.",
                "Provide shade cloth for sensitive crops. Increase watering frequency. Avoid fertilizer application during heat."
            ))

        # Frost damage: near or below freezing
        if temp and temp < 2:
            diagnoses.append((
                "frost_damage", 0.75,
                f"Near-freezing temperature ({temp}°C) may have caused frost damage to tender growth.",
                "Inspect leaves for browning/wilting at tips. Cover sensitive plants if frost is expected again tonight."
            ))

        # Overwatering / waterlogging
        if precip and precip > 10:
            diagnoses.append((
                "waterlogging", 0.65,
                f"Heavy precipitation ({precip}mm/hr) may have caused waterlogging and root oxygen deprivation.",
                "Ensure drainage channels are clear. Avoid additional irrigation for 48-72 hours."
            ))

        # Wind damage
        if wind and wind > 15:
            diagnoses.append((
                "wind_damage", 0.50,
                f"High winds ({wind} m/s) can cause mechanical damage and desiccation.",
                "Inspect for broken stems/branches. Install windbreaks for young plants."
            ))

    # Generic diagnoses based on NDVI pattern alone
    if not diagnoses:
        if drop > 0.15:
            diagnoses.append((
                "significant_decline", 0.60,
                f"NDVI is {drop:.2f} below seasonal baseline — a significant departure that warrants investigation.",
                "Send a scout to visually inspect the area. Check for discoloration, wilting, pest damage, or disease signs."
            ))
        elif drop > 0.08:
            diagnoses.append((
                "moderate_decline", 0.50,
                f"NDVI is {drop:.2f} below seasonal baseline — moderate but worth monitoring.",
                "Continue monitoring. If decline persists for 3+ consecutive readings, investigate on-site."
            ))
        else:
            diagnoses.append((
                "mild_variance", 0.40,
                f"NDVI is slightly below baseline ({drop:.2f}) — likely within normal variability.",
                "No immediate action needed. Continue scheduled monitoring."
            ))

    return diagnoses


# ═══════════════════════════════════════════════════════════════════════════
# Main Advisory Generator
# ═══════════════════════════════════════════════════════════════════════════

def generate_advisory(ndvi, seasonal_data, classification=None, weather=None,
                      historical_ndvi=None, zone_name="Field Zone"):
    """
    Generate a comprehensive farmer advisory from all available data.

    Args:
        ndvi:             float, current NDVI reading
        seasonal_data:    dict from radiometric.get_seasonal_baseline()
                          {baseline, week, season}
        classification:   dict from DINOv2 classifier
                          {class, confidence, healthy_prob, stressed_prob}
        weather:          dict from NWS
                          {temp_c, humidity_pct, precip_1h_mm, description, wind_speed_ms}
        historical_ndvi:  list of recent NDVI values (for trend)
        zone_name:        str, human-readable zone name

    Returns:
        dict with comprehensive advisory
    """
    baseline = seasonal_data["baseline"]
    season = seasonal_data["season"]
    week = seasonal_data["week"]
    ndvi_cat, ndvi_desc = interpret_ndvi(ndvi)

    # Deviation from seasonal baseline
    deviation = ndvi - baseline
    deviation_pct = (deviation / baseline * 100) if baseline > 0 else 0

    # Overall health assessment
    if deviation >= 0.05:
        health = "above_baseline"
        health_label = "✅ Above Seasonal Average"
        health_summary = (
            f"NDVI of {ndvi:.3f} is {abs(deviation):.3f} ABOVE the expected seasonal "
            f"baseline of {baseline:.3f} for week {week} ({season}). "
            f"Vegetation is performing better than average."
        )
    elif deviation >= -0.05:
        health = "on_track"
        health_label = "✅ On Track"
        health_summary = (
            f"NDVI of {ndvi:.3f} is within normal range of the seasonal "
            f"baseline {baseline:.3f} for week {week} ({season}). "
            f"{SEASON_CONTEXT[season]['expected']}"
        )
    elif deviation >= -0.10:
        health = "watch"
        health_label = "🟡 Below Average — Monitor"
        health_summary = (
            f"NDVI of {ndvi:.3f} is {abs(deviation):.3f} below the seasonal "
            f"baseline of {baseline:.3f}. This is outside normal variance. "
            f"{SEASON_CONTEXT[season]['stressed_note']}"
        )
    else:
        health = "alert"
        health_label = "🔴 Significant Stress Detected"
        health_summary = (
            f"NDVI of {ndvi:.3f} is {abs(deviation):.3f} BELOW the seasonal "
            f"baseline of {baseline:.3f} — a {abs(deviation_pct):.0f}% deficit. "
            f"{SEASON_CONTEXT[season]['stressed_note']}"
        )

    # NDVI explanation in farmer-friendly terms
    ndvi_explanation = (
        f"The Normalized Difference Vegetation Index (NDVI) measures how "
        f"green and active the vegetation is by analyzing reflected light. "
        f"Your reading of {ndvi:.3f} indicates: {ndvi_desc}."
    )

    # Seasonal context
    seasonal_explanation = (
        f"For {season} in NYC (week {week}), the historical average NDVI "
        f"from 5 years of NASA satellite data is {baseline:.3f}. "
        f"{SEASON_CONTEXT[season]['expected']}"
    )

    # Trend analysis
    trend = None
    if historical_ndvi and len(historical_ndvi) >= 3:
        recent = historical_ndvi[-3:]
        if all(recent[i] < recent[i-1] for i in range(1, len(recent))):
            trend = {
                "direction": "declining",
                "description": f"NDVI has been declining over the last {len(recent)} readings ({recent[0]:.3f} → {recent[-1]:.3f}). Monitor closely.",
                "icon": "📉"
            }
        elif all(recent[i] > recent[i-1] for i in range(1, len(recent))):
            trend = {
                "direction": "improving",
                "description": f"NDVI is improving over the last {len(recent)} readings ({recent[0]:.3f} → {recent[-1]:.3f}). Recovery underway.",
                "icon": "📈"
            }
        else:
            trend = {
                "direction": "stable",
                "description": f"NDVI is relatively stable over recent readings.",
                "icon": "➡️"
            }

    # Diagnosis and recommendations
    diagnoses = []
    primary_recommendation = None

    if health in ("watch", "alert"):
        diagnoses = diagnose_stress(ndvi, baseline, weather)
        if diagnoses:
            primary_recommendation = diagnoses[0][3]  # top recommendation
    elif health in ("above_baseline", "on_track"):
        primary_recommendation = (
            f"Vegetation in {zone_name} is healthy. Continue current management practices. "
            f"Next optimal scan time will be determined by the adaptive scheduler."
        )

    # Weather context
    weather_summary = None
    if weather:
        parts = []
        if weather.get("temp_c") is not None:
            parts.append(f"{weather['temp_c']}°C")
        if weather.get("humidity_pct") is not None:
            parts.append(f"{weather['humidity_pct']}% humidity")
        if weather.get("description"):
            parts.append(weather["description"])
        if weather.get("precip_1h_mm", 0) > 0:
            parts.append(f"{weather['precip_1h_mm']}mm/hr rain")
        weather_summary = " | ".join(parts)

    # Classification context
    ai_assessment = None
    if classification:
        cls = classification.get("class", "unknown")
        conf = classification.get("confidence", 0)
        if cls == "healthy_vegetation":
            ai_assessment = (
                f"The onboard AI classifier (DINOv2, 99.95% validated accuracy) "
                f"confirms healthy vegetation with {conf:.1f}% confidence."
            )
        elif cls == "stressed_vegetation":
            ai_assessment = (
                f"The onboard AI classifier (DINOv2) detects stressed vegetation "
                f"with {conf:.1f}% confidence. Plants are present but showing "
                f"signs of stress (yellowing, wilting, or reduced vigor)."
            )
        elif cls == "bare_soil":
            ai_assessment = (
                f"The onboard AI classifier (DINOv2) identifies this area as "
                f"bare soil with {conf:.1f}% confidence. No visible vegetation "
                f"detected — this may be exposed ground, tilled earth, or dormant "
                f"cover awaiting planting."
            )
            # Override health label for bare soil
            health_label = "🟤 Bare Soil Detected"
            health_summary = (
                f"No active vegetation detected. NDVI of {ndvi:.3f} with bare soil "
                f"classification indicates exposed ground. "
                f"Any NDVI signal may be from soil organic matter or microbial activity."
            )
            if not primary_recommendation:
                primary_recommendation = (
                    "Consider whether this area should have vegetation. If so, "
                    "check for erosion risk and consider cover crop planting."
                )
        elif cls == "non_vegetation":
            ai_assessment = (
                f"The onboard AI classifier (DINOv2) identifies this as a "
                f"non-vegetation surface with {conf:.1f}% confidence "
                f"(concrete, asphalt, rock, or water)."
            )
            health_label = "⬛ Non-Vegetation Surface"
            health_summary = (
                f"This area is not vegetation — classified as built surface or water. "
                f"NDVI readings from non-vegetation surfaces are not meaningful for "
                f"crop health assessment."
            )
            primary_recommendation = (
                "This surface does not contain vegetation. Reposition sensor "
                "to scan a vegetated area for agricultural monitoring."
            )
        else:
            ai_assessment = (
                f"The onboard AI classifier (DINOv2) classifies this as "
                f"'{cls}' with {conf:.1f}% confidence."
            )

    # Quality/confidence assessment
    data_quality = []
    if weather:
        data_quality.append("✅ Weather data available")
    else:
        data_quality.append("⚠️ No weather data — stress diagnosis is less precise")
    if classification:
        data_quality.append(f"✅ AI classification: {classification.get('class')} ({classification.get('confidence', 0):.0f}%)")
    if historical_ndvi and len(historical_ndvi) >= 3:
        data_quality.append(f"✅ Historical trend: {len(historical_ndvi)} readings")
    else:
        data_quality.append("⚠️ Limited history — trend analysis unavailable")

    # Build the full advisory
    advisory = {
        # Core results
        "ndvi": round(ndvi, 4),
        "baseline": baseline,
        "deviation": round(deviation, 4),
        "deviation_pct": round(deviation_pct, 1),

        # Health assessment
        "health_status": health,
        "health_label": health_label,
        "health_summary": health_summary,

        # Explanations (farmer-friendly)
        "ndvi_explanation": ndvi_explanation,
        "seasonal_context": seasonal_explanation,
        "ndvi_category": ndvi_cat,
        "ndvi_description": ndvi_desc,
        "season": season,
        "week": week,

        # AI assessment
        "ai_assessment": ai_assessment,

        # Weather
        "weather_summary": weather_summary,

        # Trend
        "trend": trend,

        # Diagnosis and action items
        "diagnoses": [
            {
                "cause": d[0],
                "confidence": d[1],
                "explanation": d[2],
                "recommendation": d[3],
            }
            for d in diagnoses
        ],
        "primary_recommendation": primary_recommendation,

        # Data quality
        "data_quality": data_quality,

        # Timestamp
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "zone": zone_name,
    }

    return advisory


def format_advisory_text(advisory):
    """Format advisory as human-readable text for terminal/log output."""
    lines = []
    lines.append("═" * 62)
    lines.append(f"  CROP ADVISORY — {advisory['zone']}")
    lines.append(f"  {advisory['timestamp']}")
    lines.append("═" * 62)
    lines.append("")
    lines.append(f"  {advisory['health_label']}")
    lines.append(f"  {advisory['health_summary']}")
    lines.append("")
    lines.append(f"  📊 NDVI: {advisory['ndvi']:.4f}  |  Baseline: {advisory['baseline']:.4f}  |  Deviation: {advisory['deviation']:+.4f} ({advisory['deviation_pct']:+.1f}%)")
    lines.append(f"  🌱 {advisory['ndvi_description']}")
    lines.append("")
    lines.append(f"  📅 {advisory['seasonal_context']}")
    lines.append("")

    if advisory.get("ai_assessment"):
        lines.append(f"  🤖 {advisory['ai_assessment']}")
        lines.append("")

    if advisory.get("weather_summary"):
        lines.append(f"  🌤️  Weather: {advisory['weather_summary']}")
        lines.append("")

    if advisory.get("trend"):
        t = advisory["trend"]
        lines.append(f"  {t['icon']} Trend: {t['description']}")
        lines.append("")

    if advisory["diagnoses"]:
        lines.append("  🔍 POSSIBLE CAUSES:")
        for d in advisory["diagnoses"]:
            conf_bar = "█" * int(d["confidence"] * 10) + "░" * (10 - int(d["confidence"] * 10))
            lines.append(f"     [{conf_bar}] {d['cause'].replace('_', ' ').title()}")
            lines.append(f"       {d['explanation']}")
            lines.append("")

    if advisory.get("primary_recommendation"):
        lines.append(f"  💡 RECOMMENDATION:")
        lines.append(f"     {advisory['primary_recommendation']}")
        lines.append("")

    lines.append("  📋 DATA QUALITY:")
    for q in advisory["data_quality"]:
        lines.append(f"     {q}")

    lines.append("")
    lines.append("═" * 62)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Quick Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Simulate a March reading in Central Park
    from radiometric import get_seasonal_baseline

    seasonal = get_seasonal_baseline()

    # Healthy scenario
    print("\n" + "=" * 62)
    print("  SCENARIO 1: Healthy vegetation in March")
    print("=" * 62)
    advisory = generate_advisory(
        ndvi=0.38,
        seasonal_data=seasonal,
        classification={"class": "healthy_vegetation", "confidence": 98.5},
        weather={"temp_c": 12, "humidity_pct": 55, "description": "Partly cloudy"},
        zone_name="Central Park — Great Lawn"
    )
    print(format_advisory_text(advisory))

    # Stressed scenario
    print("\n" + "=" * 62)
    print("  SCENARIO 2: Heat-stressed vegetation in summer")
    print("=" * 62)
    summer_seasonal = {"baseline": 0.72, "week": 25, "season": "summer"}
    advisory = generate_advisory(
        ndvi=0.48,
        seasonal_data=summer_seasonal,
        classification={"class": "stressed_soil", "confidence": 92.3},
        weather={"temp_c": 37, "humidity_pct": 28, "precip_1h_mm": 0,
                 "description": "Clear, extreme heat warning"},
        historical_ndvi=[0.65, 0.58, 0.52, 0.48],
        zone_name="Central Park — Sheep Meadow"
    )
    print(format_advisory_text(advisory))

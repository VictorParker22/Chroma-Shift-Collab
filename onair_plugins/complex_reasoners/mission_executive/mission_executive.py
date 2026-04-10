"""
mission_executive.py — OnAIR Complex Reasoner Plugin
=====================================================
The executive decision-maker for the Chroma-Shift SmallSat mission.
Fuses ALL upstream plugin outputs into unified mission decisions:
  - Stress detection and alert generation
  - Operating mode management (FULL > REDUCED > SAFE > EMERGENCY)
  - Downlink prioritization (bandwidth-constrained SmallSat)
  - Command generation (capture timing, filter position, power mode)
  - Triage decision logging

Extracted from jetson_main.py scan_zone() orchestration logic.

OnAIR role: Complex Reasoner (PCR)
  - Receives outputs from ALL Knowledge Representation, Learner,
    and Planner plugins
  - Produces mission-level decisions and commands
"""

import sys
import os
import json
from datetime import datetime, timezone

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Field zone definitions and stress thresholds
# (extracted from jetson_main.py constants)
# ---------------------------------------------------------------------------

FIELD_ZONES = {
    "A1": {"name": "North Orchard",  "baseline": 0.72},
    "A2": {"name": "North Meadow",   "baseline": 0.68},
    "B1": {"name": "East Ridge",     "baseline": 0.65},
    "B2": {"name": "East Valley",    "baseline": 0.71},
    "B3": {"name": "South Field",    "baseline": 0.74},
    "C1": {"name": "West Terrace",   "baseline": 0.69},
    "T1": {"name": "USS Maine Monument", "baseline": 0.6493},
}

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


# ---------------------------------------------------------------------------
# Stress detection (extracted from jetson_main.py lines 244-261)
# ---------------------------------------------------------------------------

def detect_stress(ndvi, baseline):
    """Compare NDVI to baseline and determine stress severity."""
    drop = baseline - ndvi
    for threshold, severity, status in STRESS_LEVELS:
        if drop > threshold:
            return severity, status, drop
    return "low", "healthy", drop


def build_recommendation(stress_type, drop, zone_name):
    """Generate actionable recommendation based on stress type."""
    pct = int(drop * 100)
    recs = {
        "water_stress":   f"Increase irrigation in {zone_name} by {pct}% for the next 48 hours.",
        "heat_stress":    f"Increase shading or irrigation frequency in {zone_name} during peak hours.",
        "nutrient_deficiency": f"Apply nitrogen-rich fertilizer to {zone_name}. Recheck in 7 days.",
        "disease_detected": f"Inspect {zone_name} for fungal/bacterial signs. Consider targeted fungicide.",
        "pest_damage":    f"Deploy pest traps in {zone_name}. Inspect leaf undersides for damage.",
    }
    return recs.get(stress_type, f"Investigate {zone_name}.")


class Plugin:
    """OnAIR Complex Reasoner plugin — Mission Executive."""

    def __init__(self, _name, _headers):
        self.component_name = _name
        self.headers = _headers
        self._result = None
        self._scan_count = 0
        self._alert_count = 0
        self._triage_log = []

    def update(self, low_level_data=[], high_level_data={}):
        """Fuse all plugin outputs into mission decisions.

        Args:
            low_level_data: Raw telemetry frame (unused at this level).
            high_level_data: Dict containing ALL upstream plugin outputs:
              - radiometric: corrected NDVI, seasonal, solar, quality flags
              - scene_analysis: VILA scene composition and corrections
              - vegetation_classifier: 4-class AI classification
              - anomaly_detector: anomalies, health score, operating mode
              - observation_scheduler: interval, priority, power mode
              - crop_advisor: farmer advisory
        """
        self._scan_count += 1
        now = datetime.now(timezone.utc)

        # Gather all upstream outputs
        radiometric = high_level_data.get("radiometric", {}) or {}
        scene = high_level_data.get("scene_analysis", {}) or {}
        classifier = high_level_data.get("vegetation_classifier", {}) or {}
        anomaly = high_level_data.get("anomaly_detector", {}) or {}
        scheduler = high_level_data.get("observation_scheduler", {}) or {}
        advisory = high_level_data.get("crop_advisor", {}) or {}

        # Core values
        ndvi = radiometric.get("mean_ndvi")
        raw_ndvi = radiometric.get("raw_ndvi")
        zone_id = radiometric.get("zone_id", "T1")
        seasonal = radiometric.get("seasonal", {})
        solar = radiometric.get("solar", {})
        quality_flags = radiometric.get("quality_flags", [])

        # Determine baseline (seasonal or static)
        baseline = None
        season = "unknown"
        if seasonal:
            baseline = seasonal.get("baseline")
            season = seasonal.get("season", "unknown")
        if baseline is None:
            zone_info = FIELD_ZONES.get(str(zone_id), {})
            baseline = zone_info.get("baseline", 0.6493)

        # Operating mode from anomaly detector
        operating_mode = anomaly.get("operating_mode", "full")
        health_score = anomaly.get("health_score", 100)

        # --- Stress Detection ---
        alerts = []
        status = "healthy"
        severity = "low"
        drop = 0.0

        if ndvi is not None and baseline is not None:
            severity, status, drop = detect_stress(ndvi, baseline)

            if status != "healthy":
                stress_type = STRESS_TYPES.get(severity, "water_stress")
                zone_name = FIELD_ZONES.get(str(zone_id), {}).get("name", str(zone_id))
                recommendation = build_recommendation(stress_type, drop, zone_name)

                alert = {
                    "zone_id": str(zone_id),
                    "zone_name": zone_name,
                    "severity": severity,
                    "status": status,
                    "stress_type": stress_type,
                    "ndvi": ndvi,
                    "baseline": baseline,
                    "drop": round(drop, 4),
                    "recommendation": recommendation,
                    "timestamp": now.isoformat(),
                }

                # Enrich with AI classification
                if classifier:
                    alert["ai_class"] = classifier.get("class")
                    alert["ai_confidence"] = classifier.get("confidence")

                # Enrich with VILA scene analysis
                if scene:
                    alert["vila_explanation"] = scene.get("explanation", "")[:200]
                    alert["vila_confidence"] = scene.get("confidence")

                alerts.append(alert)
                self._alert_count += 1

        # --- Downlink Prioritization ---
        # In a SmallSat, bandwidth is limited. Prioritize what to send to ground.
        downlink_queue = []

        # Always downlink: NDVI reading (lightweight)
        if ndvi is not None:
            downlink_queue.append({
                "type": "ndvi_reading",
                "priority": 1,
                "payload": {
                    "value": round(ndvi, 4),
                    "zone": str(zone_id),
                    "season": season,
                    "baseline": baseline,
                    "status": status,
                },
            })

        # High priority: alerts
        for alert in alerts:
            downlink_queue.append({
                "type": "alert",
                "priority": 0,  # highest
                "payload": alert,
            })

        # Medium priority: system health (periodic)
        if self._scan_count % 5 == 0:
            downlink_queue.append({
                "type": "system_status",
                "priority": 2,
                "payload": {
                    "operating_mode": operating_mode,
                    "health_score": health_score,
                    "scan_count": self._scan_count,
                    "alert_count": self._alert_count,
                },
            })

        # Sort by priority (0 = highest)
        downlink_queue.sort(key=lambda x: x["priority"])

        # --- Next Capture Command ---
        interval = scheduler.get("interval_sec", 600)
        power_mode = scheduler.get("power_mode", "full")

        next_command = {
            "action": "capture",
            "interval_sec": interval,
            "power_mode": power_mode,
            "skip_ai": anomaly.get("skip_ai", False),
            "skip_vila": anomaly.get("skip_vila", False),
        }

        # --- Triage Log Entry ---
        triage_entry = {
            "timestamp": now.isoformat(),
            "scan_number": self._scan_count,
            "zone_id": str(zone_id),
            "ndvi": ndvi,
            "raw_ndvi": raw_ndvi,
            "baseline": baseline,
            "season": season,
            "status": status,
            "severity": severity,
            "operating_mode": operating_mode,
            "health_score": health_score,
            "interval_sec": interval,
            "ai_class": classifier.get("class") if classifier else None,
            "ai_confidence": classifier.get("confidence") if classifier else None,
            "vila_confidence": scene.get("confidence") if scene else None,
            "anomaly_count": len(anomaly.get("anomalies", [])),
            "alert_generated": len(alerts) > 0,
            "quality_flags": quality_flags,
        }
        self._triage_log.append(triage_entry)

        self._result = {
            "alerts": alerts,
            "operating_mode": operating_mode,
            "health_score": health_score,
            "status": status,
            "severity": severity,
            "next_capture_command": next_command,
            "downlink_queue": downlink_queue,
            "triage_entry": triage_entry,
            "scan_count": self._scan_count,
            "alert_count": self._alert_count,
        }

    def render_reasoning(self):
        """Return mission executive decision."""
        return self._result

    def get_triage_log(self):
        """Return full triage log for ground control review."""
        return self._triage_log

    def export_triage_log(self, filepath):
        """Write triage log to JSONL file."""
        with open(filepath, "w") as f:
            for entry in self._triage_log:
                f.write(json.dumps(entry) + "\n")

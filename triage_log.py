#!/usr/bin/env python3
"""
triage_log.py — Autonomous Decision Triage Logger
===================================================
Records every AI decision in the Chroma-Shift pipeline so NASA can
see exactly how the system "thinks" without any human in the loop.

Each triage entry captures the FULL decision chain:

  1. SENSE   — What the sensor saw (raw DN values, exposure, gain)
  2. CORRECT — How the pipeline corrected it (7 radiometric layers)
  3. MEASURE — What NDVI value was computed
  4. COMPARE — How it compares to seasonal baseline + history
  5. CLASSIFY — What the DINOv2 AI predicted and its confidence
  6. DIAGNOSE — What stress factors were identified (if any)
  7. ADVISE  — What recommendation was generated for the farmer
  8. SCHEDULE — When to scan next (adaptive interval decision)

The log is append-only, JSON-lines format for easy parsing.
Each line is one complete autonomous triage decision.

Usage:
    from triage_log import TriageLogger

    logger = TriageLogger()
    logger.log_decision(
        zone_id="park_north",
        sense={...},
        corrections={...},
        ndvi=0.42,
        seasonal={...},
        classification={...},
        advisory={...},
        scheduling={...},
    )
"""

import os
import json
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
LOG_DIR = SCRIPT_DIR / "triage_logs"


class TriageLogger:
    """
    Append-only autonomous triage logger.

    Every scan produces one triage entry recording the complete
    autonomous decision chain from raw photons to farmer advisory.
    """

    def __init__(self, log_dir=None):
        self.log_dir = Path(log_dir) if log_dir else LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"triage_{self.session_id}.jsonl"
        self.entry_count = 0

        # Write session header
        header = {
            "type": "session_start",
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system": "Chroma-Shift Autonomous Crop Monitor",
            "hardware": "NVIDIA Jetson Orin Nano 8GB",
            "model": "DINOv2 ViT-S/14, 4-class (93.94% on 1,800 held-out samples)",
            "pipeline": "7-layer radiometric correction → NDVI → AI classify → advisory",
            "human_in_loop": False,
        }
        self._append(header)

    def _append(self, entry):
        """Append a JSON entry to the log file."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def log_decision(self, zone_id, sense, corrections, ndvi, seasonal,
                     classification=None, advisory=None, scheduling=None,
                     weather=None, vlm_analysis=None):
        """
        Log a complete autonomous triage decision.

        Args:
            zone_id:        str, field zone identifier
            sense:          dict, raw sensor data
                           {red_mean_dn, nir_mean_dn, red_exposure, nir_exposure,
                            red_gain, nir_gain, capture_time}
            corrections:    dict, radiometric pipeline corrections applied
                           {L1_exposure, L2_dos, L3_solar, L4_cloud, L5_interframe,
                            L6_seasonal, L7_kalman}
            ndvi:           float, computed NDVI value
            seasonal:       dict from get_seasonal_baseline()
                           {baseline, week, season}
            classification: dict from DINOv2 classifier (optional)
                           {class, confidence, healthy_prob, stressed_prob}
            advisory:       dict from crop_advisor (optional)
            scheduling:     dict from adaptive_sensing (optional)
                           {next_interval_sec, priority, reason}
            weather:        dict from NWS API (optional)
            vlm_analysis:   str, cloud VLM natural language analysis (optional)
        """
        self.entry_count += 1
        timestamp = datetime.now(timezone.utc)

        # ── Build the triage entry ──
        entry = {
            "type": "triage_decision",
            "entry_id": self.entry_count,
            "timestamp": timestamp.isoformat(),
            "zone_id": zone_id,
            "human_intervention": False,

            # ─── 1. SENSE: What the sensor saw ───
            "sense": {
                "description": "Raw sensor readings before any processing",
                "data": sense,
            },

            # ─── 2. CORRECT: Radiometric pipeline applied ───
            "correct": {
                "description": "7-layer radiometric correction chain",
                "layers_applied": list(corrections.keys()) if corrections else [],
                "corrections": corrections,
            },

            # ─── 3. MEASURE: NDVI computed ───
            "measure": {
                "description": "Normalized Difference Vegetation Index = (NIR - Red) / (NIR + Red)",
                "ndvi": round(ndvi, 4),
                "valid": -1.0 <= ndvi <= 1.0,
                "vegetation_detected": ndvi > 0.15,
            },

            # ─── 4. COMPARE: Seasonal baseline comparison ───
            "compare": {
                "description": "Compare against 5-year NASA HLS satellite baseline for NYC",
                "seasonal_baseline": seasonal.get("baseline"),
                "week": seasonal.get("week"),
                "season": seasonal.get("season"),
                "deviation": round(ndvi - seasonal.get("baseline", 0), 4),
                "deviation_pct": round((ndvi - seasonal.get("baseline", 0)) / max(seasonal.get("baseline", 0.01), 0.01) * 100, 1),
                "within_normal": abs(ndvi - seasonal.get("baseline", 0)) < 0.05,
            },

            # ─── 5. CLASSIFY: AI prediction ───
            "classify": {
                "description": "DINOv2 ViT-S/14 TensorRT FP16 inference",
                "model_accuracy": "93.94% on 1,800 held-out samples (4-class)",
                "prediction": classification.get("class") if classification else None,
                "confidence_pct": classification.get("confidence") if classification else None,
                "probabilities": {
                    "healthy": classification.get("healthy_prob") if classification else None,
                    "stressed": classification.get("stressed_prob") if classification else None,
                },
            } if classification else {"description": "No classifier available", "prediction": None},

            # ─── 6. DIAGNOSE: Stress analysis ───
            "diagnose": {
                "description": "Autonomous stress diagnosis from NDVI + weather + trends",
                "health_status": advisory.get("health_status") if advisory else None,
                "health_label": advisory.get("health_label") if advisory else None,
                "possible_causes": advisory.get("diagnoses") if advisory else [],
                "weather_context": weather,
            },

            # ─── 7. ADVISE: Farmer recommendation ───
            "advise": {
                "description": "Actionable recommendation generated autonomously",
                "recommendation": advisory.get("primary_recommendation") if advisory else None,
                "health_summary": advisory.get("health_summary") if advisory else None,
                "ndvi_explanation": advisory.get("ndvi_explanation") if advisory else None,
            },

            # ─── 8. SCHEDULE: Next observation decision ───
            "schedule": {
                "description": "Adaptive sensing scheduler — adjusts interval based on conditions",
                "next_interval_sec": scheduling.get("next_interval_sec") if scheduling else None,
                "next_interval_min": round(scheduling.get("next_interval_sec", 0) / 60, 1) if scheduling else None,
                "priority": scheduling.get("priority") if scheduling else None,
                "reason": scheduling.get("reason") if scheduling else None,
            } if scheduling else {"description": "Fixed interval (no adaptive data)", "next_interval_sec": None},

            # ─── VLM (if available) ───
            "vlm_analysis": vlm_analysis,
        }

        # ─── AI reasoning chain (human-readable summary) ───
        chain = self._build_reasoning_chain(entry)
        entry["reasoning_chain"] = chain

        self._append(entry)
        return entry

    def _build_reasoning_chain(self, entry):
        """Build a human-readable reasoning chain from the triage entry."""
        steps = []

        # Sense
        sense = entry.get("sense", {}).get("data", {})
        if sense:
            steps.append(
                f"SENSE: Captured Red (665nm) and NIR (865nm) frames. "
                f"Red mean DN: {sense.get('red_mean_dn', '?')}, "
                f"NIR mean DN: {sense.get('nir_mean_dn', '?')}."
            )

        # Correct
        layers = entry.get("correct", {}).get("layers_applied", [])
        if layers:
            steps.append(
                f"CORRECT: Applied {len(layers)} radiometric corrections: "
                f"{', '.join(layers)}."
            )

        # Measure
        ndvi = entry.get("measure", {}).get("ndvi")
        if ndvi is not None:
            steps.append(f"MEASURE: Computed NDVI = {ndvi:.4f}.")

        # Compare
        comp = entry.get("compare", {})
        baseline = comp.get("seasonal_baseline")
        dev = comp.get("deviation")
        if baseline is not None and dev is not None:
            direction = "above" if dev >= 0 else "below"
            steps.append(
                f"COMPARE: NDVI is {abs(dev):.4f} {direction} the seasonal baseline "
                f"of {baseline:.4f} for {comp.get('season', '?')} week {comp.get('week', '?')}."
            )

        # Classify
        cls = entry.get("classify", {})
        pred = cls.get("prediction")
        conf = cls.get("confidence_pct")
        if pred:
            steps.append(
                f"CLASSIFY: DINOv2 predicts '{pred}' with {conf:.1f}% confidence."
            )

        # Diagnose
        diag = entry.get("diagnose", {})
        causes = diag.get("possible_causes", [])
        if causes:
            cause_strs = [f"{c['cause']} ({c['confidence']*100:.0f}%)" for c in causes]
            steps.append(f"DIAGNOSE: Possible causes: {', '.join(cause_strs)}.")
        elif diag.get("health_status") in ("on_track", "above_baseline"):
            steps.append("DIAGNOSE: No stress indicators detected.")

        # Advise
        rec = entry.get("advise", {}).get("recommendation")
        if rec:
            steps.append(f"ADVISE: {rec}")

        # Schedule
        sched = entry.get("schedule", {})
        interval = sched.get("next_interval_min")
        reason = sched.get("reason")
        if interval:
            steps.append(
                f"SCHEDULE: Next scan in {interval:.0f} minutes"
                f"{f' ({reason})' if reason else ''}."
            )

        return " → ".join([f"[{i+1}] {s}" for i, s in enumerate(steps)])

    def log_rejection(self, zone_id, reason):
        """Log when a scan is rejected (e.g., solar zenith too high)."""
        self.entry_count += 1
        entry = {
            "type": "scan_rejected",
            "entry_id": self.entry_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "zone_id": zone_id,
            "reason": reason,
            "human_intervention": False,
            "reasoning_chain": f"[1] SENSE: Capture attempted. [2] REJECT: {reason}. No further processing.",
        }
        self._append(entry)

    def log_session_end(self, total_scans, total_alerts, uptime_hours):
        """Log session summary when the pipeline shuts down."""
        entry = {
            "type": "session_end",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "total_entries": self.entry_count,
            "total_scans": total_scans,
            "total_alerts": total_alerts,
            "uptime_hours": round(uptime_hours, 2),
            "human_interventions": 0,
            "log_file": str(self.log_file),
        }
        self._append(entry)

    def get_stats(self):
        """Get session statistics."""
        return {
            "session_id": self.session_id,
            "entries": self.entry_count,
            "log_file": str(self.log_file),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Demo / Test
# ═══════════════════════════════════════════════════════════════════════════

def demo():
    """Generate a sample triage log showing autonomous decision-making."""
    from crop_advisor import generate_advisory
    from radiometric import get_seasonal_baseline

    logger = TriageLogger(log_dir=LOG_DIR / "demo")
    seasonal = get_seasonal_baseline()

    # ── Decision 1: Healthy scan ──
    print("\n  📝 Logging Decision 1: Healthy vegetation scan...")
    entry1 = logger.log_decision(
        zone_id="park_north",
        sense={
            "red_mean_dn": 82.3, "nir_mean_dn": 145.7,
            "red_exposure": 0.01, "nir_exposure": 0.012,
            "red_gain": 1.0, "nir_gain": 1.2,
            "capture_time": "2026-03-17T14:30:00-04:00",
        },
        corrections={
            "L1_exposure": {"red_factor": 1.0, "nir_factor": 0.833},
            "L2_dos": {"red_dark_offset": 12, "nir_dark_offset": 8},
            "L3_solar": {"zenith_deg": 48.2, "cos_zenith": 0.668, "algorithm": "NREL_SPA"},
            "L4_cloud": {"cloud_cover_pct": 15, "transmittance": 0.963},
            "L5_interframe": {"pif_count": 342, "correction_factor": 0.98},
            "L6_seasonal": {"baseline": seasonal["baseline"], "week": seasonal["week"]},
            "L7_kalman": {"prior": 0.39, "posterior": 0.385, "gain": 0.42},
        },
        ndvi=0.385,
        seasonal=seasonal,
        classification={"class": "healthy_vegetation", "confidence": 99.2,
                        "healthy_prob": 0.992, "stressed_prob": 0.008},
        advisory=generate_advisory(
            ndvi=0.385, seasonal_data=seasonal,
            classification={"class": "healthy_vegetation", "confidence": 99.2},
            weather={"temp_c": 14, "humidity_pct": 52, "description": "Partly cloudy"},
            zone_name="Park North — Great Lawn"
        ),
        scheduling={"next_interval_sec": 1200, "priority": 0.3,
                    "reason": "stable NDVI, extending interval"},
        weather={"temp_c": 14, "humidity_pct": 52, "description": "Partly cloudy"},
    )
    print(f"  ✅ {entry1['reasoning_chain'][:120]}...")

    # ── Decision 2: Stressed scan ──
    print("\n  📝 Logging Decision 2: Stressed vegetation detected...")
    summer_seasonal = {"baseline": 0.72, "week": 26, "season": "summer"}
    entry2 = logger.log_decision(
        zone_id="park_south",
        sense={
            "red_mean_dn": 134.5, "nir_mean_dn": 118.2,
            "red_exposure": 0.008, "nir_exposure": 0.01,
            "red_gain": 1.0, "nir_gain": 1.0,
            "capture_time": "2026-07-15T13:00:00-04:00",
        },
        corrections={
            "L1_exposure": {"red_factor": 1.25, "nir_factor": 1.0},
            "L2_dos": {"red_dark_offset": 15, "nir_dark_offset": 10},
            "L3_solar": {"zenith_deg": 22.1, "cos_zenith": 0.927, "algorithm": "NREL_SPA"},
            "L4_cloud": {"cloud_cover_pct": 0, "transmittance": 1.0},
            "L5_interframe": {"pif_count": 287, "correction_factor": 1.01},
            "L6_seasonal": {"baseline": 0.72, "week": 26},
            "L7_kalman": {"prior": 0.58, "posterior": 0.52, "gain": 0.55},
        },
        ndvi=0.52,
        seasonal=summer_seasonal,
        classification={"class": "stressed_soil", "confidence": 96.8,
                        "healthy_prob": 0.032, "stressed_prob": 0.968},
        advisory=generate_advisory(
            ndvi=0.52, seasonal_data=summer_seasonal,
            classification={"class": "stressed_soil", "confidence": 96.8},
            weather={"temp_c": 36, "humidity_pct": 25, "precip_1h_mm": 0,
                     "description": "Clear, heat advisory"},
            historical_ndvi=[0.68, 0.62, 0.57, 0.52],
            zone_name="Park South — Sheep Meadow"
        ),
        scheduling={"next_interval_sec": 300, "priority": 0.85,
                    "reason": "stress detected, shortening interval to 5 min"},
        weather={"temp_c": 36, "humidity_pct": 25, "precip_1h_mm": 0,
                 "description": "Clear, heat advisory"},
    )
    print(f"  ✅ {entry2['reasoning_chain'][:120]}...")

    # ── Decision 3: Rejected scan ──
    print("\n  📝 Logging Decision 3: Scan rejected (too dark)...")
    logger.log_rejection("park_east", "Solar zenith 84° exceeds MAX_SOLAR_ZENITH (80°) — insufficient illumination for reliable NDVI")

    # ── Session end ──
    logger.log_session_end(total_scans=2, total_alerts=1, uptime_hours=4.5)

    print(f"\n  📄 Triage log saved: {logger.log_file}")
    print(f"  Total entries: {logger.entry_count}")

    # Print one entry formatted
    print(f"\n  ── Sample reasoning chain (Decision 2) ──")
    print(f"  {entry2['reasoning_chain']}")

    return logger


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Chroma-Shift — Autonomous Triage Log Demo                ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    demo()

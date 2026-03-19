#!/usr/bin/env python3
"""
adaptive_sensing.py — Adaptive Observation Scheduling for SmallSat Missions
=============================================================================
Implements intelligent, event-driven observation scheduling that adapts
capture frequency and parameters based on real-time scene analysis and
environmental conditions. Designed for SmallSat onboard processing with
SWaP (Size, Weight, and Power) constraints.

This module addresses the core NASA Space-to-Soil challenge requirement:
moving from fixed observation schedules to adaptive sensing where the
spacecraft decides WHAT to observe, WHEN to observe, and HOW to observe
based on onboard AI analysis.

Adaptive Decisions:
  1. WHEN to capture  — increase frequency during stress events, reduce
                        during stable periods or nighttime
  2. WHAT to prioritize — rank field zones by change probability
  3. HOW to process  — select which correction layers to apply based on
                        scene conditions and available power budget

SWaP Budget (Jetson Orin Nano as SmallSat proxy):
  - Power: 10W (comparable to Xiphos Q8S 4-25W range)
  - Compute: 1024 CUDA cores, 32 Tensor Cores, 8GB unified RAM
  - Bandwidth: Supabase upload ≈ satellite S-band downlink (~2 Mbps)
"""

import time
import os
import json
from datetime import datetime, timezone, timedelta

# --- Adaptive Scheduling Configuration ---

# Default observation interval (seconds)
DEFAULT_INTERVAL = 600          # 10 minutes — nominal spacing

# Adaptive interval bounds
MIN_INTERVAL = 120              # 2 min — maximum urgency (stress detected)
MAX_INTERVAL = 1800             # 30 min — low-activity periods (night, stable)

# Thresholds that trigger adaptive changes
STRESS_NDVI_DROP = 0.05         # NDVI drop below seasonal baseline
RAPID_CHANGE_DELTA = 0.03       # NDVI change between consecutive readings
ILLUMINATION_CHANGE_PCT = 15    # % brightness difference triggering re-scan
CLOUD_COVER_UNSTABLE = 50       # Cloud cover % suggesting variable illumination

# Power management (SmallSat mode)
POWER_BUDGET_WATTS = 10.0       # Total power envelope
IDLE_POWER_WATTS = 3.5          # CPU/system idle
CAPTURE_POWER_WATTS = 5.0       # Camera + filter wheel active
GPU_POWER_WATTS = 8.0           # GPU active (DINOv2 inference)
CLOUD_POWER_WATTS = 2.0         # Network radio (S-band proxy)


class AdaptiveSensingScheduler:
    """
    Manages adaptive observation scheduling based on scene analysis,
    weather, and power/bandwidth constraints.

    Designed as a SmallSat onboard scheduler that could run on
    radiation-tolerant edge processors (Xiphos Q8S, OrbitsEdge Edge1,
    or NVIDIA Jetson in a shielded enclosure).
    """

    def __init__(self, base_interval=DEFAULT_INTERVAL):
        self.base_interval = base_interval
        self.current_interval = base_interval
        self._history = []       # Recent decisions for telemetry
        self._zone_priority = {} # Zone → priority score
        self._last_ndvi = {}     # Zone → last NDVI value
        self._consecutive_stable = {} # Zone → count of stable readings
        self._power_state = "nominal"

    def compute_next_interval(self, zone_id, ndvi, seasonal_baseline,
                               vila_result=None, weather=None, solar=None,
                               classification=None):
        """
        Compute the next observation interval based on current conditions.

        This is the core adaptive sensing algorithm. It adjusts the
        observation cadence based on:
          - NDVI stress detection (shorter interval if stress emerging)
          - Scene change rate (shorter if rapid changes detected)
          - Weather stability (longer if conditions stable)
          - Solar position (longer/skip at night)
          - Power budget (longer if power-constrained)
          - VLM confidence (shorter if VLM uncertain)
          - DINOv2 classification (shorter if stressed/uncertain)

        Args:
            zone_id:            Field zone identifier
            ndvi:               Current corrected NDVI reading
            seasonal_baseline:  Expected NDVI for this week of year
            vila_result:        Dict from VilaSceneAnalyzer.analyze()
            weather:            NWS weather dict
            solar:              Solar position dict
            classification:     Dict from DINOv2 classifier
                                {class, confidence, *_prob}

        Returns:
            dict with:
                interval_sec:   Next observation interval (seconds)
                reason:         Human-readable explanation
                priority:       Zone priority score (0-1, higher = observe sooner)
                power_mode:     "full" | "reduced" | "sleep"
                skip_zones:     List of zones to skip this cycle
                telemetry:      Dict of adaptive decision metadata
        """
        t0 = time.time()

        # --- Gather signals ---
        ndvi_drop = seasonal_baseline - ndvi if ndvi is not None else 0
        prev_ndvi = self._last_ndvi.get(zone_id)
        ndvi_delta = abs(ndvi - prev_ndvi) if prev_ndvi is not None and ndvi is not None else 0
        self._last_ndvi[zone_id] = ndvi

        sun_elevation = solar.get("elevation_deg", 45) if solar else 45
        cloud_cover = weather.get("cloud_cover_pct", 0) if weather else 0
        vila_confidence = vila_result.get("confidence", 0.5) if vila_result else 0.5
        vila_haze = vila_result.get("scene", {}).get("haze", False) if vila_result else False
        vila_change = vila_result.get("scene", {}).get("interframe_change", False) if vila_result else False

        # --- Decision logic ---
        interval = self.base_interval
        reasons = []
        priority = 0.5  # neutral

        # 1. STRESS DETECTION → increase frequency
        if ndvi_drop > STRESS_NDVI_DROP:
            stress_factor = min(ndvi_drop / 0.15, 1.0)  # 0-1 scale
            interval *= (1 - 0.7 * stress_factor)  # Up to 70% reduction
            priority += 0.3 * stress_factor
            reasons.append(f"stress detected (NDVI {ndvi_drop:+.3f} below baseline)")

        # 2. RAPID CHANGE → increase frequency
        if ndvi_delta > RAPID_CHANGE_DELTA:
            interval *= 0.5
            priority += 0.2
            reasons.append(f"rapid change (Δ{ndvi_delta:.3f} between readings)")
            self._consecutive_stable[zone_id] = 0
        else:
            self._consecutive_stable[zone_id] = self._consecutive_stable.get(zone_id, 0) + 1

        # 3. STABLE PERIOD → decrease frequency
        stable_count = self._consecutive_stable.get(zone_id, 0)
        if stable_count > 5:
            interval *= 1 + min(stable_count * 0.1, 1.0)  # Up to 2× longer
            priority -= 0.2
            reasons.append(f"stable ({stable_count} consistent readings)")

        # 4. NIGHTTIME → minimal observations
        if sun_elevation < 5:
            interval = MAX_INTERVAL
            priority = 0.0
            reasons.append(f"nighttime (solar elev {sun_elevation:.1f}°)")
        elif sun_elevation < 15:
            interval *= 2
            priority -= 0.3
            reasons.append(f"dawn/dusk (solar elev {sun_elevation:.1f}°)")

        # 5. WEATHER INSTABILITY → increase frequency
        if cloud_cover > CLOUD_COVER_UNSTABLE:
            interval *= 0.7
            priority += 0.15
            reasons.append(f"variable illumination ({cloud_cover}% cloud)")

        # 6. VLM UNCERTAINTY → increase frequency
        if vila_confidence < 0.4:
            interval *= 0.6
            priority += 0.1
            reasons.append(f"low VLM confidence ({vila_confidence:.2f})")

        # 7. VLM-DETECTED SCENE CHANGE → immediate re-scan
        if vila_change:
            interval = MIN_INTERVAL
            priority = 1.0
            reasons.append("VLM detected inter-frame illumination change")

        # 8. DINOv2 CLASSIFICATION-AWARE SCHEDULING
        if classification:
            cls = classification.get("class", "unknown")
            conf = classification.get("confidence", 0)

            if cls == "stressed_vegetation":
                interval *= 0.4  # Urgent — scan 2.5× more often
                priority += 0.4
                reasons.append(f"stressed vegetation detected ({conf:.0f}%)")
            elif cls == "bare_soil":
                interval *= 1.5  # Less urgent — no vegetation to monitor
                priority -= 0.15
                reasons.append(f"bare soil — reduced priority ({conf:.0f}%)")
            elif cls == "non_vegetation":
                interval *= 2.0  # Skip zone more often
                priority -= 0.3
                reasons.append(f"non-vegetation surface — low priority ({conf:.0f}%)")

            # Low confidence → rescan sooner for verification
            if conf < 80:
                interval *= 0.6
                priority += 0.15
                reasons.append(f"uncertain classification ({conf:.0f}%) — verify sooner")

        # --- Enforce bounds ---
        interval = max(MIN_INTERVAL, min(MAX_INTERVAL, interval))
        priority = max(0.0, min(1.0, priority))

        # --- Power management ---
        if sun_elevation < 0:
            power_mode = "sleep"
        elif priority < 0.3:
            power_mode = "reduced"  # Skip GPU inference, use heuristic only
        else:
            power_mode = "full"

        # --- Update state ---
        self.current_interval = int(interval)
        self._zone_priority[zone_id] = priority

        decision = {
            "interval_sec": int(interval),
            "reason": "; ".join(reasons) if reasons else "nominal operations",
            "priority": round(priority, 2),
            "power_mode": power_mode,
            "skip_zones": [],
            "telemetry": {
                "ndvi_drop": round(ndvi_drop, 4),
                "ndvi_delta": round(ndvi_delta, 4),
                "stable_count": stable_count,
                "sun_elevation": round(sun_elevation, 1),
                "cloud_cover": cloud_cover,
                "vila_confidence": round(vila_confidence, 2),
                "compute_ms": round((time.time() - t0) * 1000, 2),
            },
        }

        self._history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "zone": zone_id,
            **decision,
        })
        # Keep last 100 decisions
        if len(self._history) > 100:
            self._history = self._history[-100:]

        return decision

    def rank_zones(self, zone_ids):
        """
        Rank zones by observation priority (highest priority first).
        Zones with no history get neutral priority.

        In a SmallSat context, this determines which ground targets
        to observe when the satellite has limited observation windows
        over a region.
        """
        return sorted(
            zone_ids,
            key=lambda z: self._zone_priority.get(z, 0.5),
            reverse=True,
        )

    def should_use_gpu(self, power_mode):
        """
        Determine if GPU inference (DINOv2 TensorRT) should run this cycle.
        In power-constrained mode, fall back to NDVI-only analysis.
        """
        return power_mode == "full"

    def should_use_cloud_vlm(self, power_mode, interval_sec):
        """
        Determine if cloud VLM (NVIDIA NIM) should be called this cycle.
        Cloud calls consume radio bandwidth — equivalent to satellite
        S-band downlink budget. Skip during reduced power or very short
        intervals (would exceed bandwidth budget).
        """
        if power_mode == "sleep":
            return False
        if interval_sec < 180:  # Very rapid sampling — skip VLM
            return False
        return True

    def get_bandwidth_estimate(self, vila_result=None, ndvi_array=None):
        """
        Estimate downlink bandwidth for this observation cycle.

        SmallSat S-band typical: 1-2 Mbps, ~10 min pass = ~75-150 MB/pass
        Our telemetry per cycle: ~2-5 KB (JSON) — extremely efficient
        Full image downlink: ~500 KB (compressed Red+NIR pair)

        Returns dict with byte estimates for each data product.
        """
        estimates = {
            "ndvi_telemetry_bytes": 512,      # Scalar NDVI + metadata
            "alert_bytes": 256 if vila_result else 0,
            "vila_json_bytes": len(json.dumps(vila_result).encode()) if vila_result else 0,
            "weather_bytes": 384,
            "heartbeat_bytes": 128,
        }
        estimates["total_bytes"] = sum(estimates.values())
        estimates["total_kb"] = round(estimates["total_bytes"] / 1024, 2)
        return estimates

    def get_swap_budget(self, power_mode="full"):
        """
        Return the current SWaP (Size, Weight, and Power) budget estimate.

        Maps the Jetson Orin Nano power profile to SmallSat equivalents.
        """
        if power_mode == "full":
            power_w = GPU_POWER_WATTS + CLOUD_POWER_WATTS
        elif power_mode == "reduced":
            power_w = CAPTURE_POWER_WATTS
        else:
            power_w = IDLE_POWER_WATTS

        return {
            "power_w": round(power_w, 1),
            "power_budget_w": POWER_BUDGET_WATTS,
            "utilization_pct": round(power_w / POWER_BUDGET_WATTS * 100, 1),
            "compute_platform": "NVIDIA Jetson Orin Nano (SmallSat proxy)",
            "gpu_cores": 1024,
            "tensor_cores": 32,
            "ram_gb": 8,
            "comparable_flight_hw": [
                "Xiphos Q8S (4-25W, ARM+FPGA, rad-tolerant)",
                "OrbitsEdge Edge1 (15 TOPS, rad-shielded)",
                "Unibap SpaceCloud iX10 (GPU-accelerated, LEO-rated)",
            ],
        }

    def get_history(self, n=10):
        """Return the last n adaptive scheduling decisions."""
        return self._history[-n:]

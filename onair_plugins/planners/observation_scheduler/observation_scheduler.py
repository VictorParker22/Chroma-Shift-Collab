"""
observation_scheduler.py — OnAIR Planner Plugin
=================================================
Wraps AdaptiveSensingScheduler from core.adaptive_sensing.
Computes optimal observation intervals and zone priorities based on
NDVI stress, weather, solar position, and AI classification results.

OnAIR role: Planner (PP)
  - Receives Knowledge Representation + Learner outputs
  - Produces observation scheduling decisions: interval, priority,
    power mode, zone rankings
"""

import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.adaptive_sensing import AdaptiveSensingScheduler


class Plugin:
    """OnAIR Planner plugin for adaptive observation scheduling."""

    def __init__(self, _name, _headers):
        self.component_name = _name
        self.headers = _headers
        self.scheduler = AdaptiveSensingScheduler()
        self._result = None

    def update(self, low_level_data=[], high_level_data={}):
        """Compute next observation interval from upstream plugin outputs.

        Args:
            low_level_data: Raw telemetry frame.
            high_level_data: Must contain radiometric, scene_analysis,
                            vegetation_classifier, and anomaly_detector outputs.
        """
        frame = self._to_dict(low_level_data)

        # Gather upstream outputs
        radiometric = high_level_data.get("radiometric", {}) or {}
        scene = high_level_data.get("scene_analysis", {}) or {}
        classifier = high_level_data.get("vegetation_classifier", {}) or {}
        anomaly = high_level_data.get("anomaly_detector", {}) or {}

        zone_id = radiometric.get("zone_id", frame.get("zone_id", "T1"))
        ndvi = radiometric.get("mean_ndvi")
        baseline = None
        seasonal = radiometric.get("seasonal")
        if seasonal:
            baseline = seasonal.get("baseline")

        # Extract weather from frame
        weather = {}
        for key in ["cloud_cover_pct", "temp_c", "humidity_pct",
                     "wind_speed_ms", "precip_1h_mm"]:
            val = frame.get(key)
            if val is not None and val != 0.0:
                weather[key] = float(val)
        weather = weather if weather else None

        solar = radiometric.get("solar")

        try:
            decision = self.scheduler.compute_next_interval(
                zone_id,
                ndvi=ndvi,
                seasonal_baseline=baseline,
                vila_result=scene,
                weather=weather,
                solar=solar,
                classification=classifier,
            )
            self._result = decision
        except Exception:
            self._result = {
                "interval_sec": 600,
                "reason": "scheduler error — using default interval",
                "priority": "normal",
                "power_mode": "full",
            }

    def render_reasoning(self):
        """Return scheduling decision."""
        return self._result

    def _to_dict(self, low_level_data):
        if isinstance(low_level_data, dict):
            return low_level_data
        if isinstance(low_level_data, (list, tuple)) and self.headers:
            return {h: low_level_data[i] for i, h in enumerate(self.headers)
                    if i < len(low_level_data)}
        return {}

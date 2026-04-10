"""
crop_advisor_pp.py — OnAIR Planner Plugin
===========================================
Wraps generate_advisory() from core.crop_advisor.
Produces context-aware farmer advisories from fused NDVI, seasonal
baseline, weather, and AI classification data.

OnAIR role: Planner (PP)
  - Receives Knowledge Representation + Learner outputs
  - Produces actionable crop management recommendations
"""

import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.crop_advisor import generate_advisory


class Plugin:
    """OnAIR Planner plugin for crop advisory generation."""

    def __init__(self, _name, _headers):
        self.component_name = _name
        self.headers = _headers
        self._result = None

    def update(self, low_level_data=[], high_level_data={}):
        """Generate crop advisory from upstream plugin outputs.

        Args:
            low_level_data: Raw telemetry frame.
            high_level_data: Must contain radiometric and vegetation_classifier outputs.
        """
        frame = self._to_dict(low_level_data)

        # Gather upstream outputs
        radiometric = high_level_data.get("radiometric", {}) or {}
        classifier = high_level_data.get("vegetation_classifier", {}) or {}

        ndvi = radiometric.get("mean_ndvi")
        seasonal = radiometric.get("seasonal")
        zone_id = radiometric.get("zone_id", frame.get("zone_id", "T1"))

        # Extract weather from frame
        weather = {}
        for key in ["cloud_cover_pct", "temp_c", "humidity_pct",
                     "wind_speed_ms", "precip_1h_mm"]:
            val = frame.get(key)
            if val is not None and val != 0.0:
                weather[key] = float(val)
        weather = weather if weather else None

        if ndvi is None:
            self._result = None
            return

        try:
            self._result = generate_advisory(
                ndvi=ndvi,
                seasonal_data=seasonal,
                weather=weather,
                classification=classifier if classifier else None,
                zone_name=zone_id,
            )
        except Exception:
            self._result = None

    def render_reasoning(self):
        """Return crop advisory."""
        return self._result

    def _to_dict(self, low_level_data):
        if isinstance(low_level_data, dict):
            return low_level_data
        if isinstance(low_level_data, (list, tuple)) and self.headers:
            return {h: low_level_data[i] for i, h in enumerate(self.headers)
                    if i < len(low_level_data)}
        return {}

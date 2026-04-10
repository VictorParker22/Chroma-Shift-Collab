"""
scene_analysis_pkr.py — OnAIR Knowledge Representation Plugin
===============================================================
Wraps the VILA Vision-Language Model scene analyzer from core.vila_analyzer.
Analyzes Red+NIR imagery to produce scene composition metadata and
radiometric correction overrides.

OnAIR role: Knowledge Representation (PKR)
  - Receives raw telemetry and radiometric PKR output
  - Produces scene analysis: vegetation/soil/shadow percentages,
    illumination quality, haze detection, correction overrides

In CSV simulation mode, falls back to the CPU heuristic analyzer
(<1ms, no GPU required).
"""

import sys
import os
import numpy as np
from datetime import datetime, timezone

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class Plugin:
    """Scene analysis Knowledge Representation plugin for OnAIR."""

    def __init__(self, _name, _headers):
        self.component_name = _name
        self.headers = _headers
        self._analyzer = None
        self._result = None
        self._init_analyzer()

    def _init_analyzer(self):
        """Initialize VILA scene analyzer with 4-mode fallback."""
        try:
            from core.vila_analyzer import VilaSceneAnalyzer
            self._analyzer = VilaSceneAnalyzer()
        except Exception:
            # Graceful degradation: create heuristic-only analyzer
            try:
                from core.vila_analyzer import VilaSceneAnalyzer
                self._analyzer = VilaSceneAnalyzer.__new__(VilaSceneAnalyzer)
                self._analyzer.model = None
                self._analyzer.model_name = ""
                self._analyzer._fallback = True
            except Exception:
                self._analyzer = None

    def update(self, low_level_data=[], high_level_data={}):
        """Analyze scene composition from sensor data.

        Args:
            low_level_data: Raw telemetry frame from adapter.
            high_level_data: Dict containing radiometric PKR output.
        """
        if self._analyzer is None:
            self._result = self._default_result()
            return

        frame = self._to_dict(low_level_data)

        # Build synthetic frames from mean DN for heuristic analysis
        red_dn = frame.get("red_mean_dn", 100.0)
        nir_dn = frame.get("nir_mean_dn", 150.0)
        width = int(frame.get("frame_width", 10))
        height = int(frame.get("frame_height", 10))

        red_frame = np.full((height, width), red_dn, dtype=np.uint8)
        nir_frame = np.full((height, width), nir_dn, dtype=np.uint8)

        # Get weather and solar from radiometric PKR or raw frame
        weather = self._extract_weather(frame, high_level_data)
        solar = None
        radiometric = high_level_data.get("radiometric", {})
        if radiometric:
            solar = radiometric.get("solar")

        # Parse timestamp
        ts_raw = frame.get("timestamp_utc", "")
        try:
            if isinstance(ts_raw, str) and ts_raw:
                timestamp = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            else:
                timestamp = datetime.now(timezone.utc)
        except (ValueError, TypeError):
            timestamp = datetime.now(timezone.utc)

        try:
            self._result = self._analyzer.analyze(
                red_frame, nir_frame, weather, solar, timestamp
            )
        except Exception:
            self._result = self._default_result()

    def render_reasoning(self):
        """Return scene analysis results for downstream plugins."""
        return self._result

    def _extract_weather(self, frame, high_level_data):
        """Build weather dict from frame telemetry."""
        weather = {}
        for key in ["cloud_cover_pct", "temp_c", "humidity_pct",
                     "wind_speed_ms", "precip_1h_mm"]:
            val = frame.get(key)
            if val is not None and val != 0.0:
                weather[key] = float(val)
        return weather if weather else None

    @staticmethod
    def _default_result():
        """Return a safe default when scene analysis is unavailable."""
        return {
            "corrections": None,
            "explanation": "Scene analysis unavailable",
            "confidence": 0.0,
            "scene": {
                "vegetation_pct": 50.0,
                "soil_pct": 25.0,
                "artificial_pct": 10.0,
                "shadow_pct": 15.0,
            },
        }

    def _to_dict(self, low_level_data):
        if isinstance(low_level_data, dict):
            return low_level_data
        if isinstance(low_level_data, (list, tuple)) and self.headers:
            return {h: low_level_data[i] for i, h in enumerate(self.headers)
                    if i < len(low_level_data)}
        return {}

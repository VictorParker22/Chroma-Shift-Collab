"""
anomaly_detector_pl.py — OnAIR Learner Plugin
===============================================
Wraps AnomalyDetector and FaultManager from core.fault_tolerance.
Monitors sensor health via statistical anomaly detection and manages
operating mode degradation.

OnAIR role: Learner (PL)
  - Receives raw telemetry + Knowledge Representation outputs
  - Detects sensor drift, stuck values, impossible readings
  - Manages operating mode: FULL > REDUCED > SAFE > EMERGENCY
"""

import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.fault_tolerance import AnomalyDetector, FaultManager


class Plugin:
    """OnAIR Learner plugin for anomaly detection and fault management."""

    def __init__(self, _name, _headers):
        self.component_name = _name
        self.headers = _headers
        self.anomaly_detector = AnomalyDetector()
        self.fault_manager = FaultManager()
        self._result = None

    def update(self, low_level_data=[], high_level_data={}):
        """Check sensor data for anomalies and update fault state.

        Args:
            low_level_data: Raw telemetry frame.
            high_level_data: Must contain 'radiometric' PKR output.
        """
        anomalies = []

        # Get NDVI from radiometric PKR
        radiometric = high_level_data.get("radiometric", {})
        if radiometric:
            ndvi = radiometric.get("mean_ndvi")
            zone_id = radiometric.get("zone_id", "T1")

            if ndvi is not None:
                ndvi_anomalies = self.anomaly_detector.check_ndvi(ndvi, zone_id)
                anomalies.extend(ndvi_anomalies)

        # Record outcome in fault manager
        if anomalies:
            for a in anomalies:
                self.fault_manager.record_error(
                    "anomaly_detector",
                    a.get("message", "Unknown anomaly"),
                    {"zone": a.get("zone_id"), "type": a.get("type")},
                )
        else:
            self.fault_manager.record_success()

        self._result = {
            "anomalies": anomalies,
            "health_score": self.anomaly_detector.get_health_score(),
            "operating_mode": self.fault_manager.mode,
            "skip_ai": self.fault_manager.skip_ai,
            "skip_vila": self.fault_manager.skip_vila,
            "error_count": self.fault_manager.total_errors,
        }

    def render_reasoning(self):
        """Return anomaly detection results and system health."""
        return self._result

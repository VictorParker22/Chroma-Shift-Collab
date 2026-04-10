"""
radiometric_pkr.py — OnAIR Knowledge Representation Plugin
============================================================
Wraps the 7-layer radiometric correction pipeline from core.radiometric.
Transforms raw sensor telemetry frames into corrected NDVI reflectance
values and seasonal/solar metadata.

OnAIR role: Knowledge Representation (PKR)
  - Receives raw telemetry from the data adapter
  - Produces high-level knowledge: corrected NDVI, quality flags,
    solar position, seasonal baseline
"""

import sys
import os
import numpy as np
from datetime import datetime, timezone

# Ensure project root is on path for core imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.radiometric import (
    apply_corrections, solar_position, get_seasonal_baseline, KalmanNDVI
)


class Plugin:
    """Radiometric correction Knowledge Representation plugin for OnAIR."""

    def __init__(self, _name, _headers):
        self.component_name = _name
        self.headers = _headers
        self.kalman_filters = {}
        self._result = None

    def update(self, low_level_data=[], high_level_data={}):
        """Process raw sensor frame through 7-layer radiometric pipeline.

        Args:
            low_level_data: Dict or list of raw telemetry values from adapter.
            high_level_data: Dict of outputs from other plugins (unused by PKR).
        """
        frame = self._to_dict(low_level_data)

        # Extract sensor values from frame
        red_dn = frame.get("red_mean_dn", 0.0)
        nir_dn = frame.get("nir_mean_dn", 0.0)
        red_exposure = frame.get("red_exposure", None)
        red_gain = frame.get("red_gain", None)
        nir_exposure = frame.get("nir_exposure", None)
        nir_gain = frame.get("nir_gain", None)
        cloud_cover = frame.get("cloud_cover_pct", None)
        zone_id = frame.get("zone_id", "T1")

        # Parse timestamp
        ts_raw = frame.get("timestamp_utc", "")
        try:
            if isinstance(ts_raw, str) and ts_raw:
                timestamp = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            else:
                timestamp = datetime.now(timezone.utc)
        except (ValueError, TypeError):
            timestamp = datetime.now(timezone.utc)

        # Build synthetic frame arrays from mean DN values
        width = int(frame.get("frame_width", 10))
        height = int(frame.get("frame_height", 10))
        is_simulation = (width <= 10 and height <= 10)

        red_exposure = float(red_exposure) if red_exposure else None
        red_gain = float(red_gain) if red_gain else None
        nir_exposure = float(nir_exposure) if nir_exposure else None
        nir_gain = float(nir_gain) if nir_gain else None
        cloud_cover = float(cloud_cover) if cloud_cover else None

        if is_simulation:
            # Simulation mode: compute NDVI directly from scalar DN values.
            # The full 7-layer pipeline requires real spatial imagery (the
            # dark-object subtraction zeros out uniform arrays). Use the
            # standard NDVI formula on the mean DN values instead.
            r = float(red_dn)
            n = float(nir_dn)
            denom = n + r
            raw_ndvi = (n - r) / denom if denom > 0 else 0.0
            mean_ndvi = raw_ndvi
            corrections_applied = ["sim_direct_ndvi"]
            quality_flags = ["simulation_mode"]
            corrected = {
                "mean_ndvi": mean_ndvi,
                "corrections_applied": corrections_applied,
                "quality_flags": quality_flags,
            }
        else:
            # Flight mode: run the full 7-layer correction pipeline
            red_frame = np.full((height, width), red_dn, dtype=np.float32)
            nir_frame = np.full((height, width), nir_dn, dtype=np.float32)
            corrected = apply_corrections(
                red_frame, nir_frame,
                red_exposure=red_exposure, red_gain=red_gain,
                nir_exposure=nir_exposure, nir_gain=nir_gain,
                timestamp=timestamp,
                cloud_cover_pct=cloud_cover,
            )
            mean_ndvi = corrected["mean_ndvi"]
            raw_ndvi = mean_ndvi

        # Kalman temporal filter (Layer 7 — per-zone)
        zone_key = str(zone_id)
        if zone_key not in self.kalman_filters:
            self.kalman_filters[zone_key] = KalmanNDVI()
        mean_ndvi = round(self.kalman_filters[zone_key].update(mean_ndvi), 4)

        # Solar position and seasonal baseline
        solar = solar_position(timestamp)
        seasonal = corrected.get("seasonal", get_seasonal_baseline(timestamp))

        self._result = {
            "mean_ndvi": mean_ndvi,
            "raw_ndvi": raw_ndvi,
            "corrections_applied": corrected.get("corrections_applied", []),
            "quality_flags": corrected.get("quality_flags", []),
            "seasonal": seasonal,
            "solar": solar,
            "zone_id": zone_key,
            "timestamp": timestamp.isoformat(),
        }

    def render_reasoning(self):
        """Return corrected NDVI and metadata for downstream plugins."""
        return self._result

    def _to_dict(self, low_level_data):
        """Convert low_level_data to a dict using self.headers."""
        if isinstance(low_level_data, dict):
            return low_level_data
        if isinstance(low_level_data, (list, tuple)) and self.headers:
            return {h: low_level_data[i] for i, h in enumerate(self.headers)
                    if i < len(low_level_data)}
        return {}

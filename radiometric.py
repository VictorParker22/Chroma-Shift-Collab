#!/usr/bin/env python3
"""
radiometric.py — Self-Calibrating Radiometric Correction Pipeline
===================================================================
Seven-layer correction stack that converts raw camera DN (digital numbers)
to normalized reflectance, enabling accurate NDVI computation without
a physical calibration panel.

Layers:
  1. Auto-exposure normalization     (camera metadata)
  2. Dark Object Subtraction         (Chavez 1988)
  3. Solar zenith angle correction   (astronomical model)
  4. Cloud transmittance modeling     (NWS weather data)
  5. Inter-frame illumination match   (pseudo-invariant features)
  6. Seasonal phenology baseline      (NASA HLS weekly curve)
  7. Kalman temporal filter           (statistical smoothing)

All layers are pure functions — no external dependencies beyond numpy
and the Python standard library.

Reference:
  Chavez, P.S. (1988). An improved dark-object subtraction technique
  for atmospheric scattering correction. Remote Sensing of Environment.
"""

import math
import numpy as np
from datetime import datetime, timezone


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

# Central Park, NYC — default observation site
DEFAULT_LAT = 40.7812
DEFAULT_LON = -73.9665

# Solar zenith rejection threshold (degrees) — below this, signal is too noisy
MAX_SOLAR_ZENITH = 80.0

# Kasten & Czeplak (1980) cloud transmittance model coefficient
# T = 1 - k * (cloud_cover / 100)^2    where k ≈ 0.75
CLOUD_COEFF = 0.75

# Inter-frame brightness tolerance — flag if frames differ by more than this
INTERFRAME_THRESHOLD = 0.10  # 10%

# Kalman filter defaults
KALMAN_PROCESS_NOISE = 0.001   # NDVI changes slowly (~0.01/day max)
KALMAN_MEASUREMENT_NOISE = 0.02  # typical capture-to-capture variance

# Seasonal phenology curve — weekly NDVI baseline for NYC Central Park
# Derived from 296 clear-sky HLS observations (2019–2023)
# Week 1 = first week of January, Week 52 = last week of December
SEASONAL_NDVI = {
    1: 0.28, 2: 0.27, 3: 0.26, 4: 0.26, 5: 0.27, 6: 0.28,
    7: 0.29, 8: 0.30, 9: 0.32, 10: 0.34, 11: 0.37, 12: 0.40,
    13: 0.43, 14: 0.47, 15: 0.51, 16: 0.55, 17: 0.59, 18: 0.62,
    19: 0.65, 20: 0.67, 21: 0.69, 22: 0.71, 23: 0.72, 24: 0.73,
    25: 0.74, 26: 0.74, 27: 0.74, 28: 0.73, 29: 0.73, 30: 0.72,
    31: 0.71, 32: 0.70, 33: 0.69, 34: 0.67, 35: 0.65, 36: 0.62,
    37: 0.59, 38: 0.55, 39: 0.51, 40: 0.47, 41: 0.43, 42: 0.40,
    43: 0.37, 44: 0.34, 45: 0.32, 46: 0.30, 47: 0.29, 48: 0.28,
    49: 0.27, 50: 0.27, 51: 0.27, 52: 0.28,
}


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1: Auto-Exposure Normalization
# ═══════════════════════════════════════════════════════════════════════════

def normalize_exposure(frame, exposure_time, gain):
    """
    Convert raw DN to relative radiance by dividing out camera settings.

    If the camera auto-adjusts exposure/gain between Red and NIR captures,
    the raw pixel values are not directly comparable. Dividing by
    (exposure × gain) removes the camera's compensation, leaving only
    the scene's reflectance.

    Args:
        frame:         np.ndarray (H, W) grayscale float32, raw DN
        exposure_time: float, camera exposure time in seconds (or arbitrary units)
        gain:          float, camera gain (1.0 = no amplification)

    Returns:
        np.ndarray (H, W) float32, normalized radiance
    """
    if exposure_time <= 0 or gain <= 0:
        return frame  # fallback: no metadata available
    return frame / (exposure_time * gain)


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2: Dark Object Subtraction (Chavez 1988)
# ═══════════════════════════════════════════════════════════════════════════

def dark_object_subtraction(frame, percentile=1.0):
    """
    Remove atmospheric scatter/haze by subtracting the darkest pixel value.

    Even at ground level, scattered skylight adds a baseline offset to every
    pixel. The darkest pixel (deep shadow, dark soil) should theoretically
    have zero reflectance — any measured value is atmospheric contribution.

    Uses the 1st percentile (not absolute minimum) to avoid sensor noise
    outliers, following Chavez (1988).

    Args:
        frame:      np.ndarray (H, W) float32
        percentile: float, percentile for dark object (default 1.0)

    Returns:
        np.ndarray (H, W) float32, haze-corrected
    """
    dark_value = np.percentile(frame, percentile)
    corrected = frame - dark_value
    corrected = np.clip(corrected, 0, None)
    return corrected


# ═══════════════════════════════════════════════════════════════════════════
# Layer 3: Solar Zenith Angle Correction
# ═══════════════════════════════════════════════════════════════════════════

def solar_position(dt, lat=DEFAULT_LAT, lon=DEFAULT_LON):
    """
    Compute solar zenith and azimuth angles using the Spencer (1971) method.

    Pure math — no external libraries required. Accuracy ~0.01° for
    solar declination, sufficient for radiometric correction.

    Args:
        dt:  datetime (UTC or timezone-aware)
        lat: float, latitude in degrees
        lon: float, longitude in degrees

    Returns:
        dict with keys:
            zenith_deg:  solar zenith angle (0° = overhead, 90° = horizon)
            elevation_deg: solar elevation (complement of zenith)
            azimuth_deg: solar azimuth (0° = north, clockwise)
            cos_zenith:  cos(zenith) for direct use in correction
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # Day of year (1-366)
    doy = dt.timetuple().tm_yday

    # Fractional hour (UTC)
    hour_utc = dt.hour + dt.minute / 60.0 + dt.second / 3600.0

    # Solar declination (Spencer 1971)
    gamma = 2.0 * math.pi * (doy - 1) / 365.0
    decl = (0.006918
            - 0.399912 * math.cos(gamma)
            + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma)
            + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma)
            + 0.00148 * math.sin(3 * gamma))

    # Equation of time (minutes)
    eqtime = (229.18 * (0.000075
              + 0.001868 * math.cos(gamma)
              - 0.032077 * math.sin(gamma)
              - 0.014615 * math.cos(2 * gamma)
              - 0.04089 * math.sin(2 * gamma)))

    # True solar time
    time_offset = eqtime + 4.0 * lon  # minutes
    true_solar_time = hour_utc * 60.0 + time_offset
    hour_angle = math.radians((true_solar_time / 4.0) - 180.0)

    # Solar zenith
    lat_rad = math.radians(lat)
    cos_zen = (math.sin(lat_rad) * math.sin(decl)
               + math.cos(lat_rad) * math.cos(decl) * math.cos(hour_angle))
    cos_zen = max(-1.0, min(1.0, cos_zen))
    zenith = math.degrees(math.acos(cos_zen))

    # Solar azimuth
    sin_zen = math.sin(math.radians(zenith))
    if sin_zen > 0.001:
        cos_az = ((math.sin(decl) - math.sin(lat_rad) * cos_zen)
                  / (math.cos(lat_rad) * sin_zen))
        cos_az = max(-1.0, min(1.0, cos_az))
        azimuth = math.degrees(math.acos(cos_az))
        if hour_angle > 0:
            azimuth = 360.0 - azimuth
    else:
        azimuth = 0.0

    return {
        "zenith_deg": zenith,
        "elevation_deg": 90.0 - zenith,
        "azimuth_deg": azimuth,
        "cos_zenith": cos_zen,
    }


def correct_solar_zenith(frame, cos_zenith):
    """
    Normalize radiance for solar angle: divide by cos(zenith).

    Low sun → dim illumination → divide by small cos(θ) → boost signal.
    High sun → bright illumination → divide by ~1.0 → no change.

    Args:
        frame:       np.ndarray (H, W) float32
        cos_zenith:  float, cos(solar zenith angle)

    Returns:
        np.ndarray (H, W) float32, angle-corrected
        None if solar zenith > MAX_SOLAR_ZENITH (reject capture)
    """
    zenith_deg = math.degrees(math.acos(max(0.001, min(1.0, cos_zenith))))
    if zenith_deg > MAX_SOLAR_ZENITH:
        return None  # signal: reject this capture
    return frame / max(cos_zenith, 0.05)  # floor at 0.05 to prevent division explosion


# ═══════════════════════════════════════════════════════════════════════════
# Layer 4: Cloud Transmittance (NWS Weather API)
# ═══════════════════════════════════════════════════════════════════════════

def cloud_transmittance(cloud_cover_pct):
    """
    Estimate ground-level solar transmittance from cloud cover percentage.

    Uses the Kasten & Czeplak (1980) empirical model:
        T = 1 - k × (C/100)^2

    where k ≈ 0.75 and C is cloud cover in percent.

    Args:
        cloud_cover_pct: float, 0-100

    Returns:
        float, transmittance factor (0.25 for overcast, 1.0 for clear)
    """
    c = max(0.0, min(100.0, cloud_cover_pct)) / 100.0
    return 1.0 - CLOUD_COEFF * (c ** 2)


def correct_cloud_transmittance(frame, cloud_cover_pct):
    """
    Normalize for cloud-reduced irradiance.

    Args:
        frame:           np.ndarray (H, W) float32
        cloud_cover_pct: float, 0-100 from NWS weather data

    Returns:
        np.ndarray (H, W) float32, cloud-normalized
    """
    t = cloud_transmittance(cloud_cover_pct)
    return frame / max(t, 0.1)  # floor at 0.1 to prevent division explosion


# ═══════════════════════════════════════════════════════════════════════════
# Layer 5: Inter-Frame Illumination Matching
# ═══════════════════════════════════════════════════════════════════════════

def match_interframe_illumination(red_frame, nir_frame, mask=None):
    """
    Scale frames so pseudo-invariant features (PIFs) have equal brightness.

    If a cloud passes between the Red and NIR captures (3-5s apart),
    one frame is dimmer. We detect this by comparing mean brightness
    of stable regions (concrete, asphalt, monuments) and scaling
    the dimmer frame to match the brighter one.

    Args:
        red_frame:  np.ndarray (H, W) float32
        nir_frame:  np.ndarray (H, W) float32
        mask:       optional np.ndarray (H, W) bool, True = PIF pixels
                    If None, uses the brightest 10% of pixels (heuristic for
                    concrete/rock which are bright in both Red and NIR)

    Returns:
        (red_corrected, nir_corrected, metadata) where metadata contains:
            delta_pct:       float, brightness difference between frames
            scale_applied:   float, scale factor used (1.0 = no change)
            flagged:         bool, True if delta > threshold
    """
    if mask is None:
        # Heuristic: use the brightest 10% of pixels as pseudo-invariant
        combined = (red_frame + nir_frame) / 2.0
        threshold = np.percentile(combined, 90)
        mask = combined >= threshold

    if mask.sum() < 100:
        # Not enough PIF pixels — skip correction
        return red_frame, nir_frame, {
            "delta_pct": 0.0, "scale_applied": 1.0, "flagged": False
        }

    mean_red = float(red_frame[mask].mean())
    mean_nir = float(nir_frame[mask].mean())

    if mean_red == 0 or mean_nir == 0:
        return red_frame, nir_frame, {
            "delta_pct": 0.0, "scale_applied": 1.0, "flagged": False
        }

    # Compute relative difference
    delta = abs(mean_red - mean_nir) / max(mean_red, mean_nir)

    if delta > INTERFRAME_THRESHOLD:
        # Scale the dimmer frame to match the brighter one
        if mean_red < mean_nir:
            scale = mean_nir / mean_red
            red_frame = red_frame * scale
        else:
            scale = mean_red / mean_nir
            nir_frame = nir_frame * scale
    else:
        scale = 1.0

    return red_frame, nir_frame, {
        "delta_pct": round(delta * 100, 1),
        "scale_applied": round(scale, 4),
        "flagged": delta > INTERFRAME_THRESHOLD,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Layer 6: Seasonal Phenology Baseline
# ═══════════════════════════════════════════════════════════════════════════

def get_seasonal_baseline(dt=None):
    """
    Return the expected NDVI baseline for the current week of year.

    Uses a phenology curve derived from NASA HLS observations over
    Central Park (2019-2023). March grass is ~0.35, August grass ~0.74.

    Args:
        dt: datetime (default: now UTC)

    Returns:
        dict with keys:
            baseline:   float, expected NDVI for this week
            week:       int, ISO week number (1-52)
            season:     str, season name
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    week = dt.isocalendar()[1]
    week = max(1, min(52, week))
    baseline = SEASONAL_NDVI.get(week, 0.6493)

    # Season classification
    month = dt.month
    if month in (3, 4, 5):
        season = "spring"
    elif month in (6, 7, 8):
        season = "summer"
    elif month in (9, 10, 11):
        season = "autumn"
    else:
        season = "winter"

    return {
        "baseline": baseline,
        "week": week,
        "season": season,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Layer 7: Kalman Temporal Filter
# ═══════════════════════════════════════════════════════════════════════════

class KalmanNDVI:
    """
    1D Kalman filter for smoothing NDVI time series.

    Vegetation changes slowly (~0.01 NDVI/day max), so a single corrupted
    reading (shadow, bird, sensor glitch) gets smoothed out. Sustained
    drops over multiple readings trigger real alerts.

    Usage:
        kf = KalmanNDVI()
        for ndvi_measurement in stream:
            smoothed = kf.update(ndvi_measurement)
    """

    def __init__(self, process_noise=KALMAN_PROCESS_NOISE,
                 measurement_noise=KALMAN_MEASUREMENT_NOISE):
        self.Q = process_noise        # process noise variance
        self.R = measurement_noise    # measurement noise variance
        self.x = None                 # state estimate (NDVI)
        self.P = 1.0                  # estimate uncertainty
        self.initialized = False

    def update(self, measurement):
        """
        Update filter with a new NDVI measurement.

        Args:
            measurement: float, raw NDVI value

        Returns:
            float, smoothed NDVI estimate
        """
        if not self.initialized:
            self.x = measurement
            self.P = self.R
            self.initialized = True
            return self.x

        # Predict
        x_pred = self.x
        P_pred = self.P + self.Q

        # Update
        K = P_pred / (P_pred + self.R)  # Kalman gain
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred

        return self.x

    def get_state(self):
        """Return current state and uncertainty."""
        return {
            "estimate": self.x,
            "uncertainty": self.P,
            "initialized": self.initialized,
        }

    def reset(self):
        """Reset filter state."""
        self.x = None
        self.P = 1.0
        self.initialized = False


# ═══════════════════════════════════════════════════════════════════════════
# Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def apply_corrections(red_frame, nir_frame,
                      red_exposure=None, red_gain=None,
                      nir_exposure=None, nir_gain=None,
                      timestamp=None,
                      cloud_cover_pct=None,
                      lat=DEFAULT_LAT, lon=DEFAULT_LON,
                      pif_mask=None,
                      vila_overrides=None):
    """
    Apply the full 7-layer correction pipeline to a Red+NIR frame pair.

    Any layer can be skipped if its input data is unavailable — the pipeline
    degrades gracefully. The vila_overrides parameter allows the VILA VLM
    to selectively enable/disable layers or adjust parameters based on
    scene analysis.

    Args:
        red_frame:       np.ndarray (H, W) uint8 or float, raw Red capture
        nir_frame:       np.ndarray (H, W) uint8 or float, raw NIR capture
        red_exposure:    float, Red frame exposure time (None = skip L1)
        red_gain:        float, Red frame gain
        nir_exposure:    float, NIR frame exposure time
        nir_gain:        float, NIR frame gain
        timestamp:       datetime, capture time (None = skip L3)
        cloud_cover_pct: float 0-100, from NWS (None = skip L4)
        lat, lon:        float, observation site coordinates
        pif_mask:        np.ndarray (H, W) bool, PIF pixels (None = auto)
        vila_overrides:  dict, VLM-generated correction overrides:
            skip_layers:       list[int], layers to skip (e.g. [3, 4])
            cloud_cover_adj:   float, adjusted cloud cover from scene analysis
            interframe_scale:  float, override scale factor
            confidence:        float 0-1, VLM confidence in scene assessment

    Returns:
        dict with keys:
            red:            np.ndarray, corrected Red frame
            nir:            np.ndarray, corrected NIR frame
            ndvi:           np.ndarray, per-pixel NDVI map
            mean_ndvi:      float, mean NDVI of valid pixels
            corrections:    dict, metadata from each applied layer
            solar:          dict, solar position info
            seasonal:       dict, phenology baseline info
            quality_flags:  list[str], any quality warnings
    """
    # Parse overrides
    skip = set()
    if vila_overrides:
        skip = set(vila_overrides.get("skip_layers", []))

    corrections = {}
    quality_flags = []

    # Convert to float32
    red = red_frame.astype(np.float32)
    nir = nir_frame.astype(np.float32)

    # ── Layer 1: Auto-Exposure Normalization ──
    if 1 not in skip and red_exposure is not None and nir_exposure is not None:
        red = normalize_exposure(red, red_exposure, red_gain or 1.0)
        nir = normalize_exposure(nir, nir_exposure, nir_gain or 1.0)
        corrections["L1_exposure"] = {
            "red": {"exposure": red_exposure, "gain": red_gain},
            "nir": {"exposure": nir_exposure, "gain": nir_gain},
        }
    else:
        corrections["L1_exposure"] = "skipped"

    # ── Layer 2: Dark Object Subtraction ──
    if 2 not in skip:
        red_dark = float(np.percentile(red, 1.0))
        nir_dark = float(np.percentile(nir, 1.0))
        red = dark_object_subtraction(red)
        nir = dark_object_subtraction(nir)
        corrections["L2_dos"] = {
            "red_dark_value": round(red_dark, 2),
            "nir_dark_value": round(nir_dark, 2),
        }
    else:
        corrections["L2_dos"] = "skipped"

    # ── Layer 3: Solar Zenith Correction ──
    solar_info = None
    if 3 not in skip and timestamp is not None:
        solar_info = solar_position(timestamp, lat, lon)
        cos_z = solar_info["cos_zenith"]

        if solar_info["zenith_deg"] > MAX_SOLAR_ZENITH:
            quality_flags.append(
                f"solar_zenith_extreme ({solar_info['zenith_deg']:.1f}°)"
            )
        else:
            red = correct_solar_zenith(red, cos_z)
            nir = correct_solar_zenith(nir, cos_z)
            if red is None or nir is None:
                quality_flags.append("capture_rejected_low_sun")
                red = red_frame.astype(np.float32)
                nir = nir_frame.astype(np.float32)

        corrections["L3_solar"] = {
            "zenith_deg": round(solar_info["zenith_deg"], 1),
            "elevation_deg": round(solar_info["elevation_deg"], 1),
            "cos_zenith": round(cos_z, 4),
        }
    else:
        corrections["L3_solar"] = "skipped"

    # ── Layer 4: Cloud Transmittance ──
    if 4 not in skip and cloud_cover_pct is not None:
        # Allow VILA to override cloud cover estimate
        cc = cloud_cover_pct
        if vila_overrides and "cloud_cover_adj" in vila_overrides:
            cc = vila_overrides["cloud_cover_adj"]

        t = cloud_transmittance(cc)
        red = correct_cloud_transmittance(red, cc)
        nir = correct_cloud_transmittance(nir, cc)
        corrections["L4_cloud"] = {
            "cloud_cover_pct": round(cc, 1),
            "transmittance": round(t, 3),
        }
    else:
        corrections["L4_cloud"] = "skipped"

    # ── Layer 5: Inter-Frame Illumination Matching ──
    if 5 not in skip:
        if vila_overrides and "interframe_scale" in vila_overrides:
            # VILA provided a scale factor directly
            scale = vila_overrides["interframe_scale"]
            if scale > 1.0:
                nir = nir * scale
            elif scale < 1.0:
                red = red * (1.0 / scale)
            corrections["L5_interframe"] = {
                "source": "vila",
                "scale_applied": round(scale, 4),
            }
        else:
            red, nir, ifm = match_interframe_illumination(red, nir, pif_mask)
            corrections["L5_interframe"] = ifm
            if ifm["flagged"]:
                quality_flags.append(
                    f"interframe_mismatch ({ifm['delta_pct']}%)"
                )
    else:
        corrections["L5_interframe"] = "skipped"

    # ── Compute NDVI ──
    denom = nir + red
    ndvi = np.where(denom > 0, (nir - red) / denom, 0.0)
    valid = denom > 0
    mean_ndvi = float(ndvi[valid].mean()) if valid.any() else 0.0

    # ── Layer 6: Seasonal Phenology Baseline ──
    seasonal = get_seasonal_baseline(timestamp)
    corrections["L6_seasonal"] = seasonal

    # ── Layer 7: Kalman filter is applied externally per-zone ──
    corrections["L7_kalman"] = "applied per-zone in scan_zone()"

    return {
        "red": red,
        "nir": nir,
        "ndvi": ndvi,
        "mean_ndvi": round(mean_ndvi, 4),
        "corrections": corrections,
        "solar": solar_info,
        "seasonal": seasonal,
        "quality_flags": quality_flags,
    }

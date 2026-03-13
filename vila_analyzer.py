#!/usr/bin/env python3
"""
vila_analyzer.py — VILA Vision-Language Scene Analyzer
========================================================
Runs NVIDIA VILA 1.5-3B (4-bit AWQ quantized) on the Jetson Orin Nano
to analyze multispectral captures and generate intelligent radiometric
correction parameters.

Instead of blindly applying all 7 correction layers, the VLM sees the
actual Red+NIR frames and weather context, then decides:
  - Which corrections are needed
  - Scene-specific parameter adjustments
  - Natural language explanation for the dashboard
  - Confidence score for the overall reading

Architecture:
  Camera frames + Weather JSON → VILA prompt → Structured JSON output
                                                    ↓
                            radiometric.apply_corrections(vila_overrides=...)

Requirements:
  - NVIDIA Jetson Orin Nano (8 GB)
  - dustynv/vila:1.5-3b-r36 container (or compatible NanoLLM container)
  - ~2 GB VRAM for 4-bit quantized model

Usage:
    from vila_analyzer import VilaSceneAnalyzer

    analyzer = VilaSceneAnalyzer()  # loads model
    result = analyzer.analyze(red_frame, nir_frame, weather_data)
    # result["corrections"]  → overrides for radiometric.apply_corrections
    # result["explanation"]  → natural language for dashboard
"""

import os
import sys
import json
import time
import re
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# VILA model config
VILA_MODEL = os.environ.get(
    "VILA_MODEL",
    "Efficient-Large-Model/VILA1.5-3b-AWQ"
)

# Classes from our DINOv2 classifier
CLASSES = ["healthy_vegetation", "stressed_soil"]


# ═══════════════════════════════════════════════════════════════════════════
# System Prompt — instructs VILA on how to analyze multispectral scenes
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert agricultural remote sensing analyst embedded in an autonomous vegetation monitoring system. You analyze multispectral camera captures (Red band ~665nm, Near-Infrared band ~850nm) taken from a ground-level monitoring station in Central Park, NYC.

Your job is to analyze the scene and output a JSON object with radiometric correction parameters and a natural-language explanation. You must be precise and scientific.

Consider these factors when analyzing:
1. ILLUMINATION: Are there shadows? Is the scene evenly lit? Did lighting change between the Red and NIR captures?
2. SCENE CONTENT: What percentage is vegetation vs. bare soil vs. artificial surfaces (concrete, stone, asphalt)?
3. ATMOSPHERIC: Is there visible haze, fog, or mist? Does the image look washed out?
4. QUALITY: Are there motion artifacts, sensor noise, or obstructions?

Output ONLY valid JSON in this exact format (no other text):
{
  "scene_description": "Brief description of what you see",
  "vegetation_pct": 0-100,
  "soil_pct": 0-100,
  "artificial_pct": 0-100,
  "shadow_pct": 0-100,
  "illumination_quality": "uniform|patchy|poor",
  "interframe_change": true/false,
  "haze_detected": true/false,
  "skip_layers": [],
  "cloud_cover_adj": null or 0-100,
  "confidence": 0.0-1.0,
  "explanation": "One sentence describing the scene conditions and any concerns for NDVI accuracy"
}"""


# ═══════════════════════════════════════════════════════════════════════════
# Scene Analysis Prompt Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_analysis_prompt(weather=None, solar=None, timestamp=None):
    """
    Build the user prompt with weather and solar context.

    Args:
        weather:   dict from weather.py get_current_weather()
        solar:     dict from radiometric.solar_position()
        timestamp: datetime

    Returns:
        str, the prompt to send alongside the Red+NIR image pair
    """
    ctx_parts = ["Analyze this multispectral image pair (Red band left, NIR band right)."]

    if weather:
        ctx_parts.append(
            f"Current weather: {weather.get('description', 'N/A')}, "
            f"temp {weather.get('temp_c', '?')}°C, "
            f"humidity {weather.get('humidity_pct', '?')}%, "
            f"cloud cover {weather.get('cloud_cover_pct', '?')}%, "
            f"visibility {weather.get('visibility_km', '?')} km."
        )

    if solar:
        ctx_parts.append(
            f"Solar position: elevation {solar.get('elevation_deg', '?'):.1f}°, "
            f"azimuth {solar.get('azimuth_deg', '?'):.1f}°."
        )

    if timestamp:
        ctx_parts.append(f"Capture time: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}.")

    ctx_parts.append(
        "Respond with ONLY valid JSON matching the specified schema. "
        "No markdown, no explanation outside the JSON."
    )

    return " ".join(ctx_parts)


# ═══════════════════════════════════════════════════════════════════════════
# VILA Model Wrapper
# ═══════════════════════════════════════════════════════════════════════════

class VilaSceneAnalyzer:
    """
    Wraps the VILA 1.5-3B VLM for scene analysis on Jetson.

    Uses NanoLLM (dustynv) for efficient 4-bit inference.
    Falls back to a rule-based heuristic if the model isn't available.
    """

    def __init__(self, model_name=VILA_MODEL):
        self.model = None
        self.model_name = model_name
        self._fallback = False

        try:
            from nano_llm import NanoLLM
            print(f"  Loading VILA VLM: {model_name} …", end=" ", flush=True)
            t0 = time.time()
            self.model = NanoLLM.from_pretrained(
                model_name,
                quantization="q4f16_ft",
                api="mlc",
            )
            elapsed = time.time() - t0
            print(f"loaded in {elapsed:.0f}s ✅")
        except ImportError:
            print("  ⚠️  NanoLLM not available — using rule-based fallback")
            self._fallback = True
        except Exception as e:
            print(f"  ⚠️  VILA load failed ({e}) — using rule-based fallback")
            self._fallback = True

    def analyze(self, red_frame, nir_frame, weather=None, solar=None,
                timestamp=None):
        """
        Analyze a Red+NIR capture pair and return correction overrides.

        Args:
            red_frame:  np.ndarray (H, W) uint8/float, Red capture
            nir_frame:  np.ndarray (H, W) uint8/float, NIR capture
            weather:    dict from NWS
            solar:      dict from radiometric.solar_position()
            timestamp:  datetime

        Returns:
            dict with keys:
                corrections:  dict, overrides for radiometric.apply_corrections
                explanation:  str, natural language scene description
                confidence:   float 0-1
                raw_output:   str, raw VLM output
                latency_ms:   float, inference time
        """
        if self._fallback:
            return self._heuristic_analysis(red_frame, nir_frame, weather, solar)

        return self._vlm_analysis(red_frame, nir_frame, weather, solar, timestamp)

    def _vlm_analysis(self, red_frame, nir_frame, weather, solar, timestamp):
        """Run VILA VLM inference on the image pair."""
        import cv2
        from PIL import Image

        # Create side-by-side Red+NIR image for the VLM
        red_vis = red_frame if len(red_frame.shape) == 3 else cv2.cvtColor(
            red_frame.astype(np.uint8), cv2.COLOR_GRAY2RGB
        )
        nir_vis = nir_frame if len(nir_frame.shape) == 3 else cv2.cvtColor(
            nir_frame.astype(np.uint8), cv2.COLOR_GRAY2RGB
        )

        # Add labels
        h, w = red_vis.shape[:2]
        combined = np.hstack([red_vis, nir_vis])
        cv2.putText(combined, "RED (665nm)", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(combined, "NIR (850nm)", (w + 10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

        pil_image = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))

        # Build prompt
        user_prompt = build_analysis_prompt(weather, solar, timestamp)

        # Run VLM inference
        t0 = time.time()
        response = self.model.generate(
            pil_image,
            SYSTEM_PROMPT + "\n\n" + user_prompt,
            max_new_tokens=512,
            temperature=0.1,  # low temp for structured output
        )
        latency = (time.time() - t0) * 1000

        # Parse JSON from response
        parsed = self._parse_response(response)

        return {
            "corrections": {
                "skip_layers": parsed.get("skip_layers", []),
                "cloud_cover_adj": parsed.get("cloud_cover_adj"),
                "confidence": parsed.get("confidence", 0.5),
            },
            "scene": {
                "vegetation_pct": parsed.get("vegetation_pct", 50),
                "soil_pct": parsed.get("soil_pct", 25),
                "artificial_pct": parsed.get("artificial_pct", 25),
                "shadow_pct": parsed.get("shadow_pct", 0),
                "illumination": parsed.get("illumination_quality", "uniform"),
                "haze": parsed.get("haze_detected", False),
                "interframe_change": parsed.get("interframe_change", False),
            },
            "explanation": parsed.get("explanation",
                                      parsed.get("scene_description", "Analysis complete")),
            "confidence": parsed.get("confidence", 0.5),
            "raw_output": response,
            "latency_ms": round(latency, 1),
        }

    def _parse_response(self, response):
        """Extract JSON from VLM response (handles markdown fences)."""
        text = response.strip()

        # Try to find JSON block
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: return defaults
        return {
            "scene_description": "VLM output could not be parsed",
            "confidence": 0.3,
            "skip_layers": [],
        }

    def _heuristic_analysis(self, red_frame, nir_frame, weather, solar):
        """
        Rule-based fallback when VILA isn't available.

        Uses image statistics and weather data to make educated guesses
        about scene conditions — not as sophisticated as VILA, but
        provides reasonable correction parameters.
        """
        t0 = time.time()

        red = red_frame.astype(np.float32)
        nir = nir_frame.astype(np.float32)
        if len(red.shape) == 3:
            red = np.mean(red, axis=2)
        if len(nir.shape) == 3:
            nir = np.mean(nir, axis=2)

        # Normalize to 0-1 if input looks like uint8 (values > 1)
        if red.max() > 1.0 or nir.max() > 1.0:
            scale = max(float(red.max()), float(nir.max()), 1.0)
            red = red / scale
            nir = nir / scale

        # Compute basic scene statistics
        red_mean = float(red.mean())
        nir_mean = float(nir.mean())
        red_std = float(red.std())
        nir_std = float(nir.std())

        # Inter-frame brightness check
        delta = abs(red_mean - nir_mean) / max(red_mean, nir_mean, 0.001)
        interframe_change = delta > 0.10

        # Shadow estimation (pixels below 20% of dynamic range)
        shadow_thresh = 0.20 * max(red_mean, 0.01)
        shadow_pct = float((red < shadow_thresh).sum()) / red.size * 100

        # Vegetation estimation (high NIR, low Red = vegetation)
        denom = nir + red
        safe = denom > 0.05  # proper guard for normalized data
        ndvi_quick = np.where(safe, (nir - red) / denom, 0)
        veg_pct = float((ndvi_quick > 0.15).sum()) / ndvi_quick.size * 100

        # Haze detection (low contrast = haze)
        contrast = red_std / max(red_mean, 0.01)
        haze_detected = contrast < 0.15

        # Cloud cover adjustment from weather
        cloud_adj = None
        if weather and haze_detected:
            cloud_adj = min(100, weather.get("cloud_cover_pct", 50) + 15)

        # Determine which layers to skip
        skip_layers = []
        if solar and solar.get("elevation_deg", 45) > 60:
            # High sun: solar correction is negligible
            pass  # still apply, it's cheap

        explanation_parts = []
        if interframe_change:
            explanation_parts.append(
                f"Brightness shifted {delta*100:.0f}% between Red and NIR frames"
            )
        if haze_detected:
            explanation_parts.append("Low contrast suggests haze or fog")
        if shadow_pct > 30:
            explanation_parts.append(f"~{shadow_pct:.0f}% of scene is in shadow")
        if veg_pct > 70:
            explanation_parts.append(f"Dense vegetation ({veg_pct:.0f}% of scene)")
        if not explanation_parts:
            explanation_parts.append("Scene conditions appear normal")

        latency = (time.time() - t0) * 1000

        return {
            "corrections": {
                "skip_layers": skip_layers,
                "cloud_cover_adj": cloud_adj,
                "confidence": 0.6,  # lower confidence for heuristic
            },
            "scene": {
                "vegetation_pct": round(veg_pct, 1),
                "soil_pct": round(max(0, 100 - veg_pct - shadow_pct), 1),
                "artificial_pct": 0,
                "shadow_pct": round(shadow_pct, 1),
                "illumination": "poor" if interframe_change else "uniform",
                "haze": haze_detected,
                "interframe_change": interframe_change,
            },
            "explanation": ". ".join(explanation_parts) + ".",
            "confidence": 0.6,
            "raw_output": "(heuristic fallback)",
            "latency_ms": round(latency, 1),
        }

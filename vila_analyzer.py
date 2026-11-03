#!/usr/bin/env python3
"""
vila_analyzer.py — VILA Vision-Language Scene Analyzer
========================================================
Runs scene analysis via NVIDIA NIM cloud VLM (preferred) or a local
heuristic fallback to generate intelligent radiometric correction
parameters for multispectral NDVI captures.

Instead of blindly applying all 7 correction layers, the VLM sees the
actual Red+NIR frames and weather context, then decides:
  - Which corrections are needed
  - Scene-specific parameter adjustments
  - Natural language explanation for the dashboard
  - Confidence score for the overall reading

Architecture:
  Camera frames + Weather JSON → Cloud VLM (NVIDIA NIM) → JSON output
                                                     ↓
                            radiometric.apply_corrections(vila_overrides=...)

Modes (auto-detected, in priority order):
  1. NVIDIA NIM Cloud API  → Set NVIDIA_NIM_API_KEY in .env
  2. Local REST API        → Set VILA_API_URL (container on LAN)
  3. Direct NanoLLM        → Inside dustynv/nano_llm container (16GB+ only)
  4. Heuristic fallback    → Rule-based, <1ms, no GPU needed

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
CLASSES = ["bare_soil", "healthy_vegetation", "non_vegetation", "stressed_vegetation"]


# ═══════════════════════════════════════════════════════════════════════════
# System Prompt — instructs VILA on how to analyze multispectral scenes
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert agricultural remote sensing analyst embedded in an autonomous vegetation monitoring system. You analyze multispectral camera captures (Red band ~665nm, Near-Infrared band ~865nm) taken from a ground-level monitoring station in Central Park, NYC.

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

    Three modes of operation (auto-detected):
      1. REST API: VILA runs in Docker container, jetson_main.py calls via HTTP
         → Set VILA_API_URL env var or pass api_url to constructor
      2. Direct: NanoLLM imported in-process (when running inside the container)
      3. Heuristic fallback: rule-based analysis when VILA isn't available

    NOTE: VILA 3B MLC compilation requires >10GB RAM and will OOM on 8GB
    Orin Nano. The heuristic fallback (mode 3) is the production path for
    8GB devices. VLM is available for 16GB+ Orin NX/AGX or cross-compiled
    models. The heuristic detects vegetation, haze, shadows, and
    inter-frame brightness shifts in <1ms per frame.
    """

    def __init__(self, model_name=VILA_MODEL, api_url=None):
        self.model = None
        self.model_name = model_name
        self._fallback = False
        self._api_url = api_url or os.environ.get("VILA_API_URL")
        self._nim_api_key = os.environ.get("NVIDIA_NIM_API_KEY")
        self._nim_model = os.environ.get(
            "NVIDIA_NIM_MODEL",
            "meta/llama-3.2-90b-vision-instruct",  # best free VLM on NIM
        )
        self._nim_base_url = os.environ.get(
            "NVIDIA_NIM_URL",
            "https://integrate.api.nvidia.com/v1",
        )

        # Mode 1: NVIDIA NIM Cloud API (preferred — offloads VLM from Jetson)
        if self._nim_api_key:
            print(f"  🌐 NVIDIA NIM Cloud VLM: {self._nim_model}")
            print(f"     Endpoint: {self._nim_base_url}")
            return

        # Mode 2: Local REST API (VILA container on same device / LAN)
        if self._api_url:
            try:
                import urllib.request
                req = urllib.request.Request(
                    f"{self._api_url.rstrip('/')}/api/health",
                    method="GET",
                )
                urllib.request.urlopen(req, timeout=3)
                print(f"  VILA REST API connected: {self._api_url} ✅")
                return
            except Exception as e:
                print(f"  ⚠️  VILA API at {self._api_url} not reachable: {e}")
                self._api_url = None

        # Mode 3: Direct NanoLLM import (inside container)
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

        if self._nim_api_key:
            return self._cloud_analysis(red_frame, nir_frame, weather, solar, timestamp)

        if self._api_url:
            return self._api_analysis(red_frame, nir_frame, weather, solar, timestamp)

        return self._vlm_analysis(red_frame, nir_frame, weather, solar, timestamp)

    def _cloud_analysis(self, red_frame, nir_frame, weather, solar, timestamp):
        """Call NVIDIA NIM cloud API for scene analysis (OpenAI-compatible)."""
        import urllib.request
        import cv2
        import base64

        t0 = time.time()

        # Create side-by-side composite image
        red_u8 = red_frame.astype(np.uint8) if red_frame.dtype != np.uint8 else red_frame
        nir_u8 = nir_frame.astype(np.uint8) if nir_frame.dtype != np.uint8 else nir_frame
        if len(red_u8.shape) == 2:
            red_u8 = cv2.cvtColor(red_u8, cv2.COLOR_GRAY2RGB)
        if len(nir_u8.shape) == 2:
            nir_u8 = cv2.cvtColor(nir_u8, cv2.COLOR_GRAY2RGB)
        combined = np.hstack([red_u8, nir_u8])
        _, img_bytes = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(img_bytes.tobytes()).decode()

        # Build user message with weather/solar context
        context_parts = []
        if weather:
            cc = weather.get('cloud_cover_pct', 'N/A')
            desc = weather.get('description', 'N/A')
            temp = weather.get('temperature_f', weather.get('temperature_c', 'N/A'))
            context_parts.append(f"Weather: {desc}, cloud cover {cc}%, temp {temp}")
        if solar:
            context_parts.append(f"Solar elevation: {solar.get('elevation_deg', 'N/A')}°")
        context_str = ". ".join(context_parts) if context_parts else "No weather data available"

        user_text = (
            f"This is a side-by-side multispectral image pair from an agricultural monitoring camera. "
            f"Left half is the Red band (665nm), right half is the NIR band (865nm). "
            f"Current conditions: {context_str}. "
            f"Analyze the scene and output ONLY valid JSON with these fields: "
            f'{{"vegetation_pct": 0-100, "soil_pct": 0-100, "artificial_pct": 0-100, '
            f'"shadow_pct": 0-100, "haze_detected": true/false, '
            f'"illumination_quality": "uniform"/"patchy"/"poor", '
            f'"interframe_change": true/false, '
            f'"skip_layers": [], "cloud_cover_adj": null or 0-100, '
            f'"confidence": 0.0-1.0, '
            f'"explanation": "one sentence describing the scene"}}'
        )

        # OpenAI-compatible chat completion request
        payload = json.dumps({
            "model": self._nim_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": user_text,
                        },
                    ],
                }
            ],
            "max_tokens": 512,
            "temperature": 0.1,
            "stream": False,
        }).encode()

        try:
            req = urllib.request.Request(
                f"{self._nim_base_url.rstrip('/')}/chat/completions",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._nim_api_key}",
                },
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=30)
            result = json.loads(resp.read().decode())
            response_text = result["choices"][0]["message"]["content"]

            latency = (time.time() - t0) * 1000
            parsed = self._parse_response(response_text)

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
                                          parsed.get("scene_description", "Cloud analysis complete")),
                "confidence": parsed.get("confidence", 0.5),
                "raw_output": response_text,
                "latency_ms": round(latency, 1),
                "source": "nvidia_nim_cloud",
            }
        except Exception as e:
            print(f"  ⚠️  NIM cloud call failed ({e}), using heuristic fallback")
            return self._heuristic_analysis(red_frame, nir_frame, weather, solar)

    def _api_analysis(self, red_frame, nir_frame, weather, solar, timestamp):
        """Call VILA via the REST API running in the Docker container."""
        import urllib.request
        import cv2
        import base64

        t0 = time.time()

        # Create side-by-side image
        red_vis = red_frame if len(red_frame.shape) == 3 else cv2.cvtColor(
            red_frame.astype(np.uint8) if red_frame.dtype != np.uint8 else red_frame,
            cv2.COLOR_GRAY2RGB,
        )
        nir_vis = nir_frame if len(nir_frame.shape) == 3 else cv2.cvtColor(
            nir_frame.astype(np.uint8) if nir_frame.dtype != np.uint8 else nir_frame,
            cv2.COLOR_GRAY2RGB,
        )
        combined = np.hstack([red_vis, nir_vis])
        _, img_bytes = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(img_bytes.tobytes()).decode()

        # Build prompt
        user_prompt = build_analysis_prompt(weather, solar, timestamp)

        # POST to REST API
        payload = json.dumps({
            "prompt": SYSTEM_PROMPT + "\n\n" + user_prompt,
            "image": img_b64,
            "max_new_tokens": 512,
            "temperature": 0.1,
        }).encode()

        try:
            req = urllib.request.Request(
                f"{self._api_url.rstrip('/')}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=30)
            result = json.loads(resp.read().decode())
            response_text = result.get("response", result.get("text", ""))

            latency = (time.time() - t0) * 1000
            parsed = self._parse_response(response_text)

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
                "raw_output": response_text,
                "latency_ms": round(latency, 1),
                "source": "vila_rest_api",
            }
        except Exception as e:
            print(f"  ⚠️  VILA API call failed ({e}), using heuristic fallback")
            return self._heuristic_analysis(red_frame, nir_frame, weather, solar)

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
        cv2.putText(combined, "NIR (865nm)", (w + 10, 30),
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
        ndvi_quick = np.where(safe, (nir - red) / (denom + 1e-10), 0)
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

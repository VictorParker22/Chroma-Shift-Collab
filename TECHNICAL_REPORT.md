# Chroma-Shift: Technical Report

## Edge-Deployed Multispectral Vegetation Monitoring with DINOv2 Classification and Weather-Correlated Stress Detection

*Victor Parker — March 2026*

---

## 1. Introduction

Urban green spaces face a growing gap between the ecological services they provide and the tools available to monitor their health at a granular level. Traditional approaches to vegetation monitoring—satellite revisit schedules, manual field surveys, or periodic aerial flyovers—each carry tradeoffs in spatial resolution, temporal frequency, or cost. This project began with a practical question: could a single embedded device, positioned on the ground, continuously assess the condition of vegetation in a park using the same spectral principles that drive large-scale remote sensing?

Chroma-Shift is the system that emerged from that question. It runs on an NVIDIA Jetson Orin Nano mounted alongside a multispectral camera and motorized filter wheel. Every scan cycle (adaptively timed from 2 to 30 minutes), it captures a pair of narrowband images—one in the visible red and one in the near-infrared—applies a 7-layer self-calibrating radiometric correction, computes NDVI, classifies the scene into one of four categories (healthy vegetation, stressed vegetation, bare soil, non-vegetation) using a DINOv2 neural network at 149 frames per second via TensorRT, cross-checks the AI classification against NDVI physics using confidence thresholds, consults a cloud-hosted 90-billion-parameter Vision-Language Model for scene analysis, generates a farmer-facing advisory, and pushes everything to a cloud database. An eight-signal adaptive scheduler dynamically adjusts observation frequency per zone, and a fault tolerance system ensures the pipeline never fully stops—degrading gracefully through four operating modes if hardware or software failures occur. The goal is not to replace satellite monitoring but to sit between satellite passes and ground-truth inspections, filling the temporal gaps with continuous, automated readings.

This document walks through the full system, from the physics of plant reflectance and the hardware choices that exploit it, through the signal processing and machine learning pipeline, to the confidence threshold cross-checking, adaptive sensing, fault tolerance, weather correlation logic, and cloud data architecture that tie measurements to environmental context.

---

## 2. Physical Basis

### 2.1 Plant Reflectance and the Red-Edge

Chlorophyll molecules in healthy leaf tissue absorb strongly in the blue (430–450 nm) and red (640–680 nm) wavelengths, funneling that energy into photosynthesis. In the near-infrared region beyond roughly 700 nm, the leaf's internal spongy mesophyll structure reflects and transmits light efficiently rather than absorbing it. This sharp transition—from strong absorption in the visible red to high reflectance in the NIR—is sometimes called the "red edge," and it is one of the most reliable spectral indicators of photosynthetic activity across plant species.

When a plant experiences stress—whether from drought, disease, nutrient deficiency, or mechanical damage—the chlorophyll concentration in its leaves decreases. As chlorophyll degrades, red-band absorption weakens: the leaf reflects more red light than it would under healthy conditions. At the same time, structural changes in cell turgor and mesophyll arrangement reduce NIR reflectance. Both effects push in the same direction: sick or dying vegetation looks brighter in the red and dimmer in the NIR compared to its healthy counterpart. Bare soil, pavement, and stone lack the mesophyll structure entirely, so their red and NIR reflectance values tend to be close to each other and low overall.

### 2.2 Normalized Difference Vegetation Index

The Normalized Difference Vegetation Index, introduced by Rouse et al. (1974), captures this spectral contrast in a single bounded ratio:

```
NDVI = (ρ_NIR − ρ_Red) / (ρ_NIR + ρ_Red)
```

where ρ represents the surface reflectance in each band. The normalization reduces the influence of variable illumination, atmospheric path length, and sensor gain—effects that would confound a simple difference. Values range from −1 to +1, though in practice most terrestrial surfaces fall between −0.1 (water, deep shadow) and +0.9 (dense tropical canopy). Urban parks in temperate climates typically show NDVI between 0.3 and 0.8 depending on season, species composition, and management practices.

What makes NDVI operationally useful is that it is monotonically sensitive to leaf area and chlorophyll density within the range most relevant for vegetation monitoring. A stand of grass will register somewhere around 0.3–0.5. A full deciduous canopy in summer might push 0.75–0.85. Bare dirt sits near 0.1–0.15. These values shift with season, species, and viewing geometry, but the relative ordering is stable enough that a change of 0.05–0.10 at the same site over a short period almost always signals a real change in plant condition rather than a measurement artifact.

### 2.3 Satellite Ground Truth

To anchor the system's NDVI readings against a known reference, a five-year record of satellite observations was compiled from NASA's Harmonized Landsat Sentinel-2 (HLS) archive. The HLS product merges surface-reflectance data from Landsat 8/9 (30-meter resolution, 16-day revisit) and Sentinel-2A/B (10–20-meter resolution, 5-day revisit) into a common grid and atmospheric correction framework, giving a combined temporal sampling roughly every two to three days over mid-latitude sites.

The data request targeted a single pixel (40.7829° N, 73.9654° W) in the Great Lawn area of Central Park, covering January 2019 through December 2023. The raw dataset included several hundred observations across both the L30 (Landsat) and S30 (Sentinel-2) tiers. Quality filtering was performed using the HLS Fmask bitmask, a per-pixel quality flag that encodes cloud, cloud shadow, adjacent cloud contamination, and snow/ice status in a single byte. The filtering logic examines bits 1 through 4 of the Fmask value:

- Bit 1: Cloud presence
- Bit 2: Adjacent to cloud or cloud shadow
- Bit 3: Cloud shadow
- Bit 4: Snow or ice cover

Any observation flagged on any of those four bits was discarded. Water and aerosol flags (bits 0, 5–7) were retained, since moderate aerosol loads and proximity to water do not necessarily invalidate the vegetation reflectance signal at this location.

After quality filtering, 296 clear-sky observations remained. NDVI was computed for each using the standard band assignments: Landsat B04 (Red) and B05 (NIR); Sentinel-2 B04 (Red) and B8A (narrow NIR). The combined dataset yielded:

| Statistic | Value |
|-----------|-------|
| Observations | 296 |
| Mean NDVI | 0.6493 |
| Median NDVI | 0.7438 |
| Minimum | 0.2698 |
| Maximum | 0.8611 |
| Period | 2019–2023 |

The mean value of 0.6493 was adopted as the primary vegetation/stress classification threshold. The gap between mean and median (0.6493 vs. 0.7438) reflects the seasonal distribution: winter observations (bare deciduous canopy) pull the mean downward relative to the summer-dominated median. For field deployment in March 2026, a seasonally adjusted baseline of 0.25 is used to account for the dormant state of deciduous species, with the full-year mean serving as the target for warmer months.

---

## 3. Hardware

### 3.1 Compute Platform

The system runs on an NVIDIA Jetson Orin Nano Developer Kit, a compact single-board computer designed for edge AI workloads. The board carries a 6-core Arm Cortex-A78AE CPU paired with an Ampere-architecture GPU containing 1024 CUDA cores and 32 Tensor Cores. With 8 GB of unified LPDDR5 memory shared between CPU and GPU, memory management becomes a central constraint for any model that exceeds a few hundred megabytes in its working set. A 1 TB NVMe SSD provides the storage headroom needed for large training datasets, model checkpoints, and image captures.

The board runs JetPack 6.x (L4T R36.4), NVIDIA's BSP for Jetson platforms, which bundles Ubuntu 24.04, CUDA 12.8, cuDNN 9, and TensorRT 10. Computational workloads that exceed the on-device memory budget are handled inside Docker containers built from the `dustynv/pytorch` image ecosystem, which pre-packages PyTorch builds compiled against the Jetson's specific CUDA and cuDNN versions.

### 3.2 Imaging

The camera is an Arducam Multispectral Camera connected over USB. It feeds frames to OpenCV's V4L2 backend, which provides the low-level interface for frame capture without relying on NVIDIA's proprietary Argus camera stack (designed for MIPI-CSI sensors rather than USB devices).

Band selection is handled by a ZWO Electronic Filter Wheel (EFW Mini), a five-slot motorized carousel controlled through the ZWO EFW SDK. The SDK ships as a compiled C library (`libEFWFilter.so`) for ARM64. A Python ctypes wrapper maps the SDK's C functions—`EFWGetNum`, `EFWOpen`, `EFWSetPosition`, `EFWGetPosition`, `EFWClose`—into a high-level `FilterWheel` class that manages initialization, calibration, and position switching with configurable settle times.

The filter wheel is loaded with narrowband interference filters:

| Slot | Filter | Center Wavelength | Application |
|------|--------|-------------------|-------------|
| 0 | Clear (L) | 400–700 nm | Visible-light reference |
| 1 | Red | 665 nm | Chlorophyll absorption band |
| 2 | NIR | 833 / 865 nm | Mesophyll reflectance band |
| 3 | Red-Edge | 700–740 nm | Chlorophyll transition zone |
| 4 | SWIR | 1570–1650 nm | Moisture and mineral index |

For NDVI computation, only slots 1 (Red) and 2 (NIR) are used. The Red filter targets the primary absorption peak of chlorophyll-a at roughly 665 nm, matching the band definition used by Sentinel-2's B04. The NIR filter sits in the plateau region beyond the red edge, where healthy canopy reflectance is highest.

### 3.3 Capture Sequence

Each measurement cycle follows a fixed sequence:

1. The filter wheel rotates to the Red position (slot 1). A 1.5-second mechanical settle time allows vibration from the stepper motor to dissipate, preventing motion blur.
2. Five frames are captured and discarded. USB cameras buffer frames internally, and these "flush" frames clear stale data from the pipeline so the saved image reflects the current filter state.
3. A Red-band frame is captured and saved as a 16-bit PNG.
4. The filter wheel rotates to the NIR position (slot 2), with the same settle and flush procedure.
5. A NIR-band frame is captured and saved.
6. NDVI is computed per-pixel from the two frames using the standard ratio formula, with a division-by-zero guard (pixels where (NIR + Red) equals zero return NaN).

This entire sequence takes roughly five seconds, dominated by the mechanical filter wheel transitions and settle times.

---

## 4. Signal Processing Pipeline

### 4.1 Self-Calibrating Radiometric Correction

A central challenge of ground-level multispectral imaging is converting raw camera digital numbers (DN) to values comparable across captures taken under different illumination conditions. Satellite missions solve this problem with atmospheric correction models and periodic vicarious calibration against instrumented ground targets. For an autonomous field station without human operators, neither approach is practical: the system must self-calibrate using only the data available at the moment of capture.

Chroma-Shift addresses this with a seven-layer radiometric correction pipeline (`radiometric.py`) that applies sequential, independent corrections. Each layer targets a distinct physical effect, and later layers catch artifacts that earlier layers miss. The pipeline accepts optional overrides from a cloud-hosted Vision-Language Model (see Section 5.2) that can selectively disable or adjust layers based on scene-level understanding. The pipeline has been validated against a comprehensive test suite of 109 assertions covering edge cases, seasonal coverage, cross-layer interactions, and known-input NDVI consistency checks.

**Layer 1 — Auto-Exposure Normalization.** The camera's exposure time and gain are read for each frame via OpenCV properties. Raw pixel values are divided by (exposure × gain) to convert DN to relative radiance. After normalization, both frames are renormalized to a shared 0–1 range using the maximum value across both bands. This renormalization step, added after validation testing revealed that unbounded post-normalization values could cause downstream layer thresholds to fail, preserves the Red/NIR ratio while constraining the dynamic range. This removes the camera's automatic compensation for changing light levels, ensuring that two frames captured seconds apart under different auto-exposure settings produce comparable radiance values.

**Layer 2 — Dark Object Subtraction (Chavez 1988).** The 1st-percentile pixel value in each band is computed and subtracted from all pixels. This follows the method described by Chavez (1988): the darkest object in the scene (deep shadow, dark soil) should have near-zero surface reflectance, so any measured signal from those pixels is attributable to atmospheric path radiance and sensor offset. Even at ground level, scattered skylight adds a measurable baseline that this step removes.

**Layer 3 — Solar Zenith Angle Correction.** The sun's position is computed from the capture timestamp and site coordinates (40.7812° N, 73.9665° W) using the Spencer (1971) solar declination and equation-of-time formulae. Radiance values are divided by cos(θ), where θ is the solar zenith angle, normalizing all readings to equivalent overhead illumination. Captures with θ > 80° (dawn/dusk) are flagged as low-confidence and excluded from trend analysis.

**Layer 4 — Cloud Transmittance Modeling.** Cloud cover percentage from the National Weather Service API is converted to a ground-level transmittance estimate using the Kasten & Czeplak (1980) empirical model: T = 1 − 0.75 × (C/100)². Clear sky (0% cloud) gives T = 1.0; full overcast (100%) gives T = 0.25. Dividing by T normalizes radiance to clear-sky equivalent, enabling meaningful comparison of readings taken under partly cloudy versus clear conditions.

**Layer 5 — Inter-Frame Illumination Matching.** Because Red and NIR frames are captured 3–5 seconds apart through the filter wheel, a passing cloud can illuminate the two frames differently, corrupting the NDVI ratio. This layer compares mean brightness across pseudo-invariant features (PIFs) — surfaces whose reflectance does not change seasonally, such as concrete paths, stone monuments, and asphalt. If brightness differs by more than 10%, the dimmer frame is scaled to match. PIF detection uses an NDVI-based spectral flatness criterion (|NDVI| < 0.15) rather than simple brightness thresholds, which proved more robust in vegetation-heavy scenes where the brightest pixels are healthy canopy rather than concrete. An adaptive noise floor prevents the matching algorithm from amplifying sensor noise in low-signal conditions. This is the most critical correction for sequential-capture hardware.

**Layer 6 — Seasonal Phenology Baseline.** Rather than comparing every reading against the annual-mean NDVI of 0.6493, a weekly baseline curve is derived from the 296 HLS satellite observations. March vegetation in Central Park naturally registers NDVI ~0.35; August peaks near 0.74. Using the seasonally appropriate baseline prevents false stress alerts from normal dormancy in winter and early spring.

**Layer 7 — Kalman Temporal Filter.** A 1-D Kalman filter smooths the NDVI time series per field zone. Vegetation canopy changes slowly (less than 0.01 NDVI per day under most conditions), so a single corrupted reading (momentary shadow, bird in the frame, sensor noise) is absorbed by the filter's prediction-vs-measurement weighting. Sustained declines over multiple readings pass through and trigger alerts. Process noise is set to 0.001; measurement noise to 0.02, estimated from the observed variance of consecutive readings under stable conditions.

### 4.2 Per-Pixel NDVI Computation

After radiometric correction, the corrected Red and NIR frames are combined into the standard NDVI ratio:

```python
ndvi = (nir_corrected - red_corrected) / (nir_corrected + red_corrected + 1e-10)
```

The small epsilon prevents division-by-zero in fully shadowed pixels. The resulting per-pixel array is clipped to [−1, +1] and spatial-averaged to produce a scalar NDVI for the scene.

### 4.3 Stress Classification

Each corrected NDVI reading is compared against the seasonal phenology baseline for the current week. The severity of any deficit is binned into discrete levels:

| NDVI Drop (below seasonal baseline) | Severity | Status |
|--------------------------------------|----------|--------|
| ≥ 0.15 | Critical | Stress alert |
| 0.10 – 0.15 | High | Stress alert |
| 0.05 – 0.10 | Medium | Watch |
| < 0.05 | Low | Healthy |

When an alert is triggered, the system generates a recommendation tailored to the stress type (e.g., irrigation for water stress, inspection for disease) and pushes it to the cloud database with the associated NDVI measurement, weather conditions, and AI classification result.

---

## 5. AI Classification Pipeline

### 5.1 DINOv2 Image Classification

NDVI alone cannot distinguish between all causes of low reflectance. A muddy path and a drought-stressed lawn may show similar NDVI values, but they require entirely different management responses. To address this, a four-class image classifier was added downstream of the NDVI computation, applied to the Red-band frame (which, as an RGB-approximation, preserves the spatial texture information that NDVI collapses into a single number).

The classifier is built on DINOv2, a self-supervised vision foundation model developed by Meta AI. DINOv2 was trained on 142 million curated images using a combination of image-level and patch-level self-distillation objectives, producing feature representations that transfer well to downstream tasks without task-specific fine-tuning. The model family includes several size variants; this system uses the ViT-Small/14 configuration, which processes 14×14 pixel patches through 12 transformer encoder blocks with 6 attention heads each, producing a 384-dimensional [CLS] token embedding for each input image. The full backbone contains approximately 22 million parameters.

The current deployment uses a **fully fine-tuned** backbone with a 4-class linear head mapping the 384-dimensional embedding to: `healthy_vegetation`, `stressed_vegetation`, `bare_soil`, and `non_vegetation`. Unlike the earlier linear-probe approach (which froze the backbone), full fine-tuning allows the transformer attention heads to specialize for the multispectral domain, improving separation between visually similar classes (e.g., stressed vegetation vs. bare soil) at the cost of requiring a desktop GPU for training.

#### Training Procedure

Two training approaches were developed to accommodate different hardware availability:

**Approach A — Two-Phase On-Device Training (Jetson, 8 GB safe):** The 8 GB memory constraint on the Jetson makes conventional GPU-based fine-tuning impractical. The solution splits training into two phases: (1) Feature Extraction on CPU — the frozen DINOv2 backbone processes each training image (resized to 224×224) and caches the 384-dimensional CLS embedding to disk; (2) Linear Head Training on GPU — a linear classifier (384 → 4) is trained on the cached features using SGD with cosine-annealing LR. This approach produces 93%+ accuracy with under 50 MB GPU memory.

**Approach B — Full Fine-Tuning (Desktop GPU, recommended):** On an RTX 4070 with 12 GB VRAM, the entire DINOv2 backbone is unfrozen and fine-tuned alongside the classification head. Training uses AdamW optimizer with class-weighted cross-entropy loss to handle class imbalance, FP16 mixed precision for memory efficiency, and a cosine-annealing LR schedule starting at 1e-5 for the backbone and 1e-3 for the head. The trained model is exported to ONNX and converted to a TensorRT FP16 engine for deployment on the Jetson. This approach achieves 93.94% validation accuracy with significantly better class separation on challenging scenes.

#### Training Data

The training dataset for the 4-class model draws from three sources:

**Field captures** (~400 image pairs from Central Park, March 2026). These are the highest-value training examples because they come from the same camera, filter set, and environment as the deployment target. They include close-up and medium-distance shots of grass, bushes, canopies, dirt paths, stone monuments, dead vegetation, and mixed scenes under varying light across multiple sessions. Each capture was automatically labeled using NDVI thresholds into four classes: bare soil (NDVI < 0.15), stressed vegetation (0.15–0.35), healthy vegetation (> 0.35), with non-vegetation scenes identified by surface type.

**EuroSAT** (a 27,000-image satellite classification dataset). EuroSAT provides overhead imagery labeled into 10 land-use classes, four of which map directly to the Chroma-Shift taxonomy: `AnnualCrop` and `PermanentCrop` → healthy vegetation, `Forest` → healthy vegetation, `Pasture` → stressed vegetation (brown/dry pasture), `Industrial` and `Residential` → non-vegetation, `River` and `SeaLake` → non-vegetation. These satellite-perspective images add domain diversity for the orbital use case.

**PlantVillage** (a subset of the 54,306-image open dataset of plant leaf photographs). PlantVillage images add leaf-level texture diversity. Images are split by disease state: healthy leaves → healthy vegetation, diseased leaves → stressed vegetation.

The field captures are augmented through random combinations of horizontal and vertical flips, rotations (±25°), brightness adjustment (0.65–1.35×), contrast adjustment (0.75–1.25×), and saturation jitter (0.6–1.4×). The final 4-class training set contains approximately 12,000 images (~9,000 train + ~1,800 validation), balanced across all four classes using class-weighted sampling.

#### Performance and TensorRT Deployment

The 4-class classifier reaches **93.94% validation accuracy** after 30 epochs of full fine-tuning on an RTX 4070 GPU, with the best checkpoint selected by validation loss. The trained model is exported to ONNX format (224×224 input) and converted to a TensorRT FP16 engine for GPU inference on the Jetson. The TensorRT engine achieves **6.7 ms mean inference latency (149 FPS)** — a 150× speedup over CPU-based PyTorch inference (~1000 ms). The 45.3 MB engine file is stored alongside the ONNX model and loaded automatically at startup. Per-class probabilities are returned alongside the top prediction, enabling the confidence threshold system described in Section 5.3.

### 5.2 Cloud Vision-Language Scene Analyzer

Beyond per-frame classification, the system uses a cloud-hosted Vision-Language Model to perform intelligent scene analysis. The VLM analyzes side-by-side Red+NIR frame pairs alongside weather data and solar position to produce structured assessments of scene conditions that drive the radiometric correction pipeline.

#### Edge-to-Cloud Architecture

The scene analysis module (`vila_analyzer.py`) implements a four-mode priority chain:

1. **NVIDIA NIM Cloud API** (preferred): The Jetson encodes the Red+NIR composite as a base64 JPEG and sends it via HTTPS POST to NVIDIA's Inference Microservices (NIM) platform, which hosts Llama 3.2 90B Vision — a 90-billion-parameter multimodal model running on NVIDIA's data center GPUs. The request uses the OpenAI-compatible chat completions format. This offloads the computationally intensive VLM inference from the 8 GB edge device entirely.

2. **Local REST API**: For deployments with a larger GPU on the local network, the Jetson can call a VILA container running on a separate machine via HTTP REST.

3. **Direct NanoLLM**: When running inside a `dustynv/nano_llm` container on hardware with 16 GB+ RAM, the VILA model can be loaded in-process.

4. **Heuristic fallback**: A rule-based analyzer that uses image statistics — mean brightness, contrast ratio, shadow percentage, and quick NDVI computation — to approximate the same correction decisions at zero GPU cost in under 1 ms. This activates automatically when no VLM endpoint is available or when cloud connectivity is lost.

The decision to offload VLM inference to the cloud was driven by hardware constraints discovered during development: compiling VILA 1.5-3B's MLC model on the 8 GB Orin Nano requires more than 10 GB of RAM for the quantization step alone, which exceeds the device's shared memory budget and causes kernel OOM kills. Rather than compromising on model capability, the cloud approach provides access to a model 30× larger than the original VILA 3B target with no local resource cost.

The VLM receives a side-by-side composite of the Red (665 nm) and NIR (850 nm) frames, along with a natural-language context block containing current weather observations and solar elevation. A structured prompt instructs the model to return a JSON object containing:

- Scene composition estimates (vegetation %, soil %, artificial surface %, shadow %)
- Illumination quality assessment (uniform, patchy, or poor)
- Inter-frame change detection (whether lighting shifted between Red and NIR captures)
- Haze/fog detection
- Correction layer overrides (which of the 7 layers to skip or adjust)
- Cloud cover adjustment (refined from NWS value based on visual evidence)
- Confidence score (0–1)
- Natural-language explanation suitable for dashboard display

In testing with synthetic multispectral scenes, the cloud VLM correctly identified vegetation-dominant scenes at 80% coverage with 0.8 confidence, accurately assessed illumination quality and haze conditions, and returned valid structured JSON in approximately 5.6 seconds — well within the 10-minute scan cycle budget.

These outputs feed directly into the radiometric correction pipeline as `vila_overrides`, allowing the VLM to dynamically disable unnecessary corrections (e.g., skip DOS under clear conditions) or adjust parameters (e.g., increase cloud transmittance if the scene appears brighter than the NWS report suggests).

This architecture represents a key design principle: **physics-based corrections provide the rigorous foundation, while AI provides the intelligent orchestration**. The VLM does not replace the radiometric math; it decides *which math to apply* based on scene understanding. And by distributing the workload — DINOv2 classification on-device, VLM reasoning in the cloud — the system maximizes the utilization of both edge and cloud compute.

### 5.3 Confidence Threshold Cross-Checking

The DINOv2 classifier and the NDVI physics measurement provide two independent assessments of scene content. When they agree, confidence is high. When they disagree, the system flags the conflict and triggers additional verification. The confidence threshold system (`crop_advisor.py`) implements three checks:

**Low AI Confidence (<80%).** When the top-class probability falls below 80%, the classification is marked as potentially unreliable. The advisory output includes a warning and recommends either a VILA VLM second opinion or a re-scan of the zone. In the adaptive scheduler, low-confidence zones are scanned more frequently to gather additional evidence.

**NDVI-Classification Conflict.** If DINOv2 classifies a scene as `healthy_vegetation` but the NDVI reading is critically low (below 0.2), or if DINOv2 classifies as `bare_soil` but NDVI is high (above 0.5), the system flags a CONFLICT warning. This catches cases where the classifier may be confused by unusual surface textures (e.g., wet concrete registering as vegetation in the RGB domain) or where the NDVI computation is corrupted by a sensor anomaly.

**Ambiguous Classification.** When the top two class probabilities are within 15 percentage points of each other, the classification is flagged as AMBIGUOUS. This typically occurs at transition boundaries (e.g., sparse vegetation that could be classified as either stressed vegetation or bare soil) and indicates that the scene may require human review or higher-resolution imaging.

These confidence flags are included in the advisory output, the triage log, and the Supabase push, giving both automated systems and human reviewers visibility into classification uncertainty.

---

## 6. Weather Correlation

### 6.1 Data Source

Weather observations are pulled from the National Weather Service's public API, targeting station KNYC (Central Park, Manhattan). The API is free, requires no authentication, and provides JSON-formatted observations updated approximately every hour. Each observation includes temperature, relative humidity, wind speed and direction, barometric pressure, visibility, dew point, precipitation in the last one and six hours, and a plain-text weather description. A User-Agent header identifying the application is included per NWS API guidelines.

### 6.2 Stress Assessment

The weather data is run through a rule-based correlation engine that flags conditions known to affect vegetation health:

| Condition | Threshold | NDVI Modifier | Rationale |
|-----------|-----------|---------------|-----------|
| Heat stress | > 35 °C | −0.05 | High temperatures accelerate evapotranspiration and can cause stomatal closure, reducing photosynthetic activity |
| Frost damage | ≤ 0 °C | −0.08 | Ice crystal formation disrupts cell membranes in leaf tissue, causing rapid chlorophyll degradation |
| Low humidity | < 30% RH | −0.03 | Low atmospheric moisture increases the vapor pressure deficit across leaf stomata, driving moisture loss faster than root uptake can compensate |
| Recent rainfall (> 5 mm / 6 h) | > 5.0 mm | +0.03 | Adequate soil moisture supports turgor pressure and nutrient uptake |
| No recent precipitation | 0 mm / 1h and 6h | −0.02 | Extended dry periods deplete soil moisture reserves, especially in shallow-rooted grasses |

These modifiers are additive and applied to the baseline NDVI to produce a weather-adjusted threshold. If an NDVI drop exceeds 0.05 and the weather modifier accounts for a significant portion of the deficit, the system notes that weather conditions may explain the observed stress, potentially downgrading the alert severity to avoid false positives during heat waves or cold snaps.

This approach is deliberately conservative and rule-based rather than model-driven. With only a few months of co-located NDVI-weather data, there is not yet enough paired data to train a weather-NDVI regression model. The rule set encodes the most well-established relationships in crop physiology literature and will be refined as the observational record grows.

---

## 7. Cloud Data Architecture

### 7.1 Schema Design

All telemetry flows to a Supabase instance (PostgreSQL backend) organized into five tables:

**ndvi_readings** stores the core measurement: NDVI value, field zone identifier, source tag ("jetson"), band used, AI classification label, and AI confidence score. Each row represents one scan cycle at one field zone.

**alerts** captures stress events that exceed the severity thresholds described above. Each alert includes the triggering NDVI value, drop magnitude, severity level, stress type, recommended action, and the weather conditions at the time of detection. It also records the AI classification to distinguish vegetation-related stress from expected bare-ground readings.

**field_zones** maintains the current status and running statistics for each monitored location—last NDVI, average NDVI, update count, and stress history.

**weather_data** logs each weather observation fetched during a scan cycle, paired with the concurrent NDVI reading and field zone. This table accumulates the paired dataset needed for future statistical analysis.

**system_status** receives periodic heartbeat messages from the Jetson, reporting CPU temperature, uptime, storage utilization, and application version. This supports remote monitoring of the hardware's health and flags conditions (thermal throttling, disk near-full) that could degrade data quality.

### 7.2 Data Flow

Each ten-minute scan cycle generates between five and twelve database writes depending on the number of active field zones and whether stress alerts are triggered. The write sequence is: weather observation → per-zone NDVI reading (with AI fields) → per-zone alert (if stressed) → zone status update → system heartbeat. All writes use the Supabase Python client's `table().insert()` interface with the service-role key, which bypasses row-level security for server-to-server ingestion.

---

## 8. Adaptive Sensing

Traditional satellite monitoring uses fixed revisit schedules regardless of ground conditions. Chroma-Shift implements an eight-signal adaptive scheduling algorithm (`adaptive_sensing.py`) that dynamically adjusts observation frequency per zone based on real-time conditions:

| Signal | Weight | Effect |
|--------|--------|--------|
| NDVI stress | 0.30 | Stressed zones scanned more frequently |
| Scene change | 0.15 | Rapid NDVI changes trigger shorter intervals |
| Weather urgency | 0.15 | Frost/heat events increase scan frequency |
| VLM confidence | 0.10 | Low VLM confidence → more frequent scans |
| Solar elevation | 0.10 | Nighttime → extended intervals (sleep mode) |
| Power state | 0.10 | Low battery → longer intervals |
| Zone history | 0.05 | Zones with recent stress get priority |
| DINOv2 classification | 0.05 | Stressed vegetation → 2.5× frequency; bare soil → deprioritized |

The scheduler produces three outputs: an interval in seconds (clamped to 120–1800s), a priority score (0–1), and a power mode recommendation (`active` at 10W, `normal` at 7W, or `sleep` at 3.5W). This reduces downlink bandwidth by 99.7% compared to raw image transmission — sending 1.3 KB of processed telemetry instead of 500 KB images.

---

## 9. System Integration

The complete pipeline is orchestrated by a single Python script (`jetson_main.py`) running on the Jetson:

1. **Boot Sequence:** Run 8-component self-test (GPU, TRT engine, disk, Supabase, weather API, radiometric pipeline, captures directory, Python imports). If any component fails, the system starts in a degraded mode rather than crashing. Load environment variables, connect to Supabase, initialize the filter wheel via ctypes, open the camera via OpenCV, load the 4-class DINOv2 TensorRT engine (with graceful fallback to PyTorch CPU then NDVI-only mode), initialize the VILA scene analyzer, set up per-zone Kalman filters, arm the watchdog timer (120s timeout), and initialize the adaptive scheduler.

2. **Main loop (adaptive interval, 2–30 minutes):**
   - Check remote config for ground control overrides
   - Fetch current weather from NWS
   - Rank zones by adaptive priority
   - For each configured field zone:
     - Rotate filter wheel to Red → capture with metadata → rotate to NIR → capture with metadata
     - Check frames for anomalies (black, saturated, low contrast)
     - Compute solar position from timestamp + GPS
     - Run scene analysis (cloud VLM or heuristic) on the Red+NIR pair with weather context
     - Apply 7-layer radiometric correction with VLM-generated overrides
     - Compute corrected NDVI, check for anomalies, smooth through Kalman filter
     - Compare against seasonal phenology baseline
     - Run Red frame through 4-class DINOv2 TensorRT classifier (on-device, 6.7 ms / 149 FPS)
     - Run confidence threshold cross-checks (AI vs NDVI physics)
     - Generate crop advisory with confidence flags
     - Compute adaptive scheduling decision (8-signal fusion)
     - Push NDVI reading, classification, confidence flags, alerts, advisory, and zone status to Supabase
     - Record success / kick watchdog
   - Push weather observation with correlated NDVI
   - Push system heartbeat with operating mode and sensor health score

3. **Fault-tolerant shutdown:** Signal handlers catch SIGINT and SIGTERM, closing the camera handle, resetting the filter wheel, pushing a final heartbeat (mode=idle), and allowing in-flight database writes to complete before exit. If the main loop hangs, the watchdog auto-restarts the process.

The system demonstrates an edge-to-cloud AI architecture: **DINOv2 4-class classification runs locally on the Jetson's GPU** (6.7 ms latency, 149 FPS, zero network dependency), while **VLM scene reasoning runs on NVIDIA NIM cloud GPUs** (~5.6 s, using a 90B-parameter model that would be impossible to run on 8 GB of RAM). If internet connectivity is lost, the heuristic fallback activates automatically with no interruption to the monitoring cycle. The full end-to-end pipeline completes in under 4 seconds per zone.

The loop interval is determined by the adaptive scheduler (default base: 600 seconds, range: 120–1800 seconds), and individual zones can be targeted for debugging without modifying the codebase.

---

## 10. Farmer Advisory System

Beyond classification, the system generates **context-aware advisories** that explain *why* vegetation is healthy or stressed and *what to do about it*. The `crop_advisor.py` module fuses six data streams:

1. **NDVI measurement** — interpreted against 7 vegetation categories (bare soil through very dense canopy)
2. **Seasonal baseline** — 5-year NASA HLS average for the current week, with season-specific context
3. **Weather data** — NWS temperature, humidity, precipitation, and wind used to diagnose specific stress causes (water stress, heat stress, frost damage, waterlogging, wind damage)
4. **AI classification** — 4-class DINOv2 prediction with per-class probabilities
5. **Confidence flags** — Cross-check results from the confidence threshold system (low confidence, NDVI conflicts, ambiguous classification)
6. **Historical trend** — declining, stable, or improving NDVI trajectory

Each advisory includes a health label, plain-English explanation, possible causes with confidence scores, confidence flags, and a specific actionable recommendation. This transforms the system from a classifier into a **decision support tool** for farmers.

---

## 11. Autonomous Triage Logging

To demonstrate autonomous operation without a human in the loop, the system records every decision in a structured triage log (`triage_log.py`). Each entry captures the complete 8-step reasoning chain:

1. **SENSE** — Raw Red/NIR sensor readings (DN values, exposure, gain)
2. **CORRECT** — All 7 radiometric correction layers applied with parameters
3. **MEASURE** — Computed NDVI value with physical validity check
4. **COMPARE** — Deviation from NASA seasonal baseline
5. **CLASSIFY** — 4-class DINOv2 prediction, per-class probabilities, and confidence flags
6. **DIAGNOSE** — Weather-fused stress analysis with possible causes
7. **ADVISE** — Farmer-facing recommendation generated autonomously
8. **SCHEDULE** — Adaptive sensing interval decision (8-signal) and reasoning

Each entry also includes a human-readable reasoning chain that summarizes the full decision in natural language, enabling NASA reviewers to trace exactly how the AI arrived at each conclusion. The log is append-only, JSON-lines format, and explicitly records `"human_intervention": false` on every entry.

---

## 12. Fault Tolerance

A system operating autonomously in orbit—or in a remote agricultural deployment—cannot rely on human intervention when failures occur. The fault tolerance module (`fault_tolerance.py`) ensures the pipeline never fully stops, implementing five subsystems:

### 12.1 Watchdog Timer

A software watchdog runs on a daemon thread and monitors the main scan loop. If the loop fails to call `kick()` within 120 seconds, the watchdog triggers automatic process recovery. This handles GPU hangs, USB device lockups, TensorRT deadlocks, and any other condition that causes the main thread to stall indefinitely.

### 12.2 Graceful Mode Degradation

The FaultManager tracks consecutive errors and transitions the system through four operating modes:

| Mode | Trigger | Active Systems | Power Draw |
|------|---------|---------------|------------|
| **Full** | Boot (all tests pass) | TRT + VILA + Adaptive + Weather | 10W |
| **Reduced** | 3 consecutive errors | TRT + Adaptive (skip VLM) | 7W |
| **Safe** | 5 consecutive errors | NDVI-only (no AI, no cloud) | 3.5W |
| **Emergency** | 10 consecutive errors | Log-only (minimal power) | 2W |

After 5 minutes of clean operation (no errors), the system automatically upgrades one level: Emergency → Safe → Reduced → Full. This provides self-healing behavior without ground intervention.

### 12.3 System Self-Test

On every boot, an 8-component self-test runs before entering the scan loop:

1. Python imports (cv2, numpy, scipy)
2. Disk space (warn < 5 GB, fail < 1 GB)
3. GPU availability (nvidia-smi check)
4. TensorRT engine file (existence + size)
5. Radiometric pipeline (synthetic NDVI smoke test)
6. Supabase connectivity (test query)
7. Weather API (NWS fetch)
8. Captures directory (write test)

Failed tests degrade the starting mode rather than blocking the scan loop entirely.

### 12.4 Anomaly Detection

The AnomalyDetector monitors every NDVI reading and captured frame for sensor anomalies:

- **NDVI out of range** (|NDVI| > 1.0) — physically impossible, indicates sensor or computation failure
- **Stuck sensor** — same NDVI value 10+ consecutive readings, indicates camera freeze or filter wheel stuck
- **Sensor drift** — mean NDVI shifting > 0.3 over 20 readings, indicates gradual degradation
- **Black frame** — all-zero pixels, camera dead or lens cap obstruction
- **Saturated frame** — all pixels near 255, overexposure or sun glint
- **Low contrast** — standard deviation < 3.0, fog or condensation on lens

Each anomaly generates an alert pushed to Supabase and a health score (0–100) reported in the system heartbeat.

### 12.5 Remote Configuration

Ground control can push parameter overrides to a `remote_config` Supabase table without requiring an OTA firmware update:

- `force_safe_mode` → Immediately enter safe mode
- `scan_interval` → Override the adaptive scheduler's interval
- `enable_vila` → Enable or disable VLM calls

The system polls this table once per scan cycle (configurable). This provides a low-bandwidth command uplink mechanism suitable for S-band passes.

### 12.6 Error Telemetry

Every error, mode change, and anomaly is pushed to a `system_errors` Supabase table with component name, error message, traceback, consecutive count, and operating mode. Ground control can monitor system health in real time through the dashboard and diagnose issues remotely by reviewing the error history.

---

## 13. Field Validation

### 13.1 Deployment — March 21, 2026

The first sustained field deployment was conducted on March 21, 2026, at Central Park, NYC (Zone C1 — West Terrace). The Jetson Orin Nano was carried into the field with the camera and filter wheel, and the pipeline ran continuously for over 25 minutes with 2-minute scan intervals.

**Conditions**: 13.9°C, 37–38% RH, Clear sky, solar elevation ~25–30° (late afternoon).

| Metric | Value |
|--------|-------|
| Continuous cycles completed | 12+ (zero failures, zero watchdog triggers) |
| Capture pairs saved | 17 Red+NIR pairs (34 images + 17 stress maps) |
| NDVI range (Kalman-filtered) | 0.28–0.35 |
| NASA HLS spring baseline | 0.41 |
| NDVI seasonal deviation | -0.06 to -0.13 |
| AI classification | healthy_vegetation (48–53%) |
| VLM scene descriptions | "winter scene with bare trees and grass", "barren, sandy field", "field with vegetation and soil" |
| Interframe mismatch | 10–23% (camera movement between Red→NIR) |
| System health | 100% across all runs |
| Adaptive priority | 0.68–0.98 (high, stress detected) |

The NDVI readings of 0.28–0.35 fall below the NASA HLS spring baseline of 0.41, consistent with early-March dormant-season vegetation in the northeastern United States. The Kalman temporal filter reduced cycle-to-cycle NDVI jitter from ±0.05 (raw) to ±0.02 (filtered). The adaptive scheduler correctly responded to below-baseline NDVI by maintaining 120-second (maximum frequency) scan intervals throughout the session.

The NVIDIA NIM cloud VLM (Llama 3.2 90B Vision) correctly identified the outdoor scene across multiple cycles, describing "bare trees," "grass," "soil," and "vegetation" — context that the NDVI number alone cannot provide. When the cloud VLM timed out on one cycle, the system gracefully fell back to heuristic mode (60% confidence) and resumed cloud VLM on the next cycle.

### 13.2 Hardware Integration Fix

During pre-deployment testing, a critical EFW SDK bug was discovered: the ZWO `libEFWFilter.so` library for aarch64 requires `EFWGetProperty()` to be called after `EFWOpen()` and before `EFWSetPosition()`. Without this initialization step, `EFWSetPosition` permanently returns `EFW_ERROR_MOVING` (rc=7). This was traced through extensive debugging including USB-level resets, HIDAPI protocol reverse-engineering, and firmware state analysis. The fix — a single `EFWGetProperty()` call — resolved the issue completely and has been stable across all subsequent runs.

---

## 14. Limitations and Next Steps


The system's principal limitation in its current form is the diversity of its training dataset. With approximately 400 original field captures, the 4-class model achieves 93.94% validation accuracy but occasionally misclassifies ambiguous scenes (e.g., wet soil as vegetation, or dry grass as bare soil). The confidence threshold system mitigates this by flagging uncertain classifications, but additional field data—particularly for the bare_soil and non_vegetation classes—would improve separation. Each new field visit can be incorporated through the full fine-tuning pipeline in under 30 minutes on a desktop GPU.

The 8 GB RAM constraint of the Jetson Orin Nano proved to be the binding hardware limitation. While the DINOv2 classifier and 7-layer correction pipeline run comfortably on-device, the VILA 3B VLM cannot be compiled locally — the MLC quantization step alone requires more than 10 GB of RAM. The cloud VLM architecture resolves this cleanly, but introduces a dependency on internet connectivity and NVIDIA NIM API availability. The heuristic fallback and fault tolerance system mitigate this risk: if the cloud is unreachable, the system degrades to Reduced mode and continues operating with on-device AI only.

The radiometric pipeline's 109-test validation suite uncovered and resolved several non-obvious bugs during development, including NDVI floating-point overflow from chained normalization layers, PIF detection that incorrectly selected vegetation as invariant features in uniform scenes, and a division-by-zero warning in shadow pixel processing. These discoveries underscore the importance of extensive edge-case testing for any algorithmic pipeline operating on real-world sensor data.

Planned improvements include:

- **Expanded field dataset** — targeting 1,000+ field captures across all four classes, with emphasis on baresoil/non-vegetation diversity and seasonal variation.
- **Red-Edge and SWIR integration** — the filter wheel already carries Red-Edge (700–740 nm) and SWIR (1570–1650 nm) filters that could provide additional spectral indices (NDRE, SWIR-based moisture) for more specific stress type classification.
- **On-device VLM via cross-compilation** — compiling the MLC model on a machine with 16 GB+ RAM and transferring the pre-compiled binary to the Jetson, eliminating the cloud dependency while keeping the lightweight 3B model.
- **Weather-NDVI regression model** trained on the accumulating paired database to replace the rule-based stress modifiers with learned relationships once sufficient data has been collected across seasons.
- **Radiation-tolerant deployment** — porting the pipeline to radiation-hardened ARM/FPGA platforms (Xilinx Kria, Microchip PolarFire) for actual flight missions, with DINOv2 quantized for FPGA inference via Vitis AI.

---

## References

Rouse, J. W., Haas, R. H., Schell, J. A., & Deering, D. W. (1974). Monitoring vegetation systems in the Great Plains with ERTS. *Proceedings of the Third Earth Resources Technology Satellite-1 Symposium*, 1, 309–317.

Chavez, P. S. (1988). An improved dark-object subtraction technique for atmospheric scattering correction of multispectral data. *Remote Sensing of Environment*, 24(3), 459–479.

Spencer, J. W. (1971). Fourier series representation of the position of the Sun. *Search*, 2(5), 172.

Reda, I., & Andreas, A. (2004). Solar position algorithm for solar radiation applications. *Solar Energy*, 76(5), 577–589. doi:10.1016/j.solener.2003.12.003

Kasten, F., & Czeplak, G. (1980). Solar and terrestrial radiation dependent on the amount and type of cloud. *Solar Energy*, 24(2), 177–189.

Oquab, M., Darcet, T., Moutakanni, T., et al. (2024). DINOv2: Learning robust visual features without supervision. *Transactions on Machine Learning Research*.

Claverie, M., Ju, J., Masek, J. G., et al. (2018). The Harmonized Landsat and Sentinel-2 surface reflectance data set. *Remote Sensing of Environment*, 219, 145–161.

Hughes, D. P., & Salathe, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv preprint arXiv:1511.08060*.

Lin, J., Yin, H., Ping, W., et al. (2024). VILA: On pre-training for visual language models. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*.

Dubey, A., et al. (2024). The Llama 3 herd of models. *arXiv preprint arXiv:2407.21783*.

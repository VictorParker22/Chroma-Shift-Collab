# Chroma-Shift: Technical Report

## Edge-Deployed Multispectral Vegetation Monitoring with DINOv2 Classification and Weather-Correlated Stress Detection

*Victor Parker — March 2026*

---

## 1. Introduction

Urban green spaces face a growing gap between the ecological services they provide and the tools available to monitor their health at a granular level. Traditional approaches to vegetation monitoring—satellite revisit schedules, manual field surveys, or periodic aerial flyovers—each carry tradeoffs in spatial resolution, temporal frequency, or cost. This project began with a practical question: could a single embedded device, positioned on the ground, continuously assess the condition of vegetation in a park using the same spectral principles that drive large-scale remote sensing?

Chroma-Shift is the system that emerged from that question. It runs on an NVIDIA Jetson Orin Nano mounted alongside a multispectral camera and motorized filter wheel. Every ten minutes, it captures a pair of narrowband images—one in the visible red and one in the near-infrared—computes a vegetation index from the reflectance difference, runs the frame through a neural network trained to distinguish healthy canopy from stressed or bare ground, pulls the latest weather data for the site, and pushes everything to a cloud database. The goal is not to replace satellite monitoring but to sit between satellite passes and ground-truth inspections, filling the temporal gaps with continuous, automated readings.

This document walks through the full system, from the physics of plant reflectance and the hardware choices that exploit it, through the signal processing and machine learning pipeline, to the weather correlation logic and cloud data architecture that tie measurements to environmental context.

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

Chroma-Shift addresses this with a seven-layer radiometric correction pipeline (`radiometric.py`) that applies additive, independent corrections. Each layer targets a distinct physical effect, and later layers catch artifacts that earlier layers miss. The pipeline accepts optional overrides from a Vision-Language Model (see Section 5.2) that can selectively disable or adjust layers based on scene-level understanding.

**Layer 1 — Auto-Exposure Normalization.** The camera's exposure time and gain are read for each frame via OpenCV properties. Raw pixel values are divided by (exposure × gain) to convert DN to relative radiance. This removes the camera's automatic compensation for changing light levels, ensuring that two frames captured seconds apart under different auto-exposure settings produce comparable radiance values.

**Layer 2 — Dark Object Subtraction (Chavez 1988).** The 1st-percentile pixel value in each band is computed and subtracted from all pixels. This follows the method described by Chavez (1988): the darkest object in the scene (deep shadow, dark soil) should have near-zero surface reflectance, so any measured signal from those pixels is attributable to atmospheric path radiance and sensor offset. Even at ground level, scattered skylight adds a measurable baseline that this step removes.

**Layer 3 — Solar Zenith Angle Correction.** The sun's position is computed from the capture timestamp and site coordinates (40.7812° N, 73.9665° W) using the Spencer (1971) solar declination and equation-of-time formulae. Radiance values are divided by cos(θ), where θ is the solar zenith angle, normalizing all readings to equivalent overhead illumination. Captures with θ > 80° (dawn/dusk) are flagged as low-confidence and excluded from trend analysis.

**Layer 4 — Cloud Transmittance Modeling.** Cloud cover percentage from the National Weather Service API is converted to a ground-level transmittance estimate using the Kasten & Czeplak (1980) empirical model: T = 1 − 0.75 × (C/100)². Clear sky (0% cloud) gives T = 1.0; full overcast (100%) gives T = 0.25. Dividing by T normalizes radiance to clear-sky equivalent, enabling meaningful comparison of readings taken under partly cloudy versus clear conditions.

**Layer 5 — Inter-Frame Illumination Matching.** Because Red and NIR frames are captured 3–5 seconds apart through the filter wheel, a passing cloud can illuminate the two frames differently, corrupting the NDVI ratio. This layer compares mean brightness across pseudo-invariant features (PIFs) — surfaces whose reflectance does not change seasonally, such as concrete paths, stone monuments, and asphalt. If brightness differs by more than 10%, the dimmer frame is scaled to match. This is the most critical correction for sequential-capture hardware.

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

NDVI alone cannot distinguish between all causes of low reflectance. A muddy path and a drought-stressed lawn may show similar NDVI values, but they require entirely different management responses. To address this, an image classifier was added downstream of the NDVI computation, applied to the Red-band frame (which, as an RGB-approximation, preserves the spatial texture information that NDVI collapses into a single number).

The classifier is built on DINOv2, a self-supervised vision foundation model developed by Meta AI. DINOv2 was trained on 142 million curated images using a combination of image-level and patch-level self-distillation objectives, producing feature representations that transfer well to downstream tasks without task-specific fine-tuning. The model family includes several size variants; this system uses the ViT-Small/14 configuration, which processes 14×14 pixel patches through 12 transformer encoder blocks with 6 attention heads each, producing a 384-dimensional [CLS] token embedding for each input image. The full backbone contains approximately 22 million parameters.

In this deployment, the backbone weights are frozen entirely. Only a single linear layer maps the 384-dimensional embedding to 2 output classes (healthy vegetation vs. stressed/bare ground), yielding 770 trainable parameters—384 weights per class plus 2 biases. This extreme parameter economy reflects the quality of the foundation model's features: a linear probe on DINOv2 embeddings routinely matches or exceeds models that fine-tune millions of parameters on similar classification tasks.

#### Training Procedure

The 8 GB memory constraint on the Jetson makes conventional GPU-based fine-tuning of even a small ViT impractical. Loading the backbone, intermediate activations, and optimizer state simultaneously exceeds the unified memory budget. The solution was to split training into two phases:

**Phase 1 — Feature Extraction (CPU, one-time):** The frozen DINOv2 backbone is loaded on the CPU. Each training image is resized to 168×168, normalized to ImageNet statistics, and passed through the backbone. The resulting 384-dimensional [CLS] embedding is stored as a NumPy array on disk. The entire training set is processed in this way, producing a compact feature cache (a few megabytes for thousands of images) that can be loaded entirely into memory for the second phase. This step runs at roughly one image per second on the Jetson's A78AE cores, taking about 50 minutes for 3,000 images.

**Phase 2 — Linear Head Training (GPU, fast):** The cached feature vectors are loaded into GPU memory as a single batch. A linear classifier (384 → 2) is trained using SGD with momentum (0.9) and a cosine-annealing learning rate schedule starting at 0.001. Training runs for 30 epochs. Because the feature cache fits entirely in memory and the classifier has only 770 parameters, each epoch completes in roughly 0.2 seconds. The entire training phase finishes in under 10 seconds.

This two-phase approach reduces the GPU memory requirement from several gigabytes (full model) to under 50 MB (feature vectors + tiny linear layer), making it practical on 8 GB hardware.

#### Training Data

The training dataset draws from two sources with different strengths:

**Field captures** (171 image pairs from Central Park, March 2026). These are the highest-value training examples because they come from the same camera, filter set, and environment as the deployment target. They include close-up and medium-distance shots of grass, bushes, canopies, dirt paths, stone monuments, dead vegetation, and mixed scenes under varying light. Each capture was automatically labeled using the NDVI seasonal threshold. The original 171 pairs provide about 342 usable images across both classes.

**PlantVillage** (a subset of the 54,306-image open dataset of plant leaf photographs). PlantVillage images are close-up, laboratory-style photographs of individual leaves under controlled conditions, covering 38 crop species and associated diseases. While the domain is very different from landscape-scale field captures, these images add diversity in leaf-level texture. Only 200 images per class are sampled from this dataset to prevent it from dominating the training distribution.

The field captures are augmented 15× through random combinations of horizontal and vertical flips, rotations (±25°), brightness adjustment (0.65–1.35×), contrast adjustment (0.75–1.25×), and saturation jitter (0.6–1.4×). Combined with PlantVillage samples, the final training set contains approximately 15,153 images (14,496 train + 657 validation).

#### Performance and TensorRT Deployment

The classifier reaches **84.17% validation accuracy** after 30 epochs, with the best checkpoint recorded at epoch 19. The trained model is exported to ONNX format and converted to a TensorRT FP16 engine for GPU inference on the Jetson. The TensorRT engine achieves **8.7 ms mean inference latency** (6.3 ms minimum, 12.5 ms maximum) — a 115× speedup over CPU-based PyTorch inference (~1000 ms). The 84.2 MB engine file is stored alongside the ONNX model and loaded automatically at startup.

### 5.2 VILA Vision-Language Scene Analyzer

Beyond per-frame classification, the system deploys NVIDIA VILA 1.5-3B — a 3-billion-parameter Vision-Language Model — to perform intelligent scene analysis. Running quantized to 4-bit (AWQ) on the Jetson's GPU, VILA examines side-by-side Red+NIR frame pairs alongside weather data and solar position to produce structured assessments of scene conditions.

The VLM receives a side-by-side composite of the Red (665 nm) and NIR (850 nm) frames annotated with band labels, along with a natural-language context block containing current weather observations, solar elevation, and capture timestamp. A structured system prompt instructs the model to return a JSON object containing:

- Scene composition estimates (vegetation %, soil %, artificial surface %, shadow %)
- Illumination quality assessment (uniform, patchy, or poor)
- Inter-frame change detection (whether lighting shifted between Red and NIR captures)
- Haze/fog detection
- Correction layer overrides (which of the 7 layers to skip or adjust)
- Cloud cover adjustment (refined from the NWS value based on visual evidence)
- Confidence score (0–1)
- Natural-language explanation suitable for dashboard display

These outputs feed directly into the radiometric correction pipeline as `vila_overrides`, allowing the VLM to dynamically disable unnecessary corrections (e.g., skip DOS under clear conditions) or adjust parameters (e.g., increase cloud transmittance if the scene appears brighter than the NWS report suggests).

When the VILA model is not available (outside the Docker container, or during development), the system falls back to a rule-based heuristic analyzer that uses image statistics — mean brightness, contrast ratio, shadow percentage, quick NDVI computation — to approximate the same correction decisions at zero GPU cost.

This architecture represents a key design principle: **physics-based corrections provide the rigorous foundation, while AI provides the intelligent orchestration**. The VLM does not replace the radiometric math; it decides *which math to apply* based on scene understanding.

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

## 8. System Integration

The complete pipeline is orchestrated by a single Python script (`jetson_main.py`) that runs inside a Docker container on the Jetson:

1. **Initialization:** Load environment variables, connect to Supabase, initialize the filter wheel via ctypes, open the camera via OpenCV, load the DINOv2 TensorRT engine (with graceful fallback to PyTorch CPU then NDVI-only mode), initialize the VILA scene analyzer (with heuristic fallback), and set up per-zone Kalman filters.

2. **Main loop (repeats every N seconds):**
   - Fetch current weather from NWS
   - For each configured field zone:
     - Rotate filter wheel to Red → capture with metadata → rotate to NIR → capture with metadata
     - Compute solar position from timestamp + GPS
     - Run VILA scene analysis on the Red+NIR pair with weather context
     - Apply 7-layer radiometric correction with VLM-generated overrides
     - Compute corrected NDVI and smooth through Kalman filter
     - Compare against seasonal phenology baseline
     - Run Red frame through DINOv2 TensorRT classifier
     - Push NDVI reading, VILA explanation, alerts, and zone status to Supabase
   - Push weather observation with correlated NDVI
   - Push system heartbeat

3. **Graceful shutdown:** Signal handlers catch SIGINT and SIGTERM, closing the camera handle, resetting the filter wheel to the clear position, and allowing in-flight database writes to complete before exit.

The loop interval is configurable from the command line (default: 600 seconds), and individual zones can be targeted for debugging without modifying the codebase.

---

## 9. Limitations and Next Steps

The system's principal limitation in its current form is the size of its training dataset. With 171 original field captures, even aggressive augmentation cannot fully capture the visual diversity of an urban park across seasons, weather conditions, and times of day. Model accuracy on mixed-texture scenes (partially shaded paths, sparse ground cover, seasonal transitions) remains below the threshold needed for fully autonomous operation. Each new field visit roughly doubles the useful training data, and the two-phase training pipeline reduces the retraining cycle to under an hour on-device.

Planned improvements include:

- **Multi-class expansion** beyond the binary healthy/stressed split to distinguish between specific stress types (drought, disease, nutrient deficiency) using the Red-Edge and SWIR filter channels that are already installed in the filter wheel but not yet incorporated into the classification pipeline.
- **VILA fine-tuning** on domain-specific multispectral imagery to improve the VLM's correction parameter estimates beyond the general-purpose baseline.
- **Weather-NDVI regression model** trained on the accumulating paired database to replace the rule-based stress modifiers with learned relationships once sufficient data has been collected across seasons.

---

## References

Rouse, J. W., Haas, R. H., Schell, J. A., & Deering, D. W. (1974). Monitoring vegetation systems in the Great Plains with ERTS. *Proceedings of the Third Earth Resources Technology Satellite-1 Symposium*, 1, 309–317.

Chavez, P. S. (1988). An improved dark-object subtraction technique for atmospheric scattering correction of multispectral data. *Remote Sensing of Environment*, 24(3), 459–479.

Spencer, J. W. (1971). Fourier series representation of the position of the Sun. *Search*, 2(5), 172.

Kasten, F., & Czeplak, G. (1980). Solar and terrestrial radiation dependent on the amount and type of cloud. *Solar Energy*, 24(2), 177–189.

Oquab, M., Darcet, T., Moutakanni, T., et al. (2024). DINOv2: Learning robust visual features without supervision. *Transactions on Machine Learning Research*.

Claverie, M., Ju, J., Masek, J. G., et al. (2018). The Harmonized Landsat and Sentinel-2 surface reflectance data set. *Remote Sensing of Environment*, 219, 145–161.

Hughes, D. P., & Salathe, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv preprint arXiv:1511.08060*.

Lin, J., Yin, H., Ping, W., et al. (2024). VILA: On pre-training for visual language models. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*.

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

### 4.1 Per-Pixel NDVI Computation

The NDVI map is computed directly from the two raw frames without atmospheric correction, since the camera and scene are at ground level (path length <100 m) and the narrowband filters already isolate wavelength regions that are minimally affected by Rayleigh scattering or ozone absorption at such short distances. The computation is a straightforward per-pixel operation on the full sensor resolution:

```python
ndvi = (nir.astype(float) - red.astype(float)) / (nir.astype(float) + red.astype(float) + 1e-10)
```

The small epsilon in the denominator prevents division-by-zero in fully shadowed pixels where both channels read near-zero. The resulting array is clipped to [−1, +1] and averaged to produce a single scalar NDVI reading for the scene.

### 4.2 Stress Classification

Each NDVI reading is compared against the field zone's baseline value, which has been calibrated from a combination of the satellite-derived ground truth and site-specific observations. The severity of any deficit is binned into discrete levels:

| NDVI Drop (below baseline) | Severity | Status |
|---------------------------|----------|--------|
| ≥ 0.15 | Critical | Stress alert |
| 0.10 – 0.15 | High | Stress alert |
| 0.05 – 0.10 | Medium | Watch |
| < 0.05 | Low | Healthy |

When an alert is triggered, the system generates a recommendation tailored to the stress type (e.g., irrigation for water stress, inspection for disease) and pushes it to the cloud database with the associated NDVI measurement, weather conditions, and AI classification result.

---

## 5. DINOv2 Classification

### 5.1 Architecture

NDVI alone cannot distinguish between all causes of low reflectance. A muddy path and a drought-stressed lawn may show similar NDVI values, but they require entirely different management responses. To address this, an image classifier was added downstream of the NDVI computation, applied to the Red-band frame (which, as an RGB-approximation, preserves the spatial texture information that NDVI collapses into a single number).

The classifier is built on DINOv2, a self-supervised vision foundation model developed by Meta AI. DINOv2 was trained on 142 million curated images using a combination of image-level and patch-level self-distillation objectives, producing feature representations that transfer well to downstream tasks without task-specific fine-tuning. The model family includes several size variants; this system uses the ViT-Small/14 configuration, which processes 14×14 pixel patches through 12 transformer encoder blocks with 6 attention heads each, producing a 384-dimensional [CLS] token embedding for each input image. The full backbone contains approximately 22 million parameters.

In this deployment, the backbone weights are frozen entirely. Only a single linear layer maps the 384-dimensional embedding to 2 output classes (healthy vegetation vs. stressed/bare ground), yielding 770 trainable parameters—384 weights per class plus 2 biases. This extreme parameter economy reflects the quality of the foundation model's features: a linear probe on DINOv2 embeddings routinely matches or exceeds models that fine-tune millions of parameters on similar classification tasks.

### 5.2 Training Procedure

The 8 GB memory constraint on the Jetson makes conventional GPU-based fine-tuning of even a small ViT impractical. Loading the backbone, intermediate activations, and optimizer state simultaneously exceeds the unified memory budget. The solution was to split training into two phases:

**Phase 1 — Feature Extraction (CPU, one-time):** The frozen DINOv2 backbone is loaded on the CPU. Each training image is resized to 168×168, normalized to ImageNet statistics, and passed through the backbone. The resulting 384-dimensional [CLS] embedding is stored as a NumPy array on disk. The entire training set is processed in this way, producing a compact feature cache (a few megabytes for thousands of images) that can be loaded entirely into memory for the second phase. This step runs at roughly one image per second on the Jetson's A78AE cores, taking about 50 minutes for 3,000 images.

**Phase 2 — Linear Head Training (GPU, fast):** The cached feature vectors are loaded into GPU memory as a single batch. A linear classifier (384 → 2) is trained using SGD with momentum (0.9) and a cosine-annealing learning rate schedule starting at 0.001. Training runs for 30 epochs. Because the feature cache fits entirely in memory and the classifier has only 770 parameters, each epoch completes in roughly 0.2 seconds. The entire training phase finishes in under 10 seconds.

This two-phase approach reduces the GPU memory requirement from several gigabytes (full model) to under 50 MB (feature vectors + tiny linear layer), making it practical on 8 GB hardware.

### 5.3 Training Data

The training dataset draws from two sources with different strengths:

**Field captures** (66 image pairs from Central Park, March 2026). These are the highest-value training examples because they come from the same camera, filter set, and environment as the deployment target. They include close-up and medium-distance shots of grass, bushes, canopies, dirt paths, stone monuments, dead vegetation, and mixed scenes under varying light. Each capture was manually labeled based on the scene content and confirmed against its NDVI value using the satellite-derived threshold. The original 66 pairs (both Red and NIR frames) provide about 198 usable images across both classes.

**PlantVillage** (a subset of the 54,306-image open dataset of plant leaf photographs). PlantVillage images are close-up, laboratory-style photographs of individual leaves under controlled conditions, covering 38 crop species and associated diseases. While the domain is very different from landscape-scale field captures, these images add diversity in leaf-level texture: the model encounters a wide variety of healthy chlorophyll patterns and disease-related discolorations. Only 200 images per class are sampled from this dataset to prevent it from dominating the training distribution.

The field captures are augmented 15× through random combinations of horizontal and vertical flips, rotations (±25°), brightness adjustment (0.65–1.35×), contrast adjustment (0.75–1.25×), and saturation jitter (0.6–1.4×). The augmented images are resized to 224×224 before saving. This produces roughly 2,970 augmented field images. Combined with the 400 PlantVillage samples, the final training set contains approximately 3,570 images, with field-origin data constituting roughly 80% of the total.

An 85/15 train/validation split is applied after augmentation and merging, yielding approximately 3,030 training images and 535 validation images.

### 5.4 Performance

The classifier reaches **85.79% validation accuracy** after 30 epochs, with the best checkpoint recorded at epoch 23. On the field captures used for deployment:

- Stone and concrete surfaces (e.g., the USS Maine Monument) are correctly classified as stressed/bare ground with 84–98% confidence
- Grass at various distances (close, medium, far) is correctly identified as healthy vegetation with 73–95% confidence
- Dirt and compacted soil paths register as stressed with 68–88% confidence
- Dead grass and dried vegetation are classified as stressed with 81–95% confidence

Remaining errors occur primarily at scene boundaries (mixed vegetation and bare ground in the same frame), in deep shade (where both classes look dark), and on unusual textures like wood mulch. These edge cases will improve as more field data is collected and fed back into the training loop.

### 5.5 Export and Deployment

The trained model (frozen backbone + trained linear head) is exported to the ONNX interchange format for potential TensorRT conversion but currently runs inference through PyTorch directly within the Docker container. Model loading and inference are performed on the CPU to avoid GPU memory pressure, which adds roughly one second of latency per classification—acceptable for a ten-minute scan cycle.

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

1. **Initialization:** Load environment variables, connect to Supabase, initialize the filter wheel via ctypes, open the camera via OpenCV, attempt to load the DINOv2 classifier (with graceful fallback to NDVI-only mode if the model file is not available).

2. **Main loop (repeats every N seconds):**
   - Fetch current weather from NWS
   - For each configured field zone:
     - Rotate filter wheel to Red → capture → rotate to NIR → capture
     - Compute per-pixel NDVI, extract scalar mean
     - Run Red frame through DINOv2 classifier (if loaded)
     - Compare NDVI against weather-adjusted baseline
     - Push NDVI reading, alerts, and zone status to Supabase
   - Push weather observation with correlated NDVI
   - Push system heartbeat

3. **Graceful shutdown:** Signal handlers catch SIGINT and SIGTERM, closing the camera handle, resetting the filter wheel to the clear position, and allowing in-flight database writes to complete before exit.

The loop interval is configurable from the command line (default: 600 seconds), and individual zones can be targeted for debugging without modifying the codebase.

---

## 9. Limitations and Next Steps

The system's principal limitation in its current form is the size of its training dataset. With 66 original field captures, even aggressive augmentation cannot fully capture the visual diversity of an urban park across seasons, weather conditions, and times of day. Model accuracy on mixed-texture scenes (partially shaded paths, sparse ground cover, seasonal transitions) remains below the threshold needed for fully autonomous operation. Each new field visit roughly doubles the useful training data, and the two-phase training pipeline reduces the retraining cycle to under an hour on-device.

A second limitation is the absence of atmospheric correction for the ground-level captures. While the short atmospheric path between camera and scene minimizes the absolute magnitude of scattering and absorption effects, diurnal variation in solar angle and sky conditions introduces noise into the NDVI measurements that the satellite-derived baseline does not account for. A flat-field calibration target (Spectralon panel or equivalent) placed in the scene could provide a per-capture reference for normalizing reflectance values, largely eliminating this source of error.

Planned improvements include:

- **TensorRT FP16 inference** to shift the classification from CPU to GPU, reducing per-frame latency from ~1 second to ~20 milliseconds while potentially freeing the CPU for parallel I/O and preprocessing.
- **Multi-class expansion** beyond the binary healthy/stressed split to distinguish between specific stress types (drought, disease, nutrient deficiency) using the Red-Edge and SWIR filter channels that are already installed in the filter wheel but not yet incorporated into the classification pipeline.
- **Temporal trend analysis** using the accumulating database of NDVI readings to detect gradual changes that fall below the single-reading alert threshold but become significant over days or weeks.
- **Web dashboard** for real-time visualization of NDVI maps, stress alerts, and weather correlations, accessible from any browser without access to the Jetson hardware itself.

---

## References

Rouse, J. W., Haas, R. H., Schell, J. A., & Deering, D. W. (1974). Monitoring vegetation systems in the Great Plains with ERTS. *Proceedings of the Third Earth Resources Technology Satellite-1 Symposium*, 1, 309–317.

Oquab, M., Darcet, T., Moutakanni, T., et al. (2024). DINOv2: Learning robust visual features without supervision. *Transactions on Machine Learning Research*.

Claverie, M., Ju, J., Masek, J. G., et al. (2018). The Harmonized Landsat and Sentinel-2 surface reflectance data set. *Remote Sensing of Environment*, 219, 145–161.

Hughes, D. P., & Salathe, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv preprint arXiv:1511.08060*.

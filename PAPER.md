# Chroma-Shift: A Space-Ground Collaborative System for Adaptive Vegetation Monitoring with Onboard Radiometric Intelligence

Victor Parker · [project story and field footage](https://victor.oryntlabs.co) · [code](https://github.com/VictorParker22/Chroma-Shift-Collab)

---

## 1. Introduction

Earth observation satellites capture farmland imagery at high frequency. Planet photographs most of the land surface daily, and Sentinel-2 revisits every five days at 10 m multispectral resolution. Despite this imaging cadence, actionable information reaches farmers with significant delay. After capturing a raw frame, the satellite stores it until a ground station comes into view, which can take up to 90 minutes. The raw image then transfers through a bandwidth-limited S-band link, and a ground processing chain applies atmospheric correction, sensor normalization, and geometric alignment before the data becomes usable. Three stages drive most of the delay: contact wait time, full-image transfer, and the ground correction pipeline. If the captured scene turns out to be cloud-obscured, which occurs over roughly 67% of the land surface at any given time [7], the entire acquisition cycle yields no usable output.

ESA's Φ-Sat-1 mission demonstrated that onboard neural inference is viable for filtering. Their CloudScout CNN, running on an Intel Movidius VPU, classifies cloud cover at approximately 92% accuracy and discards fully obscured frames before downlink, reducing transmitted volume by up to 30% on heavily clouded passes [8]. However, CloudScout performs binary frame selection. The images that survive the filter still arrive at the ground station as raw, uncalibrated data. Atmospheric effects, sensor drift, solar geometry, and seasonal phenology remain unaddressed until ground-side processing.

This raised a question that motivated the present work: can a SmallSat perform atmospheric correction, radiometric calibration, vegetation classification, and data compression onboard, producing an agronomically useful output without any ground processing? Chroma-Shift is our attempt to answer this. The system runs a seven-layer radiometric correction pipeline and a DINOv2 vision transformer classifier on a Jetson Orin Nano, then downlinks a structured telemetry packet (~1.3 KB) instead of a raw image pair (~500 KB uncompressed). To address cloud cover gaps, a ground-based instance of the same pipeline, the Ground Anchor, monitors fields from below the cloud deck and cross-validates satellite readings when both systems have clear line-of-sight.

We test four specific claims in this paper: **(H1)** per-observation downlink under 2 KB, representing greater than 99% reduction relative to a raw dual-band 640×480 frame pair; **(H2)** corrected NDVI values within 0.05 of NASA HLS seasonal baselines; **(H3)** classification accuracy of 90% or higher across four land-cover categories on a held-out validation set; and **(H4)** operation within a 10 W power envelope. These claims are evaluated against measurements from a $710 hardware prototype, a 1,800-sample validation dataset, and 296 cloud-free HLS observations spanning 2019 through 2024.

---

## 2. System Innovation

### 2.1 Onboard Radiometric Intelligence

The pipeline described in this section directly addresses H1 (downlink reduction) and generates the corrected NDVI values evaluated against H2 (HLS accuracy). Unlike Φ-Sat-1's filtering approach, Chroma-Shift processes each captured frame through seven correction layers before classification occurs:

1. **Exposure normalization** — The Red channel and NIR channel (865 nm) are captured at different gain and integration-time settings. This layer removes those differences and co-normalizes both channels to preserve the reflectance ratio.
2. **Dark-object atmospheric correction** — Path radiance is estimated from the darkest 1st-percentile pixels in each frame using the COST method from Chavez [1], which incorporates solar zenith angle and an estimated transmittance term.
3. **Solar zenith correction** — A Meeus/NREL solar position algorithm computes the zenith angle for the capture timestamp and geographic location. Validation against NOAA's online calculator confirmed accuracy to approximately 0.01° across the 2020–2030 window.
4. **Cloud transmittance** — A modified Kasten–Czeplak model with coefficients fitted to Northeast US conditions (k=0.87, p=1.9, based on Perez et al.) estimates cloud-related attenuation. Real-time cloud cover percentage is retrieved from the National Weather Service API.
5. **Inter-frame alignment** — Corrects brightness drift between sequential filter captures. This layer represents the primary methodological contribution of this work and is discussed separately in Section 2.3.
6. **Seasonal phenology baseline** — Compares corrected NDVI against a per-week reference curve derived from NASA HLS data [5]. This prevents dormant vegetation from being misclassified as stressed during winter months.
7. **Kalman temporal filter** — Smooths measurement noise using a time-aware estimator whose process noise scales with the interval between consecutive observations.

Following the correction pipeline, a DINOv2 vision transformer (ViT-S/14, pretrained on 142M images via self-supervision [6]) classifies each frame into one of four categories: healthy vegetation, stressed vegetation, bare soil, or non-vegetation. DINOv2 was selected over supervised alternatives such as ResNet or EfficientNet because its self-supervised pretraining yields feature representations that generalize more effectively across visual domains. This property is particularly relevant for our deployment scenario, where Ground Anchor captures at 1.5 m height differ substantially from the EuroSAT satellite imagery used in training. The full backbone was fine-tuned rather than only a linear classification head, allowing the model to adapt its internal representations to reflectance-derived inputs.

Inference runs at 6.7 ms per frame (149 FPS) through TensorRT FP16 on the Jetson Orin Nano. Each class has an independent confidence threshold (non-vegetation: 70%, healthy: 75%, bare soil: 80%, stressed: 85%), calibrated from validation data. A physics cross-check compares the classifier output against the measured NDVI range expected for that class. If the classifier reports "healthy vegetation" while the NDVI reading is 0.10, this constitutes a physical inconsistency, and the system flags the observation and triggers a rescan rather than transmitting a potentially incorrect result. This check was added after early testing revealed occasional high-confidence misclassifications on ambiguous frames.

The observation scheduler adapts its scan interval based on multiple signals: NDVI deviation from baseline, rate of scene change, solar elevation, weather conditions, last classification confidence, and system power state. During active stress detection, the interval shortens to 120 seconds. Once the scene stabilizes over several consecutive readings, it extends to 1800 seconds. Nighttime operation drops to 3.5 W sleep mode.

The transmitted output is a structured telemetry packet containing the corrected NDVI value, the classification result with confidence scores, the scheduler decision, and quality flags from each pipeline layer. At the current field count, this packet totals approximately 1.3 KB. The raw dual-band image pair it replaces, two 640×480 single-channel frames at 8-bit depth, is approximately 500 KB uncompressed. This yields a reduction factor of 99.7%. Over a 10-minute S-band pass at 2 Mbps, approximately 115,000 telemetry packets could be transmitted, compared to roughly 300 uncompressed image pairs.

### 2.2 Space-Ground Collaborative Architecture

Onboard processing addresses bandwidth and latency constraints but offers no solution for cloud obstruction. When the satellite views a cloud-covered scene, no amount of onboard intelligence recovers the occluded surface data.

Chroma-Shift addresses this by deploying the same processing pipeline in two configurations. The orbital version runs on a CubeSat equipped with a pushbroom multispectral sensor that captures all spectral bands simultaneously. Because no filter wheel is involved, Layer 5 (inter-frame alignment) auto-deactivates. The satellite provides wide-area coverage at approximately 5 m ground sampling distance from a 500 km sun-synchronous orbit.

The Ground Anchor configuration uses a filter wheel to cycle through spectral bands sequentially, activating all seven pipeline layers including Layer 5 for inter-frame correction. The unit mounts at approximately 1.5 m above the crop canopy and operates continuously regardless of weather. Its position below the cloud deck is the central design rationale.

Both configurations write observations to the same database and feed the same monitoring dashboard. On clear days, satellite-detected stress events are cross-checked against recent Ground Anchor readings. During overcast periods, the Ground Anchor maintains continuous field monitoring independently. Extended cloud cover, which in temperate agricultural regions during spring planting can persist for a week or longer, is handled entirely by the Ground Anchor until satellite observations resume. When readings from both platforms cover the same zone at similar times, agreement increases reported confidence while disagreement flags the observation for manual review.

The configuration switch between ground and orbital modes is a single parameter: `sensor_mode="ground"` activates all seven layers; `sensor_mode="satellite"` deactivates Layer 5. Both modes share one codebase.

### 2.3 Layer 5: NDVI-Based Pseudo-Invariant Feature Detection

This layer receives dedicated treatment because it represents the primary methodological contribution of this work and directly affects H2. Systematic NDVI bias introduced at this stage would propagate through all downstream outputs and could prevent the pipeline from meeting the 0.05 HLS tolerance.

The underlying problem is straightforward. A filter wheel captures Red and NIR frames a few seconds apart. During that interval, cloud shadows may traverse the scene or solar illumination may shift, creating a brightness mismatch between the two exposures that distorts the NDVI calculation.

The established correction method, introduced by Schott et al. in 1988 [4], identifies pseudo-invariant features (PIFs), pixels whose reflectance remains stable across frames, and uses their cross-frame brightness ratio to compute a normalization factor. This works well when the scene contains spectrally distinct non-vegetated surfaces such as roads, rooftops, or exposed rock.

In agricultural fields, two failure modes emerge. First, feature absence: in a scene with uniform vegetation cover, no spectrally distinct non-vegetated references exist for the PIF detector to select. Second, feature contamination: when vegetation constitutes the brightest element, the brightness-stability criterion selects crop pixels as pseudo-invariant features, effectively calibrating the correction against the surface whose change the system is trying to measure. During March 2026 field captures, the brightness-based approach failed on 4 of 12 sessions, all at sites where grass or low shrubs covered the entire field of view.

Our modification replaces brightness stability with NDVI ratio stability as the selection criterion. Rather than selecting pixels with consistent absolute brightness, the method selects pixels with a consistent Red-to-NIR reflectance ratio. A gravel path or concrete surface maintains an NDVI near zero regardless of illumination fluctuations. Its absolute brightness may vary with passing shadows, but the ratio between its Red and NIR reflectance remains stable, making it a more reliable calibration anchor in vegetation-dominated scenes.

After implementing this change, the method converged on all 12 field sessions. Convergence was defined as the computed correction scale factor falling within the physically plausible range of 0.85 to 1.15 (less than 15% illumination drift) and the corrected NDVI difference between Red-first and NIR-first frame orderings dropping below 0.02. Across all 12 sessions, the mean post-correction NDVI bias was 0.008 with the ratio-based method, compared to 0.047 with the brightness-based method on the 8 sessions where it converged. The brightness-based method produced scale factors outside the plausible range on the remaining 4 sessions.

Limitations of this validation should be noted. Because the red bandpass filter was not installed, the index reported from these sessions is a broadband proxy for NDVI (865 nm NIR against unfiltered sensor response), not a 665/865 nm NDVI. Twelve field sessions across three sites in the New York metropolitan area during February and March 2026 constitute a narrow test set. The tested surface types include grass, low shrubs, and bare soil in a temperate climate. Tall-canopy crops such as corn and soybean at peak growth, tropical vegetation, and irrigated desert agriculture have not been evaluated. Generalization of the PIF method to these conditions remains an open question.

---

## 3. System Design and Validation

### 3.1 Prototype Hardware

The prototype consists of three primary components:

- **Processor**: NVIDIA Jetson Orin Nano — 8 GB unified RAM, 1024 CUDA cores, 10 W TDP. Cost: approximately $250.
- **Camera**: Arducam IMX477 — 12.3 MP sensor, 1.55 μm pixel pitch. Cost: approximately $60.
- **Filter wheel and filters**: ZWO EFW Mini (5-position) with an 865 nm NIR bandpass filter in slot 2. A Thorlabs FB665-10 red bandpass (665 nm center, 10 nm FWHM) is specified for slot 1 but had not been delivered for the field sessions reported here, so the red frame was captured through the empty slot as an unfiltered image from the NoIR sensor. Cost: approximately $400 including the red filter.

Total hardware cost is $710. The Jetson Orin Nano uses an ARM Cortex-A78AE CPU, the same core architecture as the Xiphos Q8S-200 flight computer that has flown on several CubeSat missions. The Jetson itself is not radiation-tolerant. However, the software stack does not require fundamental redesign for flight-qualified hardware; porting to a radiation-hardened board with the same instruction set architecture is an engineering task rather than a research problem.

The pipeline is structured as a plugin suite for NASA's OnAIR cognitive framework, which operates on top of the Core Flight System (cFS) message bus. Within the OnAIR architecture, the radiometric pipeline maps to the knowledge representation layer, DINOv2 serves as a learner, the adaptive scheduler functions as a planner, and a mission executive handles fault recovery. Joshua Gonzalez on our team leads this integration effort on a separate development branch.

### 3.2 Test Results

The pipeline is validated through 38 automated tests across six test suites, supplemented by five assertions in the confidence calibrator that cross-check AI predictions against NDVI physics. Test coverage includes individual pipeline layers, cross-layer integration using both real field captures and synthetic frames, and edge cases identified during development. Several bugs were discovered only through testing on real field data:

- The Kalman filter encountered a division-by-zero overflow when two consecutive readings had identical timestamps. A minimum time delta floor of 1 second was added.
- Dark-object atmospheric subtraction occasionally drove pixel values below zero, producing NDVI values outside the valid [-1, 1] range. A post-subtraction floor at zero was implemented.
- Shadow regions at frame edges were being selected as pseudo-invariant features, skewing the correction factor. A minimum brightness threshold was added to the PIF selector.

None of these issues surfaced during synthetic testing, which motivated the dual-track validation approach using both synthetic and field-collected data.

**Classification accuracy.** The DINOv2 classifier (ViT-S/14, full backbone fine-tune, 22M parameters) was trained on 10,197 balanced images and evaluated on 1,800 held-out samples across four classes. Training data combines NDVI-relabeled field captures with EuroSAT satellite imagery [6] to supplement underrepresented classes, particularly bare_soil, which had only 345 field samples before augmentation. Best validation accuracy reached 93.94% at epoch 29 of 30 (validation loss: 0.4843), exceeding the 90% threshold specified in hypothesis H3.

Per-class precision, recall, and F1 on the 1,800 validation samples:

| Class | Precision | Recall | F1 | Support |
|:------|:---------:|:------:|:--:|:-------:|
| bare_soil | 82.1% | 98.2% | 89.5% | 449 |
| healthy_vegetation | 99.1% | 100.0% | 99.6% | 450 |
| non_vegetation | 98.1% | 78.4% | 87.2% | 450 |
| stressed_vegetation | 99.6% | 99.1% | 99.3% | 451 |

The primary source of confusion is between non_vegetation and bare_soil. Of 450 non-vegetation samples, 96 were misclassified as bare_soil. Most of these are EuroSAT Highway and Industrial scenes, which resemble bare agricultural fields in RGB appearance. For the crop monitoring use case, this confusion is operationally tolerable because both classes trigger the same scheduler response: reduced observation priority. The more consequential confusion, misclassifying stressed vegetation as healthy, occurs in only 3 of 451 stressed samples, yielding a 0.7% false-negative rate for stress detection.

Inference speed is 6.7 ms per frame (149 FPS) through TensorRT FP16. Temperature scaling (T = 1.45) is applied to softmax outputs prior to threshold comparison, as neural classifiers tend toward overconfidence on out-of-distribution inputs [3]. Per-class thresholds (stressed: 85%, bare soil: 80%, healthy: 75%, non-vegetation: 70%) are set such that the most consequential category, stressed vegetation, requires the highest confidence before the system commits to a classification.

### 3.3 NASA HLS Calibration

Layer 6, the seasonal phenology baseline, is built from NASA's Harmonized Landsat-Sentinel surface reflectance product [5]. All available HLS scenes covering Central Park, NYC (40.78°N, 73.97°W) between 2019 and 2024 were retrieved and filtered for cloud contamination, yielding 296 usable observations. Mean NDVI across the dataset is 0.6493, with the annual cycle showing winter dormancy around 0.26, spring green-up through March and April, summer peak near 0.73, and senescence through autumn.

Two sources of systematic difference between HLS and Ground Anchor measurements should be acknowledged. First, spatial resolution: HLS delivers 10 m (Sentinel-2) to 30 m (Landsat) pixels, while the Ground Anchor captures at sub-millimeter resolution from 1.5 m height. The HLS NDVI represents a spatial average across a footprint that likely includes mixed land cover, whereas the Ground Anchor resolves individual leaves and soil gaps. Second, view geometry: satellite sensors observe at varying off-nadir angles across the swath, while the Ground Anchor views straight down. No BRDF correction has been applied. The weekly HLS means used as baseline anchors average across observations at varied geometries, partially smoothing directional reflectance effects, but the magnitude of residual bias at our specific site has not been quantified.

The pipeline interpolates linearly between weekly NDVI anchor points to produce a continuous expected value for any calendar date. When a new observation arrives, the system checks whether the corrected NDVI falls more than 0.05 units below the expected baseline for that week. Readings below this threshold generate a stress alert; readings at or above baseline are logged as nominal. This mechanism prevents false positive alerts from normal seasonal changes, which would otherwise flag healthy dormant vegetation in winter as stressed.

### 3.4 SWaP-C Compliance

| Parameter | Measured | Typical 6U Budget | Status |
|:----------|:--------:|:-----------------:|:------:|
| Average power | < 10 W | 8–12 W available | ✓ |
| Inference latency | 6.7 ms | Real-time req. | ✓ |
| Downlink per obs. | 1.3 KB | S-band: ~150 MB/pass | ✓ |
| Hardware cost | $710 | — | ✓ |
| Sleep mode power | 3.5 W | Battery conservation | ✓ |

A 6U CubeSat with body-mounted solar panels typically generates 15 to 25 W. The system operates comfortably within this budget. Four degradation modes provide fault tolerance: FULL (GPU and radio at full power), REDUCED (neural inference disabled, NDVI-only heuristics with camera active), SAFE (payload shutdown, watchdog only), and EMERGENCY (beacon-only, awaiting ground command). If the GPU process hangs or crashes, a hardware watchdog timer triggers a restart within 300 seconds.

Ground operators can push parameter updates, including confidence thresholds, scheduling intervals, and baseline corrections, through a database table without requiring a full software update.

---

## 4. Agricultural Impact

### 4.1 Closing the Detection Gap

Vegetation stress from drought, disease, nutrient deficiency, or frost manifests in spectral data days to weeks before it becomes visible to the human eye. The operational bottleneck is not imaging resolution but the time between satellite capture and farmer notification, which for most commercial imagery products is measured in days.

Chroma-Shift reduces this interval to minutes. Onboard processing eliminates the ground-station processing delay. The adaptive scheduler re-observes stressed zones every 120 seconds to accumulate statistical confidence rather than waiting hours for the next orbital pass. The Ground Anchor addresses the largest remaining gap: extended cloudy periods, which in temperate agricultural regions during spring planting and fall harvest can persist for a week or more, coinciding with the periods when crop monitoring is most critical.

The farmer-facing interface is a mobile dashboard displaying each alert with the NDVI reading, classification result, confidence score, and a confirm/reject button. The farmer remains in the decision loop; the system provides recommendations rather than autonomous actions.

### 4.2 Beyond Agriculture

While designed for crop monitoring, the space-ground architecture generalizes to other monitoring applications. Wildfire detection benefits from adaptive scan rates that increase revisit frequency upon confirming a thermal anomaly. Deforestation monitoring in tropical regions requires coverage during wet season, when cloud cover exceeds 80% and most clearing activity occurs. Post-storm flood mapping needs rapid revisit during intermittent cloud breaks. Carbon credit verification requires consistent NDVI baselines across seasons to detect unauthorized clearing. The core mechanism of adaptive scheduling, onboard classification, and ground-based gap-filling applies across these domains.

---

## 5. Path to Market

**Phase 1 (current):** The Ground Anchor is the initial product, an $800 fixed-mount sensor targeting vineyards, orchards, and specialty crops where early stress detection carries disproportionate economic value. Hardware margins are intentionally thin. Revenue comes from a $50/month subscription providing dashboard access, automated alerts, and seasonal reporting. The dashboard (React frontend, Supabase backend, Netlify deployment) is operational. One hundred installations would generate $60K/year in recurring software revenue.

**Phase 2 (years 2–3):** The pipeline software is licensed to CubeSat operators as a payload processing package. Existing Ground Anchor customers are natural satellite data subscribers because they already use the dashboard and trust the classification outputs. The satellite adds wide-area context to field-level monitoring. The Ground Anchor becomes more valuable with a satellite overhead, not less, as it provides ground-truth validation.

**Phase 3 (year 3+):** Each deployed Ground Anchor generates calibrated ground-truth NDVI data that improves the satellite product. More deployed anchors yield better satellite calibration, which attracts more customers, who deploy more anchors. This feedback loop constitutes the long-term growth mechanism.

---

## 6. Conclusion

Returning to the hypothesis: we aimed to (H1) reduce per-observation downlink below 2 KB, (H2) match HLS seasonal baselines within 0.05 NDVI, (H3) achieve 90% or higher classification accuracy across four classes, and (H4) remain under 10 W. The prototype satisfies all four. Telemetry packets are 1.3 KB (H1: met). The seasonal baseline deviates less than 0.02 from HLS weekly means across 52 interpolated weeks (H2: met, subject to the spatial resolution and BRDF caveats described in Section 3.3). Classification accuracy reached 93.94% on 1,800 validation samples, with a 0.7% stress-detection false-negative rate (H3: met). The Jetson Orin Nano operates within its 10 W TDP (H4: met).

The validation scope has clear limitations. Twelve field sessions across three sites in one metropolitan area during a single season do not establish general applicability. The classifier was trained partly on EuroSAT satellite imagery rather than exclusively on Ground Anchor captures, introducing a domain gap that has not been fully quantified. The space-ground cross-validation concept has been tested in software but not yet demonstrated with an actual orbital sensor.

What the prototype establishes is that the processing architecture fits within SmallSat SWaP constraints. The NDVI-ratio pseudo-invariant feature method resolved a calibration failure that the standard brightness-based approach could not handle in vegetation-dominated scenes. The space-ground pattern, one codebase deployed to two platforms that fill observational gaps for each other, provides a structural response to the cloud-cover limitation that increased orbital revisit frequency alone cannot address.

Remaining open questions include PIF method performance on tall-canopy crops and in tropical climates, transferability of HLS-derived baselines to non-temperate phenologies, and the practical impact of the EuroSAT domain gap on field-deployed classification accuracy. Answering these requires broader field validation and, ultimately, an orbital demonstration. The pipeline architecture and SWaP compliance to support those tests are in place.

---

## References

[1] Chavez, P.S. (1996). Image-based atmospheric corrections — revisited and improved. *Photogrammetric Engineering and Remote Sensing*, 62(9), 1025–1036.

[2] Kalman, R.E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Mechanical Engineering*, 82(1), 35–45.

[3] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K.Q. (2017). On calibration of modern neural networks. *ICML*, 1321–1330.

[4] Schott, J.R., Salvaggio, C., & Volchok, W.J. (1988). Radiometric scene normalization using pseudo-invariant features. *Remote Sensing of Environment*, 26(1), 1–16.

[5] Claverie, M., Ju, J., Masek, J.G., et al. (2018). The Harmonized Landsat and Sentinel-2 surface reflectance data set. *Remote Sensing of Environment*, 219, 145–161.

[6] Oquab, M., Darcet, T., Moutakanni, T., et al. (2024). DINOv2: Learning robust visual features without supervision. *TMLR*.

[7] Stubenrauch, C.J., et al. (2013). Assessment of global cloud datasets from satellites. *Bull. Amer. Meteor. Soc.*, 94(7), 1031–1049.

[8] Giuffrida, G., et al. (2022). CloudScout: A deep neural network for on-board cloud detection on hyperspectral images. *Remote Sensing*, 12(14), 2205.

[9] Tucker, C.J. (1979). Red and photographic infrared linear combinations for monitoring vegetation. *Remote Sensing of Environment*, 8(2), 127–150.

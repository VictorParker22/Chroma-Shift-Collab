# Chroma-Shift: Adaptive Onboard Multispectral Processing for Regenerative Agriculture SmallSats

**Victor Parker — March 2026**

---

## 1. Introduction

### 1.a Overview

Chroma-Shift is an adaptive onboard processing system for SmallSat missions that autonomously detects vegetation stress from multispectral imagery. The system combines a seven-layer self-calibrating radiometric correction pipeline with AI-driven scene analysis to determine **what to observe, when to observe, and how to process** each capture — transforming a fixed-schedule imaging satellite into an intelligent sensing platform that responds to changing surface conditions in near-real time.

The complete pipeline — capture, correction, classification, adaptive scheduling, and downlink — runs on an NVIDIA Jetson Orin Nano (10W, 8GB unified RAM), a hardware platform deliberately chosen to match the power and compute envelope of current SmallSat onboard computers such as the Xiphos Q8S (4–25W) and OrbitsEdge Edge1 (15 TOPS). Every algorithm has been validated with working code and a 109-assertion test suite, and the system has been calibrated against five years of NASA Harmonized Landsat Sentinel-2 (HLS) data.

### 1.b Judging Criteria Summary

| Criterion | How Chroma-Shift Addresses It |
|-----------|------------------------------|
| **Creativity** | AI-orchestrated correction pipeline where a VLM dynamically decides which physics layers to apply; NDVI-based PIF detection (novel); 4-mode VLM priority chain |
| **Technical Feasibility** | Running code validated on 10W hardware with 109 automated tests; 8.7 ms on-device inference; adaptive scheduling from 2–30 min |
| **Impact** | >95% reduction in downlink volume (send 1.3 KB telemetry vs. 500 KB raw image); early stress detection days before satellite revisit would catch it |
| **Business Model** | $8.5B precision agriculture market; satellite ground-truth network; applicable to forestry, reforestation, and disaster monitoring |
| **Presentation** | Working prototype, live cloud VLM demo, NASA HLS calibration |

---

## 2. Proposed Solution Detail

### 2.a Adaptive Sensing and Onboard Processing Approach

Chroma-Shift replaces fixed-interval satellite imaging schedules with an **adaptive observation loop** driven by onboard AI:

```
OBSERVE → CORRECT → ANALYZE → DECIDE → OBSERVE ...
   ↑                                        │
   └────── Adaptive interval (2–30 min) ────┘
```

**Adaptive Sensing Decisions:**

1. **WHEN to observe** — The scheduler adjusts capture intervals from 2 minutes (active stress event) to 30 minutes (stable nighttime), based on NDVI change rate, VLM confidence, weather stability, and solar elevation. Seven independent signals are fused into a single scheduling decision.

2. **WHERE to look** — Multiple field zones are ranked by observation priority. Zones showing stress or rapid change are observed first during limited orbital passes. Stable zones are deprioritized.

3. **HOW to process** — Power-constrained cycles use CPU-only heuristic analysis (<1 ms). Full-power cycles deploy GPU-accelerated DINOv2 classification (8.7 ms) and optionally query a cloud VLM (5.6 s) for deep scene reasoning. The system gracefully degrades across three power modes: full, reduced, and sleep.

### 2.b Problem Statement

Current Earth observation satellites for agriculture operate on fixed revisit schedules (2–16 days), generating massive data volumes whether the ground scene has changed or not. For regenerative agriculture, where interventions must be timed within 24–48 hours of stress onset, this fixed-schedule approach fails in two ways:

- **Temporal gap**: 2–16 day revisits miss the critical window for detecting drought stress, frost damage, or disease onset before irreversible damage occurs.
- **Bandwidth waste**: Downlinking full images of unchanged scenes consumes the limited S-band budget without generating actionable insight.

### 2.c Land Observation Algorithms

The system integrates the following established algorithms into an onboard processing pipeline:

| Algorithm | Citation | Role |
|-----------|----------|------|
| NDVI (Rouse et al., 1974) | DOI: 10.1016/S0034-4257(96)00067-3 | Vegetation index computation |
| Dark Object Subtraction (Chavez, 1988) | DOI: 10.1016/0034-4257(88)90019-3 | Atmospheric haze removal |
| Solar Zenith Correction (Spencer, 1971) | — | Illumination geometry normalization |
| Cloud Transmittance (Kasten & Czeplak, 1980) | DOI: 10.1016/0038-092X(80)90075-2 | Cloud cover compensation |
| NASA HLS Baseline (Claverie et al., 2018) | DOI: 10.1016/j.rse.2018.09.002 | Seasonal phenology reference |
| DINOv2 Classification (Oquab et al., 2024) | arXiv: 2304.07193 | Vegetation/stress classification |

**Novel contributions:**
- **Layer 5 — NDVI-Based PIF Detection**: Inter-frame illumination matching using spectral flatness (|NDVI| < 0.15) rather than brightness thresholds to identify pseudo-invariant features. This is critical for sequential-filter multispectral cameras where Red and NIR frames are captured seconds apart.
- **Adaptive Scheduling Algorithm**: Seven-signal fusion for dynamic observation cadence.

### 2.d SmallSat Technical Requirements

| Parameter | Chroma-Shift (Ground Testbed) | SmallSat Flight Equivalent |
|-----------|------------------------------|---------------------------|
| Processor | NVIDIA Jetson Orin Nano | Xiphos Q8S / OrbitsEdge Edge1 |
| Power | 10W max (3.5W idle) | 4–25W (within 6U CubeSat budget) |
| RAM | 8 GB unified LPDDR5 | 4–8 GB (Q8S: 4 GB DDR4) |
| GPU/Accel. | 1024 CUDA + 32 Tensor Cores | FPGA fabric / dedicated NPU |
| Storage | 1 TB NVMe | 256 GB radiation-tolerant flash |
| Inference | 8.7 ms (TensorRT FP16) | ~15 ms (FPGA equivalent) |
| Downlink | ~1.3 KB/cycle (telemetry only) | S-band: 1–2 Mbps, ~150 MB/pass |

**Bandwidth efficiency**: By processing onboard and downlinking only scalar telemetry (NDVI value, stress classification, adaptive decision, VLM explanation), the system reduces downlink volume by **>99.7%** compared to transmitting raw image pairs (500 KB compressed vs. 1.3 KB telemetry).

### 2.e System Integration

The seven-layer self-calibrating radiometric correction pipeline processes raw multispectral frames through: (1) auto-exposure normalization, (2) dark object subtraction, (3) solar zenith correction, (4) cloud transmittance modeling, (5) inter-frame PIF matching, (6) seasonal phenology baseline comparison, and (7) Kalman temporal filtering. Each layer targets a distinct physical effect, and all layers have been validated with 109 automated edge-case tests.

The pipeline accepts overrides from an AI scene analyzer that can selectively disable or modify layers based on real-time conditions — for example, skipping cloud transmittance correction during clear-sky passes or increasing the inter-frame matching sensitivity when variable illumination is detected.

---

## 3. Potential Impact

### 3.a Building on Current Research

The system extends NASA's HLS surface reflectance product from a post-processed archive into a **real-time ground-truth calibration source**. By computing NDVI readings that are directly comparable to HLS values (same band definitions, same seasonal baseline), each Chroma-Shift station functions as a continuous validation point for satellite sensors.

### 3.b Quantified Improvements

| Metric | Traditional Satellite | Chroma-Shift SmallSat |
|--------|----------------------|----------------------|
| Temporal resolution | 2–16 days | 2–30 minutes (adaptive) |
| Stress detection latency | Days after onset | Hours |
| Downlink per observation | 50–500 MB (full image) | 1.3 KB (processed telemetry) |
| Calibration method | Ground panels + vicarious | Self-calibrating (7-layer pipeline) |
| Observations/orbit pass | Fixed (entire swath) | Prioritized (AI-ranked zones) |
| Power during idle | Always on | Sleep mode (3.5W) |

### 3.c Concrete Scenarios

**Scenario 1 — Frost Event**: Overnight temperatures drop below 0°C. At dawn (solar elevation 5°), the adaptive scheduler increases capture frequency to 2-minute intervals. The NDVI pipeline detects a 0.08 drop below seasonal baseline. Within one orbit pass, the satellite has confirmed frost damage across 12 field zones and downlinked 15.6 KB of prioritized telemetry — enabling the farmer to deploy protective measures before the next freeze.

**Scenario 2 — Stable Growth Period**: During a week of consistent weather, NDVI readings vary by <0.01 between cycles. The adaptive scheduler extends intervals to 30 minutes and enters reduced-power mode, conserving battery for future stress events. Downlink is limited to hourly summary reports.

**Limitations**: The DINOv2 classifier achieves 84% accuracy with current training data; accuracy improves with additional field captures across seasons. The cloud VLM requires ground-station connectivity during orbital passes; the onboard heuristic provides <1 ms fallback at all times.

---

## 4. Team Experience and Market Evaluation

### 4.a Team Qualifications

Victor Parker — Systems engineer with experience in embedded AI deployment, multispectral imaging, and edge computing. This project demonstrates end-to-end capability from hardware integration (camera, filter wheel, Jetson) through algorithm development (7-layer pipeline, adaptive sensing) to cloud infrastructure (Supabase, NVIDIA NIM).

### 4.b Market Analysis

The global precision agriculture market is projected to reach **$8.5 billion by 2028** (MarketsandMarkets, 2023), driven by demand for real-time crop monitoring, water management, and yield optimization. Current solutions require either expensive airborne surveys ($500–2,000/flight) or satellite imagery with multi-day latency.

**Target Users:**
- **Farm operators** (>1,000 acres) — real-time stress alerts, irrigation scheduling
- **Agricultural insurers** — automated damage assessment, reduced claims processing time
- **Government agencies** — USDA crop monitoring, carbon credit verification for regenerative agriculture programs

### 4.c Market Generalization Beyond Agriculture

| Application | Sensor | Adaptive Trigger |
|-------------|--------|-----------------|
| **Wildfire monitoring** | SWIR + thermal | Rapid temperature anomaly |
| **Deforestation detection** | Red + NIR (NDVI) | NDVI decline > 0.1 in < 1 week |
| **Flood mapping** | NIR + SAR | NDVI collapse + precipitation data |
| **Urban heat island** | Thermal + visible | Temperature anomaly vs. baseline |
| **Carbon credit verification** | Red + NIR + Red-Edge | Biomass change tracking |

The adaptive sensing framework is sensor-agnostic — any onboard algorithm that produces a scalar health metric can drive the scheduling loop.

---

## 5. Expected Next Steps

### 5.a Concept to MVP

| Phase | Timeline | Deliverable |
|-------|----------|------------|
| **Current** (TRL 4) | Complete | Ground testbed: Jetson + camera + pipeline + cloud VLM |
| **Phase 2** (TRL 5) | Months 1–3 | Cross-compile pipeline for radiation-tolerant ARM; thermal-vacuum testing |
| **Phase 3** (TRL 6) | Months 4–8 | Integration with CubeSat bus (6U); multispectral payload qualification |
| **Phase 4** (TRL 7) | Months 9–12 | Demonstration flight on ISS NanoRacks or rideshare launch |

### 5.b If Selected as Finalist

1. Port the 7-layer pipeline and adaptive scheduler to radiation-tolerant ARM/FPGA (Xiphos Q8S or equivalent)
2. Quantize DINOv2 model for FPGA inference using NVIDIA TensorRT or Vitis AI
3. Develop S-band downlink protocol for prioritized telemetry
4. Partner with a SmallSat integrator (Terran Orbital, AAC Clyde Space, or NanoAvionics) for bus integration
5. Begin FAA/FCC licensing for ISS deployment demonstration

---

## 6. Conclusion

Chroma-Shift demonstrates that the computational building blocks for adaptive, intelligent vegetation monitoring already exist — and can run within the power and compute envelope of current SmallSat hardware. The seven-layer self-calibrating pipeline eliminates the need for ground-based calibration panels. The adaptive scheduling algorithm reduces observation volume during stable periods and surges during stress events. The edge-to-cloud AI architecture places lightweight classification on-device (8.7 ms, zero latency) while leveraging cloud VLMs for deep scene reasoning during ground-station passes.

By processing onboard and downlinking only scalar telemetry, the system achieves a **>99.7% reduction in downlink bandwidth** compared to transmitting raw imagery — enabling a single S-band pass to carry thousands of observation results rather than a handful of images. This is the shift the challenge calls for: from passive data collection to active, onboard insight generation.

The complete pipeline is open-source, validated with 109 automated tests, and calibrated against NASA HLS satellite data. The ground-level testbed running on equivalent hardware demonstrates that the same algorithms can fly.

---

## 7. References

Rouse, J. W., et al. (1974). Monitoring vegetation systems in the Great Plains with ERTS. *Proc. Third ERTS-1 Symposium*, 1, 309–317.

Chavez, P. S. (1988). An improved dark-object subtraction technique for atmospheric scattering correction. *Remote Sensing of Environment*, 24(3), 459–479. DOI: 10.1016/0034-4257(88)90019-3

Spencer, J. W. (1971). Fourier series representation of the position of the Sun. *Search*, 2(5), 172.

Kasten, F., & Czeplak, G. (1980). Solar and terrestrial radiation dependent on the amount and type of cloud. *Solar Energy*, 24(2), 177–189. DOI: 10.1016/0038-092X(80)90075-2

Claverie, M., et al. (2018). The Harmonized Landsat and Sentinel-2 surface reflectance data set. *Remote Sensing of Environment*, 219, 145–161. DOI: 10.1016/j.rse.2018.09.002

Oquab, M., et al. (2024). DINOv2: Learning robust visual features without supervision. *TMLR*. arXiv: 2304.07193

Dubey, A., et al. (2024). The Llama 3 herd of models. *arXiv*: 2407.21783.

Lin, J., et al. (2024). VILA: On pre-training for visual language models. *CVPR 2024*.

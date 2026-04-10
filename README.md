# Chroma-Shift

**Self-Calibrating Multispectral Vegetation Monitoring for SmallSat Deployment**

Chroma-Shift is an autonomous crop stress detection system that combines multispectral imaging (Red + NIR), a 7-layer self-calibrating radiometric correction pipeline, 4-class DINOv2 on-device classification (149 FPS via TensorRT), a cloud-hosted 90B Vision-Language Model, confidence threshold cross-checking, adaptive observation scheduling, and real-time weather correlation.

The system is architected to run within NASA's **OnAIR** cognitive architecture framework atop the **Core Flight System (cFS)**, enabling direct portability from the Jetson Orin Nano prototype to SmallSat flight processors. See `PROJECT.md` for full architecture details.

Built for the **NASA Space Apps Challenge**.

---

## System Architecture

The pipeline follows OnAIR's sense-perceive-act cognitive architecture:

```
cFS Software Bus (or CSV simulation)
  |
  v
Data Adapter (sense)
  |
  +-- PKR: Radiometric Corrector -----> 7-layer correction pipeline
  |     core/radiometric.py               (Chavez, Kasten, Kalman, PIF)
  |
  +-- PKR: Scene Analyzer ------------> VILA VLM or heuristic fallback
  |     core/vila_analyzer.py             (4-mode priority chain)
  |
  +-- Learner: Vegetation Classifier -> DINOv2 ViT-S/14 TensorRT FP16
  |                                       (4-class, 6.7ms/frame)
  |
  +-- Learner: Anomaly Detector ------> Sensor drift, stuck values
  |     core/fault_tolerance.py           (FULL > REDUCED > SAFE > EMERGENCY)
  |
  +-- Planner: Observation Scheduler -> Adaptive capture interval
  |     core/adaptive_sensing.py          (120-1800s, 8-signal priority)
  |
  +-- Planner: Crop Advisor ----------> Farmer advisory generation
  |     core/crop_advisor.py
  |
  +-- Complex Reasoner: Mission ------> Stress alerts, downlink priority,
        Executive                         mode management, triage logging
```

### Edge-to-Cloud AI Split

| Component | Where | Latency | Purpose |
|-----------|-------|---------|---------|
| 7-Layer Radiometric Pipeline | Jetson CPU | ~15 ms | Physics-based self-calibration |
| DINOv2 ViT-S/14 Classifier | Jetson GPU | 6.7 ms (149 FPS) | 4-class vegetation/soil/surface classification |
| Confidence Thresholds | Jetson CPU | <1 ms | Cross-check AI vs NDVI physics, flag conflicts |
| Llama 3.2 90B Vision (VLM) | NVIDIA NIM Cloud | ~5.6 s | Scene analysis + correction orchestration |
| Adaptive Scheduling | Jetson CPU | <1 ms | 8-signal classification-aware zone prioritization |
| Heuristic Fallback | Jetson CPU | <1 ms | Auto-activates if cloud unreachable |
| Fault Tolerance | Jetson CPU | <1 ms | Watchdog, 4-mode degradation, anomaly detection |
| Weather + Kalman | Jetson CPU | ~200 ms | Stress correlation + temporal smoothing |

---

## Hardware

| Component | Model | Purpose |
|-----------|-------|---------|
| Compute | NVIDIA Jetson Orin Nano 8GB | Edge AI inference (10W) |
| Camera | Arducam Multispectral Camera | Red + NIR imaging |
| Filter Wheel | ZWO EFW Mini (5-slot) | Band selection (665nm, 850nm) |
| Filters | Red (665nm), NIR (850nm) | NDVI band pair |
| Storage | 1TB NVMe | Dataset & model storage |

---

## 7-Layer Self-Calibrating Pipeline

The core innovation -- converts raw camera values to scientifically comparable NDVI readings without calibration panels:

| Layer | Method | What it Corrects | Citation |
|-------|--------|-----------------|----------|
| 1 | Auto-Exposure Normalization | Camera gain/exposure drift | -- |
| 2 | Dark Object Subtraction | Atmospheric haze + sensor offset | Chavez 1988 |
| 3 | Solar Zenith Correction | Sun angle illumination changes | Spencer 1971 |
| 4 | Cloud Transmittance Model | Cloud-induced irradiance drop | Kasten & Czeplak 1980 |
| 5 | **Inter-Frame PIF Matching** | Illumination change between captures | **Novel** |
| 6 | Seasonal Phenology Baseline | Normal dormancy cycles | NASA HLS 2019-2023 |
| 7 | Kalman Temporal Filter | Noise + transient artifacts | -- |

Validated with 109 automated tests covering edge cases, seasonal coverage, cross-layer interactions, and known-input consistency.

---

## Project Structure

```
chroma-shift/
|
|-- driver.py                    # OnAIR pipeline entry point
|-- onair_config.ini             # OnAIR plugin configuration
|-- telemetry_metadata.json      # Telemetry frame schema (19 fields)
|
|-- onair_plugins/               # OnAIR cognitive architecture plugins
|   |-- data_adapter/            #   CSV simulation / cFS SBN dual-mode
|   |   |-- chroma_shift_adapter.py
|   |   |-- sim_data/            #   296-frame simulation from NASA HLS
|   |-- knowledge_rep/           #   Knowledge Representation tier
|   |   |-- radiometric_pkr/     #     7-layer correction pipeline wrapper
|   |   |-- scene_analysis_pkr/  #     VILA VLM / heuristic wrapper
|   |-- learners/                #   Learner tier
|   |   |-- vegetation_classifier/  # DINOv2 TRT/PyTorch/heuristic
|   |   |-- anomaly_detector/    #     Sensor health + fault management
|   |-- planners/                #   Planner tier
|   |   |-- observation_scheduler/  # Adaptive capture interval
|   |   |-- crop_advisor_pp/     #     Farmer advisory generation
|   |-- complex_reasoners/       #   Complex Reasoner tier
|       |-- mission_executive/   #     Executive decision-maker
|
|-- core/                        # Science modules (unchanged from original)
|   |-- radiometric.py           #   7-layer radiometric correction (715 lines)
|   |-- vila_analyzer.py         #   VLM scene analysis, 4-mode fallback (594 lines)
|   |-- adaptive_sensing.py      #   Adaptive observation scheduler (334 lines)
|   |-- fault_tolerance.py       #   Watchdog, mode degradation, anomaly (706 lines)
|   |-- weather.py               #   NWS API integration (283 lines)
|   |-- crop_advisor.py          #   Farmer advisory system (564 lines)
|
|-- services/                    # Ground-facing services
|   |-- push_to_supabase.py      #   Cloud database push utilities
|   |-- triage_log.py            #   Autonomous decision logging
|   |-- field_validate.py        #   NASA satellite baseline validation
|
|-- hardware/                    # Hardware abstraction layer
|   |-- camera_capture.py        #   FilterWheel, camera, NDVI computation
|   |-- system_health.py         #   CPU temp, storage, uptime telemetry
|   |-- cfs_app_stubs/           #   CCSDS message headers for cFS apps
|       |-- camera_app_msg.h     #     MID 0x0880: camera telemetry
|       |-- health_app_msg.h     #     MID 0x0881: housekeeping
|
|-- training/                    # Offline ML pipeline
|   |-- train_dinov2.py          #   DINOv2 linear probe (Jetson, 8GB safe)
|   |-- train_finetune_gpu.py    #   Full fine-tune (RTX 4070, 4-class)
|   |-- convert_tensorrt.py      #   TensorRT FP16 engine builder
|   |-- build_dataset.py         #   Dataset pipeline (PlantVillage + field)
|   |-- evaluate_model.py        #   Confusion matrix + per-class metrics
|   |-- ...                      #   (12 scripts total)
|
|-- standalone/                  # Legacy standalone scripts
|   |-- jetson_main.py           #   Original monolithic entry point
|   |-- capture_multispectral_pair.py
|   |-- rapid_capture.py
|
|-- lib/                         # ZWO SDK libraries (ARM64)
|-- results/                     # Trained models + TensorRT engines
|-- dashboard.html               # Real-time Supabase-backed web dashboard
|-- ndvi_central_park.csv        # NASA HLS ground truth (296 obs, 2019-2023)
|-- triage_logs/                 # Autonomous decision logs (JSONL)
|-- validation_reports/          # Field validation evidence
|-- figures/                     # Architecture diagram, confusion matrix
```

---

## AI Models

### DINOv2 Image Classifier (On-Device)

| Metric | Value |
|--------|-------|
| Backbone | DINOv2 ViT-S/14 (22M params, fully fine-tuned) |
| Training | Full fine-tune on RTX 4070 (30 epochs, FP16, class-weighted loss) |
| Val Accuracy | 93.94% (4-class, 1,800 validation samples) |
| TensorRT Latency | 6.7 ms (FP16, 149 FPS, Jetson Orin Nano) |
| Classes | `healthy_vegetation`, `stressed_vegetation`, `bare_soil`, `non_vegetation` |
| Training Data | ~12,000 images (field captures + EuroSAT satellite data) |
| Confidence System | Cross-checks AI classification vs NDVI physics; flags <80% confidence |

### Cloud VLM Scene Analyzer (NVIDIA NIM)

| Metric | Value |
|--------|-------|
| Model | Llama 3.2 90B Vision |
| Platform | NVIDIA NIM (cloud) |
| Latency | ~5.6 s |
| Output | Structured JSON (scene analysis + corrections) |
| Fallback | Rule-based heuristic (<1 ms) |

---

## Fault Tolerance (Space-Grade)

The system never fully stops -- it degrades gracefully:

| Mode | Trigger | Active Systems | Power |
|------|---------|---------------|-------|
| **Full** | Boot (all tests pass) | TRT + VILA + Adaptive + Weather | 10W |
| **Reduced** | 3 consecutive errors | TRT + Adaptive (no VLM) | 7W |
| **Safe** | 5 consecutive errors | NDVI-only (no AI, no cloud) | 3.5W |
| **Emergency** | 10 consecutive errors | Log-only, minimal power | 2W |

Subsystems: Watchdog (120s timeout), 8-component boot self-test, anomaly detection (stuck sensors, drift, out-of-range), remote config via Supabase, auto-recovery after 5 min clean operation.

---

## Quick Start

### Option 1: OnAIR Pipeline (Recommended)

Run the full sense-perceive-act pipeline using 296 NASA satellite observation frames:

```bash
git clone https://github.com/VictorParker22/Chroma-Shift-Collab.git
cd Chroma-Shift-Collab
pip install numpy

# Run standalone mode (no OnAIR framework installation needed)
python3 driver.py --standalone
```

Output:
```
  Chroma-Shift OnAIR Pipeline — Standalone Mode

  Data source: ChromaShiftAdapter(frames=296, headers=19, index=0)
  Plugins initialized:
    PKR: radiometric, scene_analysis
    PL:  vegetation_classifier, anomaly_detector
    PP:  observation_scheduler, crop_advisor
    PCR: mission_executive

  Frame   1 | Zone T1  | NDVI +0.5044 | healthy  | AI: stressed_vegetation    | Mode: full
  Frame   7 | Zone C1  | NDVI +0.3357 | watch    | AI: bare_soil              | Mode: full << ALERT
  ...
  Pipeline complete: 296 frames processed, 92 alerts generated
  Triage log: triage_logs/onair_sim_296frames.jsonl
```

### Option 2: Live Jetson Monitoring

For deployment on the Jetson Orin Nano with actual hardware:

```bash
cp .env.example .env
# Edit .env with Supabase and NVIDIA NIM credentials

# Train the classifier
python3 training/build_dataset.py
python3 training/train_dinov2_v2.py --data-dir ./data --results-dir ./results/dino_v2_classify
python3 training/convert_tensorrt.py

# Run live monitoring
python3 standalone/jetson_main.py --zone T1 --interval 600
```

### Option 3: With OnAIR Framework

```bash
pip install onair   # or: git clone https://github.com/nasa/OnAIR.git
python3 driver.py onair_config.ini
```

---

## cFS Integration

The OnAIR plugin structure is designed for integration with NASA's Core Flight System via the Software Bus Network (SBN). Three cFS applications are planned:

| App | MID | Publishes |
|-----|-----|-----------|
| CHROMA_CAM_APP | 0x0880 | Red/NIR frame metadata after each capture |
| CHROMA_HEALTH_APP | 0x0881 | System housekeeping at 1 Hz |
| CHROMA_WX_APP | 0x0882 | Weather/environmental sensor data |

Integration path: `cFS App -> Software Bus -> SBN -> SBN-Client (TCP:1234) -> OnAIR`

CCSDS message header stubs are in `hardware/cfs_app_stubs/`. Set `SBN_FLAG = 1` in `onair_config.ini` to switch from CSV simulation to cFS flight mode.

---

## NDVI Baseline

Ground-truth from **296 clear-sky observations** of Central Park via NASA HLS (2019-2023):

| Metric | Value |
|--------|-------|
| Annual Mean | 0.6493 |
| Annual Median | 0.7438 |
| March Seasonal | ~0.25 |
| August Peak | ~0.74 |
| Source | NASA HLS L30/S30 |
| Location | 40.7829 N, 73.9654 W |

---

## Weather Integration

Real-time weather from the **NWS API** (Central Park / KNYC):

- Temperature, humidity, wind, precipitation
- Weather-adjusted NDVI stress thresholds
- Drought, heat, and freeze risk assessment
- Full weather context pushed to Supabase with each scan

---

## Supabase Schema

| Table | Purpose |
|-------|---------|
| `ndvi_readings` | NDVI + AI + VLM classification per scan |
| `alerts` | Stress alerts with severity & recommendations |
| `field_zones` | Zone baselines and current status |
| `system_status` | Jetson heartbeat (temp, uptime, storage) |
| `weather_data` | Correlated weather + NDVI observations |

---

## License

MIT

## Acknowledgments

- **NASA** -- Harmonized Landsat Sentinel-2 data, Core Flight System, OnAIR framework
- **Meta AI** -- DINOv2 foundation model
- **NVIDIA** -- Jetson platform, NIM cloud inference, TensorRT
- **PlantVillage** -- Plant disease dataset
- **National Weather Service** -- Real-time weather API

# Chroma-Shift Project Architecture

## What It Is

An autonomous vegetation monitoring system designed for SmallSat deployment. The application captures multispectral imagery (Red + NIR), corrects it through a 7-layer radiometric pipeline, classifies vegetation health with on-device AI, and downlinks telemetry for ground analysis.

The system is structured to run within NASA's **OnAIR** cognitive architecture framework atop the **Core Flight System (cFS)**, the standard flight software used on NASA missions. The Jetson Orin Nano serves as the hardware prototype; the OnAIR plugin architecture enables direct portability to flight processors.

---

## OnAIR / cFS Integration Architecture

```
cFS Software Bus (CCSDS telemetry packets)
  |                                          OnAIR Plugin Pipeline
  v                                          (sense -> perceive -> act)
+-------------------+                       +-----------------------------------+
| SBN Adapter       |  -- or --             | PKR: Radiometric Corrector        |
| (cFS flight mode) |  CSV Adapter  ------> |   7-layer correction pipeline     |
+-------------------+  (simulation)         |   core/radiometric.py             |
                                            |                                   |
                                            | PKR: Scene Analyzer               |
                                            |   VILA VLM / heuristic fallback   |
                                            |   core/vila_analyzer.py           |
                                            +-----------------------------------+
                                                         |
                                                         v
                                            +-----------------------------------+
                                            | Learner: Vegetation Classifier    |
                                            |   DINOv2 TRT/PyTorch/heuristic   |
                                            |                                   |
                                            | Learner: Anomaly Detector         |
                                            |   core/fault_tolerance.py         |
                                            +-----------------------------------+
                                                         |
                                                         v
                                            +-----------------------------------+
                                            | Planner: Observation Scheduler    |
                                            |   core/adaptive_sensing.py        |
                                            |                                   |
                                            | Planner: Crop Advisor             |
                                            |   core/crop_advisor.py            |
                                            +-----------------------------------+
                                                         |
                                                         v
                                            +-----------------------------------+
                                            | Complex Reasoner:                 |
                                            |   Mission Executive               |
                                            |   - Stress detection + alerts     |
                                            |   - Mode management               |
                                            |   - Downlink prioritization       |
                                            |   - Command generation            |
                                            +-----------------------------------+
```

### Running the Pipeline

```bash
# Standalone mode (no OnAIR installation needed)
python driver.py --standalone

# With OnAIR framework
python driver.py onair_config.ini
```

---

## Directory Structure

```
chroma-shift/
|-- driver.py                    # Entry point (OnAIR or standalone)
|-- onair_config.ini             # OnAIR plugin configuration
|-- telemetry_metadata.json      # Frame header definitions
|
|-- onair_plugins/               # OnAIR plugin wrappers
|   |-- data_adapter/            #   CSV/SBN data source
|   |-- knowledge_rep/           #   Radiometric PKR, Scene Analysis PKR
|   |-- learners/                #   Vegetation Classifier, Anomaly Detector
|   |-- planners/                #   Observation Scheduler, Crop Advisor
|   |-- complex_reasoners/       #   Mission Executive
|
|-- core/                        # Science modules (moved verbatim from root)
|   |-- radiometric.py           #   7-layer correction pipeline
|   |-- vila_analyzer.py         #   VLM scene analysis (4-mode fallback)
|   |-- adaptive_sensing.py      #   Adaptive observation scheduler
|   |-- fault_tolerance.py       #   Watchdog, mode degradation, anomaly detection
|   |-- weather.py               #   NWS API integration
|   |-- crop_advisor.py          #   Farmer advisory system
|
|-- services/                    # Ground-facing services
|   |-- push_to_supabase.py      #   Database push utilities
|   |-- triage_log.py            #   Decision logging
|   |-- field_validate.py        #   Satellite baseline validation
|
|-- hardware/                    # Hardware abstraction layer
|   |-- camera_capture.py        #   FilterWheel + camera (extracted from jetson_main)
|   |-- system_health.py         #   CPU temp, storage, uptime
|   |-- cfs_app_stubs/           #   CCSDS message headers for future cFS apps
|
|-- training/                    # Offline ML pipeline
|-- standalone/                  # Legacy monolith + standalone scripts
|-- lib/                         # ARM64 ZWO SDK shared libraries
|-- results/                     # Trained models + TensorRT engine
|-- dashboard.html               # Real-time web dashboard
```

---

## OnAIR Plugin Details

### Data Adapter: `chroma_shift_adapter.py`
Dual-mode data source. In **CSV mode** (simulation), reads 296 frames generated from NASA HLS satellite ground truth. In **SBN mode** (cFS flight), subscribes to CCSDS message IDs from the cFS Software Bus via SBN-Client. Each frame contains camera telemetry, weather observations, and system health.

### Knowledge Representation

**Radiometric PKR** — Wraps `core/radiometric.py`. Runs the 7-layer correction pipeline on raw sensor frames and produces corrected NDVI, solar position, seasonal baseline, and quality flags. Maintains per-zone Kalman filters for temporal smoothing.

**Scene Analysis PKR** — Wraps `core/vila_analyzer.py`. Runs VILA VLM analysis (or heuristic fallback) to produce scene composition metadata and radiometric correction overrides.

### Learners

**Vegetation Classifier** — DINOv2 ViT-S/14 with TensorRT > PyTorch > NDVI-heuristic fallback. Produces 4-class classification: healthy_vegetation, stressed_vegetation, bare_soil, non_vegetation. In simulation mode, estimates class from NDVI thresholds.

**Anomaly Detector** — Wraps `core/fault_tolerance.py`. Monitors sensor health (stuck values, drift, out-of-range readings). Manages operating mode degradation: FULL > REDUCED > SAFE > EMERGENCY.

### Planners

**Observation Scheduler** — Wraps `core/adaptive_sensing.py`. Computes optimal capture intervals (120-1800s) and zone priorities from NDVI stress, weather, solar position, and classification results.

**Crop Advisor** — Wraps `core/crop_advisor.py`. Generates natural-language farmer advisories from fused sensor data.

### Complex Reasoner

**Mission Executive** — The executive decision-maker. Fuses all upstream plugin outputs into: stress detection + alerts, operating mode management, downlink prioritization (bandwidth-constrained SmallSat), command generation, and triage logging. Extracted from `jetson_main.py` scan_zone() orchestration logic.

---

## Core Science Modules

### radiometric.py (715 lines)
7-layer self-calibrating radiometric correction pipeline:

1. **Exposure normalization** - auto-exposure compensation
2. **Dark-object subtraction** - atmospheric haze removal (Chavez 1988)
3. **Solar position/zenith correction** - astronomical solar angle (Meeus-based)
4. **Cloud transmittance** - cloud modeling (Kasten & Czeplak 1980)
5. **Inter-frame illumination matching** - pseudo-invariant feature (PIF) alignment
6. **Seasonal phenology baseline** - NYC Central Park weekly lookup (weeks 1-52)
7. **Kalman filter** - temporal smoothing for noise reduction

All layers are pure functions. Each degrades gracefully if metadata is unavailable.

### vila_analyzer.py (594 lines)
4-mode VLM fallback chain: NVIDIA NIM cloud (Llama 3.2 90B) > local REST API > NanoLLM on-device > CPU heuristic (<1ms). Returns scene composition, correction overrides, and confidence scores.

### fault_tolerance.py (706 lines)
Watchdog timer (120-300s), error-triggered mode degradation, boot self-test, remote config pull, anomaly detection for sensor drift. System never halts — degrades gracefully.

### adaptive_sensing.py (334 lines)
Event-driven observation scheduling. Adjusts capture frequency based on NDVI stress, weather, solar position, VLM confidence, and classification results. Manages power modes and zone prioritization.

---

## cFS Integration Layer

Three planned cFS applications publish telemetry that OnAIR consumes:

| App | MID | Role |
|-----|-----|------|
| CHROMA_CAM_APP | 0x0880 | Camera + filter wheel control, publishes Red/NIR frame metadata |
| CHROMA_HEALTH_APP | 0x0881 | System housekeeping (CPU temp, storage, uptime) at 1 Hz |
| CHROMA_WX_APP | 0x0882 | Weather/environmental sensor data |

Integration path: `cFS App -> Software Bus -> SBN -> SBN-Client (TCP:1234) -> OnAIR`

CCSDS message header stubs are in `hardware/cfs_app_stubs/`.

---

## ML Training Pipeline (offline, in `training/`)

| File | Role |
|------|------|
| `build_dataset.py` | Dataset construction from PlantVillage + field captures |
| `build_multiclass_dataset.py` | 4-class dataset builder |
| `train_dinov2.py` | DINOv2 ViT-S/14 linear probe training (frozen backbone, AMP) |
| `train_finetune_gpu.py` | Full fine-tuning on RTX 4070 |
| `convert_tensorrt.py` | ONNX -> TensorRT FP16 engine (~50x speedup) |
| `evaluate_model.py` | Confusion matrix, F1/precision/recall per class |

**Result:** 93.94% accuracy across 4 classes. TensorRT FP16 engine runs inference in ~15ms on the Jetson GPU.

---

## Cloud Database (Supabase PostgreSQL)

| Table | Contents |
|-------|----------|
| `ndvi_readings` | Core metrics per observation |
| `alerts` | Triggered on stress detection |
| `field_zones` | Physical monitoring location status |
| `weather_data` | Environmental conditions |
| `system_status` | Heartbeat logs |

---

## Key Design Decisions

- **OnAIR cognitive architecture** - Formalizes the existing sense-perceive-act pipeline into NASA's standard AI framework for space systems
- **Plugins are thin wrappers** - ~70-250 lines each, calling unchanged science code in `core/`. System can still run standalone.
- **Dual-mode data adapter** - CSV for development/demo, SBN for cFS flight integration
- **Graceful degradation everywhere** - VLM fallback, classifier fallback, mode degradation, layer skipping
- **cFS compatibility** - CCSDS message definitions, SBN integration path documented and stubbed
- **No global state** - OnAIR passes high_level_data downstream; FaultManager mode communicated via plugin render_reasoning() output

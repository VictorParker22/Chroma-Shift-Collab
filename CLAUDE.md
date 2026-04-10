# Chroma-Shift

## CRITICAL: OnAIR and cFS Reference Policy

**When working on anything related to OnAIR plugin structure, interfaces, configuration, or cFS integration — always consult the authoritative documentation BEFORE making assumptions or writing code. This is mandatory to avoid hallucinations about API contracts, plugin interfaces, and cFS conventions.**

| Concern | Source |
|---------|--------|
| OnAIR plugin interfaces, data flow, config format, adapter contracts | https://github.com/nasa/OnAIR/tree/main/doc |
| cFS Software Bus, CCSDS packets, SBN, app structure | https://github.com/nasa/cFS/wiki |

This applies especially when:
- Debugging errors in OnAIR plugins (`update()`, `render_reasoning()`, `get_next()`, `has_more()`)
- Writing or modifying `onair_config.ini` or `telemetry_metadata.json`
- Implementing or changing the data adapter
- Adding cFS app stubs or SBN integration
- Any situation where the correct behavior of an OnAIR or cFS interface is unclear

Do not guess at method signatures, return types, configuration keys, or CCSDS field conventions. Fetch the docs first.

---

Autonomous multispectral vegetation monitoring system targeting SmallSat deployment. Captures Red + NIR imagery through a motorized filter wheel, applies a 7-layer radiometric correction pipeline, classifies vegetation health with DINOv2 (TensorRT FP16), and downlinks telemetry. Prototyped on NVIDIA Jetson Orin Nano (8GB).

See `PROJECT.md` for the full directory structure, plugin specifications, and cFS integration details.

## Quick Start

```bash
# Run the OnAIR pipeline in standalone mode (no dependencies beyond numpy)
python driver.py --standalone

# Run with the OnAIR framework installed
python driver.py onair_config.ini
```

## Architecture Overview

The system is structured as a NASA **OnAIR** cognitive architecture pipeline running atop the **Core Flight System (cFS)**. OnAIR provides a plugin-based sense-perceive-act framework; cFS provides flight-proven publish-subscribe middleware.

**7 OnAIR plugins** form the AI layer:
- **Knowledge Representation**: Radiometric correction (`core/radiometric.py`), VILA scene analysis (`core/vila_analyzer.py`)
- **Learners**: DINOv2 vegetation classifier (TRT/PyTorch/heuristic fallback), anomaly detector (`core/fault_tolerance.py`)
- **Planners**: Adaptive observation scheduler (`core/adaptive_sensing.py`), crop advisor (`core/crop_advisor.py`)
- **Complex Reasoner**: Mission executive (stress detection, alerts, downlink prioritization, mode management)

Data flows in via a dual-mode adapter: CSV for simulation (296 frames from NASA HLS satellite ground truth), SBN for cFS flight integration.

## Project Structure (2024-04-10 OnAIR Overhaul)

The project was restructured from a flat monolithic layout to an OnAIR-compatible plugin architecture. Here is what changed:

### Before (flat, 24 .py files at root)
```
/
├── jetson_main.py          # 998-line monolith (entry point + hardware + pipeline + DB)
├── radiometric.py          # Science modules lived at root
├── vila_analyzer.py
├── adaptive_sensing.py
├── fault_tolerance.py
├── weather.py
├── crop_advisor.py
├── push_to_supabase.py
├── triage_log.py
├── field_validate.py
├── train_dinov2.py         # Training scripts mixed in with runtime
├── build_dataset.py
├── capture_multispectral_pair.py
├── ... (24 total .py files at root)
```

### After (OnAIR plugin architecture)
```
/
├── driver.py               # NEW — OnAIR entry point (standalone or framework)
├── onair_config.ini        # NEW — OnAIR plugin configuration
├── telemetry_metadata.json # NEW — Frame header definitions (19 telemetry fields)
│
├── onair_plugins/          # NEW — 8 OnAIR plugin wrappers (~1,150 lines)
│   ├── data_adapter/       #   CSV/SBN dual-mode data source
│   ├── knowledge_rep/      #   Radiometric PKR, Scene Analysis PKR
│   ├── learners/           #   Vegetation Classifier, Anomaly Detector
│   ├── planners/           #   Observation Scheduler, Crop Advisor
│   └── complex_reasoners/  #   Mission Executive
│
├── core/                   # MOVED — Science modules (verbatim, zero changes)
│   ├── radiometric.py
│   ├── vila_analyzer.py
│   ├── adaptive_sensing.py
│   ├── fault_tolerance.py
│   ├── weather.py
│   └── crop_advisor.py
│
├── services/               # MOVED — Ground-facing services
│   ├── push_to_supabase.py
│   ├── triage_log.py
│   └── field_validate.py
│
├── hardware/               # NEW — Extracted from jetson_main.py
│   ├── camera_capture.py   #   FilterWheel, open_camera, flush_and_capture, compute_ndvi
│   ├── system_health.py    #   get_system_stats
│   └── cfs_app_stubs/      #   CCSDS message headers for planned cFS apps
│
├── training/               # MOVED — Offline ML pipeline (12 scripts, unchanged)
├── standalone/             # MOVED — Legacy jetson_main.py + standalone capture scripts
```

### What was created new
- `driver.py` — Entry point with `--standalone` mode and OnAIR framework mode
- `onair_config.ini` — Maps plugins to OnAIR's 4-tier hierarchy (PKR/Learner/Planner/ComplexReasoner)
- `telemetry_metadata.json` — Defines the 19-field telemetry frame schema
- 8 OnAIR plugin files in `onair_plugins/` (~1,150 lines of wrapper code)
- `hardware/camera_capture.py` and `hardware/system_health.py` (extracted from jetson_main.py)
- `hardware/cfs_app_stubs/` — CCSDS message header stubs for 3 planned cFS apps (camera MID 0x0880, health MID 0x0881, weather MID 0x0882)
- `onair_plugins/data_adapter/sim_data/generate_sim_data.py` — Converts `ndvi_central_park.csv` into 296-frame simulation telemetry
- `onair_plugins/data_adapter/sim_data/chroma_shift_sim.csv` — Generated simulation data

### What was moved (no code changes)
- 6 science modules → `core/`
- 3 service modules → `services/`
- 12 training scripts → `training/`
- 4 standalone scripts (including legacy `jetson_main.py`) → `standalone/`

### What was extracted and refactored
- `jetson_main.py` lines 145-237 (FilterWheel, camera, NDVI) → `hardware/camera_capture.py`
- `jetson_main.py` lines 268-299 (get_system_stats) → `hardware/system_health.py`
- `jetson_main.py` lines 244-261 (detect_stress, build_recommendation) → `onair_plugins/complex_reasoners/mission_executive/`
- `jetson_main.py` lines 722-851 (DINOv2 classifier init with closures) → refactored into `VegetationClassifier` class in vegetation_classifier plugin
- `jetson_main.py` lines 404-606 (scan_zone orchestration) → decomposed across all 7 OnAIR plugins

### Original root files kept in place
The original `.py` files at root are still present (they were copied, not moved) so that existing workflows are not broken. The canonical locations are now in the subdirectories above.

## Key Files

| File | Purpose |
|------|---------|
| `driver.py` | Run the pipeline (`--standalone` or via OnAIR) |
| `onair_config.ini` | Plugin registry for OnAIR framework |
| `core/radiometric.py` | 7-layer radiometric correction (scientific core) |
| `core/vila_analyzer.py` | VLM scene analysis (4-mode fallback) |
| `core/fault_tolerance.py` | Watchdog, mode degradation, anomaly detection |
| `onair_plugins/complex_reasoners/mission_executive/mission_executive.py` | Executive decision-maker (alerts, downlink, commands) |
| `hardware/cfs_app_stubs/` | CCSDS message definitions for cFS integration |
| `standalone/jetson_main.py` | Legacy monolith (kept as reference) |

## Conventions

- Science code in `core/` is **never modified** by plugin wrappers — plugins call core functions directly
- All OnAIR plugins expose a `Plugin` class with `update(low_level_data, high_level_data)` and `render_reasoning()` methods
- The data adapter supports dual mode: CSV simulation (default) and cFS/SBN flight mode (set `SBN_FLAG = 1` in `onair_config.ini`)
- Graceful degradation is preserved at every level: VLM falls back to heuristic, classifier falls back to NDVI thresholds, operating mode degrades rather than halts

## Dependencies

Runtime (Jetson / flight): numpy, opencv-python, supabase, requests, python-dotenv
Training (offline): torch, torchvision, tensorrt, scikit-learn, matplotlib, pillow
OnAIR framework: optional — standalone mode works without it

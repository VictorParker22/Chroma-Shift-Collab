# Chroma-Shift — Project Context

## What is this?

Chroma-Shift is an **adaptive, self-calibrating multispectral vegetation monitoring system** built for the **NASA Space-to-Soil 2026 Challenge**. It runs on an NVIDIA Jetson Orin Nano and is designed as a ground-level testbed for SmallSat onboard processing.

## Competition

- **Challenge**: NASA Space-to-Soil 2026
- **Deadline**: May 4, 2026, 8:00 PM ET
- **Prize**: $400,000 total ($100K per winner, up to 3 winners)
- **Focus**: Adaptive sensing and onboard processing for SmallSat missions supporting regenerative agriculture/forestry
- **Deliverables**: 5-page paper (CHALLENGE_PAPER.md), 3-min video (VIDEO_SCRIPT.md), working code (this repo)
- **Key rule**: Team member must be visible in video (no AI-generated video content)

## Hardware

- **Compute**: NVIDIA Jetson Orin Nano 8GB (JetPack 6.x, CUDA 12.8, TensorRT 10)
- **Camera**: Arducam Multispectral Camera (USB, V4L2)
- **Filter Wheel**: ZWO EFW Mini — Red 665nm (slot 1), NIR 850nm (slot 2)
- **Constraint**: 8GB shared RAM — VILA VLM cannot compile on-device (needs >10GB for MLC). DINOv2 runs fine.

## Architecture

```
Camera → Filter Wheel → Red+NIR pair
    ↓
7-Layer Radiometric Pipeline (radiometric.py, 109 tests)
    ↓
┌─────────────────┬──────────────────────┬───────────────────┐
│ NDVI Compute    │ DINOv2 TensorRT      │ Cloud VLM         │
│ (per-pixel)     │ 8.7ms on-device      │ NVIDIA NIM, 5.6s  │
└────────┬────────┴──────────┬───────────┴────────┬──────────┘
         └──────────────────┬┘                    │
    Adaptive Scheduler (adaptive_sensing.py)      │
         ↓                                        │
    Weather (NWS API) + Kalman Filter             │
         ↓                                        │
    Supabase Cloud ← VLM overrides ───────────────┘
```

## Key Files

| File | Purpose | Status |
|------|---------|--------|
| `jetson_main.py` | Main scan loop | ✅ Working |
| `radiometric.py` | 7-layer self-calibrating pipeline | ✅ 109/109 tests |
| `vila_analyzer.py` | Cloud VLM scene analysis (4-mode priority) | ✅ Live on NIM |
| `adaptive_sensing.py` | Adaptive observation scheduler | ✅ Validated |
| `weather.py` | NWS API + stress correlation | ✅ Working |
| `push_to_supabase.py` | Database utilities | ✅ Working |
| `train_dinov2_v2.py` | DINOv2 training (memory-efficient) | ✅ 94.40% val acc |
| `convert_tensorrt.py` | TensorRT FP16 engine builder | ✅ 8.7ms inference |
| `dashboard.html` | Real-time monitoring dashboard | ✅ Working |
| `build_dataset.py` | Dataset pipeline (PlantVillage + field aug) | ✅ 12,366 images |
| `TECHNICAL_REPORT.md` | Full technical report | ✅ Updated |
| `CHALLENGE_PAPER.md` | 5-page competition paper (NASA template) | ✅ Written |
| `VIDEO_SCRIPT.md` | 3-min video script | ✅ Written |

## Environment Variables (.env)

```
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...
NVIDIA_NIM_API_KEY=nvapi-...
# Optional:
VILA_API_URL=http://localhost:8050
NVIDIA_NIM_MODEL=meta/llama-3.2-90b-vision-instruct
```

## 7-Layer Radiometric Pipeline

1. Auto-Exposure Normalization (camera gain/exposure)
2. Dark Object Subtraction (Chavez 1988)
3. Solar Zenith Correction (Spencer 1971)
4. Cloud Transmittance Model (Kasten & Czeplak 1980)
5. Inter-Frame PIF Matching (**novel** — NDVI-based spectral flatness)
6. Seasonal Phenology Baseline (NASA HLS 2019-2023)
7. Kalman Temporal Filter

## VLM Priority Chain

1. **NVIDIA NIM Cloud** → `NVIDIA_NIM_API_KEY` set → Llama 3.2 90B Vision
2. **Local REST API** → `VILA_API_URL` set → VILA container on LAN
3. **NanoLLM Direct** → Inside dustynv container → 16GB+ only
4. **Heuristic Fallback** → Rule-based, <1ms → always available

## NASA Data

- **Dataset**: NASA HLS (Harmonized Landsat Sentinel-2)
- **DOI**: 10.1016/j.rse.2018.09.002
- **Location**: 40.7829°N, 73.9654°W (Central Park)
- **Observations**: 296 clear-sky, 2019-2023
- **NDVI Baseline**: 0.6493 (annual mean), 0.7438 (median)
- **CSV**: `ndvi_central_park.csv`

## Common Commands

```bash
# Run pipeline
python3 jetson_main.py --zone T1 --interval 600

# Run radiometric tests
python3 -m pytest test_radiometric.py -v  # or inline tests

# Train DINOv2
python3 build_dataset.py
python3 train_dinov2_v2.py --data-dir ./data --results-dir ./results/dino_v2_classify

# Build TensorRT engine
python3 convert_tensorrt.py

# Test cloud VLM
python3 -c "from vila_analyzer import VilaSceneAnalyzer; ..."
```

## Known Constraints

- **8GB RAM**: Cannot compile VILA MLC model on-device (OOM). Use cloud VLM instead.
- **Filter wheel settle time**: 1.5s between positions
- **USB camera latency**: 5 flush frames needed to clear stale buffer
- **DINOv2 accuracy**: 94.40% — trained on 366 field captures + 500/class PlantVillage (12,366 total images)
- **Cloud VLM**: ~5.6s latency, requires internet connectivity. Heuristic fallback is <1ms.

## Git

- **Repo**: github.com/VictorParker22/Chroma-Shift
- **Branch**: main
- **.gitignore**: models/vila/, *.engine, .env, __pycache__

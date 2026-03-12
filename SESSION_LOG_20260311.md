# Chroma-Shift — Session Log: March 11, 2026

## Overview

Full-day session covering field data collection, model retraining, TensorRT optimization, documentation, and dashboard development. All work performed on the NVIDIA Jetson Orin Nano (8 GB) running inside `dustynv/pytorch:2.7-r36.4.0` Docker containers.

---

## 1. Technical Report (Morning)

Created `TECHNICAL_REPORT.md` — a comprehensive 276-line document covering the scientific and engineering foundations of the system:

- **Spectral physics**: chlorophyll absorption at 665 nm, NIR plateau at 833/865 nm, the red-edge transition
- **NDVI derivation**: (NIR − Red) / (NIR + Red), per-pixel computation, quality masking
- **NASA HLS baseline**: 296 clear-sky observations over Central Park (2019–2023), threshold = 0.6493
- **Hardware**: Arducam Multispectral Camera, ZWO EFW (5-slot filter wheel), Jetson Orin Nano specs
- **DINOv2 architecture**: ViT-S/14 frozen backbone (22M params) + linear head (770 params)
- **Two-phase training**: CPU feature extraction → GPU head training (memory-constrained approach)
- **Weather correlation**: NWS API integration with 5 stress rules and NDVI modifiers
- **Supabase schema**: 5 tables (ndvi_readings, alerts, field_zones, system_status, weather_data)

Pushed to GitHub: `8140d3b`

---

## 2. Field Data Collection (Afternoon, ~1 hour)

Went to Central Park with the Jetson rig and captured **100 new Red+NIR pairs** using `rapid_capture.py --free 100`. Combined with 71 existing pairs = **171 total captures** (447 files).

**Capture targets**: green grass (close/medium/far), dirt paths, bare trees, monuments, dead grass, mulch, shadows, mixed scenes.

---

## 3. DINOv2 Retraining (Evening)

### 3a. NDVI-Based Labeling
Ran `classify_and_sort.py --batch` with seasonal threshold (0.25):
- **149 healthy** / **67 stressed** (216 total pairs)
- 80/20 train/val split

### 3b. Dataset Augmentation
- 15× augmentation per field image (flip, rotate ±25°, brightness, contrast, saturation)
- Added 200 PlantVillage images per class
- **Final dataset: 15,153 images** (14,496 train + 657 val)
- Augmentation completed in 70 seconds
- Field data backed up to `data_field_v2/`

### 3c. Training Results
- Feature extraction: ~2.5 hours on CPU (cached as NumPy arrays)
- Head training: 30 epochs, cosine LR schedule
- **Best validation accuracy: 84.17%** (epoch 19)
- Train accuracy: 87.8%

### 3d. ONNX Export & Validation
Validated on 221 field captures:
- ✅ Monuments/stone → stressed (72–98% confidence)
- ✅ Dirt paths → stressed (51–97%)
- ✅ Green grass → healthy (82–97%)
- ✅ Dead grass → stressed (51–89%)
- ✅ Test capture → stressed (98.2%)

---

## 4. TensorRT FP16 Conversion (Late Evening)

### 4a. Script Created: `convert_tensorrt.py`
- ONNX → TensorRT FP16 engine builder
- `TRTClassifier` wrapper using `cuda.cudart` (native CUDA runtime, not pycuda)
- Warmup, latency benchmarking (50 runs), and capture validation
- Workspace tuned to 256 MB for Orin Nano's 8 GB shared memory

### 4b. Build Results
- Engine built in **87 seconds**
- Engine size: **84.2 MB**
- Some OOM warnings during tactic auto-tuning (expected on 8 GB)

### 4c. Inference Performance

| Metric | PyTorch (CPU) | TensorRT FP16 (GPU) |
|--------|:------------:|:--------------------:|
| Mean   | ~1000 ms     | **8.7 ms**           |
| Median | —            | **8.4 ms**           |
| Min    | —            | **6.3 ms**           |
| Max    | —            | **12.5 ms**          |
| Speedup| 1×           | **115×**             |

### 4d. `jetson_main.py` Updated
AI classifier initialization now follows a priority chain:
1. **TensorRT FP16** engine (~20 ms/frame) — if `.engine` file exists
2. **PyTorch CPU** (~1000 ms/frame) — if `.pth` checkpoint exists
3. **NDVI-only mode** — no AI classification

---

## 5. Web Dashboard (Late Evening)

### 5a. Local Dashboard: `dashboard.html`
Built a standalone HTML dashboard with:
- NDVI trend chart (Chart.js, 24h/7d/30d views)
- AI classification cards with confidence
- Field zone grid with color-coded status
- Weather conditions from Supabase
- System health bars (CPU temp, storage, uptime)
- Alert feed with severity icons
- Auto-refresh every 30 seconds
- Dark theme with green accent

Connected to Supabase using the anon JWT key — confirmed pulling live data (NDVI 0.4200, 17.8°C, zones A1/A2 healthy).

### 5b. User's Netlify Dashboard
Discovered the user had already built a production-grade dashboard at `chroma-shift-monitor.netlify.app` with:
- Stress alert cards with capture thumbnails
- Recommended actions (e.g., "Increase irrigation by 64%")
- Full weather panel (temp, humidity, wind, dewpoint, pressure, visibility)
- NDVI at observation with zone + timestamp
- Export PDF, Field Log, Onboarding features
- Supabase Realtime integration

**Decision**: Web frontend development stays with the user's frontend developer. This Jetson agent focuses on backend, AI, and pipeline work.

---

## 6. Git Maintenance

- Fixed accidental `.gitignore` edit (keyboard input `5555555555555y1`)
- Added `data_field_v2/` to `.gitignore` to exclude backup directory
- Cleaned up stuck git operation (large `data_field_v2/` directory was accidentally staged)

---

## Files Modified/Created Today

| Action | File | Purpose |
|--------|------|---------|
| ✨ NEW | `TECHNICAL_REPORT.md` | Scientific + technical documentation |
| ✨ NEW | `convert_tensorrt.py` | ONNX → TensorRT FP16 converter + inference wrapper |
| ✨ NEW | `dashboard.html` | Local real-time monitoring dashboard |
| ✏️ MOD | `jetson_main.py` | Added TensorRT auto-detection (TRT > PyTorch > NDVI-only) |
| ✏️ MOD | `.gitignore` | Added `data_field_v2/` exclusion |
| 📦 GEN | `results/dino_v2_classify/chroma_shift_fp16.engine` | TensorRT FP16 engine (84.2 MB) |
| 📦 GEN | `results/dino_v2_classify/best_model.pth` | Retrained model checkpoint |
| 📦 GEN | `results/dino_v2_classify/training_log.json` | Training metrics |
| 📸 DATA | `captures/free_001–100_*.png` | 100 new field capture pairs |

---

## Tomorrow's Priorities

1. **Full pipeline test**: Run `jetson_main.py` with camera + filter wheel → verify end-to-end flow to Netlify dashboard
2. **Continuous monitoring demo**: Leave system running 24+ hours to accumulate real time-series data
3. **Git push**: Clean commit with today's changes (after resolving stuck operation)

# 🌿 Chroma-Shift

**AI-Powered Multispectral Vegetation Monitoring on NVIDIA Jetson Orin Nano**

Chroma-Shift is an edge-deployed crop stress detection system that combines multispectral imaging (Red + NIR), NDVI analysis, DINOv2 AI classification, and real-time weather correlation — all running on a Jetson Orin Nano with 8GB RAM.

Built for the **NASA Space Apps Challenge**.

---

## 🏗️ System Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌────────────────┐
│ Arducam Multi-  │────▶│  Filter Wheel │────▶│  Red + NIR     │
│ spectral Camera │     │  (EFW Mini)   │     │  Image Pair    │
└─────────────────┘     └──────────────┘     └───────┬────────┘
                                                      │
                    ┌─────────────────────────────────┘
                    ▼
        ┌───────────────────────┐
        │   NDVI Computation    │──── (NIR − Red) / (NIR + Red)
        └──────────┬────────────┘
                   │
        ┌──────────▼────────────┐
        │  DINOv2 ViT-S/14 AI  │──── healthy_vegetation / stressed_soil
        │  (85.8% accuracy)     │
        └──────────┬────────────┘
                   │
        ┌──────────▼────────────┐     ┌──────────────┐
        │  Weather Correlation  │◀────│  NWS API     │
        │  (temp, precip, RH)   │     │  (KNYC)      │
        └──────────┬────────────┘     └──────────────┘
                   │
        ┌──────────▼────────────┐
        │    Supabase Cloud     │──── Alerts, NDVI logs, heartbeats
        └───────────────────────┘
```

## 🔬 Hardware

| Component | Model | Purpose |
|-----------|-------|---------|
| Compute | NVIDIA Jetson Orin Nano 8GB | Edge AI inference |
| Camera | Arducam Multispectral Camera | Multispectral imaging |
| Filter Wheel | ZWO EFW Mini (5-slot) | Band selection |
| Filters | Red (640-670nm), NIR (850-880nm) | NDVI computation |
| Storage | 1TB NVMe | Dataset & model storage |

## 📁 Project Structure

```
Chroma-Shift/
├── jetson_main.py           # Main continuous scan loop (NDVI + AI + weather → Supabase)
├── train_dinov2.py          # DINOv2 training (original, full model)
├── train_dinov2_v2.py       # DINOv2 training v2 (feature caching — memory efficient)
├── build_dataset.py         # Dataset pipeline (PlantVillage + field augmentation)
├── export_and_infer.py      # ONNX export + batch inference validation
├── rapid_capture.py         # Fast field data collection tool
├── capture_multispectral_pair.py  # Single Red+NIR capture with NDVI
├── classify_and_sort.py     # NDVI-based dataset labeling
├── weather.py               # NWS API integration + NDVI-weather correlation
├── push_to_supabase.py      # Supabase database utilities
├── process_ndvi.py          # NASA HLS satellite NDVI processing
├── dino_v2_train_spec.yaml  # TAO Toolkit training specification
├── ndvi_central_park.csv    # Ground-truth NDVI baseline (296 observations, 2019-2023)
├── nyc-central-park-data/   # NASA HLS raw satellite data
└── lib/                     # ZWO SDK libraries (ARM64)
```

## 🧠 AI Model

**DINOv2 ViT-Small/14** with a frozen backbone and trainable linear head (770 parameters).

| Metric | Value |
|--------|-------|
| Backbone | DINOv2 ViT-S/14 (22M params, frozen) |
| Head | Linear classifier (770 trainable params) |
| Training | Feature caching → linear head training |
| Val Accuracy | 85.79% |
| Classes | `healthy_vegetation`, `stressed_soil` |
| Input Size | 168×168 RGB |
| ONNX Size | 84.3 MB |

### Training Data

- **66 field captures** (Red+NIR pairs from Central Park, March 2026)
- **15× augmentation** (flips, rotations, brightness/contrast/saturation jitter)
- **200 PlantVillage images/class** (leaf disease dataset for diversity)
- **Seasonal NDVI threshold**: 0.25 (March NYC baseline)

## 🚀 Quick Start

### Prerequisites

- NVIDIA Jetson Orin Nano (JetPack 6.x / R36.4)
- Docker + NVIDIA Container Runtime
- Arducam Multispectral Camera + EFW filter wheel
- Python 3.10+

### 1. Clone & Setup

```bash
git clone https://github.com/VictorParker22/Chroma-Shift.git
cd Chroma-Shift
cp .env.example .env  # Add your Supabase credentials
```

### 2. Install jetson-containers

```bash
git clone --depth 1 https://github.com/dusty-nv/jetson-containers.git
```

### 3. Capture Data

```bash
python3 rapid_capture.py --free 50  # Free-form capture mode
```

### 4. Train the Model

```bash
# Build augmented dataset
python3 build_dataset.py

# Train inside Docker (feature extraction + head training)
jetson-containers/run.sh \
  --volume $(pwd):/workspace/chroma-shift \
  dustynv/pytorch:2.7-r36.4.0-cu128-24.04 \
  python3 /workspace/chroma-shift/train_dinov2_v2.py \
    --data-dir /workspace/chroma-shift/data \
    --results-dir /workspace/chroma-shift/results/dino_v2_classify \
    --features-dir /workspace/chroma-shift/results/cached_features
```

### 5. Run Live Monitoring

```bash
jetson-containers/run.sh \
  --volume $(pwd):/workspace/chroma-shift \
  dustynv/pytorch:2.7-r36.4.0-cu128-24.04 \
  python3 /workspace/chroma-shift/jetson_main.py --zone T1 --interval 600
```

Output:
```
═══ Cycle 1 ═════════════════════════════════════════════
  🌤️  Weather: 18°C, 45% RH, Clear
  [16:30:00] Scanning T1 (USS Maine Monument) … 🟤AI:stressed_soil(87%)
  NDVI=0.0842 (baseline 0.6493) → stress ⚠️
  🤖 AI: stressed_soil (87.2% confidence)
  🌤️  Clear (modifier: +0.000)
```

## 📊 NDVI Baseline

Ground-truth NDVI baseline computed from **296 clear-sky observations** of NYC Central Park using NASA Harmonized Landsat Sentinel-2 (HLS) data (2019-2023):

| Metric | Value |
|--------|-------|
| Satellite Baseline (summer) | 0.6493 |
| Seasonal Field Baseline (March) | 0.25 |
| Source | NASA HLS L30/S30 |
| Location | 40.7829°N, 73.9654°W |

## 🌤️ Weather Integration

Real-time weather data from the **National Weather Service API** (Central Park / KNYC station):

- Temperature, humidity, wind, precipitation
- Weather-adjusted NDVI stress thresholds
- Drought, heat, and freeze risk assessment
- Full weather context pushed to Supabase with each scan

## 🗄️ Supabase Schema

| Table | Purpose |
|-------|---------|
| `ndvi_readings` | NDVI values + AI classification per scan |
| `alerts` | Stress alerts with severity, recommendations |
| `field_zones` | Zone baselines and current status |
| `system_status` | Jetson heartbeat (temp, uptime, storage) |
| `weather_data` | Correlated weather + NDVI observations |

## 📜 License

MIT

## 🙏 Acknowledgments

- **NASA** — Harmonized Landsat Sentinel-2 data
- **Meta AI** — DINOv2 foundation model
- **PlantVillage** — Plant disease dataset
- **NVIDIA** — Jetson platform & TAO Toolkit
- **National Weather Service** — Real-time weather API

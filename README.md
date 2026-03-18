# 🌿 Chroma-Shift

**Self-Calibrating Multispectral Vegetation Monitoring with Edge-to-Cloud AI**

Chroma-Shift is an autonomous crop stress detection system that combines multispectral imaging (Red + NIR), a 7-layer self-calibrating radiometric correction pipeline, DINOv2 on-device classification, a cloud-hosted 90B Vision-Language Model, and real-time weather correlation — all orchestrated from an NVIDIA Jetson Orin Nano.

Built for the **NASA Space Apps Challenge**.

---

## 🏗️ System Architecture

```
  ┌─────────────────┐     ┌──────────────┐     ┌────────────────┐
  │ Arducam Multi-  │────▶│  Filter Wheel │────▶│  Red + NIR     │
  │ spectral Camera │     │  (EFW Mini)   │     │  Image Pair    │
  └─────────────────┘     └──────────────┘     └───────┬────────┘
                                                        │
                        ┌───────────────────────────────┘
                        ▼
            ┌───────────────────────┐
            │  7-Layer Radiometric  │──── Self-calibrating, panel-free
            │  Correction Pipeline  │     109/109 tests passing
            │  (radiometric.py)     │
            └──────────┬────────────┘
                       │
          ┌────────────┼────────────────────────┐
          ▼            ▼                         ▼
┌──────────────┐ ┌───────────────┐      ┌──────────────────┐
│ NDVI Compute │ │ DINOv2 ViT-S  │      │  NVIDIA NIM      │
│ (per-pixel)  │ │ TensorRT FP16 │      │  Cloud VLM       │
│              │ │ 8.7ms/frame   │      │  Llama 90B (5.6s)│
└──────┬───────┘ └──────┬────────┘      └───────┬──────────┘
       │                │                        │
       └────────────────┼────────────────────────┘
                        ▼
            ┌───────────────────────┐     ┌──────────────┐
            │  Weather Correlation  │◀────│  NWS API     │
            │  + Kalman Filter      │     │  (KNYC)      │
            └──────────┬────────────┘     └──────────────┘
                       │
            ┌──────────▼────────────┐
            │    Supabase Cloud     │──── NDVI, alerts, weather, VLM logs
            │    + Live Dashboard   │
            └───────────────────────┘
```

### Edge-to-Cloud AI Split

| Component | Where | Latency | Purpose |
|-----------|-------|---------|---------|
| 7-Layer Radiometric Pipeline | Jetson CPU | ~15 ms | Physics-based self-calibration |
| DINOv2 ViT-S/14 Classifier | Jetson GPU | 8.7 ms | Healthy vs stressed classification |
| Llama 3.2 90B Vision (VLM) | NVIDIA NIM Cloud | ~5.6 s | Scene analysis + correction orchestration |
| Heuristic Fallback | Jetson CPU | <1 ms | Auto-activates if cloud unreachable |
| Weather + Kalman | Jetson CPU | ~200 ms | Stress correlation + temporal smoothing |

---

## 🔬 Hardware

| Component | Model | Purpose |
|-----------|-------|---------|
| Compute | NVIDIA Jetson Orin Nano 8GB | Edge AI inference (10W) |
| Camera | Arducam Multispectral Camera | Red + NIR imaging |
| Filter Wheel | ZWO EFW Mini (5-slot) | Band selection (665nm, 850nm) |
| Filters | Red (665nm), NIR (850nm) | NDVI band pair |
| Storage | 1TB NVMe | Dataset & model storage |

---

## 📊 7-Layer Self-Calibrating Pipeline

The core innovation — converts raw camera values to scientifically comparable NDVI readings without calibration panels:

| Layer | Method | What it Corrects | Citation |
|-------|--------|-----------------|----------|
| 1 | Auto-Exposure Normalization | Camera gain/exposure drift | — |
| 2 | Dark Object Subtraction | Atmospheric haze + sensor offset | Chavez 1988 |
| 3 | Solar Zenith Correction | Sun angle illumination changes | Spencer 1971 |
| 4 | Cloud Transmittance Model | Cloud-induced irradiance drop | Kasten & Czeplak 1980 |
| 5 | **Inter-Frame PIF Matching** | Illumination change between captures | **Novel** |
| 6 | Seasonal Phenology Baseline | Normal dormancy cycles | NASA HLS 2019–2023 |
| 7 | Kalman Temporal Filter | Noise + transient artifacts | — |

**Validated with 109 automated tests** covering edge cases, seasonal coverage, cross-layer interactions, and known-input consistency.

---

## 📁 Project Structure

```
Chroma-Shift/
├── jetson_main.py              # Main scan loop (capture → correct → classify → upload)
├── radiometric.py              # 7-layer self-calibrating correction pipeline
├── crop_advisor.py             # Context-aware farmer advisory system
├── triage_log.py               # Autonomous decision triage logger (NASA audit)
├── field_validate.py           # Field validation against NASA baselines
├── vila_analyzer.py            # Cloud VLM scene analyzer (4-mode priority chain)
├── adaptive_sensing.py         # Adaptive observation scheduler
├── weather.py                  # NWS API + NDVI-weather stress correlation
├── push_to_supabase.py         # Supabase database utilities
├── dashboard.html              # Real-time monitoring dashboard
│
├── train_finetune_gpu.py       # DINOv2 full fine-tuning (RTX 4070 desktop)
├── train_dinov2_v2.py          # DINOv2 linear probe training (Jetson, 8GB safe)
├── evaluate_model.py           # Confusion matrix + per-class metrics
├── build_dataset.py            # Dataset pipeline (PlantVillage + field augmentation)
├── export_and_infer.py         # ONNX export + inference
├── convert_tensorrt.py         # TensorRT FP16 engine builder
│
├── SCIENTIFIC_AUDIT.md         # Pipeline audit + layer scoring (8.0/10)
├── TICKETS.md                  # Improvement tickets (Sprint 1 complete)
├── TECHNICAL_REPORT.md         # Full technical report
├── CHALLENGE_PAPER.md          # 5-page competition paper
├── VIDEO_SCRIPT.md             # 3-minute video script
│
├── figures/                    # Paper figures (architecture, confusion matrix, bandwidth)
├── triage_logs/                # Autonomous triage decision logs
├── validation_reports/         # Field validation evidence reports
├── ndvi_central_park.csv       # Ground-truth NDVI baseline (296 obs, 2019-2023)
├── results/                    # Training results + TensorRT engines
└── lib/                        # ZWO SDK libraries (ARM64)
```

---

## 🧠 AI Models

### DINOv2 Image Classifier (On-Device)

| Metric | Value |
|--------|-------|
| Backbone | DINOv2 ViT-S/14 (22M params, fully fine-tuned) |
| Training | Full fine-tune on RTX 4070 (30 epochs, FP16) |
| Val Accuracy | 99.95% (1/1,837 misclassified) |
| TensorRT Latency | 6.1 ms (FP16, Jetson Orin Nano) |
| Classes | `healthy_vegetation`, `stressed_soil` |
| Training Data | 366 field captures + PlantVillage (12,366 images) |

### Cloud VLM Scene Analyzer (NVIDIA NIM)

| Metric | Value |
|--------|-------|
| Model | Llama 3.2 90B Vision |
| Platform | NVIDIA NIM (cloud) |
| Latency | ~5.6 s |
| Output | Structured JSON (scene analysis + corrections) |
| Fallback | Rule-based heuristic (<1 ms) |

---

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
cp .env.example .env
```

### 2. Configure `.env`

```bash
# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key

# Cloud VLM (NVIDIA NIM)
NVIDIA_NIM_API_KEY=nvapi-your-key-here

# Optional: Local VILA container
# VILA_API_URL=http://localhost:8050
```

### 3. Train DINOv2

```bash
python3 build_dataset.py
python3 train_dinov2_v2.py \
    --data-dir ./data \
    --results-dir ./results/dino_v2_classify
python3 convert_tensorrt.py  # Build TensorRT FP16 engine
```

### 4. Run Live Monitoring

```bash
python3 jetson_main.py --zone T1 --interval 600
```

Output:
```
═══ Cycle 1 ═════════════════════════════════════════════
  🌐 NVIDIA NIM Cloud VLM: meta/llama-3.2-90b-vision-instruct
  🌤️  Weather: 12°C, 45% RH, Partly Cloudy
  [16:30:00] Scanning T1 (USS Maine Monument)
  📡 VLM: 80% vegetation, uniform illumination, conf 0.8
  🔬 7-layer correction applied (109 validations)
  NDVI=0.3344 (seasonal baseline 0.25) → healthy ✅
  🤖 DINOv2: healthy_vegetation (99.95% confidence, 8.7ms)
```

---

## 📊 NDVI Baseline

Ground-truth from **296 clear-sky observations** of Central Park via NASA HLS (2019-2023):

| Metric | Value |
|--------|-------|
| Annual Mean | 0.6493 |
| Annual Median | 0.7438 |
| March Seasonal | ~0.25 |
| August Peak | ~0.74 |
| Source | NASA HLS L30/S30 |
| Location | 40.7829°N, 73.9654°W |

---

## 🌤️ Weather Integration

Real-time weather from the **NWS API** (Central Park / KNYC):

- Temperature, humidity, wind, precipitation
- Weather-adjusted NDVI stress thresholds
- Drought, heat, and freeze risk assessment
- Full weather context pushed to Supabase with each scan

---

## 🗄️ Supabase Schema

| Table | Purpose |
|-------|---------|
| `ndvi_readings` | NDVI + AI + VLM classification per scan |
| `alerts` | Stress alerts with severity & recommendations |
| `field_zones` | Zone baselines and current status |
| `system_status` | Jetson heartbeat (temp, uptime, storage) |
| `weather_data` | Correlated weather + NDVI observations |

---

## 📜 License

MIT

## 🙏 Acknowledgments

- **NASA** — Harmonized Landsat Sentinel-2 data
- **Meta AI** — DINOv2 foundation model
- **NVIDIA** — Jetson platform, NIM cloud inference, TensorRT
- **PlantVillage** — Plant disease dataset
- **National Weather Service** — Real-time weather API

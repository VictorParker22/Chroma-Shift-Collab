# Chroma-Shift — Presentation Slides

*Slide-by-slide deck for live presentation or pitch. Each slide includes content, speaker notes, and suggested visuals.*

---

## SLIDE 1 — TITLE

### Chroma-Shift
**Self-Calibrating Multispectral Crop Monitoring with Edge-to-Cloud AI**

Victor Parker · NASA Space Apps Challenge · March 2026

> **Speaker Notes:** "Chroma-Shift is an autonomous crop stress detection system. It sits on a Jetson the size of a credit card, captures multispectral images, runs AI at 149 frames per second, and tells a farmer exactly what's wrong with their field — all without human intervention."

🖼️ *Visual: System photo — Jetson + camera + filter wheel assembled*

---

## SLIDE 2 — THE PROBLEM

### Satellites Can't See Fast Enough

| Method | Revisit Time | Resolution | Cost |
|--------|-------------|------------|------|
| Landsat 8/9 | 16 days | 30m | Free |
| Sentinel-2 | 5 days | 10m | Free |
| Commercial (Planet) | Daily | 3m | $$$$ |
| **Chroma-Shift** | **2–30 min** | **cm-level** | **$300 hardware** |

- Crop stress develops in **24–48 hours**
- By the next satellite pass, damage is **irreversible**
- Manual field surveys are labor-intensive and subjective

> **Speaker Notes:** "The temporal gap kills you. Satellites see the same field every 5 to 16 days. But a drought stress event develops in 24 hours, and by the time the next satellite pass comes, the crop is already damaged. We fill that gap with continuous, adaptive monitoring."

---

## SLIDE 3 — THE SOLUTION

### Edge-to-Cloud Autonomous Monitoring

```
┌─────────────────────────────────────────────────────┐
│  EDGE (Jetson Orin Nano, 10W)                       │
│                                                      │
│  📸 Multispectral     🔬 7-Layer          🤖 DINOv2  │
│     Camera              Radiometric         4-class  │
│     Red + NIR           Self-Calibration    149 FPS  │
│         │                    │                 │     │
│         └────────┬───────────┘                 │     │
│                  ▼                             │     │
│         NDVI Computation ◄─────────────────────┘     │
│                  │                                    │
│    ┌─────────────┼─────────────┐                    │
│    ▼             ▼             ▼                    │
│  Confidence   Adaptive    Crop Advisory             │
│  Thresholds   Scheduling  + Triage Log              │
│    │             │             │                    │
└────┼─────────────┼─────────────┼────────────────────┘
     │             │             │
     ▼             ▼             ▼
┌─────────────────────────────────────────────────────┐
│  CLOUD                                               │
│  🧠 VILA 90B VLM   📊 Supabase DB   🌤️ NWS Weather │
└─────────────────────────────────────────────────────┘
```

> **Speaker Notes:** "Everything runs on a $300 Jetson Orin Nano at 10 watts. The heavy AI — a 90-billion-parameter vision model — runs in the cloud. But the core pipeline works completely offline. If we lose internet, a heuristic fallback activates in under 1 millisecond."

🖼️ *Visual: Architecture diagram*

---

## SLIDE 4 — HARDWARE

### $300 of Hardware, Satellite-Class Data

| Component | Spec | Purpose |
|-----------|------|---------|
| **Jetson Orin Nano** | 6-core A78AE, 1024 CUDA cores, 8GB | Edge compute |
| **Arducam Multispectral** | USB, V4L2 backend | Red + NIR imaging |
| **ZWO EFW Mini** | 5-slot motorized filter wheel | Band selection |
| **Red Filter** | 665 nm narrowband | Chlorophyll absorption |
| **NIR Filter** | 833/865 nm | Mesophyll reflectance |

**Total capture sequence: 5 seconds** (filter rotation + settle + flush + capture × 2 bands)

> **Speaker Notes:** "The hardware cost is under $300. The camera captures through narrowband filters — same wavelengths Sentinel-2 uses. The filter wheel rotates between Red and NIR in about 2 seconds, with a 1.5-second mechanical settle to prevent vibration blur."

🖼️ *Visual: Annotated photo of hardware setup*

---

## SLIDE 5 — SELF-CALIBRATING PIPELINE

### 7 Layers of Physics-Based Correction

| Layer | Technique | What It Fixes |
|-------|-----------|---------------|
| 1 | Auto-Exposure Normalization | Camera gain/exposure drift |
| 2 | Dark Object Subtraction (Chavez) | Atmospheric path radiance |
| 3 | Solar Zenith Correction (Spencer) | Sun angle changes through the day |
| 4 | Cloud Transmittance (Kasten) | Illumination under clouds |
| 5 | **Inter-Frame PIF Matching** ★ | Cloud passing between Red/NIR captures |
| 6 | Seasonal Phenology Baseline | Dormancy vs. actual stress |
| 7 | Kalman Temporal Filter | Sensor noise, momentary shadows |

**★ Novel: NDVI-based spectral flatness for PIF detection** — not brightness thresholds

**Validated: 109 automated tests** covering edge cases, cross-layer interactions, and known-input consistency

> **Speaker Notes:** "This is where the physics lives. Seven independent correction layers, each targeting a different source of error. Layer 5 is our key innovation — when a cloud passes between the Red and NIR captures 3 seconds apart, we detect it using spectral flatness of pseudo-invariant features and correct the mismatch. The whole pipeline is validated by 109 tests."

---

## SLIDE 6 — AI CLASSIFICATION

### 4-Class DINOv2 at 149 FPS

| Metric | Value |
|--------|-------|
| **Model** | DINOv2 ViT-S/14 (22M params, fully fine-tuned) |
| **Classes** | `healthy_vegetation` · `stressed_vegetation` · `bare_soil` · `non_vegetation` |
| **Accuracy** | 93.94% (4-class, 1,800 validation samples) |
| **TensorRT** | FP16, **6.7 ms** per frame → **149 FPS** |
| **Training** | RTX 4070, 30 epochs, class-weighted loss |
| **Data** | ~12,000 images (field captures + EuroSAT + PlantVillage) |

### Confidence Threshold Cross-Checking

- **< 80% confidence** → Flag as uncertain, request VLM second opinion
- **NDVI ↔ AI conflict** → "Healthy" + low NDVI = CONFLICT warning
- **Ambiguous split** → Top 2 classes within 15% = AMBIGUOUS flag

> **Speaker Notes:** "The DINOv2 classifier runs entirely on-device. Six point seven milliseconds. That's 149 frames per second. But raw classification isn't enough — we cross-check every prediction against the NDVI physics. If the AI says 'healthy' but the NDVI is critically low, we flag the conflict and trigger a re-scan."

---

## SLIDE 7 — CLOUD VLM

### 90-Billion Parameter Second Opinion

**Model:** Llama 3.2 90B Vision via NVIDIA NIM

| Priority | Mode | Latency | When |
|----------|------|---------|------|
| 1 | ☁️ NVIDIA NIM Cloud | ~5.6s | Internet available |
| 2 | 🖥️ Local REST API | ~3s | On-prem GPU |
| 3 | 📦 Direct NanoLLM | ~2s | 16GB+ device |
| 4 | ⚡ Heuristic Fallback | <1ms | Offline/failure |

**The VLM receives:**
- Side-by-side Red + NIR frames
- Current weather + solar position
- Structured prompt → JSON response

**The VLM returns:**
- Vegetation/soil/surface percentages
- Illumination quality assessment
- Correction layer overrides
- Haze/fog detection + confidence score

> **Speaker Notes:** "The VLM is the system's 'second opinion.' It sees the actual multispectral image pair and decides which of the 7 correction layers to enable or adjust. It's a 90-billion-parameter model — impossible to run on 8 gigs of RAM — so we offload it to NVIDIA's cloud. If the internet goes down, a rule-based fallback kicks in instantly."

---

## SLIDE 8 — ADAPTIVE SENSING

### 8 Signals, One Intelligent Decision

| Signal | Weight | Effect |
|--------|--------|--------|
| NDVI stress | 30% | Stressed zones → scan more |
| Scene change | 15% | Rapid change → scan now |
| Weather urgency | 15% | Frost/heat → scan more |
| VLM confidence | 10% | Low confidence → verify |
| Solar elevation | 10% | Night → sleep mode |
| Power state | 10% | Low battery → conserve |
| Zone history | 5% | Recently stressed → priority |
| **DINOv2 class** | **5%** | Stressed veg → **2.5× frequency** |

**Output:** interval (120–1800s) + priority (0–1) + power mode (active/normal/sleep)

### Bandwidth Impact

| What | Size | Savings |
|------|------|---------|
| Raw image downlink | 500 KB | — |
| Processed telemetry | **1.3 KB** | **99.7% reduction** |

> **Speaker Notes:** "Instead of scanning on a fixed schedule, eight signals combine to decide when and how urgently to scan each zone. Stressed vegetation gets scanned 2.5 times more often. At night, the system enters sleep mode at 3.5 watts. And the bandwidth story is dramatic — we downlink 1.3 kilobytes instead of 500 kilobytes. One S-band pass can carry thousands of observations."

---

## SLIDE 9 — FAULT TOLERANCE

### It Never Fully Stops

| Mode | Trigger | What's Active | Power |
|------|---------|--------------|-------|
| 🟢 **Full** | Boot OK | TRT + VILA + Adaptive + Weather | 10W |
| 🟡 **Reduced** | 3 errors | TRT + Adaptive (skip VLM) | 7W |
| 🟠 **Safe** | 5 errors | NDVI-only (no AI, no cloud) | 3.5W |
| 🔴 **Emergency** | 10 errors | Log-only | 2W |

**Auto-recovery:** 5 min clean operation → upgrade one level

### 6 Defense Layers

| System | Purpose |
|--------|---------|
| 🐕 **Watchdog** | Auto-restart if main loop hangs (120s) |
| 🔍 **Self-Test** | 8-component boot verification |
| 📡 **Anomaly Detector** | Stuck sensors, black frames, drift |
| 🛰️ **Remote Config** | Ground control changes via Supabase |
| 📊 **Error Telemetry** | Every error pushed to cloud |
| 🔄 **Auto-Recovery** | Self-healing mode upgrades |

> **Speaker Notes:** "When you're 400 kilometers up, there's no SSH. So the system never fully stops — it degrades gracefully. Camera fails? Drop to NDVI-only. GPU hangs? Watchdog restarts the process. Ground control can push parameter changes through a database table — no OTA update needed. And after 5 minutes of clean operation, it heals itself and returns to full mode."

---

## SLIDE 10 — FARMER ADVISORY

### From Data to Decisions

```
══════════════════════════════════════════════════════════════
  CROP ADVISORY — Great Lawn B3
══════════════════════════════════════════════════════════════

  🔴 Significant Stress Detected
  NDVI of 0.291 is 28% BELOW the seasonal baseline of 0.405

  📊 NDVI: 0.2913  |  Baseline: 0.4046  |  Deviation: -28.0%

  🤖 DINOv2: healthy_vegetation (90.4% confidence, 6.7ms)

  🌤️  Weather: 6°C | 42% humidity | Clear

  💡 RECOMMENDATION:
     Continue monitoring. If decline persists for 3+ consecutive
     readings, investigate on-site.

  ⏱️  Adaptive: next scan in 568s, priority 0.43
══════════════════════════════════════════════════════════════
```

**6 fused data streams:** NDVI + seasonal baseline + weather + AI classification + confidence flags + trend history

> **Speaker Notes:** "This is what the farmer actually sees. Plain English. Not just 'NDVI is 0.29' — but why it matters, what the AI thinks, what the weather says, and what to do about it. It's a decision support tool, not just a classifier."

---

## SLIDE 11 — LIVE RESULTS

### Real Field Data — Central Park, March 2026

| Metric | Value |
|--------|-------|
| Radiometric pipeline | **109/109 tests passing** |
| DINOv2 validation accuracy | **93.94% (4-class)** |
| TensorRT inference | **6.7 ms / 149 FPS** |
| End-to-end pipeline | **< 4 seconds per zone** |
| Bandwidth reduction | **99.7%** (1.3 KB vs 500 KB) |
| Adaptive interval range | **2–30 minutes** |
| Power modes | **3.5W sleep → 10W active** |
| Fault tolerance modes | **4 (Full → Emergency)** |
| NASA HLS baseline | **5 years, 296 observations** |
| Field captures | **400+ multispectral pairs** |

> **Speaker Notes:** "This isn't a simulation. This is real field data from Central Park. 109 tests passing. 93.94% accuracy on four classes. Six point seven milliseconds. The full pipeline — capture, correct, classify, advise, schedule — completes in under 4 seconds."

🖼️ *Visual: Dashboard screenshot, NDVI heatmap, confusion matrix*

---

## SLIDE 12 — FROM GROUND TO ORBIT

### SmallSat Deployment Roadmap

```
     GROUND STATION          →          SMALLSAT (LEO)
     ──────────────                     ──────────────
     Jetson Orin Nano                   Rad-tolerant ARM/FPGA
     USB camera + EFW                   Integrated multispectral
     WiFi uplink                        S-band downlink
     Supabase cloud                     Onboard storage + burst
     10W power                          Solar + battery, 3.5W avg

     ✅ VALIDATED TODAY                 🚀 FLIGHT-READY PATH
     ─────────────────                 ─────────────────────
     • 7-layer correction              • Port to Vitis AI / FPGA
     • 4-class DINOv2 TRT              • Quantize for rad-hard
     • Adaptive scheduling             • S-band telemetry format
     • Fault tolerance                 • Radiation testing
     • Cloud VLM fallback              • Onboard VLM (3B)
```

> **Speaker Notes:** "Everything we've built runs on flight-equivalent hardware at flight-equivalent power budgets. The Jetson's Cortex-A78 and Ampere GPU are architecturally identical to what you'd find on a rad-tolerant SmallSat processor. The pipeline ports directly. DINOv2 can be quantized for FPGA inference. And the fault tolerance system already handles the kind of failures you'd see from radiation-induced single-event upsets."

---

## SLIDE 13 — CLOSING

### Why Chroma-Shift?

| | Traditional | Chroma-Shift |
|--|------------|-------------|
| **Revisit** | 5–16 days | 2–30 minutes |
| **Processing** | Cloud batch, days | On-device, 4 seconds |
| **AI** | None or basic | 4-class DINOv2 + 90B VLM |
| **Calibration** | Ground panels | Self-calibrating (no panels) |
| **Cost** | $10K+ per satellite image | $300 hardware |
| **Fault handling** | Manual recovery | Autonomous 4-mode degradation |
| **Output** | Raw imagery | Farmer-ready advisory |

### Key Numbers

> **149 FPS** · **93.94%** · **109 tests** · **8 signals** · **4 modes** · **<4s end-to-end** · **99.7% bandwidth reduction**

**Working code. Real field data. Flight-equivalent hardware.**

> **Speaker Notes:** "Most proposals are PowerPoints. We have a running prototype on flight-equivalent hardware with 109 tests passing, real field data from Central Park, and a complete edge-to-cloud pipeline validated end-to-end. Thank you."

---

## Q&A BACKUP SLIDES

### Q: "How does adaptive sensing reduce costs?"
99.7% bandwidth reduction — 1.3 KB telemetry vs 500 KB raw images. During stable periods, the system sleeps at 3.5W. Over a mission lifetime, this extends battery life by 40–60%.

### Q: "What about radiation tolerance?"
The algorithms are pure Python math. The pipeline ports directly to radiation-tolerant ARM/FPGA. DINOv2 can be quantized for FPGA via Vitis AI. The fault tolerance system handles single-event upsets — watchdog restarts, graceful degradation, auto-recovery.

### Q: "Why not just use commercial satellite data?"
Temporal resolution. Satellites revisit every 2–16 days. Our adaptive scheduler captures at 2–30 minute intervals. Stress events develop and become irreversible within 24–48 hours.

### Q: "What's your unfair advantage?"
Working code. 109 tests passing. 4-class DINOv2 at 149 FPS. Live cloud VLM. Confidence threshold cross-checking. Fault tolerance. NASA HLS calibration. Full pipeline validated end-to-end in 3.8 seconds on $300 of hardware.

### Q: "Why 93.94% and not 99%?"
Because we went from 2 classes to 4, and the model now distinguishes between stressed vegetation and bare soil — which look very similar spectrally. The confidence threshold system catches the uncertain cases and flags them. More field data will push accuracy past 97%.

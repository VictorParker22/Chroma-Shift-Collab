# Chroma-Shift — 3-Minute Video Script
## From Prototype to Space: Edge-to-Cloud Vegetation Intelligence

**Total runtime: 3:00**

---

## PRE-PRODUCTION NOTES

**Tone:** Confident, scientific, but accessible. Think NASA JPL press conference meets startup demo day.

**Visual style:** Clean cuts. Real hardware footage intercut with screen recordings of data flowing. No flashy transitions — let the tech speak.

**Music:** Subtle ambient/electronic underscore. Mute during all voiceover. Swell at the finale.

---

## SCENE 1 — THE HOOK (0:00 – 0:20)

### Visual
Open on a sweeping establishing shot of Central Park from above (drone footage or stock). Quick cut to a close-up of green leaves, then a slow push into a browning leaf edge — the first visible sign of stress.

### Voiceover
> *"By the time you can see a plant dying, it's already too late. Disease, drought, frost damage — they start at the molecular level, days before any human eye can detect them. But there's a wavelength of light that reveals the truth long before the damage is visible."*

### On-screen text
**CHROMA-SHIFT** — *Seeing What the Eye Cannot*

---

## SCENE 2 — THE SCIENCE (0:20 – 0:50)

### Visual
Split-screen animation: Left side shows a healthy green leaf. Right side shows the same leaf through a near-infrared filter — it glows bright white. Then show a stressed leaf: dimmer in NIR, brighter in Red. Overlay the NDVI formula.

### Voiceover
> *"Healthy plants absorb red light for photosynthesis but reflect near-infrared light from their internal cell structure. When a plant is stressed, this contrast collapses — it reflects more red, less infrared. The Normalized Difference Vegetation Index captures this ratio in a single number, from negative one to positive one."*

### Visual
Show the NASA HLS data — a graph of 296 satellite observations over 5 years, the NDVI baseline of 0.6493 for Central Park. Highlight the seasonal curve.

### Voiceover
> *"We calibrated our system against five years of NASA's Harmonized Landsat Sentinel-2 satellite data — 296 clear-sky observations over Central Park. This gives us a ground-truth baseline to measure against."*

### On-screen text
**296 satellite observations · 5-year NDVI baseline: 0.6493**

---

## SCENE 3 — THE HARDWARE (0:50 – 1:15)

### Visual
Close-up beauty shots of the actual hardware on a desk or mounted in the field:
1. The Jetson Orin Nano — close up of the board, NVIDIA logo visible
2. The Arducam multispectral camera
3. The ZWO filter wheel spinning between positions — you can see the filters click into place
4. The assembled system — camera + filter wheel + Jetson

### Voiceover
> *"The system runs on an NVIDIA Jetson Orin Nano — 8 gigabytes of memory, 1024 CUDA cores, and 32 Tensor Cores, all in a package that draws just 10 watts. An Arducam multispectral camera captures images through a motorized filter wheel with narrowband filters at 665 nanometers — the red absorption peak of chlorophyll — and 850 nanometers in the near-infrared."*

### Visual
Screen recording showing the capture sequence: filter wheel rotating, Red frame appearing, then NIR frame, then the NDVI heatmap calculated in real-time.

### Voiceover
> *"Every ten minutes, it captures a Red and NIR pair, corrects for every environmental variable, and computes a scientifically calibrated NDVI reading. No human intervention, no calibration panels, fully autonomous."*

### On-screen text
**665 nm Red · 850 nm NIR · 10-min cycles · 10W power**

---

## SCENE 4 — THE SELF-CALIBRATING PIPELINE (1:15 – 1:50)

### Visual
Animated diagram of the 7-layer pipeline. Each layer lights up as it's described. Show arrows flowing through the pipeline with sample numbers. This is the technical centerpiece — make it look clean and impressive.

```
RAW CAPTURE
    ↓
Layer 1: Auto-Exposure Normalization
    ↓
Layer 2: Dark Object Subtraction (Chavez 1988)
    ↓
Layer 3: Solar Zenith Correction (Spencer 1971)
    ↓
Layer 4: Cloud Transmittance (Kasten & Czeplak 1980)
    ↓
Layer 5: Inter-Frame PIF Matching ★ NOVEL
    ↓
Layer 6: Seasonal Phenology (NASA HLS Baseline)
    ↓
Layer 7: Kalman Temporal Filter
    ↓
CALIBRATED NDVI
```

### Voiceover
> *"The core innovation is a seven-layer self-calibrating radiometric correction pipeline. It corrects for camera exposure, atmospheric haze, solar angle, cloud cover, illumination changes between captures, seasonal vegetation cycles, and temporal noise — all without calibration panels. Layer five is particularly novel: it uses NDVI-based spectral flatness to detect concrete and stone surfaces as pseudo-invariant features, then matches illumination between the Red and NIR frames captured seconds apart. This is critical because a passing cloud during the three-second filter change would otherwise corrupt the entire NDVI calculation."*

### Visual
Show the test results: **109/109 TESTS PASSING** in green. Brief clip of test output scrolling.

### Voiceover
> *"The pipeline is validated against 109 automated tests covering every layer with edge cases, seasonal coverage, and known-input consistency checks."*

### On-screen text
**7-layer pipeline · Panel-free · 109/109 tests · Published methodologies**

---

## SCENE 5 — THE AI BRAIN (1:50 – 2:20)

### Visual
Animation showing the edge-to-cloud architecture. The Jetson on the left, NVIDIA cloud on the right, with data flowing between them.

```
    JETSON ORIN NANO                    NVIDIA CLOUD
  ┌───────────────────┐            ┌──────────────────┐
  │  Camera Capture   │            │  NVIDIA NIM API  │
  │  7-Layer Pipeline │            │                  │
  │  DINOv2 (8.7ms)  │───HTTPS───▶│  Llama 3.2 90B   │
  │  Kalman Filter    │◀──JSON────│  Vision (5.6s)   │
  │  Supabase Push    │            │                  │
  └───────────────────┘            └──────────────────┘
```

### Voiceover
> *"Two AI systems work together. On-device, a DINOv2 neural network — trained by Meta on 142 million images and optimized with NVIDIA TensorRT — classifies each frame as healthy or stressed vegetation in just 8.7 milliseconds. In the cloud, a 90-billion-parameter Llama Vision model hosted on NVIDIA's NIM platform analyzes the scene: it sees the multispectral image pair alongside weather data and solar position, then decides which correction layers to apply and how to adjust them. It returns structured JSON — vegetation percentage, haze detection, illumination quality, confidence scores — all feeding directly into the radiometric pipeline."*

### Visual
Show the actual cloud VLM response: the JSON output with 80% vegetation, 0.8 confidence. Show it feeding into the pipeline.

### Voiceover
> *"If the internet goes down, a rule-based heuristic takes over in under one millisecond. Zero downtime, graceful degradation — the system never stops monitoring."*

### On-screen text
**DINOv2 on-device: 8.7ms · Llama 90B cloud: 5.6s · Heuristic fallback: <1ms**

---

## SCENE 6 — THE DATA (2:20 – 2:35)

### Visual
Screen recording of the Supabase dashboard showing real-time NDVI data flowing in. Show the dashboard with charts, stress alerts, weather correlations. Make it look alive with data updating.

### Voiceover
> *"Every reading — NDVI values, AI classifications, weather correlations, VLM scene analyses — flows to a Supabase cloud database in real-time. The dashboard correlates vegetation stress with temperature, humidity, rainfall, and cloud cover from the National Weather Service. When stress exceeds the seasonal threshold, the system generates automated alerts with specific recommendations."*

### On-screen text
**Real-time telemetry · Weather correlation · Automated alerts**

---

## SCENE 7 — FROM PROTOTYPE TO SPACE (2:35 – 2:55)

### Visual
Start with the prototype on the desk. Dissolve to a rendering or map showing multiple stations across a city. Then a globe view showing stations across agricultural regions. Finally, show a satellite in orbit with ground stations feeding up validation data.

### Voiceover
> *"Today, Chroma-Shift monitors one park. But the architecture is designed to scale. A network of autonomous stations — self-calibrating, weather-aware, AI-driven — could blanket agricultural regions, reforestation sites, or urban forests with continuous ground-truth data. This data doesn't just stay on Earth. Each station's NDVI readings, calibrated against the same NASA HLS baseline that satellite sensors use, become ground-truth validation points for next-generation orbital instruments. The same technology that monitors a single tree in Central Park could help calibrate the sensors that monitor every forest on the planet."*

### On-screen text
**One station → City network → Global deployment → Satellite validation**

---

## SCENE 8 — CLOSING (2:55 – 3:00)

### Visual
Final shot: the Jetson system in the field, LED glowing, camera pointed at vegetation. Fade to the Chroma-Shift logo.

### Voiceover
> *"Chroma-Shift. Seeing what the eye cannot."*

### On-screen text
```
CHROMA-SHIFT
Self-Calibrating Multispectral Vegetation Intelligence
NVIDIA Jetson Orin Nano · DINOv2 · NVIDIA NIM · NASA HLS
github.com/VictorParker22/Chroma-Shift
```

---

## TIMING SUMMARY

| Scene | Duration | Cumulative | Content |
|-------|----------|------------|---------|
| 1 - Hook | 20s | 0:20 | Problem statement, emotional hook |
| 2 - Science | 30s | 0:50 | NDVI physics, satellite baseline |
| 3 - Hardware | 25s | 1:15 | Jetson, camera, filter wheel demo |
| 4 - Pipeline | 35s | 1:50 | 7-layer correction, validation |
| 5 - AI Brain | 30s | 2:20 | DINOv2 + cloud VLM architecture |
| 6 - Data | 15s | 2:35 | Dashboard, weather, alerts |
| 7 - Vision | 20s | 2:55 | Scale from prototype to space |
| 8 - Close | 5s | 3:00 | Logo, tagline, GitHub |

---

## B-ROLL SHOT LIST

Capture these specific shots for the video:

1. **Central Park establishing** — wide shot showing trees/grass (phone or drone)
2. **Leaf close-up** — healthy green leaf, then browning/stressed leaf
3. **Jetson hero shot** — board on desk, clean background, NVIDIA logo visible
4. **Camera + filter wheel** — close-up of the assembled imaging system
5. **Filter wheel spinning** — slow-motion or close-up of filters clicking into position
6. **Screen: capture sequence** — terminal showing Red/NIR capture in real-time
7. **Screen: test results** — 109/109 TESTS PASSED in terminal
8. **Screen: VLM response** — JSON output from NVIDIA NIM
9. **Screen: dashboard** — Supabase dashboard with live NDVI charts
10. **System in field** — the complete system outdoors near vegetation
11. **Night shot** (optional) — LED glow indicating autonomous operation

---

## KEY TALKING POINTS FOR JUDGES

If there's a Q&A section, be prepared to discuss:

1. **"Why not just use a satellite?"** — Temporal resolution. Satellites revisit every 2-16 days. We capture every 10 minutes. Stress can develop and cause irreversible damage within 48 hours.

2. **"Why not use calibration panels?"** — Panels require maintenance, can be vandalized or weathered, and don't scale. Our self-calibrating pipeline uses physics + AI to achieve the same goal autonomously.

3. **"How accurate is it compared to satellite?"** — We calibrate against the same HLS baseline. Our NDVI readings are directly comparable, and our pipeline is validated against 109 automated tests.

4. **"What about the cloud dependency?"** — The heuristic fallback activates in under 1ms. DINOv2 runs entirely on-device. The system never stops monitoring, even offline.

5. **"How does this scale?"** — Each station is fully autonomous. Add a station, plug it in, it self-calibrates. The cloud VLM and Supabase can handle thousands of stations.

6. **"What's novel?"** — NDVI-based PIF detection (Layer 5), the 4-mode VLM priority chain, and the concept of using a 90B cloud VLM to orchestrate a physics-based correction pipeline on an edge device.

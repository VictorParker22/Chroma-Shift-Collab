# Chroma-Shift — 3-Minute Video Script
## Adaptive SmallSat Intelligence for Regenerative Agriculture

**Total runtime: 3:00**

> **⚠️ CRITICAL RULE**: A team member must be **visible on camera** throughout the video. AI-generated video content is NOT allowed. Use slides, props, prototypes, and screen recordings as visual aids while you present.

---

## PRE-PRODUCTION NOTES

**Tone:** Confident, passionate, clear. Think NASA pitch day — substance over style.

**Setup:** You (Victor) presenting to camera with the hardware prototype visible. Cut to screen recordings and slides as visual aids. You should be the primary visual throughout.

**Music:** None or very subtle ambient. Your voice is the focus.

---

## SCENE 1 — THE PROBLEM (0:00 – 0:25)

### [On camera: Victor, speaking to camera with hardware on desk]

> *"Every two weeks, a satellite flies over a farm and takes a picture. But crops don't wait two weeks. A frost event, a disease outbreak, a failed irrigation line — by the time the next satellite image arrives, the damage is done. The real question isn't whether we can observe from space — it's whether we can observe SMART from space."*

> *"I'm Victor, and Chroma-Shift is an onboard processing system that turns a SmallSat from a passive camera into an adaptive observer — one that decides what to look at, when to look, and what to tell the ground."*

### On-screen text
**CHROMA-SHIFT** — *Adaptive SmallSat Intelligence for Regenerative Agriculture*

---

## SCENE 2 — THE SCIENCE (0:25 – 0:50)

### [Cut to slide showing Red vs NIR reflectance diagram, then NDVI formula]

> *"The physics is straightforward. Healthy plants absorb red light and reflect near-infrared. Stressed plants do the opposite. The ratio between these two bands — the NDVI — has been the standard vegetation index since 1974."*

### [Cut to slide showing NASA HLS data — 296 observations graph]

> *"We calibrated our system against five years of NASA's Harmonized Landsat Sentinel-2 data — 296 clear-sky observations over Central Park, New York. This gives us a ground-truth seasonal baseline that's directly comparable to satellite measurements."*

### [Hold up the filter wheel or point to it on the desk]

> *"Our camera captures through narrowband filters at 665 and 850 nanometers — the same bands Sentinel-2 uses."*

---

## SCENE 3 — THE HARDWARE (0:50 – 1:10)

### [Pick up the Jetson, show it to camera]

> *"Everything runs on this — an NVIDIA Jetson Orin Nano. Ten watts, eight gigabytes of RAM, a thousand CUDA cores. This has nearly identical specs to satellite-grade computers like the Xiphos Q8S. If it runs here, it can fly."*

### [Show the complete system: camera + filter wheel + Jetson]

> *"Camera, filter wheel, edge processor — assembled for under a thousand dollars. The entire pipeline — capture, correction, classification, and adaptive scheduling — runs on this hardware."*

---

## SCENE 4 — THE SELF-CALIBRATING PIPELINE (1:10 – 1:40)

### [Cut to slide showing 7-layer pipeline diagram]

> *"Here's what makes this different. Satellite missions use ground calibration panels and atmospheric models that don't work for autonomous, continuous monitoring. We built a seven-layer self-calibrating pipeline that corrects for everything — exposure drift, atmospheric haze, solar angle, cloud cover, illumination changes between frames, seasonal vegetation cycles, and measurement noise."*

### [Cut to slide: "109/109 TESTS PASSING"]

> *"Every layer is validated. 109 automated tests covering edge cases, seasonal coverage, and cross-layer interactions. We found and fixed bugs that wouldn't show up in simple testing — floating-point overflow, pseudo-invariant feature misdetection, shadow pixel artifacts."*

> *"Layer five is our novel contribution: using NDVI-based spectral flatness to detect invariant surfaces. Traditional methods use brightness, which fails when vegetation is the brightest thing in the scene."*

---

## SCENE 5 — ADAPTIVE SENSING (1:40 – 2:10)

### [Cut to slide showing adaptive scheduling diagram]

> *"Current satellites observe on fixed schedules. Chroma-Shift observes on demand. Seven signals drive the scheduling — NDVI stress, scene change rate, weather stability, solar position, AI confidence, illumination variability, and power state."*

### [Cut to screen recording showing adaptive scheduler output]

> *"During a frost event, the system drops to two-minute intervals and ranks the most stressed zones first. During stable growth, it extends to thirty minutes and enters sleep mode at 3.5 watts. This means the satellite observes MORE when it matters and conserves power when it doesn't."*

> *"And here's the bandwidth story: instead of downlinking a 500-kilobyte image, we downlink 1.3 kilobytes of processed telemetry. That's a 99.7% reduction. One S-band pass can carry thousands of observations instead of a handful of images."*

---

## SCENE 6 — EDGE-TO-CLOUD AI (2:10 – 2:35)

### [Cut to slide showing DINOv2 on-device + cloud VLM architecture]

> *"Two AI systems work together. On-device, a DINOv2 classifier runs at 8.7 milliseconds per frame — fast enough for real-time processing. In the cloud — or on a ground station — a 90-billion-parameter Vision Language Model analyzes the scene and decides which correction layers to apply. If the radio link is down, a rule-based heuristic takes over in under one millisecond. Zero downtime."*

### [Cut to screen recording of live VLM response: JSON output]

> *"The VLM sees the actual multispectral image pair plus weather data, and returns structured JSON — vegetation percentage, haze detection, illumination quality, correction overrides. Physics provides the foundation; AI provides the orchestration."*

---

## SCENE 7 — FROM GROUND TO ORBIT (2:35 – 2:52)

### [Back on camera: Victor, speaking directly to camera]

> *"Right now, this is a ground-level testbed producing real NDVI data in Central Park. But the algorithms, the power budget, and the compute requirements are identical to what would run on a 6U CubeSat. The path from here to orbit is thermal-vacuum qualification, radiation-tolerant ARM porting, and payload integration — engineering steps, not research unknowns."*

> *"And the market is ready. Precision agriculture is an 8.5 billion dollar market. But this framework isn't limited to agriculture — wildfire monitoring, deforestation detection, carbon credit verification, flood mapping — any application that needs adaptive, intelligent observation from space."*

---

## SCENE 8 — CLOSING (2:52 – 3:00)

### [Victor, holding the Jetson + camera prototype]

> *"The question isn't whether we can observe from space. It's whether we can observe smart. Chroma-Shift is the onboard intelligence that makes that happen."*

### On-screen text
```
CHROMA-SHIFT
Adaptive Onboard Processing for Regenerative Agriculture SmallSats
NVIDIA Jetson · DINOv2 · NASA HLS · Self-Calibrating Pipeline
github.com/VictorParker22/Chroma-Shift
```

---

## TIMING SUMMARY

| Scene | Duration | Cumulative | Content |
|-------|----------|------------|---------|
| 1 - Problem | 25s | 0:25 | Fixed schedules fail, intro yourself |
| 2 - Science | 25s | 0:50 | NDVI physics, NASA HLS calibration |
| 3 - Hardware | 20s | 1:10 | Jetson = SmallSat proxy, show hardware |
| 4 - Pipeline | 30s | 1:40 | 7-layer correction, 109 tests, Layer 5 novelty |
| 5 - Adaptive | 30s | 2:10 | Smart scheduling, bandwidth reduction |
| 6 - AI | 25s | 2:35 | DINOv2 on-device + cloud VLM |
| 7 - Vision | 17s | 2:52 | Ground to orbit path, market |
| 8 - Close | 8s | 3:00 | Call to action, GitHub |

---

## SLIDES TO PREPARE

1. **Title slide**: "Chroma-Shift — Adaptive SmallSat Intelligence for Regenerative Agriculture"
2. **Red vs NIR diagram**: Healthy leaf showing absorption/reflection
3. **NASA HLS graph**: 296 observations, seasonal NDVI curve, 0.6493 mean
4. **7-layer pipeline diagram**: Flow chart with layer names
5. **Test results**: "109/109 Assertions Passing" screenshot
6. **Adaptive scheduling diagram**: Signals → scheduler → interval/priority
7. **Adaptive scheduler output**: Terminal screenshot showing 4 scenarios
8. **Bandwidth comparison**: Table — raw image (500KB) vs telemetry (1.3KB)
9. **Edge-to-cloud architecture**: Jetson ←→ NVIDIA NIM diagram
10. **VLM JSON response**: Screenshot of actual cloud analysis output
11. **SWAP comparison**: Jetson vs Xiphos Q8S vs OrbitsEdge table
12. **TRL roadmap**: Phase 1–4 timeline (TRL 4 → 7)
13. **Market**: $8.5B precision ag, 5 beyond-agriculture applications

---

## B-ROLL TO CAPTURE

1. **You presenting** — multiple angles, clear audio, good lighting
2. **Hardware hero shots** — Jetson, camera, filter wheel close-ups
3. **Filter wheel spinning** — show the physical mechanism
4. **Screen: pipeline test results** — 109/109 passing in terminal
5. **Screen: adaptive scheduler** — 4 scenarios running
6. **Screen: VLM response** — live JSON from NVIDIA NIM
7. **Screen: dashboard** — Supabase with live data

---

## JUDGE Q&A PREPARATION

1. **"This is a ground station, not a satellite."**
   → "The Jetson Orin Nano has nearly identical specs to the Xiphos Q8S flight computer. Everything we demonstrated — the pipeline, the adaptive scheduler, the AI — runs within SmallSat power and compute constraints. The path to orbit is engineering integration, not algorithmic research."

2. **"How does adaptive sensing actually reduce costs?"**
   → "99.7% bandwidth reduction — 1.3 KB telemetry vs 500 KB raw images. During stable periods, the system sleeps at 3.5W. Over a mission lifetime, this extends battery life by 40-60% compared to fixed-schedule observation."

3. **"What about radiation tolerance?"**
   → "The algorithms are pure Python math with no GPU-specific dependencies except DINOv2 inference. The pipeline ports directly to radiation-tolerant ARM/FPGA platforms. DINOv2 can be quantized for FPGA inference using Vitis AI."

4. **"Why not just use existing satellite data?"**
   → "Temporal resolution. Satellites revisit every 2-16 days. Our adaptive scheduler captures at 2-30 minute intervals. Stress events develop and become irreversible within 24-48 hours."

5. **"What's your unfair advantage?"**
   → "Working code. 109 tests passing. Live cloud VLM integration. NASA HLS calibration. Most proposals are PowerPoints — we have a running prototype on flight-equivalent hardware."

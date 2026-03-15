# Chroma-Shift — Improvement Tickets

> Source: [SCIENTIFIC_AUDIT.md](file:///home/chroma-shift/Documents/Chroma-Shift/SCIENTIFIC_AUDIT.md)
> Sprint goal: Raise pipeline score from **6.8 → 8.0+** before NASA challenge deadline

---

## Team

| Member | Role | Strengths |
|--------|------|-----------|
| **Victor** | Lead / Hardware | Jetson, cameras, filter wheel, system integration |
| **Laurel** | Botanist | Plant physiology, phenology, stress classification, ground truth |
| **Joshua** | AI Scientist | ML models, DINOv2, foundation models, statistical methods |
| **AI Agent** | Engineering | Code implementation, testing, deployment, DevOps |

---

## 🔴 Sprint 1 — Quick Wins (Est. 1-2 days)

### CS-001: Replace Spencer 1971 with NREL SPA
- **Priority**: High | **Layer**: L3 | **Score impact**: 7→9
- **Assignee**: AI Agent
- **Effort**: ~50 lines Python, pure math
- **What**: Replace `solar_position()` in `radiometric.py` with NREL Solar Position Algorithm (Reda & Andreas, 2004). Currently ±0.28° error → ±0.0003° accuracy, 1000× improvement.
- **Why**: Spencer 1971 causes up to 0.5% NDVI error at low solar angles. NREL SPA is equally simple — no external dependencies.
- **Acceptance criteria**:
  - [ ] Solar position matches NOAA calculator to <0.01° for NYC
  - [ ] All 109 pipeline tests still pass
  - [ ] No new dependencies added

---

### CS-002: Calibrate Cloud Transmittance for NYC
- **Priority**: High | **Layer**: L4 | **Score impact**: 4→7
- **Assignee**: Joshua + AI Agent
- **Effort**: 1-2 hours research + 1 line code change
- **What**: Replace Hamburg k=0.75 coefficient with NYC-calibrated value using NASA CERES surface shortwave radiation data.
- **Why**: Kasten & Czeplak (1980) coefficients were fit in Hamburg. Literature shows MBE of -28 to -62 W/m² without local calibration. We're **systematically overcorrecting** cloudy captures, inflating NDVI.
- **Steps**:
  1. Download NASA CERES monthly GHI data for NYC (40.78°N, -73.97°W)
  2. Fit k coefficient: T = 1 - k × (C/100)² against observed vs clear-sky GHI
  3. Optionally fit k per season (winter vs summer cloud types differ)
- **Acceptance criteria**:
  - [ ] k value justified with CERES data source citation
  - [ ] Seasonal k values documented (if significantly different)
  - [ ] Cloud correction test cases updated

---

### CS-003: Add Phenology Interpolation
- **Priority**: Medium | **Layer**: L6 | **Score impact**: 7→8
- **Assignee**: AI Agent
- **Effort**: ~3 lines of code
- **What**: Replace staircase lookup with linear interpolation between weeks in `get_seasonal_baseline()`.
- **Why**: Currently NDVI baseline jumps 0.03 overnight at week boundaries, potentially triggering false stress alerts.
- **Acceptance criteria**:
  - [ ] Smooth baseline across week boundaries
  - [ ] `get_seasonal_baseline()` accepts fractional day-of-year
  - [ ] No more than ±0.001 NDVI difference from lookup at mid-week

---

### CS-004: Scale Kalman Q by Time Delta
- **Priority**: Medium | **Layer**: L7 | **Score impact**: 7→8
- **Assignee**: AI Agent
- **Effort**: ~5 lines of code
- **What**: Make process noise adaptive: `Q_eff = Q_base × (dt_seconds / 600)` in `KalmanNDVI.update()`.
- **Why**: The adaptive sensing scheduler now varies intervals from 2-30 minutes. Fixed Q assumes constant 10-min intervals. Longer gaps should allow more process noise; shorter gaps should be smoother.
- **Acceptance criteria**:
  - [ ] `update()` accepts optional `dt` parameter
  - [ ] Default behavior unchanged when dt=600s (backward compatible)
  - [ ] 2-min interval produces smoother output than 30-min interval

---

## 🟡 Sprint 2 — Scientific Rigor (Est. 1 week)

### CS-005: Upgrade DOS to COST
- **Priority**: Medium | **Layer**: L2 | **Score impact**: 5→7
- **Assignee**: AI Agent
- **Effort**: ~20 lines
- **What**: Upgrade from DOS (Chavez 1988) to COST (Chavez 1996). COST adds cos(θ) multiplicative correction on top of DOS.
- **Why**: DOS only handles additive haze. COST also compensates for path radiance attenuation, giving better results at no compute cost (we already have cos(θ) from L3).
- **Reference**: Chavez, P.S. (1996). Image-based atmospheric corrections — revisited and improved. *PE&RS*, 62(9).
- **Acceptance criteria**:
  - [ ] Dark object model accounts for cos(solar_zenith)
  - [ ] NDVI values more stable across different solar angles
  - [ ] Literature reference cited in code comments

---

### CS-006: Validate Phenology Curve with In-Situ Data
- **Priority**: High | **Layer**: L6
- **Assignee**: 🌿 Laurel
- **Effort**: Field work + analysis
- **What**: Ground-truth the 52-week NDVI phenology curve against actual Central Park vegetation.
- **Why**: Our curve is from NASA HLS satellite pixel-level NDVI, which averages vegetation + paths + buildings. Laurel's expertise can validate whether the weekly baselines match actual plant phenology stages.
- **Steps**:
  1. Identify 3-5 monitoring zones in Central Park (pure grass, mixed, tree canopy)
  2. Record weekly visual assessments: green-up date, peak, senescence
  3. Cross-reference with our NDVI seasonal curve
  4. Adjust weekly values where satellite average diverges from ground-level camera view
- **Acceptance criteria**:
  - [ ] 3+ zones validated against phenology curve
  - [ ] Documented green-up and senescence dates for 2026
  - [ ] Adjusted phenology table if satellite-vs-ground discrepancies found

---

### CS-007: Expand Classification Beyond Binary
- **Priority**: High | **Layer**: Classifier
- **Assignee**: 🌿 Laurel + Joshua
- **Effort**: Multi-session
- **What**: Design a 4-6 class taxonomy beyond "healthy" vs "stressed":
  - Healthy vegetation (active growth)
  - Water stress (wilting, curling)
  - Nutrient deficiency (chlorosis, yellowing)
  - Disease/pest damage
  - Bare soil / mineral
  - Shadow / low-signal
- **Why**: Binary classification is too coarse for agronomic value. Distinguishing stress types enables actionable recommendations.
- **Steps**:
  1. **Laurel**: Define class taxonomy with visual identification keys
  2. **Laurel**: Label 200+ captured images into new classes
  3. **Joshua**: Retrain DINOv2 with multi-class head
  4. **Joshua**: Evaluate per-class accuracy and confusion matrix
- **Acceptance criteria**:
  - [ ] Taxonomy documented with example images per class
  - [ ] ≥200 labeled images per class
  - [ ] Multi-class accuracy ≥85%

---

### CS-008: Calibrate DINOv2 Confidence Scores
- **Priority**: Medium | **Layer**: Classifier
- **Assignee**: Joshua
- **Effort**: ~2 hours
- **What**: Apply temperature scaling (Guo et al., 2017) to calibrate softmax outputs as true probabilities.
- **Why**: The current softmax output (e.g., "94% confidence") is NOT a calibrated probability. A 94% softmax score might correspond to only 85% actual accuracy. NASA reviewers will notice this.
- **Steps**:
  1. Compute Expected Calibration Error (ECE) on val set
  2. Fit temperature parameter T using NLL minimization on val set
  3. Apply `softmax(logits / T)` at inference time
- **Acceptance criteria**:
  - [ ] ECE < 0.05 (well-calibrated)
  - [ ] Reliability diagram generated and documented
  - [ ] Temperature parameter saved with model checkpoint

---

### CS-009: Adaptive PIF Threshold
- **Priority**: Low | **Layer**: L5
- **Assignee**: AI Agent
- **Effort**: ~10 lines
- **What**: Make the |NDVI| < 0.15 threshold adaptive: widen to 0.20 if too few PIFs, narrow to 0.10 if too many.
- **Why**: Dense vegetation scenes may have zero concrete/asphalt pixels, causing the current code to skip L5 entirely.
- **Acceptance criteria**:
  - [ ] At least 200 PIFs found in 95% of captures
  - [ ] Threshold adjustment logged in correction metadata

---

## 🟢 Sprint 3 — Flight Preparation (Est. 2-4 weeks)

### CS-010: Simplified 6S Lookup Table
- **Priority**: Low | **Layer**: L2
- **Assignee**: Joshua + AI Agent
- **What**: Pre-compute atmospheric correction factors for 665nm and 850nm bands across AOD 0.05–1.0 and solar zenith 0°–80°. Store as ~100-entry lookup table. Use NWS visibility as AOD proxy.
- **Acceptance criteria**: RMSE vs full 6S < 2% reflectance

### CS-011: BRDF Correction (Ross-Li Model)
- **Priority**: Low | **Layer**: L3
- **Assignee**: Joshua
- **What**: Add bidirectional reflectance correction using published Ross-Li parameters for grassland and mixed vegetation.
- **Acceptance criteria**: NDVI variance reduced under varying solar angles

### CS-012: Spectral Foundation Model Evaluation
- **Priority**: Low | **Layer**: Classifier
- **Assignee**: Joshua
- **What**: Benchmark DINOv2 against SatMAE, Prithvi, or SkySense on our dataset. These models natively understand spectral bands.
- **Acceptance criteria**: Side-by-side accuracy comparison on same val set

### CS-013: Global Phenology Parameterization
- **Priority**: Low | **Layer**: L6
- **Assignee**: 🌿 Laurel + AI Agent
- **What**: Parameterize the phenology curve by latitude band using NASA HLS global data, enabling deployment beyond NYC.
- **Acceptance criteria**: Phenology curves generated for 5+ test latitudes

---

## Ticket Summary

| Sprint | Tickets | Total Effort | Score Impact |
|--------|---------|--------------|-------------|
| **Sprint 1** | CS-001 to CS-004 | 1-2 days | 6.8 → **8.0** |
| **Sprint 2** | CS-005 to CS-009 | ~1 week | 8.0 → **8.8** |
| **Sprint 3** | CS-010 to CS-013 | 2-4 weeks | 8.8 → **9.2+** |

---

## Assignment Matrix

| Member | Sprint 1 | Sprint 2 | Sprint 3 |
|--------|----------|----------|----------|
| **Victor** | — (focus on video/demo) | Integration testing | Hardware validation |
| **Laurel** | — | CS-006, CS-007 | CS-013 |
| **Joshua** | CS-002 | CS-007, CS-008 | CS-010, CS-011, CS-012 |
| **AI Agent** | CS-001, CS-003, CS-004 | CS-005, CS-009 | CS-010, CS-013 |

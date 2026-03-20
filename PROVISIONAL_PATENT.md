# PROVISIONAL PATENT APPLICATION

> ⚠️ **DISCLAIMER**: This document is a DRAFT prepared by an AI assistant. It is NOT legal advice. Before filing, you MUST have this reviewed by a registered patent attorney or agent. Filing a provisional patent without professional review is risky — a poorly written provisional can actually hurt your later patent claims.

---

## COVER SHEET INFORMATION

**Title of Invention:**
Real-Time Inter-Frame Illumination Correction for Sequential-Filter Multispectral Imaging Using Normalized Difference Vegetation Index Spectral Flatness for Pseudo-Invariant Feature Detection

**Inventor(s):**
Victor Parker
[Your Full Legal Address]
[City, State, ZIP]
[Citizenship]

**Correspondence Address:**
[Same as above or attorney address]

**Filing Entity Status:** ☐ Micro Entity ($80) ☐ Small Entity ($160) ☐ Large Entity ($320)

> **Note:** Most solo inventors qualify as **Micro Entity** ($80 filing fee). Requirements: (1) not named as inventor on more than 4 previous US patent applications, (2) gross income did not exceed 3× US median household income (~$225,000) in the prior year, (3) not obligated to assign the invention to an entity that exceeds either limit.

**Filing Date:** _____________ (gives you 12 months to file a non-provisional)

---

## FIELD OF THE INVENTION

This invention relates to methods and systems for radiometric correction of multispectral imagery acquired by sequential-filter imaging systems, and more particularly to a real-time method for detecting and correcting inter-frame illumination changes using spectral index-derived pseudo-invariant features.

---

## BACKGROUND OF THE INVENTION

### The Problem of Sequential-Filter Capture

Multispectral imaging systems used in precision agriculture, environmental monitoring, and remote sensing require images captured at multiple discrete wavelength bands — typically at least one visible band (e.g., Red, ~665 nm) and one near-infrared (NIR, ~850 nm) band. These paired images are used to compute vegetation indices such as the Normalized Difference Vegetation Index (NDVI), which quantifies photosynthetic activity:

```
NDVI = (ρ_NIR − ρ_Red) / (ρ_NIR + ρ_Red)
```

Where ρ represents surface reflectance in each band.

Many cost-effective and embedded multispectral imaging systems use a **single sensor paired with a rotating filter wheel** (e.g., ZWO Electronic Filter Wheel) rather than a sensor with physically separate band-pass detector arrays (as found on satellite instruments like Sentinel-2 MSI). In such systems, the Red and NIR frames are captured **sequentially**, with a temporal gap of typically 3–10 seconds between acquisitions as the filter wheel mechanically rotates and stabilizes.

During this temporal gap, ambient illumination conditions may change — most commonly due to cloud transit across the solar disk. If a cloud shadow falls on the scene between the Red and NIR captures, or if a cloud clears during the gap, the two frames will record different irradiance levels for reasons unrelated to surface reflectance. Because NDVI is a ratio of these two bands, such illumination mismatches directly corrupt the computed vegetation index, producing erroneous readings that can trigger false stress alerts or mask genuine vegetation decline.

### Limitations of Prior Art

Several methods for identifying pseudo-invariant features (PIFs) — surfaces with temporally stable reflectance properties — have been described in the remote sensing literature:

1. **Principal Component Analysis (PCA)** and **Multivariate Alteration Detection (MAD)** (Canty, 2004): These statistical methods identify unchanged pixels across a multi-temporal stack. They require multiple historical observations of the same scene, making them unsuitable for real-time, single-pair correction.

2. **Brightness-based thresholding**: Selecting the brightest or darkest pixels in a scene as PIF candidates. This approach fails in vegetation-dominated scenes (e.g., urban parks, agricultural fields) where the brightest pixels may be healthy canopy rather than spectrally invariant surfaces.

3. **Manual PIF designation**: Marking known invariant surfaces (e.g., painted calibration panels, specific rooftops) by GPS coordinates. This requires human presence at the deployment site and does not generalize to autonomous or orbital platforms.

4. **Calibration panels**: Placing physical reflectance targets (e.g., Spectralon panels) in the camera's field of view. This is impractical for autonomous field stations, SmallSat platforms, or drone-based systems operating over large or inaccessible areas.

None of these methods provide **real-time, single-frame-pair, autonomous PIF detection** suitable for embedded edge computing on resource-constrained hardware (e.g., NVIDIA Jetson, 8 GB RAM, 10W power envelope).

---

## SUMMARY OF THE INVENTION

The present invention provides a method and system for real-time correction of inter-frame illumination changes in sequential-filter multispectral imaging systems. The method uses the **spectral flatness** of the Normalized Difference Vegetation Index (NDVI) itself as the criterion for identifying pseudo-invariant features — surfaces suitable for inter-frame brightness calibration.

The key insight is that surfaces where the absolute value of NDVI is below a threshold (e.g., |NDVI| < 0.15) exhibit approximately equal reflectance in the Red and NIR bands. Such spectral flatness is a physical property of non-vegetated surfaces — concrete, asphalt, stone, bare dry soil, metal structures — whose reflectance is determined by mineral composition rather than biological processes. These surfaces do not change reflectance on the timescale of seconds to minutes, making them reliable inter-frame calibration references.

Unlike prior art PIF detection methods, this approach:

1. **Requires no historical data** — operates on a single Red/NIR frame pair
2. **Requires no external calibration targets** — uses surfaces naturally present in the scene
3. **Is physically motivated** — spectral flatness (|NDVI| < threshold) directly identifies non-biological surfaces by their defining spectral property
4. **Executes in real time** — computable in under 1 millisecond on embedded hardware
5. **Is self-validating** — the same index being corrected (NDVI) provides the PIF identification criterion, creating a closed-loop correction system

---

## DETAILED DESCRIPTION OF THE INVENTION

### System Configuration

The method operates within a multispectral imaging system comprising:

- A digital imaging sensor capable of capturing frames in at least two spectral bands (Red and NIR)
- A band-selection mechanism that introduces a temporal gap between band captures (e.g., rotating filter wheel, tunable filter, sequential LED illumination)
- A computing device capable of performing per-pixel arithmetic operations on the captured frames (e.g., embedded GPU, FPGA, CPU)

### Method Steps

The inter-frame illumination correction method comprises the following steps:

**Step 1 — Frame Acquisition:**
Capture a first frame (F_Red) through a Red bandpass filter (center wavelength ~640–680 nm) and a second frame (F_NIR) through a near-infrared bandpass filter (center wavelength ~800–900 nm), with a temporal gap Δt between captures.

**Step 2 — Preliminary NDVI Computation:**
Compute a preliminary per-pixel NDVI image from the raw (uncorrected) frames:

```
NDVI_prelim(x,y) = (F_NIR(x,y) − F_Red(x,y)) / (F_NIR(x,y) + F_Red(x,y) + ε)
```

Where ε is a small constant (e.g., 1×10⁻¹⁰) to prevent division by zero.

**Step 3 — PIF Mask Generation:**
Generate a binary mask M identifying pseudo-invariant feature pixels:

```
M(x,y) = 1  if |NDVI_prelim(x,y)| < T_flat
M(x,y) = 0  otherwise
```

Where T_flat is the spectral flatness threshold (preferred range: 0.10–0.20; default: 0.15). Pixels where |NDVI| < T_flat exhibit approximately equal Red and NIR reflectance, indicating a non-vegetated, spectrally stable surface.

**Step 4 — PIF Count Validation:**
Count the number of PIF pixels: N_PIF = Σ M(x,y).

If N_PIF is below a minimum threshold (e.g., 100 pixels, or 0.1% of total frame pixels), the correction step is skipped and the frames are passed through uncorrected. This prevents amplification of noise when insufficient calibration surfaces are present (e.g., dense forest canopy with no visible paths or structures).

**Step 5 — Inter-Frame Brightness Comparison:**
Compute the mean brightness of PIF pixels in each frame:

```
μ_Red = mean(F_Red(x,y) where M(x,y) = 1)
μ_NIR = mean(F_NIR(x,y) where M(x,y) = 1)
```

**Step 6 — Illumination Mismatch Detection:**
Compute the brightness ratio: R = μ_NIR / μ_Red.

If |R − 1.0| > T_mismatch (e.g., T_mismatch = 0.10, representing a 10% brightness difference), an illumination change is detected.

**Step 7 — Adaptive Noise Floor:**
To prevent amplification of sensor noise in low-signal conditions, a minimum brightness floor is enforced:

```
If min(μ_Red, μ_NIR) < T_noise_floor:
    Skip correction (signal too low for reliable matching)
```

Where T_noise_floor is set based on the sensor's known noise characteristics (e.g., 5 DN for 8-bit sensors, 20 DN for 16-bit sensors).

**Step 8 — Frame Scaling:**
If an illumination mismatch is detected and passes the noise floor check, scale the dimmer frame to match the brighter frame:

```
If μ_Red < μ_NIR:
    F_Red_corrected = F_Red × (μ_NIR / μ_Red)
    F_NIR_corrected = F_NIR
Else:
    F_NIR_corrected = F_NIR × (μ_Red / μ_NIR)
    F_Red_corrected = F_Red
```

This applies a uniform multiplicative correction across the entire frame, matching the global illumination level between captures.

**Step 9 — Corrected NDVI Computation:**
Compute the final NDVI from the corrected frames:

```
NDVI_final(x,y) = (F_NIR_corrected(x,y) − F_Red_corrected(x,y)) /
                  (F_NIR_corrected(x,y) + F_Red_corrected(x,y) + ε)
```

### Preferred Embodiment

In the preferred embodiment, the method is implemented on an NVIDIA Jetson Orin Nano (8 GB, 10W) paired with an Arducam multispectral camera and ZWO Electronic Filter Wheel (EFW Mini). The computing platform executes the method as part of a larger radiometric correction pipeline with seven sequential layers, of which this inter-frame PIF matching constitutes Layer 5.

The method executes in under 1 millisecond of compute time, contributing negligible overhead to the overall pipeline execution time of approximately 3.8 seconds per observation cycle.

### Alternative Embodiments

The method is applicable to any imaging system with sequential multi-band capture, including but not limited to:

- **SmallSat and CubeSat platforms** with filter wheel-based multispectral cameras
- **Drone-based precision agriculture** systems with rotating filter carousels
- **Ground-based environmental monitoring** stations
- **Industrial quality inspection** systems using sequential spectral imaging
- **Medical imaging** systems using sequential wavelength illumination (e.g., hyperspectral wound analysis)
- **Underwater imaging** systems where ambient illumination varies with wave-induced caustics

The spectral flatness threshold T_flat may be adapted for different index computations beyond NDVI (e.g., NDRE, NDWI, NDBI) by selecting surfaces that are spectrally flat in the relevant band pair.

### Adaptive Threshold Variant

In an enhanced embodiment, the spectral flatness threshold T_flat is made adaptive:

- If fewer than N_min PIF pixels are found at the default T_flat, the threshold is widened (e.g., to 0.20) to include more candidate surfaces
- If more than N_max PIF pixels are found, the threshold is narrowed (e.g., to 0.10) to select only the most spectrally stable surfaces
- This adaptive behavior ensures robust operation across diverse scene compositions, from urban environments (many PIFs) to dense agricultural fields (few PIFs)

---

## CLAIMS

> **Note:** Claims in a provisional patent application are optional but recommended. These draft claims should be reviewed and refined by a patent attorney.

**Claim 1.** A method for correcting inter-frame illumination differences in a sequential-filter multispectral imaging system, comprising:
- (a) capturing a first image frame through a first spectral bandpass filter;
- (b) capturing a second image frame through a second spectral bandpass filter, with a temporal gap between said first and second captures;
- (c) computing a preliminary normalized difference index from said first and second frames;
- (d) identifying pseudo-invariant feature pixels as those pixels where the absolute value of said preliminary normalized difference index falls below a spectral flatness threshold;
- (e) computing mean pixel brightness of said pseudo-invariant feature pixels in each of said first and second frames;
- (f) detecting an illumination mismatch when the ratio of said mean brightness values deviates from unity by more than a mismatch threshold; and
- (g) scaling the dimmer frame by the ratio of said mean brightness values to match the brighter frame.

**Claim 2.** The method of Claim 1, wherein said normalized difference index is the Normalized Difference Vegetation Index (NDVI) computed as (NIR − Red) / (NIR + Red).

**Claim 3.** The method of Claim 1, wherein said spectral flatness threshold is in the range of 0.10 to 0.20.

**Claim 4.** The method of Claim 1, further comprising a validation step wherein the correction is skipped if the number of identified pseudo-invariant feature pixels falls below a minimum count threshold.

**Claim 5.** The method of Claim 1, further comprising an adaptive noise floor wherein the correction is skipped if the mean brightness of pseudo-invariant feature pixels in either frame falls below a sensor noise threshold.

**Claim 6.** The method of Claim 1, wherein said spectral flatness threshold is adaptive, being widened when fewer than a minimum number of pseudo-invariant feature pixels are identified and narrowed when more than a maximum number are identified.

**Claim 7.** A system for performing real-time inter-frame illumination correction, comprising:
- a multispectral imaging sensor;
- a spectral band selection mechanism introducing a temporal gap between band captures;
- a processor configured to execute the method of Claim 1; and
- a memory storing instructions for said method.

**Claim 8.** The system of Claim 7, wherein said system is an embedded edge computing device with 10 watts or less power consumption.

**Claim 9.** The system of Claim 7, wherein said system is deployed on a satellite, unmanned aerial vehicle, or autonomous ground station.

**Claim 10.** The method of Claim 1, applied to spectral indices other than NDVI, including but not limited to NDRE (Normalized Difference Red-Edge), NDWI (Normalized Difference Water Index), and NDBI (Normalized Difference Built-Up Index).

---

## ABSTRACT

A method and system for correcting inter-frame illumination differences in sequential-filter multispectral imaging systems. The method identifies pseudo-invariant features (PIFs) in real time by computing a preliminary normalized difference vegetation index (NDVI) from an uncorrected frame pair and selecting pixels where |NDVI| falls below a spectral flatness threshold, indicating non-vegetated surfaces with temporally stable reflectance. Mean brightness of PIF pixels in each frame is compared; if a mismatch exceeds a detection threshold, the dimmer frame is multiplicatively scaled to match the brighter frame before final NDVI computation. The method requires no historical imagery, no external calibration targets, and no manual intervention, executing in under 1 millisecond on embedded edge hardware. Applications include satellite-based crop monitoring, drone-based precision agriculture, and autonomous environmental sensing.

---

## FILING INSTRUCTIONS

### To file this provisional patent with the USPTO:

1. **Create a USPTO account** at [https://patentcenter.uspto.gov](https://patentcenter.uspto.gov)
2. **Select**: New Submission → Provisional Patent Application
3. **Upload**: This document as a PDF
4. **Fill in**: Inventor info, entity status, correspondence address
5. **Pay**: Filing fee ($80 micro entity / $160 small entity)
6. **Receive**: Filing receipt with application number and filing date

### Important Deadlines:
- **12 months from filing date**: You MUST file a non-provisional (full) patent application that claims priority to this provisional, or the provisional expires and the filing date is lost.
- **Before filing**: Have a registered patent attorney review this document. A badly written provisional can limit or invalidate your later claims.

### Recommended Patent Attorneys for Solo Inventors:
- Search the USPTO Patent Attorney/Agent database: [https://oedci.uspto.gov/OEDCI](https://oedci.uspto.gov/OEDCI)
- Look for attorneys specializing in "image processing," "remote sensing," or "computer vision"
- Many offer free 30-minute consultations
- Expect $1,500–$3,000 for a provisional review and cleanup

---

## PRIOR ART SEARCH RESULTS

The following US patents and publications were considered during preparation of this application:

| Reference | Relevance | Distinction |
|-----------|-----------|-------------|
| Canty (2004) — IR-MAD for change detection | PIF identification | Requires multi-temporal stack; not real-time |
| Chavez (1988) — Dark Object Subtraction | Atmospheric correction | Does not address inter-frame mismatch |
| US Patent 7,725,258 — Atmospheric correction | Radiometric pipeline | Uses radiative transfer, not spectral flatness |
| US Patent 9,928,579 — Multispectral image fusion | Sequential capture | Does not use NDVI for PIF selection |
| US Patent 10,354,376 — Vegetation index calculation | NDVI computation | Does not address inter-frame illumination |

> The novel contribution of the present invention is the use of **the spectral index being computed (NDVI) as the criterion for identifying calibration surfaces**, creating a self-referential correction loop that requires no external data, no historical imagery, and no manual calibration.

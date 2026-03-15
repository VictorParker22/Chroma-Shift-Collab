---
description: Capture multispectral field data in Central Park
---

# Capture Field Data

Collect Red+NIR multispectral image pairs for training data and pipeline validation.

## Prerequisites
- Camera connected via USB
- Filter wheel connected and calibrated
- Sufficient storage (each pair ~2-4 MB)

## Rapid Capture Mode (for training data)

1. Start rapid capture session:
```bash
cd /home/chroma-shift/Documents/Chroma-Shift
python3 rapid_capture.py --free 50
```
This captures 50 free-form image pairs. Point at various vegetation, soil, concrete, and mixed scenes.

## Single Capture (for testing)

// turbo
2. Capture a single Red+NIR pair with NDVI:
```bash
cd /home/chroma-shift/Documents/Chroma-Shift
python3 capture_multispectral_pair.py
```

## After Capture

3. Sort captured images into healthy/stressed classes using NDVI:
```bash
cd /home/chroma-shift/Documents/Chroma-Shift
python3 classify_and_sort.py
```

4. Rebuild the augmented dataset:
```bash
cd /home/chroma-shift/Documents/Chroma-Shift
python3 build_dataset.py
```

5. Retrain the model (see `/train-model` workflow).

## Tips
- Capture in varied lighting: sunny, cloudy, morning, afternoon
- Include diverse surfaces: grass, bushes, trees, dirt, concrete, stone
- Aim for 50/50 healthy/stressed class balance
- Current dataset: 366 captures from March 2026

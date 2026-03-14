---
description: Run the radiometric pipeline test suite (109 assertions)
---

# Test Radiometric Pipeline

Validate the 7-layer self-calibrating correction pipeline.

## Quick Test (smoke test)

// turbo
1. Run a quick inline validation:
```bash
cd /home/chroma-shift/Documents/Chroma-Shift
python3 -c "
from radiometric import apply_corrections, get_seasonal_baseline
import numpy as np

# Test basic pipeline
red = np.random.randint(30, 200, (100, 100), dtype=np.uint8).astype(np.float32)
nir = np.random.randint(90, 230, (100, 100), dtype=np.uint8).astype(np.float32)
meta = {'exposure': 10000, 'gain': 1.0}
weather = {'cloud_cover_pct': 30}
solar = {'elevation_deg': 45, 'zenith_deg': 45}

r_c, n_c = apply_corrections(red, nir, meta, meta, weather=weather, solar=solar)
ndvi = (n_c - r_c) / (n_c + r_c + 1e-10)
print(f'NDVI range: [{ndvi.min():.3f}, {ndvi.max():.3f}]')
print(f'Seasonal baseline (week 11): {get_seasonal_baseline(11):.3f}')
print('✅ Pipeline OK')
"
```

## Full Test Suite (109 assertions)

// turbo
2. Run the comprehensive test suite:
```bash
cd /home/chroma-shift/Documents/Chroma-Shift
python3 test_radiometric.py 2>&1 | tail -20
```

If `test_radiometric.py` doesn't exist as a standalone file, the tests may be inline in `radiometric.py`:
```bash
cd /home/chroma-shift/Documents/Chroma-Shift
python3 radiometric.py
```

## What the tests cover
- All 7 layers with edge-case inputs (zero, NaN, overflow)
- 12-month seasonal phenology curve
- Cross-layer skip combinations
- Kalman filter stress detection
- Known-input NDVI consistency
- PIF detection accuracy
- Division-by-zero guards

---
description: Test cloud VLM integration with NVIDIA NIM
---

# Test Cloud VLM

Test the NVIDIA NIM cloud VLM (Llama 3.2 90B Vision) integration.

## Prerequisites
- `NVIDIA_NIM_API_KEY` set in `.env`
- Internet connectivity

## Quick Test

// turbo
1. Test VLM with synthetic scene:
```bash
cd /home/chroma-shift/Documents/Chroma-Shift
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()

from vila_analyzer import VilaSceneAnalyzer
import numpy as np

analyzer = VilaSceneAnalyzer()

# Synthetic vegetation scene
red = np.random.randint(30, 70, (100, 100), dtype=np.uint8)
nir = np.random.randint(150, 230, (100, 100), dtype=np.uint8)

result = analyzer.analyze(red, nir,
    weather={'cloud_cover_pct': 30},
    solar={'elevation_deg': 45})

print(f'Source: {result.get(\"source\", \"N/A\")}')
print(f'Latency: {result[\"latency_ms\"]}ms')
print(f'Vegetation: {result[\"scene\"][\"vegetation_pct\"]}%')
print(f'Confidence: {result[\"confidence\"]}')
print(f'Explanation: {result[\"explanation\"]}')
"
```

## Test Heuristic Fallback

// turbo
2. Test without API key (forces heuristic):
```bash
cd /home/chroma-shift/Documents/Chroma-Shift
python3 -c "
from vila_analyzer import VilaSceneAnalyzer
import numpy as np

# Force heuristic
a = VilaSceneAnalyzer.__new__(VilaSceneAnalyzer)
a.model = None; a._fallback = True
a._api_url = None; a._nim_api_key = None

red = np.random.randint(30, 200, (100, 100), dtype=np.uint8)
nir = np.random.randint(90, 230, (100, 100), dtype=np.uint8)
r = a.analyze(red, nir, weather={'cloud_cover_pct': 30}, solar={'elevation_deg': 45})
print(f'Heuristic: veg={r[\"scene\"][\"vegetation_pct\"]}%, {r[\"latency_ms\"]}ms')
"
```

## VLM Configuration
- Default model: `meta/llama-3.2-90b-vision-instruct`
- Override with: `NVIDIA_NIM_MODEL=your-model-name`
- Default endpoint: `https://integrate.api.nvidia.com/v1`
- Override with: `NVIDIA_NIM_URL=your-endpoint`

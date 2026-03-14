---
description: Run the full monitoring pipeline on the Jetson
---

# Run Pipeline

Run the Chroma-Shift monitoring pipeline for continuous NDVI + AI + weather + VLM analysis.

## Prerequisites
- `.env` file configured with `SUPABASE_URL`, `SUPABASE_KEY`, and optionally `NVIDIA_NIM_API_KEY`
- Camera and filter wheel connected (or running in simulated mode)
- DINOv2 TensorRT engine built (`results/dino_v2_classify/chroma_shift_fp16.engine`)

## Steps

1. Load environment variables and verify setup:
```bash
cd /home/chroma-shift/Documents/Chroma-Shift
source .env 2>/dev/null || true
python3 -c "from dotenv import load_dotenv; load_dotenv(); import os; print(f'Supabase: {\"✅\" if os.getenv(\"SUPABASE_URL\") else \"❌\"}'); print(f'NIM: {\"✅\" if os.getenv(\"NVIDIA_NIM_API_KEY\") else \"⚠️ heuristic fallback\"}')"
```

// turbo
2. Run the pipeline (single zone, 10-min interval):
```bash
cd /home/chroma-shift/Documents/Chroma-Shift
python3 jetson_main.py --zone T1 --interval 600
```

For faster testing use `--interval 30`.

3. To run all zones:
```bash
cd /home/chroma-shift/Documents/Chroma-Shift
python3 jetson_main.py --interval 600
```

4. Stop with `Ctrl+C` — the pipeline handles SIGINT gracefully (resets filter wheel, flushes DB writes).

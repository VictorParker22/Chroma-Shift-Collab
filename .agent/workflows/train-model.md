---
description: Train the DINOv2 classifier with feature caching (8GB safe)
---

# Train DINOv2

Two-phase training optimized for the 8GB Jetson:
- Phase 1: Extract 384-dim embeddings on CPU (slow, one-time)
- Phase 2: Train linear head on GPU (<10 seconds)

## Steps

1. Build the augmented dataset (PlantVillage + field captures):
// turbo
```bash
cd /home/chroma-shift/Documents/Chroma-Shift
python3 build_dataset.py
```

2. Run training:
```bash
cd /home/chroma-shift/Documents/Chroma-Shift
python3 train_dinov2_v2.py \
    --data-dir ./data \
    --results-dir ./results/dino_v2_classify \
    --features-dir ./results/cached_features
```

If running inside Docker:
```bash
jetson-containers/run.sh \
    --volume $(pwd):/workspace/chroma-shift \
    dustynv/pytorch:2.7-r36.4.0-cu128-24.04 \
    python3 /workspace/chroma-shift/train_dinov2_v2.py \
        --data-dir /workspace/chroma-shift/data \
        --results-dir /workspace/chroma-shift/results/dino_v2_classify \
        --features-dir /workspace/chroma-shift/results/cached_features
```

3. Build TensorRT FP16 engine:
```bash
cd /home/chroma-shift/Documents/Chroma-Shift
python3 convert_tensorrt.py
```

4. Verify inference speed:
// turbo
```bash
cd /home/chroma-shift/Documents/Chroma-Shift
python3 export_and_infer.py
```

## Notes
- Current accuracy: 84.17% validation
- Training data: 171 field captures (15× aug) + PlantVillage
- Feature extraction: ~1 image/sec on CPU (~50 min for 3K images)
- Linear head training: <10 seconds on GPU
- TensorRT engine: 8.7ms mean inference (FP16)

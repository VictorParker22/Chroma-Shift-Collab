#!/usr/bin/env python3
"""
export_and_infer.py — Export DINOv2 to ONNX + run inference on captures
=======================================================================
Uses PyTorch for inference (works natively in jetson-containers).
ONNX export available for future TensorRT conversion.

Usage:
    python export_and_infer.py --export                   # Export to ONNX
    python export_and_infer.py --infer path/to/image.png  # Classify one image
    python export_and_infer.py --infer-dir captures/      # Classify all captures
"""

import os
import sys
import glob
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "dino_v2_classify")
BEST_MODEL = os.path.join(RESULTS_DIR, "best_model.pth")
ONNX_MODEL = os.path.join(RESULTS_DIR, "chroma_shift_classifier.onnx")

CLASSES = ["healthy_vegetation", "stressed_soil"]
IMAGE_SIZE = 168

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ═══════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════

class DINOv2Classifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.1):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14",
            pretrained=True
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        embed_dim = self.backbone.embed_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def load_model(model_path=BEST_MODEL, device=None):
    """Load trained model for inference."""
    if device is None:
        device = torch.device("cpu")  # CPU for export/batch — saves GPU RAM

    model = DINOv2Classifier(num_classes=len(CLASSES))
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device


# ═══════════════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════════════

def export_onnx(model_path=BEST_MODEL, onnx_path=ONNX_MODEL):
    """Export trained model to ONNX."""
    print(f"\n  Loading model: {model_path}")
    model, device = load_model(model_path)

    print(f"  Exporting to ONNX: {onnx_path}")
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)

    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  ✅ ONNX exported: {size_mb:.1f} MB")


# ═══════════════════════════════════════════════════════════════════════════
# Inference (PyTorch)
# ═══════════════════════════════════════════════════════════════════════════

_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


@torch.no_grad()
def classify_image(model, device, image_path):
    """Classify a single image. Returns dict with class, confidence, probs."""
    img = Image.open(image_path).convert("RGB")
    tensor = _transform(img).unsqueeze(0).to(device)

    logits = model(tensor)
    probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    return {
        "class": CLASSES[pred_idx],
        "confidence": float(probs[pred_idx]) * 100,
        "healthy_prob": float(probs[0]),
        "stressed_prob": float(probs[1]),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true", help="Export to ONNX")
    parser.add_argument("--infer", type=str, help="Classify a single image")
    parser.add_argument("--infer-dir", type=str, help="Classify all *_red*.png in dir")
    parser.add_argument("--model", type=str, default=BEST_MODEL)
    args = parser.parse_args()

    if args.export:
        export_onnx(args.model)

    if args.infer or args.infer_dir:
        print("\n  Loading model for inference …")
        model, device = load_model(args.model)
        print(f"  Device: {device}")

    if args.infer:
        result = classify_image(model, device, args.infer)
        print(f"\n  Image: {args.infer}")
        print(f"  Class: {result['class']} ({result['confidence']:.1f}%)")
        print(f"  Probs: healthy={result['healthy_prob']:.3f}  stressed={result['stressed_prob']:.3f}\n")

    if args.infer_dir:
        red_files = sorted(
            glob.glob(os.path.join(args.infer_dir, "*_red*.png")) +
            glob.glob(os.path.join(args.infer_dir, "*_red*.jpg"))
        )
        if not red_files:
            print(f"  No *_red*.png files found in {args.infer_dir}")
            return

        print(f"\n  Classifying {len(red_files)} captures …\n")
        healthy, stressed = 0, 0

        for f in red_files:
            result = classify_image(model, device, f)
            label = "🌿" if result["class"] == "healthy_vegetation" else "🟤"
            name = os.path.basename(f).split("_20")[0]
            print(f"  {label} {name:40s} → {result['class']:22s} ({result['confidence']:5.1f}%)")
            if result["class"] == "healthy_vegetation":
                healthy += 1
            else:
                stressed += 1

        print(f"\n  Summary: 🌿 healthy={healthy}  🟤 stressed={stressed}  total={len(red_files)}\n")


if __name__ == "__main__":
    main()

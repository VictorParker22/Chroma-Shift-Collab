"""
vegetation_classifier.py — OnAIR Learner Plugin
=================================================
DINOv2 ViT-S/14 vegetation classifier with TensorRT > PyTorch > None
fallback chain. Extracted and refactored from jetson_main.py.

OnAIR role: Learner (PL)
  - Receives raw telemetry + Knowledge Representation outputs
  - Produces 4-class vegetation classification:
    healthy_vegetation, stressed_vegetation, bare_soil, non_vegetation

In simulation mode (no GPU), classification is estimated from NDVI
values provided by the radiometric PKR plugin.
"""

import sys
import os
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 4-class labels (must match training order)
AI_CLASSES = ["bare_soil", "healthy_vegetation", "non_vegetation", "stressed_vegetation"]
AI_IMAGE_SIZE = 224
AI_MEAN = [0.485, 0.456, 0.406]
AI_STD = [0.229, 0.224, 0.225]


class VegetationClassifier:
    """DINOv2 classifier with TensorRT > PyTorch > NDVI-heuristic fallback."""

    def __init__(self):
        self._mode = "ndvi_heuristic"
        self._trt_classifier = None
        self._torch_model = None
        self._torch_transform = None
        self._torch_device = None
        self._try_load_tensorrt()
        if self._mode == "ndvi_heuristic":
            self._try_load_pytorch()

    def _try_load_tensorrt(self):
        """Attempt to load TensorRT FP16 engine."""
        engine_path = os.path.join(
            _PROJECT_ROOT, "results", "dino_v2_classify", "chroma_shift_fp16.engine"
        )
        if not os.path.isfile(engine_path):
            return
        try:
            from training.convert_tensorrt import TRTClassifier
            self._trt_classifier = TRTClassifier(engine_path)
            # Warmup
            dummy = np.random.randn(1, 3, AI_IMAGE_SIZE, AI_IMAGE_SIZE).astype(np.float32)
            for _ in range(3):
                self._trt_classifier.predict(dummy)
            self._mode = "tensorrt"
        except Exception:
            pass

    def _try_load_pytorch(self):
        """Attempt to load PyTorch DINOv2 model."""
        model_path = os.path.join(
            _PROJECT_ROOT, "results", "dino_v2_classify", "best_model.pth"
        )
        if not os.path.isfile(model_path):
            return
        try:
            import torch
            import torch.nn as nn
            from torchvision import transforms as T

            class _DINOv2Cls(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.backbone = torch.hub.load(
                        "facebookresearch/dinov2", "dinov2_vits14", pretrained=True
                    )
                    for p in self.backbone.parameters():
                        p.requires_grad = False
                    self.backbone.eval()
                    self.head = nn.Sequential(
                        nn.Dropout(0.1),
                        nn.Linear(self.backbone.embed_dim, len(AI_CLASSES)),
                    )

                def forward(self, x):
                    return self.head(self.backbone(x))

            device = torch.device("cpu")
            model = _DINOv2Cls()
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)
            model.eval()

            self._torch_model = model
            self._torch_device = device
            self._torch_transform = T.Compose([
                T.Resize((AI_IMAGE_SIZE, AI_IMAGE_SIZE)),
                T.ToTensor(),
                T.Normalize(AI_MEAN, AI_STD),
            ])
            self._mode = "pytorch"
        except Exception:
            pass

    def classify_from_frame(self, frame_array):
        """Classify an image array (H, W, 3) uint8 BGR."""
        if self._mode == "tensorrt":
            return self._classify_trt(frame_array)
        elif self._mode == "pytorch":
            return self._classify_pytorch(frame_array)
        return None

    def classify_from_ndvi(self, ndvi):
        """Heuristic classification from NDVI value alone."""
        if ndvi >= 0.65:
            cls = "healthy_vegetation"
        elif ndvi >= 0.40:
            cls = "stressed_vegetation"
        elif ndvi >= 0.20:
            cls = "bare_soil"
        else:
            cls = "non_vegetation"

        # Synthetic confidence based on distance from thresholds
        confidence = min(95.0, 50.0 + abs(ndvi - 0.40) * 100)
        result = {"class": cls, "confidence": confidence}
        for c in AI_CLASSES:
            result[f"{c}_prob"] = 0.9 if c == cls else 0.033
        return result

    def _classify_trt(self, frame_array):
        """Classify via TensorRT FP16 engine."""
        from PIL import Image as PILImage
        import cv2
        rgb = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
        img = PILImage.fromarray(rgb).resize(
            (AI_IMAGE_SIZE, AI_IMAGE_SIZE), PILImage.BILINEAR
        )
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - np.array(AI_MEAN)) / np.array(AI_STD)
        arr = arr.transpose(2, 0, 1)[np.newaxis, ...]
        probs = self._trt_classifier.predict(arr)
        idx = int(np.argmax(probs))
        result = {"class": AI_CLASSES[idx], "confidence": float(probs[idx]) * 100}
        for i, c in enumerate(AI_CLASSES):
            result[f"{c}_prob"] = float(probs[i])
        return result

    def _classify_pytorch(self, frame_array):
        """Classify via PyTorch CPU inference."""
        import torch
        import torch.nn.functional as F
        from PIL import Image
        import cv2
        rgb = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tensor = self._torch_transform(img).unsqueeze(0).to(self._torch_device)
        with torch.no_grad():
            logits = self._torch_model(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())
        result = {"class": AI_CLASSES[idx], "confidence": float(probs[idx]) * 100}
        for i, c in enumerate(AI_CLASSES):
            result[f"{c}_prob"] = float(probs[i])
        return result

    @property
    def mode(self):
        return self._mode


class Plugin:
    """OnAIR Learner plugin for DINOv2 vegetation classification."""

    def __init__(self, _name, _headers):
        self.component_name = _name
        self.headers = _headers
        self.classifier = VegetationClassifier()
        self._result = None

    def update(self, low_level_data=[], high_level_data={}):
        """Classify vegetation from frame data or NDVI fallback.

        Args:
            low_level_data: Raw telemetry frame.
            high_level_data: Must contain 'radiometric' PKR output with mean_ndvi.
        """
        # Get NDVI from radiometric PKR for heuristic fallback
        radiometric = high_level_data.get("radiometric", {})
        ndvi = None
        if radiometric:
            ndvi = radiometric.get("mean_ndvi")

        # If we have a GPU classifier and actual frame data, use it
        if self.classifier.mode in ("tensorrt", "pytorch"):
            frame = self._to_dict(low_level_data)
            # In simulation, we don't have real frame arrays — use NDVI heuristic
            if ndvi is not None:
                self._result = self.classifier.classify_from_ndvi(ndvi)
                self._result["mode"] = "ndvi_heuristic_sim"
            else:
                self._result = None
        elif ndvi is not None:
            # NDVI-only heuristic classification
            self._result = self.classifier.classify_from_ndvi(ndvi)
            self._result["mode"] = "ndvi_heuristic"
        else:
            self._result = None

    def render_reasoning(self):
        """Return classification result."""
        return self._result

    def _to_dict(self, low_level_data):
        if isinstance(low_level_data, dict):
            return low_level_data
        if isinstance(low_level_data, (list, tuple)) and self.headers:
            return {h: low_level_data[i] for i, h in enumerate(self.headers)
                    if i < len(low_level_data)}
        return {}

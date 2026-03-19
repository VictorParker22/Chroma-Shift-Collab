#!/usr/bin/env python3
"""
convert_tensorrt.py — Convert ONNX model to TensorRT FP16 engine
===================================================================
Converts the DINOv2 ONNX model to a TensorRT engine optimized for
the Jetson Orin Nano's GPU (Ampere, 1024 CUDA cores, 32 Tensor Cores).

FP16 inference on Tensor Cores gives ~50x speedup vs CPU PyTorch:
  CPU (PyTorch):  ~1000 ms/image
  GPU (TensorRT): ~15-25 ms/image

Usage:
    # Inside jetson-containers (TensorRT pre-installed):
    python convert_tensorrt.py                        # default paths
    python convert_tensorrt.py --onnx path/to/model.onnx --engine out.engine
    python convert_tensorrt.py --validate            # convert + validate
"""

import os
import sys
import time
import argparse

import numpy as np

# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "dino_v2_classify")
ONNX_MODEL = os.path.join(RESULTS_DIR, "chroma_shift_classifier.onnx")
TRT_ENGINE = os.path.join(RESULTS_DIR, "chroma_shift_fp16.engine")

CLASSES = ["bare_soil", "healthy_vegetation", "non_vegetation", "stressed_vegetation"]
IMAGE_SIZE = 224


# ═══════════════════════════════════════════════════════════════════════════
# TensorRT Conversion
# ═══════════════════════════════════════════════════════════════════════════

def convert_onnx_to_trt(onnx_path, engine_path, fp16=True, max_batch=1):
    """
    Convert an ONNX model to a TensorRT engine.

    Args:
        onnx_path:   Path to input ONNX model
        engine_path: Path to save the serialized TensorRT engine
        fp16:        Enable FP16 precision (uses Tensor Cores on Ampere)
        max_batch:   Maximum batch size for the engine
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # ── Parse ONNX ──
    print(f"  Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        onnx_data = f.read()

    if not parser.parse(onnx_data):
        for i in range(parser.num_errors):
            print(f"  ❌ ONNX parse error: {parser.get_error(i)}")
        sys.exit(1)

    print(f"  ✅ ONNX parsed: {network.num_layers} layers, "
          f"{network.num_inputs} inputs, {network.num_outputs} outputs")

    # ── Print network info ──
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"     Input[{i}]: {inp.name} shape={inp.shape} dtype={inp.dtype}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"     Output[{i}]: {out.name} shape={out.shape} dtype={out.dtype}")

    # ── Build config ──
    config = builder.create_builder_config()

    # Memory pool — use 256 MB (Jetson Orin Nano has 8 GB shared)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  🚀 FP16 precision enabled (Tensor Cores)")
    elif fp16:
        print("  ⚠️  FP16 not supported on this platform, using FP32")

    # ── Optimization profile for batch size ──
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    input_shape = input_tensor.shape  # [batch, 3, 168, 168]

    # Fixed shape (batch=1 for real-time inference)
    min_shape = (1, input_shape[1], input_shape[2], input_shape[3])
    opt_shape = (1, input_shape[1], input_shape[2], input_shape[3])
    max_shape = (max_batch, input_shape[1], input_shape[2], input_shape[3])

    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    # ── Build engine ──
    print(f"\n  Building TensorRT engine (this takes 2-5 minutes on Jetson)...")
    t0 = time.time()

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        sys.exit("  ❌ Failed to build TensorRT engine")

    elapsed = time.time() - t0
    print(f"  ✅ Engine built in {elapsed:.0f}s")

    # ── Save ──
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"  ✅ Engine saved: {engine_path} ({size_mb:.1f} MB)")

    return engine_path


# ═══════════════════════════════════════════════════════════════════════════
# TensorRT Inference
# ═══════════════════════════════════════════════════════════════════════════

class TRTClassifier:
    """Wraps a TensorRT engine for classification inference.

    Uses TensorRT's native CUDA runtime (cudart) instead of pycuda
    to avoid CUDA context conflicts on Jetson.
    """

    def __init__(self, engine_path):
        import tensorrt as trt
        from cuda import cudart  # TensorRT's bundled CUDA runtime

        self.trt = trt
        self.cudart = cudart

        self.logger = trt.Logger(trt.Logger.WARNING)
        print(f"  Loading TensorRT engine: {engine_path}")

        with open(engine_path, "rb") as f:
            engine_data = f.read()

        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        # Create CUDA stream
        err, self.stream = cudart.cudaStreamCreate()
        assert err == cudart.cudaError_t.cudaSuccess, f"Stream create failed: {err}"

        # Allocate host + device buffers
        self.inputs = []
        self.outputs = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = list(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            # Handle dynamic batch dim
            if shape[0] == -1:
                shape[0] = 1
                self.context.set_input_shape(name, shape)

            size = int(np.prod(shape))
            host_mem = np.empty(size, dtype=dtype)

            # Allocate device memory
            err, device_mem = cudart.cudaMalloc(host_mem.nbytes)
            assert err == cudart.cudaError_t.cudaSuccess, f"cudaMalloc failed: {err}"

            # Set tensor address
            self.context.set_tensor_address(name, device_mem)

            mode = self.engine.get_tensor_mode(name)
            buf = {"host": host_mem, "device": device_mem, "shape": shape, "name": name}
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append(buf)
            else:
                self.outputs.append(buf)

        size_mb = os.path.getsize(engine_path) / (1024 * 1024)
        print(f"  ✅ TensorRT engine loaded ({size_mb:.1f} MB)")

    def predict(self, input_array):
        """
        Run inference on a preprocessed numpy array [1, 3, H, W] (float32).
        Returns class probabilities [num_classes].
        """
        cudart = self.cudart

        # Flatten input and copy to host buffer
        flat = input_array.ravel().astype(np.float32)
        np.copyto(self.inputs[0]["host"][:len(flat)], flat)

        # Host → Device
        cudart.cudaMemcpyAsync(
            self.inputs[0]["device"],
            self.inputs[0]["host"].ctypes.data,
            self.inputs[0]["host"].nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self.stream,
        )

        # Execute
        self.context.execute_async_v3(stream_handle=self.stream)

        # Device → Host
        cudart.cudaMemcpyAsync(
            self.outputs[0]["host"].ctypes.data,
            self.outputs[0]["device"],
            self.outputs[0]["host"].nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            self.stream,
        )

        # Sync
        cudart.cudaStreamSynchronize(self.stream)

        # Softmax on output logits
        logits = self.outputs[0]["host"][:np.prod(self.outputs[0]["shape"])].reshape(
            self.outputs[0]["shape"]
        )
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        return probs[0]

    def classify_image(self, image_path):
        """
        Classify a single image file.
        Returns dict with class, confidence, and probabilities.
        """
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

        # Normalize to ImageNet stats
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std

        # HWC → NCHW
        arr = arr.transpose(2, 0, 1)[np.newaxis, ...]

        probs = self.predict(arr)
        pred_idx = int(np.argmax(probs))

        result = {
            "class": CLASSES[pred_idx],
            "confidence": float(probs[pred_idx]) * 100,
        }
        # Per-class probabilities
        for i, cls in enumerate(CLASSES):
            result[f"{cls}_prob"] = float(probs[i])
        # Backward compat
        result["healthy_prob"] = float(probs[0])
        result["stressed_prob"] = float(probs[min(1, len(probs)-1)])
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════

def validate_engine(engine_path, captures_dir=None):
    """Quick validation of the TensorRT engine against sample images."""
    classifier = TRTClassifier(engine_path)

    # Warmup (first inference is slower due to CUDA context init)
    dummy = np.random.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    for _ in range(5):
        classifier.predict(dummy)
    print("  Warmup complete (5 iterations)")

    # Benchmark
    times = []
    for _ in range(50):
        t0 = time.time()
        classifier.predict(dummy)
        times.append((time.time() - t0) * 1000)

    print(f"\n  ⏱️  Latency (50 runs):")
    print(f"     Mean:   {np.mean(times):.1f} ms")
    print(f"     Median: {np.median(times):.1f} ms")
    print(f"     Min:    {np.min(times):.1f} ms")
    print(f"     Max:    {np.max(times):.1f} ms")

    # Classify captures if provided
    if captures_dir and os.path.isdir(captures_dir):
        import glob
        red_files = sorted(
            glob.glob(os.path.join(captures_dir, "*_red*.png")) +
            glob.glob(os.path.join(captures_dir, "*_red*.jpg"))
        )[:20]  # first 20 for quick validation

        if red_files:
            print(f"\n  Classifying {len(red_files)} captures with TensorRT:\n")
            for f in red_files:
                result = classifier.classify_image(f)
                icon = "🌿" if result["class"] == "healthy_vegetation" else "🟤"
                name = os.path.basename(f).split("_20")[0]
                print(f"  {icon} {name:40s} → {result['class']:22s} ({result['confidence']:5.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ONNX → TensorRT FP16 converter")
    parser.add_argument("--onnx", type=str, default=ONNX_MODEL,
                        help=f"Input ONNX model (default: {ONNX_MODEL})")
    parser.add_argument("--engine", type=str, default=TRT_ENGINE,
                        help=f"Output TensorRT engine (default: {TRT_ENGINE})")
    parser.add_argument("--fp32", action="store_true",
                        help="Use FP32 instead of FP16")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation after conversion")
    parser.add_argument("--captures-dir", type=str, default=None,
                        help="Directory of captures for validation")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Chroma-Shift — TensorRT FP16 Conversion                 ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    if not os.path.isfile(args.onnx):
        sys.exit(f"\n  ❌ ONNX model not found: {args.onnx}\n"
                 f"     Run: python export_and_infer.py --export")

    convert_onnx_to_trt(args.onnx, args.engine, fp16=not args.fp32)

    if args.validate:
        print("\n  ── Validation ──")
        validate_engine(args.engine, args.captures_dir)


if __name__ == "__main__":
    main()

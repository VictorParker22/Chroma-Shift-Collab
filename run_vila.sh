#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# run_vila.sh — Launch VILA 1.5-3B for Chroma-Shift scene analysis
# ─────────────────────────────────────────────────────────────────
# Usage:
#   ./run_vila.sh              # interactive chat mode
#   ./run_vila.sh --api rest   # REST API server mode (for jetson_main.py)
#   ./run_vila.sh --help       # show NanoLLM options
#
# Requirements:
#   - NVIDIA Jetson Orin Nano (JetPack 6.x / L4T R36.4+)
#   - Docker with NVIDIA container runtime
#   - ~2 GB VRAM for 4-bit AWQ model
#   - ~15 GB disk for the container + model weights
# ─────────────────────────────────────────────────────────────────

set -euo pipefail

CONTAINER="dustynv/nano_llm:r36.4.0"
MODEL="Efficient-Large-Model/VILA1.5-3b-AWQ"
QUANTIZATION="q4f16_ft"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Default: REST API server mode for integration with jetson_main.py
MODE="${1:---api}"
PORT="${VILA_PORT:-8050}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     Chroma-Shift — VILA Scene Analyzer                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Container:  ${CONTAINER}                    ║"
echo "║  Model:      VILA 1.5-3B (4-bit AWQ)                       ║"
echo "║  Mode:       ${MODE}                                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Check if container exists ──
if ! docker image inspect "${CONTAINER}" &>/dev/null; then
    echo ""
    echo "❌ Container not found: ${CONTAINER}"
    echo "   Run: docker pull ${CONTAINER}"
    exit 1
fi

# ── Create model cache directory ──
mkdir -p "${PROJECT_DIR}/models/vila"

# ── Set swap to max (important for 8GB Orin Nano) ──
if [ -f /etc/systemd/nvzramconfig.sh ]; then
    echo "🔧 Ensuring swap is configured..."
    sudo sysctl -w vm.swappiness=60 2>/dev/null || true
fi

# ── Run the container ──
echo ""
echo "🚀 Launching VILA... (first run downloads ~2GB model weights)"
echo ""

if [[ "${MODE}" == "--api" || "${MODE}" == "rest" || "${MODE}" == "--api=rest" ]]; then
    # REST API server mode — jetson_main.py connects via HTTP
    echo "📡 Starting REST API server on port ${PORT}..."
    echo "   Endpoint: http://localhost:${PORT}/api/generate"
    echo ""
    docker run --rm -it \
        --runtime nvidia \
        --network host \
        -v "${PROJECT_DIR}/models/vila:/data/models/vila" \
        -v /tmp:/tmp \
        -e VILA_PORT="${PORT}" \
        "${CONTAINER}" \
        python3 -m nano_llm.chat \
            --api mlc \
            --model "${MODEL}" \
            --quantization "${QUANTIZATION}" \
            --max-new-tokens 512 \
            --api rest \
            --rest-port "${PORT}"
else
    # Interactive chat mode — for testing prompts
    echo "💬 Interactive mode — type your prompts..."
    echo ""
    docker run --rm -it \
        --runtime nvidia \
        --network host \
        -v "${PROJECT_DIR}/models/vila:/data/models/vila" \
        -v /tmp:/tmp \
        "${CONTAINER}" \
        python3 -m nano_llm.chat \
            --api mlc \
            --model "${MODEL}" \
            --quantization "${QUANTIZATION}" \
            --max-new-tokens 512 \
            "$@"
fi

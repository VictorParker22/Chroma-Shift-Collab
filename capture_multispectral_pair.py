"""
capture_multispectral_pair.py — Chroma-Shift Multispectral Capture
====================================================================
Captures a Red / NIR image pair using:
  • USB Arducam camera  → OpenCV with V4L2 backend
  • ZWO Electronic Filter Wheel (EFW) → libEFWFilter.so via ctypes

Filter wheel slot mapping:
  Slot 1 → Red  (665 nm)
  Slot 2 → NIR  (865 nm)

Usage:
    python capture_multispectral_pair.py                  # interactive
    python capture_multispectral_pair.py --name sample_01 # single capture
"""

import os
import sys
import time
import ctypes
import argparse
from datetime import datetime

import cv2

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(SCRIPT_DIR, "lib")
EFW_LIB_PATH = os.path.join(LIB_DIR, "libEFWFilter.so")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0                     # /dev/video0 (USB Arducam)
FILTER_SLOT_RED = 1                  # 665 nm band-pass on order; slot empty in 2026 field sessions
FILTER_SLOT_NIR = 2                  # 865 nm band-pass
SETTLE_TIME_SEC = 1.5                # mechanical settle after wheel rotation
FLUSH_FRAMES = 5                     # frames to discard (clear USB buffer)
EFW_POLL_INTERVAL = 0.3              # seconds between position polls
EFW_POLL_TIMEOUT = 30                # max seconds to wait for move/calibration
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "captures")


# ═══════════════════════════════════════════════════════════════════════════
# ZWO EFW — ctypes interface to libEFWFilter.so
# ═══════════════════════════════════════════════════════════════════════════
class EFW_INFO(ctypes.Structure):
    """Mirrors the EFW_INFO struct from the ZWO EFW SDK."""
    _fields_ = [
        ("ID", ctypes.c_int),
        ("Name", ctypes.c_char * 64),
        ("slotNum", ctypes.c_int),
    ]


class FilterWheel:
    """High-level wrapper around the ZWO EFW C library."""

    def __init__(self, lib_path=EFW_LIB_PATH):
        # libudev must be loaded first (runtime dependency of libEFWFilter)
        ctypes.CDLL("libudev.so", mode=ctypes.RTLD_GLOBAL)
        self._lib = ctypes.CDLL(lib_path)
        self._id = None
        self._slots = 0

    # --- lifecycle --------------------------------------------------------

    def open(self):
        """Detect the first EFW on USB, open it, and wait for calibration."""
        num = self._lib.EFWGetNum()
        if num == 0:
            sys.exit("ERROR: No ZWO Electronic Filter Wheel detected on USB.")

        efw_id = ctypes.c_int()
        self._lib.EFWGetID(0, ctypes.byref(efw_id))
        self._id = efw_id.value

        rc = self._lib.EFWOpen(self._id)
        if rc != 0:
            sys.exit(f"ERROR: EFWOpen failed (rc={rc}). "
                     "Try unplugging and re-plugging the USB cable.")

        info = EFW_INFO()
        self._lib.EFWGetProperty(self._id, ctypes.byref(info))
        self._slots = info.slotNum
        name = info.Name.decode("utf-8").strip() or "EFW"
        print(f"  Filter wheel : {name}  ({self._slots} slots)")

        # Wait for initial calibration
        self._wait_for_position("Calibrating")

    def close(self):
        """Release the filter wheel."""
        if self._id is not None:
            self._lib.EFWClose(self._id)
            self._id = None

    # --- movement ---------------------------------------------------------

    def get_position(self):
        """Return current slot index, or -1 if still moving."""
        pos = ctypes.c_int()
        self._lib.EFWGetPosition(self._id, ctypes.byref(pos))
        return pos.value

    def move_to(self, slot):
        """
        Command the wheel to the given slot.
        Blocks until the wheel confirms arrival (position == slot),
        then waits an additional SETTLE_TIME_SEC for mechanical vibration
        to damp out.
        """
        print(f"  Rotating → slot {slot} …", end=" ", flush=True)
        self._lib.EFWSetPosition(self._id, slot)
        self._wait_for_position(target=slot, silent=True)
        time.sleep(SETTLE_TIME_SEC)
        print("done")

    # --- internal ---------------------------------------------------------

    def _wait_for_position(self, target=None, silent=False):
        """Poll until position is >= 0 (or == target)."""
        deadline = time.time() + EFW_POLL_TIMEOUT
        while time.time() < deadline:
            pos = self.get_position()
            if target is None and pos >= 0:
                if not silent:
                    print(f"  Calibrated → slot {pos}")
                return pos
            if target is not None and pos == target:
                return pos
            time.sleep(EFW_POLL_INTERVAL)
        # Timeout
        current = self.get_position()
        if not silent:
            print(f"  ⚠️  Timeout (position={current})")
        return current


# ═══════════════════════════════════════════════════════════════════════════
# Camera helpers (Arducam via V4L2)
# ═══════════════════════════════════════════════════════════════════════════

def open_camera(index=CAMERA_INDEX):
    """Open the Arducam via the V4L2 backend."""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"ERROR: Cannot open camera at /dev/video{index}.")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera opened : /dev/video{index}  ({w}×{h})")
    return cap


def flush_and_capture(cap):
    """
    Read and discard FLUSH_FRAMES frames to clear the USB pipeline,
    then return the next clean frame.
    """
    for _ in range(FLUSH_FRAMES):
        cap.read()
    ret, frame = cap.read()
    if not ret:
        sys.exit("ERROR: Failed to capture frame from camera.")
    return frame


# ═══════════════════════════════════════════════════════════════════════════
# Core capture function
# ═══════════════════════════════════════════════════════════════════════════

def capture_multispectral_pair(sample_name, cap, efw, output_dir=OUTPUT_DIR):
    """
    Capture a Red + NIR image pair for the given sample name.

    Steps for each band:
      1. Rotate filter wheel to the correct slot.
      2. Wait for confirmed arrival + 1.5 s mechanical settle.
      3. Flush 5 frames to clear the USB buffer.
      4. Capture and save the final clean frame.

    Returns:
        (red_path, nir_path) — absolute paths to the saved images.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    paths = {}
    for label, slot in [("red_665nm", FILTER_SLOT_RED),
                        ("nir_865nm", FILTER_SLOT_NIR)]:
        # --- Rotate wheel ---
        efw.move_to(slot)

        # --- Flush buffer & capture ---
        frame = flush_and_capture(cap)

        # --- Save ---
        filename = f"{sample_name}_{timestamp}_{label}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"  Saved → {filepath}")
        paths[label] = filepath

    return paths["red_665nm"], paths["nir_865nm"]


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Capture a Red/NIR image pair with Arducam + ZWO EFW."
    )
    parser.add_argument(
        "--name", "-n",
        default=None,
        help="Sample name (used in the filename). "
             "If omitted, you will be prompted interactively.",
    )
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     Chroma-Shift — Multispectral Pair Capture              ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # --- Initialise hardware ---
    print("\n[1/3] Initialising hardware …")
    cap = open_camera()
    efw = FilterWheel()
    efw.open()

    try:
        # --- Determine sample name ---
        if args.name:
            sample_name = args.name
        else:
            sample_name = input("\n  Enter sample name: ").strip()
            if not sample_name:
                sample_name = "sample"

        # --- Capture ---
        print(f"\n[2/3] Capturing pair for '{sample_name}' …")
        red_path, nir_path = capture_multispectral_pair(
            sample_name, cap, efw
        )

        print(f"\n[3/3] Capture complete ✅")
        print(f"  Red (665 nm) : {red_path}")
        print(f"  NIR (865 nm) : {nir_path}\n")

    finally:
        # --- Cleanup ---
        cap.release()
        efw.close()
        print("  Hardware released.\n")


if __name__ == "__main__":
    main()

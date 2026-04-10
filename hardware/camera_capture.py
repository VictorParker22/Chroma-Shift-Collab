"""
hardware/camera_capture.py — Camera and Filter Wheel Hardware Interface
========================================================================
Extracted from jetson_main.py for modularity.

Provides:
  - EFW_INFO: ctypes struct mirroring ZWO EFW SDK
  - FilterWheel: ctypes wrapper for ZWO Electronic Filter Wheel
  - open_camera(): Open V4L2 camera via OpenCV
  - flush_and_capture(): Flush stale frames and capture one fresh frame
  - compute_ndvi(): Per-pixel NDVI from Red + NIR frames
"""

import os
import time
import ctypes

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMERA_INDEX = 0
FILTER_SLOT_RED = 1          # 640-670 nm
FILTER_SLOT_NIR = 2          # 850-880 nm
SETTLE_TIME_SEC = 1.5
FLUSH_FRAMES = 5
EFW_POLL_INTERVAL = 0.3
EFW_POLL_TIMEOUT = 30

FILTER_BANDS = {
    0: ("Clear",     "400-700"),
    1: ("Red",       "640-670"),
    2: ("NIR",       "850-880"),
    3: ("Red-Edge",  "700-740"),
    4: ("SWIR",      "1570-1650"),
}


# ---------------------------------------------------------------------------
# EFW — ctypes wrapper
# ---------------------------------------------------------------------------

class EFW_INFO(ctypes.Structure):
    """Mirrors the EFW_INFO struct from the ZWO EFW SDK."""
    _fields_ = [
        ("ID", ctypes.c_int),
        ("Name", ctypes.c_char * 64),
        ("slotNum", ctypes.c_int),
    ]


class FilterWheel:
    def __init__(self):
        lib_path = os.path.join(SCRIPT_DIR, "lib", "libEFWFilter.so")
        ctypes.CDLL("libudev.so", mode=ctypes.RTLD_GLOBAL)
        self._lib = ctypes.CDLL(lib_path)
        self._lib.EFWOpen.restype = ctypes.c_int
        self._lib.EFWSetPosition.restype = ctypes.c_int
        self._lib.EFWGetPosition.restype = ctypes.c_int
        self._lib.EFWGetProperty.restype = ctypes.c_int
        self._lib.EFWClose.restype = ctypes.c_int
        self._id = None

    def open(self):
        if self._lib.EFWGetNum() == 0:
            raise RuntimeError("No EFW detected")
        eid = ctypes.c_int()
        self._lib.EFWGetID(0, ctypes.byref(eid))
        self._id = eid.value
        rc = self._lib.EFWOpen(self._id)
        if rc != 0:
            raise RuntimeError(f"EFWOpen failed (rc={rc})")
        # REQUIRED: EFWGetProperty must be called after Open, before SetPosition
        info = EFW_INFO()
        self._lib.EFWGetProperty(self._id, ctypes.byref(info))
        # Wait for calibration
        self._wait_ready()

    def close(self):
        if self._id is not None:
            self._lib.EFWClose(self._id)
            self._id = None

    def move_to(self, slot):
        self._lib.EFWSetPosition(self._id, slot)
        deadline = time.time() + EFW_POLL_TIMEOUT
        pos = ctypes.c_int()
        while time.time() < deadline:
            self._lib.EFWGetPosition(self._id, ctypes.byref(pos))
            if pos.value == slot:
                break
            time.sleep(EFW_POLL_INTERVAL)
        time.sleep(SETTLE_TIME_SEC)

    def _wait_ready(self):
        pos = ctypes.c_int()
        deadline = time.time() + EFW_POLL_TIMEOUT
        while time.time() < deadline:
            self._lib.EFWGetPosition(self._id, ctypes.byref(pos))
            if pos.value >= 0:
                return
            time.sleep(EFW_POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def open_camera(index=CAMERA_INDEX):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera /dev/video{index}")
    return cap


def flush_and_capture(cap, flush_frames=FLUSH_FRAMES):
    for _ in range(flush_frames):
        cap.read()
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Frame capture failed")
    return frame


# ---------------------------------------------------------------------------
# NDVI
# ---------------------------------------------------------------------------

def compute_ndvi(red_frame, nir_frame):
    red = cv2.cvtColor(red_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    nir = cv2.cvtColor(nir_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    denom = nir + red
    ndvi = np.where(denom > 0, (nir - red) / denom, 0.0)
    valid = denom > 0
    mean = float(ndvi[valid].mean()) if valid.any() else 0.0
    return ndvi, mean

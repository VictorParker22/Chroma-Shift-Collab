#!/usr/bin/env python3
"""
fault_tolerance.py — Space-Grade Fault Tolerance for Chroma-Shift
==================================================================
When a bug happens 400 km up, there's no SSH. This module provides:

  1. WATCHDOG       — Auto-restart if the main loop hangs
  2. SAFE MODE      — Minimal NDVI-only operation when errors pile up
  3. SELF-TEST      — Verify all components on boot before scanning
  4. ERROR TELEMETRY — Push crash logs to Supabase (ground sees them)
  5. REMOTE CONFIG  — Pull parameter overrides from Supabase
  6. ANOMALY DETECT — Sensor drift, stuck values, impossible readings

Design Philosophy:
  - The system NEVER fully stops. It degrades gracefully.
  - Every error is logged to Supabase so ground control can see it.
  - Ground control can push config changes via Supabase table.
  - After N consecutive errors, the system enters safe mode.
  - After a cooldown period in safe mode, it tries full mode again.

Usage in jetson_main.py:
    from fault_tolerance import (
        FaultManager, SystemSelfTest, RemoteConfig, AnomalyDetector
    )
"""

import os
import sys
import time
import json
import signal
import threading
import traceback
from datetime import datetime, timezone
from collections import deque


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

# Error thresholds
MAX_CONSECUTIVE_ERRORS = 5       # Enter safe mode after this many
SAFE_MODE_COOLDOWN_SEC = 300     # Try full mode again after 5 min
WATCHDOG_TIMEOUT_SEC = 120       # Kill/restart if no heartbeat in 2 min
ERROR_LOG_MAX = 100              # Keep last 100 errors in memory

# Anomaly detection
NDVI_MIN_VALID = -1.0
NDVI_MAX_VALID = 1.0
NDVI_STUCK_THRESHOLD = 10        # Flag if same NDVI 10 times in a row
SENSOR_DRIFT_WINDOW = 20         # Compare last 20 readings for drift
SENSOR_DRIFT_THRESHOLD = 0.3     # Flag if mean shifts > 0.3 in window

# Operating modes
MODE_FULL = "full"               # All systems: TRT + VILA + Adaptive + Weather
MODE_REDUCED = "reduced"         # TRT + basic corrections, no VLM
MODE_SAFE = "safe"               # NDVI-only, no AI, no cloud
MODE_EMERGENCY = "emergency"     # Log-only, minimal power


# ═══════════════════════════════════════════════════════════════════════════
# 1. WATCHDOG — Auto-restart if main loop hangs
# ═══════════════════════════════════════════════════════════════════════════

class Watchdog:
    """
    Hardware-style watchdog timer. The main loop must call `kick()`
    within WATCHDOG_TIMEOUT_SEC or the watchdog triggers recovery.

    In space, this prevents a hung process from wasting an entire orbit.
    """

    def __init__(self, timeout_sec=WATCHDOG_TIMEOUT_SEC, on_timeout=None):
        self.timeout = timeout_sec
        self.on_timeout = on_timeout or self._default_recovery
        self._last_kick = time.time()
        self._running = True
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def kick(self):
        """Call this every scan cycle to reset the watchdog timer."""
        self._last_kick = time.time()

    def stop(self):
        self._running = False

    def _monitor(self):
        while self._running:
            elapsed = time.time() - self._last_kick
            if elapsed > self.timeout:
                print(f"\n  🐕 WATCHDOG: No heartbeat for {elapsed:.0f}s! "
                      f"Triggering recovery...")
                self.on_timeout()
                self._last_kick = time.time()  # Reset after recovery
            time.sleep(5)

    @staticmethod
    def _default_recovery():
        """Default recovery: restart this process."""
        print("  🐕 WATCHDOG: Restarting process...")
        os.execv(sys.executable, [sys.executable] + sys.argv)


# ═══════════════════════════════════════════════════════════════════════════
# 2. FAULT MANAGER — Error tracking + safe mode transitions
# ═══════════════════════════════════════════════════════════════════════════

class FaultManager:
    """
    Tracks errors, manages operating mode, and decides when to degrade.

    Operating Modes:
      FULL     → All systems active (TRT + VILA + Adaptive + Weather)
      REDUCED  → Skip VLM, use heuristic fallback
      SAFE     → NDVI-only computation, no AI, no cloud calls
      EMERGENCY → Log-only mode, minimal power draw

    Transition Rules:
      - 3 consecutive errors → REDUCED
      - 5 consecutive errors → SAFE
      - 10 consecutive errors → EMERGENCY
      - After SAFE_MODE_COOLDOWN_SEC of clean operation → upgrade one level
    """

    def __init__(self, supabase_client=None):
        self.mode = MODE_FULL
        self.consecutive_errors = 0
        self.total_errors = 0
        self.total_recoveries = 0
        self.error_log = deque(maxlen=ERROR_LOG_MAX)
        self.sb = supabase_client
        self._last_mode_change = time.time()
        self._safe_mode_entered = None
        self._component_status = {}

    @property
    def is_safe_mode(self):
        return self.mode in (MODE_SAFE, MODE_EMERGENCY)

    @property
    def skip_vila(self):
        return self.mode in (MODE_REDUCED, MODE_SAFE, MODE_EMERGENCY)

    @property
    def skip_ai(self):
        return self.mode in (MODE_SAFE, MODE_EMERGENCY)

    @property
    def skip_supabase(self):
        return self.mode == MODE_EMERGENCY

    def record_success(self):
        """Call after a successful scan cycle."""
        self.consecutive_errors = 0
        self._try_upgrade()

    def record_error(self, component, error, context=None):
        """
        Record an error from a specific component.
        Returns the current operating mode after processing.
        """
        self.consecutive_errors += 1
        self.total_errors += 1

        error_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": component,
            "error": str(error)[:500],
            "traceback": traceback.format_exc()[:1000],
            "context": context or {},
            "consecutive": self.consecutive_errors,
            "mode": self.mode,
        }
        self.error_log.append(error_entry)

        # Print to local console (captured by triage log)
        print(f"\n  ❌ FAULT [{component}] (#{self.consecutive_errors}): {error}")

        # Push to Supabase so ground control can see it
        self._push_error_telemetry(error_entry)

        # Check if we need to degrade
        self._evaluate_mode()

        return self.mode

    def _evaluate_mode(self):
        """Decide if we need to change operating mode."""
        old_mode = self.mode

        if self.consecutive_errors >= 10:
            self.mode = MODE_EMERGENCY
        elif self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            self.mode = MODE_SAFE
        elif self.consecutive_errors >= 3:
            self.mode = MODE_REDUCED
        # Don't auto-upgrade here — that happens in _try_upgrade()

        if self.mode != old_mode:
            self._last_mode_change = time.time()
            if self.mode == MODE_SAFE:
                self._safe_mode_entered = time.time()
            print(f"\n  🔄 MODE CHANGE: {old_mode} → {self.mode}")
            print(f"     Consecutive errors: {self.consecutive_errors}")
            self._push_mode_change(old_mode, self.mode)

    def _try_upgrade(self):
        """Try to upgrade operating mode after cooldown period."""
        if self.mode == MODE_FULL:
            return

        elapsed = time.time() - self._last_mode_change
        if elapsed > SAFE_MODE_COOLDOWN_SEC:
            old_mode = self.mode
            # Upgrade one level
            upgrades = {
                MODE_EMERGENCY: MODE_SAFE,
                MODE_SAFE: MODE_REDUCED,
                MODE_REDUCED: MODE_FULL,
            }
            self.mode = upgrades.get(self.mode, MODE_FULL)
            self._last_mode_change = time.time()
            self.total_recoveries += 1
            print(f"\n  ✅ MODE UPGRADE: {old_mode} → {self.mode} "
                  f"(clean for {elapsed:.0f}s)")
            self._push_mode_change(old_mode, self.mode)

    def _push_error_telemetry(self, error_entry):
        """Push error to Supabase so ground control can see it."""
        if not self.sb or self.mode == MODE_EMERGENCY:
            return
        try:
            self.sb.table("system_errors").insert({
                "component": error_entry["component"],
                "error_message": error_entry["error"],
                "traceback": error_entry["traceback"],
                "consecutive_count": error_entry["consecutive"],
                "operating_mode": self.mode,
                "context": json.dumps(error_entry.get("context", {})),
            }).execute()
        except Exception:
            pass  # Don't error on error reporting

    def _push_mode_change(self, old_mode, new_mode):
        """Notify Supabase of mode transition."""
        if not self.sb:
            return
        try:
            self.sb.table("system_errors").insert({
                "component": "fault_manager",
                "error_message": f"Mode change: {old_mode} → {new_mode}",
                "consecutive_count": self.consecutive_errors,
                "operating_mode": new_mode,
                "context": json.dumps({
                    "total_errors": self.total_errors,
                    "total_recoveries": self.total_recoveries,
                }),
            }).execute()
        except Exception:
            pass

    def get_status(self):
        """Return current fault manager status dict."""
        return {
            "mode": self.mode,
            "consecutive_errors": self.consecutive_errors,
            "total_errors": self.total_errors,
            "total_recoveries": self.total_recoveries,
            "last_errors": [
                {"component": e["component"], "error": e["error"][:100],
                 "time": e["timestamp"]}
                for e in list(self.error_log)[-5:]
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════
# 3. SYSTEM SELF-TEST — Verify everything on boot
# ═══════════════════════════════════════════════════════════════════════════

class SystemSelfTest:
    """
    Run on boot to verify all components before entering the scan loop.
    Each test returns (passed: bool, message: str).
    Failed tests degrade the system gracefully rather than blocking.
    """

    def __init__(self):
        self.results = {}

    def run_all(self):
        """Run all self-tests and return summary."""
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  SYSTEM SELF-TEST                                          ║")
        print("╚══════════════════════════════════════════════════════════════╝")

        tests = [
            ("python_imports",   self._test_imports),
            ("disk_space",       self._test_disk),
            ("gpu_available",    self._test_gpu),
            ("trt_engine",       self._test_trt_engine),
            ("radiometric",      self._test_radiometric),
            ("supabase",         self._test_supabase),
            ("weather_api",      self._test_weather),
            ("captures_dir",     self._test_captures_dir),
        ]

        passed = 0
        failed = 0
        degraded = []

        for name, test_fn in tests:
            try:
                ok, msg = test_fn()
                self.results[name] = {"passed": ok, "message": msg}
                icon = "✅" if ok else "⚠️"
                print(f"  {icon} {name}: {msg}")
                if ok:
                    passed += 1
                else:
                    failed += 1
                    degraded.append(name)
            except Exception as e:
                self.results[name] = {"passed": False, "message": str(e)}
                print(f"  ❌ {name}: EXCEPTION — {e}")
                failed += 1
                degraded.append(name)

        print(f"\n  Results: {passed} passed, {failed} degraded")
        if degraded:
            print(f"  Degraded: {', '.join(degraded)}")

        # Determine starting mode based on test results
        mode = MODE_FULL
        if "trt_engine" in degraded:
            mode = MODE_REDUCED
        if "radiometric" in degraded:
            mode = MODE_SAFE
        if "disk_space" in degraded and not self.results["disk_space"]["passed"]:
            mode = MODE_EMERGENCY

        print(f"  Starting mode: {mode}\n")
        return mode, self.results

    def _test_imports(self):
        try:
            import cv2, numpy, scipy
            return True, f"cv2={cv2.__version__}, numpy={numpy.__version__}"
        except ImportError as e:
            return False, f"Missing: {e}"

    def _test_disk(self):
        st = os.statvfs("/")
        free_gb = (st.f_bfree * st.f_frsize) / (1024**3)
        if free_gb < 1.0:
            return False, f"CRITICAL: Only {free_gb:.1f} GB free!"
        elif free_gb < 5.0:
            return True, f"Low: {free_gb:.1f} GB free (warning)"
        return True, f"{free_gb:.1f} GB free"

    def _test_gpu(self):
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return True, result.stdout.strip()
            return False, "nvidia-smi failed"
        except Exception as e:
            return False, str(e)

    def _test_trt_engine(self):
        engine_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "results", "dino_v2_classify", "chroma_shift_fp16.engine"
        )
        if os.path.isfile(engine_path):
            size_mb = os.path.getsize(engine_path) / (1024 * 1024)
            return True, f"Found ({size_mb:.1f} MB)"
        return False, "Not found — AI classification unavailable"

    def _test_radiometric(self):
        try:
            from radiometric import apply_corrections, solar_position
            import numpy as np
            # Quick smoke test with synthetic data
            red = np.full((10, 10), 100, dtype=np.uint8)
            nir = np.full((10, 10), 200, dtype=np.uint8)
            result = apply_corrections(red, nir, 0.01, 1.0, 0.01, 1.0,
                                       datetime.now(timezone.utc), 0)
            ndvi = result["mean_ndvi"]
            if -1.0 <= ndvi <= 1.0:
                return True, f"Pipeline OK (test NDVI={ndvi:.3f})"
            return False, f"Invalid NDVI: {ndvi}"
        except Exception as e:
            return False, str(e)

    def _test_supabase(self):
        try:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")
            if not url or not key:
                return False, "SUPABASE_URL/KEY not set"
            from supabase import create_client
            sb = create_client(url, key)
            # Quick connectivity test
            sb.table("system_status").select("id").limit(1).execute()
            return True, "Connected"
        except Exception as e:
            return False, f"Connection failed: {e}"

    def _test_weather(self):
        try:
            from weather import get_current_weather
            w = get_current_weather()
            if w and "temp_c" in w:
                return True, f"{w['temp_c']}°C, {w.get('description', 'OK')}"
            return False, "No data returned"
        except Exception as e:
            return False, str(e)

    def _test_captures_dir(self):
        cap_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "captures")
        os.makedirs(cap_dir, exist_ok=True)
        # Test write
        test_file = os.path.join(cap_dir, ".selftest")
        try:
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
            return True, f"Writable ({cap_dir})"
        except Exception as e:
            return False, f"Not writable: {e}"


# ═══════════════════════════════════════════════════════════════════════════
# 4. REMOTE CONFIG — Pull parameter overrides from Supabase
# ═══════════════════════════════════════════════════════════════════════════

class RemoteConfig:
    """
    Pull configuration overrides from a Supabase table.
    Ground control can change scan intervals, enable/disable components,
    or force safe mode — all without an OTA update.

    Expected Supabase table: `remote_config`
      - key: text (primary)
      - value: text (JSON-encoded)
      - updated_at: timestamptz

    Example rows:
      key="scan_interval"     value="300"
      key="force_safe_mode"   value="true"
      key="enable_vila"       value="false"
      key="ndvi_alert_threshold"  value="0.10"
    """

    def __init__(self, supabase_client=None):
        self.sb = supabase_client
        self._cache = {}
        self._last_fetch = 0
        self.fetch_interval = 60  # Check every 60 seconds

    def fetch(self):
        """Pull latest config from Supabase."""
        if not self.sb:
            return self._cache

        now = time.time()
        if now - self._last_fetch < self.fetch_interval:
            return self._cache

        try:
            resp = self.sb.table("remote_config").select("*").execute()
            for row in resp.data:
                self._cache[row["key"]] = row["value"]
            self._last_fetch = now
        except Exception:
            pass  # Use cached values on failure

        return self._cache

    def get(self, key, default=None, cast=None):
        """
        Get a config value. Supports type casting.

        Usage:
            config.get("scan_interval", 600, cast=int)
            config.get("force_safe_mode", False, cast=bool)
            config.get("ndvi_alert_threshold", 0.10, cast=float)
        """
        self.fetch()
        val = self._cache.get(key)
        if val is None:
            return default
        if cast is not None:
            try:
                if cast == bool:
                    return val.lower() in ("true", "1", "yes")
                return cast(val)
            except (ValueError, TypeError):
                return default
        return val

    def get_overrides(self):
        """Return all current overrides as a dict."""
        self.fetch()
        return dict(self._cache)


# ═══════════════════════════════════════════════════════════════════════════
# 5. ANOMALY DETECTOR — Sensor drift, stuck values, impossible readings
# ═══════════════════════════════════════════════════════════════════════════

class AnomalyDetector:
    """
    Detects sensor anomalies that indicate hardware failure or
    environmental extremes requiring intervention.

    Checks:
      - NDVI out of physical range [-1, 1]
      - Same NDVI value N times in a row (stuck sensor)
      - Mean NDVI drifting rapidly (sensor degradation)
      - All-zero or all-saturated frames (camera failure)
      - Frame capture timeout (camera hang)
    """

    def __init__(self):
        self._ndvi_history = deque(maxlen=SENSOR_DRIFT_WINDOW)
        self._last_ndvi = None
        self._stuck_count = 0
        self._anomalies = deque(maxlen=50)

    def check_ndvi(self, ndvi, zone_id="default"):
        """
        Check an NDVI reading for anomalies.
        Returns list of anomaly dicts (empty if clean).
        """
        anomalies = []

        # Out of physical range
        if ndvi < NDVI_MIN_VALID or ndvi > NDVI_MAX_VALID:
            anomalies.append({
                "type": "out_of_range",
                "severity": "critical",
                "message": f"NDVI {ndvi:.4f} outside physical range [-1, 1]",
                "zone": zone_id,
            })

        # Stuck sensor (same value repeatedly)
        if self._last_ndvi is not None and abs(ndvi - self._last_ndvi) < 0.0001:
            self._stuck_count += 1
            if self._stuck_count >= NDVI_STUCK_THRESHOLD:
                anomalies.append({
                    "type": "stuck_sensor",
                    "severity": "high",
                    "message": f"NDVI stuck at {ndvi:.4f} for "
                               f"{self._stuck_count} readings",
                    "zone": zone_id,
                })
        else:
            self._stuck_count = 0
        self._last_ndvi = ndvi

        # Sensor drift detection
        self._ndvi_history.append(ndvi)
        if len(self._ndvi_history) >= SENSOR_DRIFT_WINDOW:
            half = len(self._ndvi_history) // 2
            first_half = list(self._ndvi_history)[:half]
            second_half = list(self._ndvi_history)[half:]
            drift = abs(
                sum(second_half) / len(second_half)
                - sum(first_half) / len(first_half)
            )
            if drift > SENSOR_DRIFT_THRESHOLD:
                anomalies.append({
                    "type": "sensor_drift",
                    "severity": "medium",
                    "message": f"NDVI mean shifted {drift:.3f} over last "
                               f"{SENSOR_DRIFT_WINDOW} readings",
                    "zone": zone_id,
                })

        # Record all anomalies
        for a in anomalies:
            a["timestamp"] = datetime.now(timezone.utc).isoformat()
            self._anomalies.append(a)

        return anomalies

    def check_frame(self, frame, label=""):
        """
        Check a captured frame for camera anomalies.
        Returns list of anomaly dicts (empty if clean).
        """
        anomalies = []
        if frame is None:
            anomalies.append({
                "type": "null_frame",
                "severity": "critical",
                "message": f"Camera returned null frame ({label})",
            })
            return anomalies

        # All-zero (black frame — lens cap on, or camera dead)
        if frame.max() == 0:
            anomalies.append({
                "type": "black_frame",
                "severity": "critical",
                "message": f"All-zero frame ({label}) — camera may be dead",
            })

        # All-saturated (white frame — sensor overexposure)
        if frame.min() >= 250:
            anomalies.append({
                "type": "saturated_frame",
                "severity": "high",
                "message": f"Saturated frame ({label}) — overexposure or "
                           f"sun glint",
            })

        # Very low contrast (fog, condensation on lens)
        std = float(frame.std())
        if std < 3.0:
            anomalies.append({
                "type": "low_contrast",
                "severity": "medium",
                "message": f"Very low contrast (std={std:.1f}) on {label} "
                           f"— possible fog/condensation",
            })

        for a in anomalies:
            a["timestamp"] = datetime.now(timezone.utc).isoformat()
            self._anomalies.append(a)

        return anomalies

    def get_recent_anomalies(self, n=10):
        """Return last N anomalies."""
        return list(self._anomalies)[-n:]

    def get_health_score(self):
        """
        Return sensor health score 0-100.
        100 = perfect, 0 = critical failure.
        """
        recent = list(self._anomalies)[-20:]
        if not recent:
            return 100

        severity_weights = {
            "critical": 25,
            "high": 15,
            "medium": 5,
        }
        penalty = sum(severity_weights.get(a["severity"], 5) for a in recent)
        return max(0, 100 - penalty)


# ═══════════════════════════════════════════════════════════════════════════
# Convenience — run self-test standalone
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # Run self-test
    test = SystemSelfTest()
    mode, results = test.run_all()

    # Test anomaly detector
    print("\n  Testing anomaly detector...")
    detector = AnomalyDetector()

    # Normal
    a = detector.check_ndvi(0.45, "B3")
    print(f"  Normal NDVI (0.45): {len(a)} anomalies")

    # Out of range
    a = detector.check_ndvi(1.5, "B3")
    print(f"  Out-of-range NDVI (1.5): {a[0]['message']}")

    # Stuck sensor
    for _ in range(12):
        a = detector.check_ndvi(0.33, "B3")
    print(f"  Stuck sensor (0.33 × 12): {len(a)} anomalies, "
          f"health={detector.get_health_score()}")

    # Fault manager
    print("\n  Testing fault manager...")
    fm = FaultManager()
    for i in range(6):
        fm.record_error("test", f"Simulated error #{i+1}")
    print(f"  After 6 errors: mode={fm.mode}, "
          f"consecutive={fm.consecutive_errors}")

    fm.record_success()
    print(f"  After success: consecutive={fm.consecutive_errors}")
    print(f"  Status: {json.dumps(fm.get_status(), indent=2)}")

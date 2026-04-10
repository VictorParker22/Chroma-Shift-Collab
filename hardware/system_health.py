"""
hardware/system_health.py — Jetson System Health Telemetry
==========================================================
Extracted from jetson_main.py for modularity.
Reads CPU temperature, storage usage, and uptime from /proc and /sys.
"""

import os


def get_system_stats():
    stats = {
        "online": True,
        "uptime_seconds": 0,
        "cpu_temp_c": 0.0,
        "gpu_usage_pct": 0.0,
        "storage_used_gb": 0.0,
        "storage_total_gb": 0.0,
    }
    try:
        with open("/proc/uptime") as f:
            stats["uptime_seconds"] = int(float(f.read().split()[0]))
    except Exception:
        pass
    try:
        for zone in sorted(os.listdir("/sys/class/thermal/")):
            tf = f"/sys/class/thermal/{zone}/temp"
            if os.path.isfile(tf):
                with open(tf) as f:
                    stats["cpu_temp_c"] = round(int(f.read().strip()) / 1000.0, 1)
                break
    except Exception:
        pass
    try:
        st = os.statvfs("/")
        total = st.f_blocks * st.f_frsize
        used = total - st.f_bfree * st.f_frsize
        stats["storage_total_gb"] = round(total / (1024**3), 1)
        stats["storage_used_gb"] = round(used / (1024**3), 1)
    except Exception:
        pass
    return stats

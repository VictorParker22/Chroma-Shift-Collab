"""
chroma_shift_adapter.py — OnAIR Data Adapter for Chroma-Shift
==============================================================
Bridges either CSV simulation data or live cFS/SBN telemetry into
OnAIR frames. Each frame is a dict mapping telemetry header names
to their current values.

CSV mode:  Reads from chroma_shift_sim.csv (generated from satellite
           NDVI ground truth + synthetic sensor metadata).
SBN mode:  Subscribes to CCSDS message IDs for camera, health,
           and weather telemetry from cFS Software Bus.
"""

import os
import csv
import json
from datetime import datetime, timezone


class Plugin:
    """OnAIR DataSource adapter for Chroma-Shift telemetry."""

    def __init__(self, data_file, meta_file, ss_breakdown=False):
        self.all_headers = []
        self.sim_data = {}
        self.binning_configs = {}
        self._frames = []
        self._index = 0

        # Load telemetry metadata (header definitions)
        if meta_file and os.path.isfile(meta_file):
            with open(meta_file, "r") as f:
                meta = json.load(f)
            self.all_headers = meta.get("order", [])
            if "subsystems" in meta:
                self.binning_configs = meta["subsystems"]

        # Load data source
        if data_file and os.path.isfile(data_file):
            self._load_csv(data_file)

    def _load_csv(self, filepath):
        """Load CSV simulation data into frame buffer."""
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame = {}
                for header in self.all_headers:
                    raw = row.get(header, "")
                    frame[header] = self._cast_value(raw)
                self._frames.append(frame)

    @staticmethod
    def _cast_value(raw):
        """Attempt to cast string to numeric, falling back to string."""
        if raw == "" or raw is None:
            return 0.0
        try:
            return float(raw)
        except (ValueError, TypeError):
            return raw

    def get_next(self):
        """Return the next telemetry frame as a {header: value} dict.

        Returns:
            dict: Telemetry frame mapping header names to values.
            None: If no more frames are available.
        """
        if self._index >= len(self._frames):
            return None
        frame = self._frames[self._index]
        self._index += 1
        return frame

    def has_more(self):
        """Check if more frames are available.

        Returns:
            bool: True if frames remain in the buffer.
        """
        return self._index < len(self._frames)

    def get_headers(self):
        """Return the list of telemetry header names."""
        return self.all_headers

    def get_current_time(self):
        """Return the timestamp from the most recently read frame."""
        if self._index > 0 and self._index <= len(self._frames):
            frame = self._frames[self._index - 1]
            ts = frame.get("timestamp_utc", "")
            if isinstance(ts, str) and ts:
                try:
                    return datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except ValueError:
                    pass
        return datetime.now(timezone.utc)

    def reset(self):
        """Reset the frame index to replay data."""
        self._index = 0

    def __len__(self):
        return len(self._frames)

    def __repr__(self):
        return (f"ChromaShiftAdapter(frames={len(self._frames)}, "
                f"headers={len(self.all_headers)}, "
                f"index={self._index})")

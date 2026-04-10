# cFS Application Stubs for Chroma-Shift

This directory contains CCSDS message header definitions for the three
planned cFS applications that would run on the SmallSat flight processor.
These apps publish telemetry onto the cFS Software Bus, which OnAIR
consumes via the SBN (Software Bus Network) data adapter.

## Planned cFS Applications

### CHROMA_CAM_APP (MID 0x0880)
Controls the multispectral camera and motorized filter wheel.
Publishes Red+NIR frame metadata after each capture cycle.

- Filter wheel position control (Red 640-670nm, NIR 850-880nm)
- Camera exposure and gain management
- Frame capture timing synchronized to observation scheduler
- Publishes: mean DN values, exposure settings, frame dimensions

### CHROMA_HEALTH_APP (MID 0x0881)
Publishes spacecraft housekeeping telemetry at 1 Hz.

- CPU/GPU temperature monitoring
- Storage usage tracking
- Uptime and operating mode reporting
- Watchdog heartbeat

### CHROMA_WX_APP (MID 0x0882)
Manages weather/environmental sensor data.

On a SmallSat, this would publish onboard sensor readings or cached
environmental data from the last ground contact. For the Jetson
prototype, it wraps the NWS API integration.

- Temperature, humidity, pressure
- Cloud cover estimation (from onboard imager or ground upload)
- Precipitation probability

## Integration with OnAIR

These apps publish telemetry as CCSDS packets on the cFS Software Bus.
The SBN (Software Bus Network) bridges these to OnAIR via TCP:

```
cFS App --> Software Bus --> SBN --> SBN-Client (TCP:1234) --> OnAIR
```

OnAIR's SBN data adapter (available in the OnAIR open-source release)
subscribes to the message IDs defined in these headers and translates
the CCSDS payloads into OnAIR telemetry frames.

## Current Status

These are **header-only stubs** documenting the planned packet structures.
The actual C implementations would be developed for the target flight
processor (e.g., SpaceCube, RAD750, or Jetson-class SBC with cFS port).

For development and demonstration, the CSV data adapter simulates
these telemetry streams using satellite ground truth data.

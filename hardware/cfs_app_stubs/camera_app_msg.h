/**
 * @file camera_app_msg.h
 * @brief CCSDS message definitions for CHROMA_CAM_APP
 *
 * Defines the telemetry packet structure published by the Chroma-Shift
 * multispectral camera application on the cFS Software Bus.
 *
 * Message ID: 0x0880
 * Rate: Event-driven (after each capture cycle, typically every 120-1800s)
 *
 * This is a header-only stub for documentation purposes. The actual
 * implementation would use cFS message macros and CCSDS packet headers.
 */

#ifndef CHROMA_CAM_APP_MSG_H
#define CHROMA_CAM_APP_MSG_H

#include "cfe_sb.h"

/* Message ID */
#define CHROMA_CAM_TLM_MID        0x0880

/* Command codes */
#define CHROMA_CAM_CAPTURE_CC     0   /* Trigger a capture cycle */
#define CHROMA_CAM_SET_FILTER_CC  1   /* Move filter wheel to position */
#define CHROMA_CAM_SET_EXPOSURE_CC 2  /* Set camera exposure */

/**
 * Camera telemetry packet payload.
 * Published after each Red+NIR capture pair completes.
 */
typedef struct {
    /* CCSDS header is prepended by cFE automatically */

    /* Timestamp (UTC seconds since J2000 epoch) */
    double  timestamp_utc;

    /* Zone identifier (0-6 mapping to field zones) */
    uint8_t zone_id;

    /* Red band capture metadata */
    float   red_mean_dn;        /* Mean digital number [0-255] */
    float   red_exposure;       /* Exposure time (seconds) */
    float   red_gain;           /* Sensor gain multiplier */

    /* NIR band capture metadata */
    float   nir_mean_dn;        /* Mean digital number [0-255] */
    float   nir_exposure;       /* Exposure time (seconds) */
    float   nir_gain;           /* Sensor gain multiplier */

    /* Frame dimensions */
    uint16_t frame_width;       /* Pixels */
    uint16_t frame_height;      /* Pixels */

    /* Filter wheel state */
    uint8_t filter_position;    /* Current filter slot (0-4) */

    /* Padding for alignment */
    uint8_t spare[3];

} CHROMA_CAM_TlmPayload_t;

/**
 * Full telemetry message (header + payload).
 */
typedef struct {
    CFE_MSG_TelemetryHeader_t  TelemetryHeader;
    CHROMA_CAM_TlmPayload_t   Payload;
} CHROMA_CAM_TlmMsg_t;

/**
 * Capture command payload.
 */
typedef struct {
    uint8_t zone_id;            /* Target zone (0-6) */
    uint8_t filter_slot_red;    /* Red filter position (default 1) */
    uint8_t filter_slot_nir;    /* NIR filter position (default 2) */
    uint8_t spare;
} CHROMA_CAM_CaptureCmd_t;

typedef struct {
    CFE_MSG_CommandHeader_t    CommandHeader;
    CHROMA_CAM_CaptureCmd_t    Payload;
} CHROMA_CAM_CaptureCmdMsg_t;

#endif /* CHROMA_CAM_APP_MSG_H */

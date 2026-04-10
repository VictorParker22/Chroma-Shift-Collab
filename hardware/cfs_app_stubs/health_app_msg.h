/**
 * @file health_app_msg.h
 * @brief CCSDS message definitions for CHROMA_HEALTH_APP
 *
 * Defines the housekeeping telemetry packet structure published by the
 * Chroma-Shift system health monitoring application on the cFS Software Bus.
 *
 * Message ID: 0x0881
 * Rate: 1 Hz (continuous housekeeping)
 *
 * This is a header-only stub for documentation purposes.
 */

#ifndef CHROMA_HEALTH_APP_MSG_H
#define CHROMA_HEALTH_APP_MSG_H

#include "cfe_sb.h"

/* Message ID */
#define CHROMA_HEALTH_TLM_MID     0x0881

/**
 * System health telemetry packet payload.
 * Published at 1 Hz for continuous spacecraft monitoring.
 */
typedef struct {
    /* CCSDS header is prepended by cFE automatically */

    /* Thermal */
    float   cpu_temp_c;         /* CPU temperature (Celsius) */
    float   gpu_temp_c;         /* GPU temperature (Celsius) */

    /* Storage */
    float   storage_used_gb;    /* Used storage (GB) */
    float   storage_total_gb;   /* Total storage (GB) */

    /* System */
    uint32_t uptime_seconds;    /* Time since boot */
    uint8_t  operating_mode;    /* 0=full, 1=reduced, 2=safe, 3=emergency */
    uint8_t  online;            /* 1=online, 0=offline */

    /* Counters */
    uint32_t scan_count;        /* Total scans completed */
    uint32_t alert_count;       /* Total alerts generated */
    uint32_t error_count;       /* Total errors recorded */

    /* Health */
    uint8_t  health_score;      /* Overall health 0-100% */
    uint8_t  spare[3];          /* Padding */

} CHROMA_HEALTH_TlmPayload_t;

/**
 * Full telemetry message (header + payload).
 */
typedef struct {
    CFE_MSG_TelemetryHeader_t     TelemetryHeader;
    CHROMA_HEALTH_TlmPayload_t   Payload;
} CHROMA_HEALTH_TlmMsg_t;

#endif /* CHROMA_HEALTH_APP_MSG_H */

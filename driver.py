#!/usr/bin/env python3
"""
driver.py — Chroma-Shift OnAIR Entry Point
============================================
Runs the Chroma-Shift vegetation monitoring pipeline through NASA's
OnAIR cognitive architecture framework.

Usage:
    python driver.py                        # uses default onair_config.ini
    python driver.py onair_config.ini       # specify config file
    python driver.py --standalone           # run standalone demo without OnAIR

The standalone mode demonstrates the plugin pipeline directly, useful
for development and testing when the OnAIR framework is not installed.
"""

import sys
import os
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def run_standalone(config_path):
    """Run the Chroma-Shift plugin pipeline without OnAIR framework.

    This demonstrates the full sense-perceive-act loop using the same
    plugins that would run inside OnAIR, but orchestrated directly.
    Useful for development and integration testing.
    """
    print("=" * 70)
    print("  Chroma-Shift OnAIR Pipeline — Standalone Mode")
    print("=" * 70)
    print()

    # Load telemetry metadata
    with open(os.path.join(PROJECT_ROOT, "telemetry_metadata.json")) as f:
        meta = json.load(f)
    headers = meta["order"]

    # Initialize data adapter
    from onair_plugins.data_adapter.chroma_shift_adapter import Plugin as Adapter
    data_source = Adapter(
        os.path.join(PROJECT_ROOT,
                     "onair_plugins/data_adapter/sim_data/chroma_shift_sim.csv"),
        os.path.join(PROJECT_ROOT, "telemetry_metadata.json"),
    )
    print(f"  Data source: {data_source}")
    print()

    # Initialize plugins (matching OnAIR plugin hierarchy)
    from onair_plugins.knowledge_rep.radiometric_pkr.radiometric_pkr import Plugin as RadiometricPKR
    from onair_plugins.knowledge_rep.scene_analysis_pkr.scene_analysis_pkr import Plugin as ScenePKR
    from onair_plugins.learners.vegetation_classifier.vegetation_classifier import Plugin as ClassifierPL
    from onair_plugins.learners.anomaly_detector.anomaly_detector_pl import Plugin as AnomalyPL
    from onair_plugins.planners.observation_scheduler.observation_scheduler import Plugin as SchedulerPP
    from onair_plugins.planners.crop_advisor_pp.crop_advisor_pp import Plugin as AdvisorPP
    from onair_plugins.complex_reasoners.mission_executive.mission_executive import Plugin as ExecutivePCR

    pkr_radiometric = RadiometricPKR("radiometric", headers)
    pkr_scene = ScenePKR("scene_analysis", headers)
    pl_classifier = ClassifierPL("vegetation_classifier", headers)
    pl_anomaly = AnomalyPL("anomaly_detector", headers)
    pp_scheduler = SchedulerPP("observation_scheduler", headers)
    pp_advisor = AdvisorPP("crop_advisor", headers)
    pcr_executive = ExecutivePCR("mission_executive", headers)

    print("  Plugins initialized:")
    print("    PKR: radiometric, scene_analysis")
    print("    PL:  vegetation_classifier, anomaly_detector")
    print("    PP:  observation_scheduler, crop_advisor")
    print("    PCR: mission_executive")
    print()

    # Run the pipeline frame-by-frame
    frame_count = 0
    alert_total = 0

    while data_source.has_more():
        frame = data_source.get_next()
        if frame is None:
            break

        frame_count += 1

        # === SENSE === (data adapter provides the frame)

        # === PERCEIVE: Knowledge Representation ===
        pkr_radiometric.update(frame, {})
        radiometric_out = pkr_radiometric.render_reasoning()

        pkr_scene.update(frame, {"radiometric": radiometric_out})
        scene_out = pkr_scene.render_reasoning()

        # === PERCEIVE: Learners ===
        high_level_kr = {"radiometric": radiometric_out, "scene_analysis": scene_out}

        pl_classifier.update(frame, high_level_kr)
        classifier_out = pl_classifier.render_reasoning()

        pl_anomaly.update(frame, high_level_kr)
        anomaly_out = pl_anomaly.render_reasoning()

        # === PERCEIVE: Planners ===
        high_level_learn = {
            **high_level_kr,
            "vegetation_classifier": classifier_out,
            "anomaly_detector": anomaly_out,
        }

        pp_scheduler.update(frame, high_level_learn)
        scheduler_out = pp_scheduler.render_reasoning()

        pp_advisor.update(frame, high_level_learn)
        advisor_out = pp_advisor.render_reasoning()

        # === ACT: Complex Reasoner ===
        high_level_all = {
            **high_level_learn,
            "observation_scheduler": scheduler_out,
            "crop_advisor": advisor_out,
        }

        pcr_executive.update(frame, high_level_all)
        executive_out = pcr_executive.render_reasoning()

        # Print summary for this frame
        ndvi = radiometric_out.get("mean_ndvi", 0) if radiometric_out else 0
        zone = frame.get("zone_id", "?")
        status = executive_out.get("status", "?") if executive_out else "?"
        ai_class = classifier_out.get("class", "?") if classifier_out else "?"
        mode = executive_out.get("operating_mode", "?") if executive_out else "?"
        alerts = len(executive_out.get("alerts", [])) if executive_out else 0
        alert_total += alerts

        if frame_count <= 10 or frame_count % 25 == 0 or alerts > 0:
            alert_marker = " << ALERT" if alerts else ""
            print(f"  Frame {frame_count:3d} | Zone {zone:3s} | "
                  f"NDVI {ndvi:+.4f} | {status:8s} | "
                  f"AI: {ai_class:22s} | Mode: {mode}{alert_marker}")

    print()
    print(f"  Pipeline complete: {frame_count} frames processed, "
          f"{alert_total} alerts generated")
    print(f"  Final health: {anomaly_out.get('health_score', '?')}% | "
          f"Mode: {executive_out.get('operating_mode', '?')}")

    # Export triage log
    triage_path = os.path.join(PROJECT_ROOT, "triage_logs",
                                f"onair_sim_{frame_count}frames.jsonl")
    os.makedirs(os.path.dirname(triage_path), exist_ok=True)
    pcr_executive.export_triage_log(triage_path)
    print(f"  Triage log: {triage_path}")
    print()


def run_onair(config_path):
    """Run the pipeline through the OnAIR framework."""
    try:
        from onair.src.run_scripts.execution_engine import ExecutionEngine
        engine = ExecutionEngine(config_path)
        engine.run_sim()
    except ImportError:
        print("ERROR: OnAIR framework not installed.")
        print("  Install with: pip install onair")
        print("  Or clone: git clone https://github.com/nasa/OnAIR.git")
        print()
        print("  Running standalone mode instead...")
        print()
        run_standalone(config_path)


def main():
    parser = argparse.ArgumentParser(
        description="Chroma-Shift OnAIR Pipeline Driver"
    )
    parser.add_argument("config", nargs="?", default="onair_config.ini",
                        help="OnAIR configuration file (default: onair_config.ini)")
    parser.add_argument("--standalone", action="store_true",
                        help="Run standalone demo without OnAIR framework")
    args = parser.parse_args()

    config_path = os.path.join(PROJECT_ROOT, args.config)

    if args.standalone:
        run_standalone(config_path)
    else:
        run_onair(config_path)


if __name__ == "__main__":
    main()

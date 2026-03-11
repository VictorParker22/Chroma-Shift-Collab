#!/usr/bin/env python3
"""
rapid_capture.py — Chroma-Shift Rapid Dataset Builder
======================================================
Guided capture session for building the DINOv2 training dataset.
Initializes hardware once, then walks through a structured sequence
of captures with on-screen prompts telling the operator what to shoot.

Press ENTER to capture, 's' to skip, 'q' to quit early.

Usage:
    python rapid_capture.py              # full guided session
    python rapid_capture.py --free 20    # 20 free-form captures after guided
"""

import os
import sys
import time
import argparse
from datetime import datetime

import cv2

from capture_multispectral_pair import (
    open_camera, flush_and_capture, FilterWheel,
    FILTER_SLOT_RED, FILTER_SLOT_NIR,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURES_DIR = os.path.join(SCRIPT_DIR, "captures")

# ═══════════════════════════════════════════════════════════════════════════
# Capture Sequence — each entry: (label, category, instruction)
#   label:       filename prefix (e.g. "healthy_grass_close")
#   category:    "healthy" or "stressed" (for your reference)
#   instruction: what to point the camera at
# ═══════════════════════════════════════════════════════════════════════════

CAPTURE_SEQUENCE = [
    # ── HEALTHY VEGETATION (dense, green, alive) ──────────────────────────
    ("healthy_grass_close_01",    "healthy",  "🌿 HEALTHY: Point at a DENSE PATCH OF GREEN GRASS, close up (1-2 ft away)"),
    ("healthy_grass_close_02",    "healthy",  "🌿 HEALTHY: Same green grass — shift angle slightly to the LEFT"),
    ("healthy_grass_close_03",    "healthy",  "🌿 HEALTHY: Same area — tilt camera DOWN so it's looking straight at the ground"),
    ("healthy_grass_med_01",      "healthy",  "🌿 HEALTHY: Step back — point at a WIDER patch of green grass (3-5 ft away)"),
    ("healthy_grass_med_02",      "healthy",  "🌿 HEALTHY: Find a DIFFERENT green grass patch nearby, medium distance"),
    ("healthy_grass_far_01",      "healthy",  "🌿 HEALTHY: Point at a LARGE LAWN AREA from further away (8-10 ft)"),
    ("healthy_bush_01",           "healthy",  "🌿 HEALTHY: Find a GREEN BUSH or SHRUB — point directly at the leaves"),
    ("healthy_bush_02",           "healthy",  "🌿 HEALTHY: Different bush/shrub — get close to the GREEN LEAVES"),
    ("healthy_bush_03",           "healthy",  "🌿 HEALTHY: Another green bush — try from a DIFFERENT ANGLE"),
    ("healthy_canopy_01",         "healthy",  "🌿 HEALTHY: Point UP at a TREE CANOPY with green leaves (if any have them)"),
    ("healthy_canopy_02",         "healthy",  "🌿 HEALTHY: Different tree — aim at the DENSEST cluster of green leaves"),
    ("healthy_ivy_01",            "healthy",  "🌿 HEALTHY: Find any IVY, GROUND COVER, or LOW GREEN PLANTS"),
    ("healthy_ivy_02",            "healthy",  "🌿 HEALTHY: More ground cover — a different patch of low green plants"),
    ("healthy_mixed_green_01",    "healthy",  "🌿 HEALTHY: Find a spot where GRASS MEETS BUSHES — point at the greenest part"),
    ("healthy_mixed_green_02",    "healthy",  "🌿 HEALTHY: Any other area with THICK GREEN VEGETATION"),
    ("healthy_shade_01",          "healthy",  "🌿 HEALTHY: Find green vegetation IN SHADE (under a tree)"),
    ("healthy_shade_02",          "healthy",  "🌿 HEALTHY: More shaded green vegetation — different spot"),
    ("healthy_sun_01",            "healthy",  "🌿 HEALTHY: Green vegetation in DIRECT SUNLIGHT — bright and vivid"),
    ("healthy_sun_02",            "healthy",  "🌿 HEALTHY: Another sunlit green patch — make it the greenest thing you can see"),
    ("healthy_sun_03",            "healthy",  "🌿 HEALTHY: One more sunlit green — try grass this time if last was a bush"),
    ("healthy_variety_01",        "healthy",  "🌿 HEALTHY: Walk to a TOTALLY NEW SPOT — find the greenest vegetation there"),
    ("healthy_variety_02",        "healthy",  "🌿 HEALTHY: Another new spot — different type of green plant if possible"),
    ("healthy_variety_03",        "healthy",  "🌿 HEALTHY: One more new location — any healthy green vegetation"),
    ("healthy_variety_04",        "healthy",  "🌿 HEALTHY: Keep moving — find more green, different texture if you can"),
    ("healthy_variety_05",        "healthy",  "🌿 HEALTHY: Last healthy capture — find the BEST, densest green you can see"),

    # ── STRESSED / BARE (dirt, dead grass, pavement, dormant) ─────────────
    ("stressed_dirt_01",          "stressed", "🟤 STRESSED: Point at BARE DIRT or EXPOSED SOIL"),
    ("stressed_dirt_02",          "stressed", "🟤 STRESSED: Different patch of bare dirt/soil — close up"),
    ("stressed_dirt_03",          "stressed", "🟤 STRESSED: More bare soil — try a dry, compacted area"),
    ("stressed_dead_grass_01",    "stressed", "🟤 STRESSED: Find DEAD or BROWN GRASS — dried out patch"),
    ("stressed_dead_grass_02",    "stressed", "🟤 STRESSED: More dead/brown grass — different spot"),
    ("stressed_dead_grass_03",    "stressed", "🟤 STRESSED: Another brown/dormant grass area"),
    ("stressed_path_01",          "stressed", "🟤 STRESSED: Point at a DIRT PATH or WORN TRAIL"),
    ("stressed_path_02",          "stressed", "🟤 STRESSED: Another dirt/gravel path area"),
    ("stressed_mulch_01",         "stressed", "🟤 STRESSED: Find MULCH, WOOD CHIPS, or dead leaf litter on the ground"),
    ("stressed_mulch_02",         "stressed", "🟤 STRESSED: More mulch/leaf litter — different spot"),
    ("stressed_bare_tree_01",     "stressed", "🟤 STRESSED: Point at BARE TREE BRANCHES (no leaves, dormant)"),
    ("stressed_bare_tree_02",     "stressed", "🟤 STRESSED: Different bare tree — aim at the densest bare branches"),
    ("stressed_sparse_01",        "stressed", "🟤 STRESSED: Find an area where grass is THIN and SPARSE — mostly dirt showing"),
    ("stressed_sparse_02",        "stressed", "🟤 STRESSED: More sparse/patchy vegetation with lots of bare ground"),
    ("stressed_edge_01",          "stressed", "🟤 STRESSED: Point where PAVEMENT MEETS DIRT — the transition zone"),
    ("stressed_edge_02",          "stressed", "🟤 STRESSED: Another edge — where grass dies out near a path"),
    ("stressed_dry_01",           "stressed", "🟤 STRESSED: Find the DRIEST looking patch of ground you can see"),
    ("stressed_dry_02",           "stressed", "🟤 STRESSED: More dry/stressed ground — different location"),
    ("stressed_shadow_01",        "stressed", "🟤 STRESSED: Bare/dead ground IN SHADOW"),
    ("stressed_shadow_02",        "stressed", "🟤 STRESSED: More bare ground in shade"),
    ("stressed_variety_01",       "stressed", "🟤 STRESSED: Walk to a NEW AREA — find bare/dead/stressed ground there"),
    ("stressed_variety_02",       "stressed", "🟤 STRESSED: Another new spot — any non-green ground surface"),
    ("stressed_variety_03",       "stressed", "🟤 STRESSED: More variety — different texture of bare/stressed ground"),
    ("stressed_variety_04",       "stressed", "🟤 STRESSED: Keep moving — find more stressed/bare areas"),
    ("stressed_variety_05",       "stressed", "🟤 STRESSED: Last stressed capture — driest, most barren spot you can find"),
]


def do_capture(cap, efw, label, output_dir):
    """Capture a Red+NIR pair and save."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    efw.move_to(FILTER_SLOT_RED)
    red_frame = flush_and_capture(cap)

    efw.move_to(FILTER_SLOT_NIR)
    nir_frame = flush_and_capture(cap)

    os.makedirs(output_dir, exist_ok=True)
    red_path = os.path.join(output_dir, f"{label}_{ts}_red_665nm.png")
    nir_path = os.path.join(output_dir, f"{label}_{ts}_nir_833nm.png")

    cv2.imwrite(red_path, red_frame)
    cv2.imwrite(nir_path, nir_frame)

    return red_path, nir_path


def main():
    parser = argparse.ArgumentParser(description="Rapid guided capture session.")
    parser.add_argument("--free", type=int, default=0,
                        help="Additional free-form captures after guided sequence.")
    parser.add_argument("--start", type=int, default=1,
                        help="Start at capture number N (to resume a session).")
    args = parser.parse_args()

    total = len(CAPTURE_SEQUENCE)
    start_idx = max(0, args.start - 1)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Chroma-Shift — Rapid Dataset Capture Session             ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║   Captures planned: {total:3d} guided + {args.free} free-form{' ' * 19}║")
    print("║   Controls: ENTER=capture  s=skip  q=quit                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Init hardware once
    print("\n  Initialising hardware …")
    cap = open_camera()
    efw = FilterWheel()
    efw.open()

    captured = 0
    skipped = 0
    healthy_count = 0
    stressed_count = 0

    try:
        # ── Guided sequence ───────────────────────────────────────────
        for i, (label, category, instruction) in enumerate(CAPTURE_SEQUENCE[start_idx:], start=start_idx + 1):
            print(f"\n{'─' * 62}")
            print(f"  [{i}/{total}]  {instruction}")
            print(f"{'─' * 62}")

            action = input("  Ready? [ENTER=capture / s=skip / q=quit]: ").strip().lower()

            if action == 'q':
                print("\n  Ending session early.")
                break
            elif action == 's':
                skipped += 1
                print("  ⏭️  Skipped")
                continue

            red_p, nir_p = do_capture(cap, efw, label, CAPTURES_DIR)
            captured += 1
            if category == "healthy":
                healthy_count += 1
            else:
                stressed_count += 1

            print(f"  ✅ Captured #{captured}  ({category})")
            print(f"     Red: {os.path.basename(red_p)}")
            print(f"     NIR: {os.path.basename(nir_p)}")
            print(f"     Running total — 🌿 healthy: {healthy_count}  🟤 stressed: {stressed_count}")

        # ── Free-form captures ────────────────────────────────────────
        if args.free > 0:
            print(f"\n{'═' * 62}")
            print(f"  FREE-FORM MODE — {args.free} additional captures")
            print(f"  Point at anything you want. Type a short label or ENTER for auto.")
            print(f"{'═' * 62}")

            for j in range(1, args.free + 1):
                print(f"\n  [Free {j}/{args.free}]")
                action = input("  Label (or ENTER for 'free_NNN', q=quit): ").strip()

                if action.lower() == 'q':
                    break

                label = action.replace(" ", "_") if action else f"free_{j:03d}"
                red_p, nir_p = do_capture(cap, efw, label, CAPTURES_DIR)
                captured += 1
                print(f"  ✅ Captured #{captured}: {label}")

    finally:
        cap.release()
        efw.close()
        print(f"\n{'═' * 62}")
        print(f"  SESSION COMPLETE")
        print(f"  Total captured: {captured}")
        print(f"  Skipped: {skipped}")
        print(f"  🌿 Healthy: {healthy_count}  🟤 Stressed: {stressed_count}")
        print(f"  Hardware released.")
        print(f"{'═' * 62}\n")
        print(f"  Next step: python classify_and_sort.py --batch")
        print()


if __name__ == "__main__":
    main()

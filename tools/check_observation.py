#!/usr/bin/env python3
"""
Checks the policy observation (observation.py) against ground truth.

Flies the scripted mission without a window and, at 10 Hz, builds the
observation from exactly what the sub received (measured sensors and simulated
detector boxes). Compares the gate's bearing and range with the truth while it's
in view. Then checks that a remembered bearing is corrected as the sub turns.

Usage: .venv/bin/python tools/check_observation.py [--detector noisy|perfect] [--seed N]
"""
import argparse
import io
import math
import os
import random
import statistics
import sys
from contextlib import redirect_stdout

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.submarine import Submarine                     # noqa: E402
from main import build_mission                         # noqa: E402
from observation import (ObservationBuilder, OWN_FEATURES, BEARING_SCALE,  # noqa: E402
                         RANGE_SCALE, MAX_UNSEEN)
from simulator import SubmarineSimulator               # noqa: E402

DT = 1 / 60
POLICY_PERIOD = 0.1
GATE = OWN_FEATURES   # Index of the gate's features in a frame


def true_gate(core):
    s, g = core.state, core.gate
    dx, dy = g.x - s.x, g.center_y - s.y
    bearing = (math.degrees(math.atan2(dy, dx)) - s.heading + 180) % 360 - 180
    return bearing, math.hypot(dx, dy)


def summarize(name, values, unit):
    if not values:
        print(f"  {name}: no samples")
        return
    a = sorted(abs(v) for v in values)
    print(f"  {name}: n={len(a)}  median |err| {statistics.median(a):.3f}{unit}  "
          f"95th pct {a[int(0.95 * (len(a) - 1))]:.3f}{unit}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--detector", choices=["noisy", "perfect"], default="noisy")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    with redirect_stdout(io.StringIO()):
        sim = SubmarineSimulator(Submarine(mission_plan=build_mission()), detector_mode=args.detector)
    sim.renderCamera = False
    core, ai = sim.core, sim.submarineAI
    builder = ObservationBuilder(core.camera)
    print(f"Observation: {builder.frame_size} values per frame x {builder.stack} frames = {builder.size}")

    seen_b, seen_r, finite = [], [], True
    t, next_obs = 0.0, 0.0
    while t < 300 and ai.current_task_index < len(ai.mission_plan):
        with redirect_stdout(io.StringIO()):
            sim.step(DT)
        t += DT
        if t < next_obs:
            continue
        next_obs += POLICY_PERIOD
        s = ai._latest_sensors
        obs = builder.update(s.depth, s.depth_rate, s.heading, s.pitch, s.imu.gyro_y, s.imu.gyro_z,
                             s.detections.red_blobs, last_action=[0.0, 0.0, 0.0], dt=POLICY_PERIOD)
        finite &= bool(all(math.isfinite(v) for v in obs))
        found, bearing, rng, _, unseen = obs[GATE:GATE + 5]
        if found:
            tb, tr = true_gate(core)
            seen_b.append(bearing * BEARING_SCALE - tb)
            seen_r.append((rng * RANGE_SCALE - tr) / tr)

    print(f"Mission {'completed' if ai.current_task_index >= len(ai.mission_plan) else 'did NOT complete'}, "
          f"{t:.0f} s, gate passes {core.gate_passes}. All values finite: {finite}\n")
    print("Gate while in view:")
    summarize("bearing error", seen_b, " deg")
    summarize("range error (fraction of true range)", seen_r, "")
    check_memory(core.camera)


def check_memory(camera):
    """See the gate dead ahead at heading 100, lose it, turn to 130: it should now be 30 deg right."""
    b = ObservationBuilder(camera)
    w, h = camera.width, camera.height
    poles = [{'center_x': x, 'center_y': h / 2, 'width': 6, 'height': 200, 'min_x': x - 3, 'max_x': x + 3,
              'min_y': h / 2 - 100, 'max_y': h / 2 + 100, 'area': 1000} for x in (w / 2 - 100, w / 2 + 100)]
    b.update(1.5, 0.0, 100.0, 0.0, 0.0, 0.0, poles, [0, 0, 0], 0.1)
    obs = b.update(1.5, 0.0, 130.0, 0.0, 0.0, 0.0, [], [0, 0, 0], 0.1)
    found, bearing, _, _, unseen = obs[GATE:GATE + 5]
    bearing_deg, unseen_s = bearing * BEARING_SCALE, unseen * MAX_UNSEEN
    ok = not found and abs(bearing_deg + 30) < 1e-3 and abs(unseen_s - 0.1) < 1e-6
    print(f"\nMemory while turning: found={found:.0f}, bearing {bearing_deg:+.1f} deg (expect -30, i.e. right), "
          f"unseen {unseen_s:.1f} s -> {'OK' if ok else 'WRONG'}")


if __name__ == "__main__":
    main()

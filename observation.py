#!/usr/bin/env python3
"""
Builds the policy's observation from what the real sub measures: the pressure
sensor, IMU/magnetometer and the camera detector's boxes. Shared with the
Raspberry Pi (numpy + standard library only), so training and the real sub
compute exactly the same thing.

One frame of the observation (all roughly in [-1, 1]):
  own state   depth, depth rate, sin/cos of heading relative to the start heading,
              pitch, pitch rate, yaw rate
  per object  found now, bearing, range, range confidence, time since last seen
              for the gate (the pair of red poles) and for the single most
              prominent red pole (which helps once the gate no longer fits in view)
  last action whatever the policy commanded last step
The policy sees the last `stack` frames concatenated, newest first, so it can
sense its own motion from how the detections change.

When an object isn't seen, its last bearing and range are kept (the bearing
corrected for how much the sub has turned since), with found = 0 and the time
since it was seen counting up. Before the first sighting everything is 0 and
the time since seen is at its maximum.

Heading is relative to the heading at reset, not absolute, so the policy can't
memorize which compass direction the gate is in.
"""
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from perception import CameraModel, Estimate, estimate_gate, estimate_pole, find_gate_pair

# Scales that bring each value to roughly [-1, 1]
DEPTH_SCALE = 2.0         # m
DEPTH_RATE_SCALE = 0.5    # m/s
PITCH_SCALE = 30.0        # deg
RATE_SCALE = 1.0          # rad/s
BEARING_SCALE = 35.0      # deg (half the field of view); held bearings can exceed 1
RANGE_SCALE = 10.0        # m
MAX_UNSEEN = 5.0          # s; time since seen is clipped here

# How much each range method is trusted (perception.Estimate.method)
RANGE_CONFIDENCE = {"poles": 1.0, "bottom": 1.0, "height": 0.8, "separation": 0.6, "width": 0.3, "none": 0.0}

OBJECT_FEATURES = 5       # found, bearing, range, confidence, time since seen
OWN_FEATURES = 7
OBJECTS = ("gate", "pole")


@dataclass
class CourseGeometry:
    """What the sub knows about the course in advance."""
    gate_width_m: float = 2.0
    pole_diameter_m: float = 0.0334     # 1in PVC
    pole_height_m: float = 1.09         # Visible length, top bar to floor
    pool_depth_m: float = 2.09          # Where the poles' bottom ends are


@dataclass
class _Track:
    found: bool = False
    seen_ever: bool = False
    abs_bearing: float = 0.0     # deg, heading-compensated: heading + bearing at sighting
    range_m: float = 0.0
    confidence: float = 0.0
    unseen: float = MAX_UNSEEN

    def update(self, est: Optional[Estimate], heading: float, dt: float):
        if est is None or est.range_m is None:
            self.found = False
            self.unseen = min(MAX_UNSEEN, self.unseen + dt)
            return
        self.found = self.seen_ever = True
        self.abs_bearing = heading + est.bearing_deg
        self.range_m = est.range_m
        self.confidence = RANGE_CONFIDENCE.get(est.method, 0.0)
        self.unseen = 0.0

    def features(self, heading: float) -> list:
        if not self.seen_ever:
            return [0.0, 0.0, 0.0, 0.0, 1.0]
        bearing = (self.abs_bearing - heading + 180) % 360 - 180
        return [float(self.found),
                float(np.clip(bearing / BEARING_SCALE, -180 / BEARING_SCALE, 180 / BEARING_SCALE)),
                float(np.clip(self.range_m / RANGE_SCALE, 0.0, 3.0)),
                self.confidence,
                self.unseen / MAX_UNSEEN]


class ObservationBuilder:
    def __init__(self, camera: CameraModel, course: Optional[CourseGeometry] = None,
                 action_dim: int = 3, stack: int = 4):
        self.camera = camera
        self.course = course or CourseGeometry()
        self.action_dim = action_dim
        self.stack = stack
        self.frame_size = OWN_FEATURES + OBJECT_FEATURES * len(OBJECTS) + action_dim
        self.size = self.frame_size * stack
        self.reset()

    def reset(self, start_heading: Optional[float] = None):
        """start_heading: heading (deg) that relative heading is measured from; None = first reading."""
        self.start_heading = start_heading
        self.tracks = {name: _Track() for name in OBJECTS}
        self.frames: deque = deque(maxlen=self.stack)

    # --- Detections -> estimates ---
    def _estimate_pole(self, box: dict, depth: float, pitch: float) -> Estimate:
        c = self.course
        return estimate_pole(box, self.camera, diameter_m=c.pole_diameter_m, height_m=c.pole_height_m,
                             bottom_depth_m=c.pool_depth_m, camera_depth_m=depth, pitch_deg=pitch)

    def _estimates(self, red_boxes: Sequence[dict], depth: float, pitch: float):
        gate_est, pole_est = None, None
        if red_boxes:
            biggest = max(red_boxes, key=lambda b: b['area'])
            pole_est = self._estimate_pole(biggest, depth, pitch)
        pair = find_gate_pair(list(red_boxes), self.camera.height)
        if pair:
            left, right = pair
            gate_est = estimate_gate(left, right, self.camera, self.course.gate_width_m,
                                     self._estimate_pole(left, depth, pitch),
                                     self._estimate_pole(right, depth, pitch))
        return gate_est, pole_est

    # --- Main entry point ---
    def update(self, depth: float, depth_rate: float, heading: float, pitch: float,
               pitch_rate: float, yaw_rate: float, red_boxes: Sequence[dict],
               last_action: Sequence[float], dt: float) -> np.ndarray:
        """Adds this step's frame and returns the stacked observation (float32, length self.size)."""
        if self.start_heading is None:
            self.start_heading = heading
        gate_est, pole_est = self._estimates(red_boxes, depth, pitch)
        self.tracks["gate"].update(gate_est, heading, dt)
        self.tracks["pole"].update(pole_est, heading, dt)

        rel_heading = math.radians(heading - self.start_heading)
        frame = [depth / DEPTH_SCALE, depth_rate / DEPTH_RATE_SCALE,
                 math.sin(rel_heading), math.cos(rel_heading),
                 pitch / PITCH_SCALE, pitch_rate / RATE_SCALE, yaw_rate / RATE_SCALE]
        for name in OBJECTS:
            frame += self.tracks[name].features(heading)
        frame += [float(a) for a in last_action]
        assert len(frame) == self.frame_size, (len(frame), self.frame_size)

        frame = np.asarray(frame, dtype=np.float32)
        if not self.frames:
            for _ in range(self.stack - 1):
                self.frames.append(frame)   # Fill the history with the first frame
        self.frames.append(frame)
        return np.concatenate(list(reversed(self.frames)))   # Newest first

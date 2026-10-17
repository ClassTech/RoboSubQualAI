#!/usr/bin/env python3
"""
Simulated object detector. Computes the bounding boxes the camera pipeline would
report, straight from the sub's true pose and the camera geometry, in the same
format as find_blobs_hsv(). This is much faster than rendering an image and
running OpenCV on it, which matters for training.

Noise, in order: box edge jitter, missed detections (range-dependent, plus
dropouts that last a few frames), false positives, and latency. Each reset()
draws new noise levels, so the policy sees a range of camera conditions.

Not modeled: occlusion (one object hiding another) and the gray top bar, which
the red/green detectors ignore anyway.
"""
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config import DetectorNoiseConfig
from perception import CameraModel
from world import PrequalGate, PrequalMarker, SubmarinePhysicsState

NEAR_PLANE = 0.2  # m, same cutoff as the simulator's project3D()


@dataclass
class _Pole:
    name: str
    color: str        # 'red' or 'green'
    x: float
    y: float
    z_top: float
    z_bottom: float
    radius: float


@dataclass
class Detections:
    red_blobs: List[Dict]
    green_blobs: List[Dict]


def _to_camera(state: SubmarinePhysicsState, p: Tuple[float, float, float]):
    """World point -> camera frame (forward, left, up), matching project3D()."""
    dx, dy, dz = p[0] - state.x, p[1] - state.y, p[2] - state.z
    h, pt = math.radians(-state.heading), math.radians(-state.pitch)
    ch, sh, cp, sp = math.cos(h), math.sin(h), math.cos(pt), math.sin(pt)
    x_yaw, y_yaw = dx * ch - dy * sh, dx * sh + dy * ch
    forward = x_yaw * cp + dz * sp
    up = x_yaw * sp - dz * cp
    return forward, y_yaw, up


def make_box(min_x: float, min_y: float, max_x: float, max_y: float) -> Dict:
    x, y = int(round(min_x)), int(round(min_y))
    w, h = int(round(max_x)) - x, int(round(max_y)) - y
    return {
        'center_x': x + w / 2, 'center_y': y + h / 2,
        'height': h, 'width': w,
        'min_x': x, 'max_x': x + w, 'min_y': y, 'max_y': y + h,
        # cv2.contourArea of a filled w x h rectangle
        'area': max(0, (w - 1) * (h - 1)),
    }


class DetectorModel:
    def __init__(self, cam: CameraModel, gate: Optional[PrequalGate], marker: Optional[PrequalMarker],
                 pole_bottom_z: float, gate_pole_radius: float,
                 min_red_area: int = 50, min_green_area: int = 20,
                 noise: Optional[DetectorNoiseConfig] = None, noisy: bool = True,
                 rng: Optional[random.Random] = None):
        self.cam = cam
        self.min_area = {'red': min_red_area, 'green': min_green_area}
        self.noise_cfg = noise or DetectorNoiseConfig()
        self.noisy = noisy
        self.rng = rng or random.Random()
        self.poles: List[_Pole] = []
        if gate:
            for name, side in (("gate left pole", -1), ("gate right pole", 1)):
                self.poles.append(_Pole(name, 'red', gate.x, gate.center_y + side * gate.width / 2,
                                        gate.z_top, pole_bottom_z, gate_pole_radius))
        if marker:
            self.poles.append(_Pole("marker", 'green', marker.x, marker.y,
                                    marker.z_top, marker.z_bottom, marker.radius))
        self.reset()

    def reset(self):
        n, r = self.noise_cfg, self.rng
        self.edge_px = r.uniform(*n.edgeJitterPx)
        self.edge_frac = r.uniform(*n.edgeJitterFrac)
        self.dropout_rate = r.uniform(*n.dropoutRatePerSec)
        self.clear_range = r.uniform(*n.clearRange)
        self.max_range = max(self.clear_range + 0.1, r.uniform(*n.maxRange))
        self.fp_rate = r.uniform(*n.falsePositivesPerSec)
        self.latency = r.uniform(*n.latencySec)
        self.time = 0.0
        self.dropout_left = {p.name: 0.0 for p in self.poles}
        self.history: deque = deque()

    # --- Geometry ---
    def project_pole(self, state: SubmarinePhysicsState, pole: _Pole) -> Optional[Dict]:
        """Noise-free bounding box of a pole, clipped to the image, or None if not in view."""
        top = _to_camera(state, (pole.x, pole.y, pole.z_top))
        bot = _to_camera(state, (pole.x, pole.y, pole.z_bottom))
        # Clip the pole's axis to the part in front of the near plane.
        if top[0] < NEAR_PLANE and bot[0] < NEAR_PLANE:
            return None
        if top[0] < NEAR_PLANE or bot[0] < NEAR_PLANE:
            front, back = (top, bot) if top[0] >= NEAR_PLANE else (bot, top)
            t = (front[0] - NEAR_PLANE) / (front[0] - back[0])
            back = tuple(front[i] + t * (back[i] - front[i]) for i in range(3))
            top, bot = front, back

        w, h, f = self.cam.width, self.cam.height, self.cam.focal_px
        xs, ys = [], []
        for fwd, left, up in (top, bot):
            u = w / 2 - f * left / fwd
            v = h / 2 - f * up / fwd
            half_w = f * pole.radius / fwd
            xs += [u - half_w, u + half_w]
            ys.append(v)
        min_x, max_x = max(0.0, min(xs)), min(float(w), max(xs))
        min_y, max_y = max(0.0, min(ys)), min(float(h), max(ys))
        if max_x - min_x < 1 or max_y - min_y < 1:
            return None
        return make_box(min_x, min_y, max_x, max_y)

    # --- Noise ---
    def _jitter(self, box: Dict) -> Optional[Dict]:
        g = self.rng.gauss
        sx = self.edge_px + self.edge_frac * box['width']
        sy = self.edge_px + self.edge_frac * box['height']
        min_x, max_x = box['min_x'] + g(0, sx), box['max_x'] + g(0, sx)
        min_y, max_y = box['min_y'] + g(0, sy), box['max_y'] + g(0, sy)
        w, h = self.cam.width, self.cam.height
        min_x, max_x = max(0.0, min_x), min(float(w), max_x)
        min_y, max_y = max(0.0, min_y), min(float(h), max_y)
        if max_x - min_x < 1 or max_y - min_y < 1:
            return None
        return make_box(min_x, min_y, max_x, max_y)

    def _detected(self, pole: _Pole, state: SubmarinePhysicsState, dt: float) -> bool:
        """Range-dependent misses plus dropouts lasting several frames."""
        if self.dropout_left[pole.name] > 0:
            self.dropout_left[pole.name] -= dt
            return False
        if self.rng.random() < self.dropout_rate * dt:
            self.dropout_left[pole.name] = self.rng.expovariate(1 / self.noise_cfg.dropoutMeanSec)
            return False
        dist = math.hypot(pole.x - state.x, pole.y - state.y)
        if dist <= self.clear_range:
            return True
        p_miss = (dist - self.clear_range) / (self.max_range - self.clear_range)
        return self.rng.random() >= p_miss

    def _false_positive(self) -> Dict:
        r = self.rng
        bw, bh = r.uniform(3, 20), r.uniform(10, 120)   # Pole-shaped, to be tempting
        x, y = r.uniform(0, self.cam.width - bw), r.uniform(0, self.cam.height - bh)
        return make_box(x, y, x + bw, y + bh)

    # --- Main entry point ---
    def detect(self, state: SubmarinePhysicsState, dt: float) -> Detections:
        self.time += dt
        blobs = {'red': [], 'green': []}
        for pole in self.poles:
            box = self.project_pole(state, pole)
            if box is None:
                continue
            if self.noisy:
                if not self._detected(pole, state, dt):
                    continue
                box = self._jitter(box)
                if box is None:
                    continue
            if box['area'] >= self.min_area[pole.color]:
                blobs[pole.color].append(box)
        if self.noisy:
            for color in blobs:
                if self.rng.random() < self.fp_rate * dt:
                    fp = self._false_positive()
                    if fp['area'] >= self.min_area[color]:
                        blobs[color].append(fp)
        current = Detections(blobs['red'], blobs['green'])
        if not self.noisy:
            return current

        # Latency: report the newest detections at least `latency` seconds old.
        self.history.append((self.time, current))
        cutoff = self.time - self.latency
        while len(self.history) > 1 and self.history[1][0] <= cutoff:
            self.history.popleft()
        if self.history[0][0] <= cutoff:
            return self.history[0][1]
        return Detections([], [])   # Nothing old enough yet (start of episode)

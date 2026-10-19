#!/usr/bin/env python3
"""
The sub's low-level control loops: depth (PID), pitch (PD) and heading (PD),
plus the mixer that turns surge/sway/yaw/heave/pitch efforts into the six
thruster commands. Shared by the scripted tasks, the RL environment and the
Raspberry Pi (numpy + standard library only).

Only measured values are used: depth and depth rate from the pressure sensor,
heading from the magnetometer, pitch and rates from the IMU.
"""
from dataclasses import dataclass

import numpy as np

from core_types import ThrusterCommands


def angle_diff(a1: float, a2: float) -> float:
    """Shortest difference a1 - a2 between two angles in degrees, in (-180, 180]."""
    d = (a1 - a2) % 360
    return d - 360 if d > 180 else d


@dataclass
class ControllerGains:
    depth_p: float = 2.5
    depth_i: float = 0.4
    depth_d: float = 5.0
    depth_i_clamp: float = 2.0
    pitch_p: float = 1.0
    pitch_d: float = 1.5
    heading_p: float = 0.1      # Per degree of heading error
    yaw_d: float = 4.0          # Per rad/s of yaw rate


def depth_pitch_efforts(gains: ControllerGains, depth: float, depth_rate: float, depth_integral: float,
                        target_depth: float, pitch: float, pitch_rate: float, target_pitch: float = 0.0):
    """Heave and pitch efforts in [-1, 1]. depth_integral is the accumulated depth error (m s)."""
    heave = (target_depth - depth) * gains.depth_p + depth_integral * gains.depth_i - depth_rate * gains.depth_d
    pitch_effort = angle_diff(target_pitch, pitch) * gains.pitch_p - pitch_rate * gains.pitch_d
    return float(np.clip(heave, -1.0, 1.0)), float(np.clip(pitch_effort, -1.0, 1.0))


def yaw_effort(gains: ControllerGains, heading: float, yaw_rate: float, target_heading: float) -> float:
    return float(np.clip(angle_diff(target_heading, heading) * gains.heading_p - yaw_rate * gains.yaw_d, -1.0, 1.0))


def mix(surge: float, sway: float, yaw: float, heave: float, pitch: float) -> ThrusterCommands:
    """
    Mixes 5-axis efforts into 6 thruster commands. The horizontal and vertical
    thrusters are separate hardware, so each group is scaled down on its own if
    any of its commands exceeds 1. (Scaling all six together meant hard turning
    and sliding cut the vertical thrust, and the buoyant sub floated up.)
    """
    cos_45 = 0.7071
    surge_cmd, sway_cmd = surge * cos_45, sway * cos_45
    c = ThrusterCommands(
        hfl=surge_cmd + sway_cmd + yaw, hfr=surge_cmd - sway_cmd - yaw,
        hal=surge_cmd - sway_cmd + yaw, har=surge_cmd + sway_cmd - yaw,
        vf=heave + pitch, vr=heave - pitch)
    h_max = max(1.0, abs(c.hfl), abs(c.hfr), abs(c.hal), abs(c.har))
    c.hfl /= h_max; c.hfr /= h_max; c.hal /= h_max; c.har /= h_max
    v_max = max(1.0, abs(c.vf), abs(c.vr))
    c.vf /= v_max; c.vr /= v_max
    return c


class SetpointController:
    """Holds a heading and depth at given surge and sway efforts. Used by the policy (in sim and on the Pi)."""
    def __init__(self, gains: ControllerGains = None):
        self.gains = gains or ControllerGains()
        self.reset()

    def reset(self):
        self.depth_integral = 0.0

    def update(self, dt: float, depth: float, depth_rate: float, heading: float, pitch: float,
               pitch_rate: float, yaw_rate: float, target_heading: float, target_depth: float,
               surge: float, sway: float = 0.0) -> ThrusterCommands:
        g = self.gains
        self.depth_integral = float(np.clip(self.depth_integral + (target_depth - depth) * dt,
                                            -g.depth_i_clamp, g.depth_i_clamp))
        heave, pitch_effort = depth_pitch_efforts(g, depth, depth_rate, self.depth_integral,
                                                  target_depth, pitch, pitch_rate)
        yaw = yaw_effort(g, heading, yaw_rate, target_heading)
        return mix(float(np.clip(surge, -1.0, 1.0)), float(np.clip(sway, -1.0, 1.0)), yaw, heave, pitch_effort)

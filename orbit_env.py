#!/usr/bin/env python3
"""
Gymnasium environment for the orbit skill: start just past the gate, go around
the marker pole (either side), and come back facing the gate. Actions,
observation and the episode loop are in sub_env.py; this skill adds sway as a
fourth action and a marker track to the observation.

The only camera faces forward (±35°), so the sub can't see the marker while
circling it nose-first. With sway it can face the marker and slide around it,
keeping it in view, which is how the scripted orbit does it.

Success needs both:
  - truth: the sub has swept at least SWEEP_DEG around the marker, either way, so
    it really went around (not out and back on one side)
  - handoff, measurable on the real sub: the gate is in view and the sub's
    heading has turned 180° (± HANDOFF_HEADING_DEG) since the start of the orbit
Failures: a collision, crossing back through the gate plane early, or a timeout.
"""
import math
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from sub_env import SubTaskEnv

SWEEP_DEG = 270.0
HANDOFF_HEADING_DEG = 45.0


def _wrap(deg: float) -> float:
    return (deg + 180) % 360 - 180


@dataclass
class OrbitStartConfig:
    """Where episodes start, relative to the gate. Each value is drawn uniformly from (min, max)."""
    past_gate_m: tuple = (1.0, 5.0)       # Beyond the gate, toward the marker
    lateral_m: tuple = (-1.0, 1.0)        # Sideways from the gate's center line
    heading_deg: tuple = (-30.0, 30.0)    # Relative to facing the marker's direction
    depth_m: tuple = (1.2, 1.7)


@dataclass
class OrbitRewardConfig:
    sweep: float = 2.0             # Per radian swept around the marker, up to SWEEP_DEG
    face_gate: float = 2.0         # For facing the gate, phased in over the second half of the sweep
    depth_weight: float = 1.0      # Per meter from target_depth_m
    target_depth_m: float = 1.5
    far_weight: float = 0.5        # Per meter farther than far_m from the marker
    far_m: float = 3.0
    success_bonus: float = 20.0
    collision_penalty: float = 5.0
    wrong_way_penalty: float = 5.0
    timeout_penalty: float = 5.0
    time_penalty: float = 0.01     # Per step
    energy_penalty: float = 0.05   # Per second at full thrust on every thruster


class OrbitEnv(SubTaskEnv):
    OUTCOMES = ("is_success", "collision", "wrong_way", "timeout")
    INCLUDE_MARKER = True
    ACTION_DIM = 4

    def __init__(self, detector_mode: str = "perfect", control_period: float = 0.1,
                 max_seconds: float = 90.0, start: Optional[OrbitStartConfig] = None,
                 reward: Optional[OrbitRewardConfig] = None):
        self.start_cfg = start or OrbitStartConfig()
        self.reward_cfg = rc = reward or OrbitRewardConfig()
        super().__init__(detector_mode, control_period, max_seconds, energy_penalty=rc.energy_penalty,
                         time_penalty=rc.time_penalty, progress_weight=1.0,
                         timeout_penalty=rc.timeout_penalty)

    # --- Episode ---
    def _place_sub(self, rng: random.Random):
        s, g, sc = self.core.state, self.core.gate, self.start_cfg
        s.x = g.x + rng.uniform(*sc.past_gate_m)
        s.y = g.center_y + rng.uniform(*sc.lateral_m)
        s.heading = rng.uniform(*sc.heading_deg) % 360
        s.z = rng.uniform(*sc.depth_m)

    def _marker_angle(self) -> float:
        """The sub's angular position around the marker (deg), ground truth."""
        s, m = self.core.state, self.core.marker
        return math.degrees(math.atan2(s.y - m.y, s.x - m.x))

    def _on_reset(self):
        self.swept = 0.0                       # Signed degrees swept around the marker
        self._last_angle = self._marker_angle()

    def _physics_outcome(self) -> Optional[str]:
        angle = self._marker_angle()
        self.swept += _wrap(angle - self._last_angle)
        self._last_angle = angle
        if self.new_crossings():
            return "wrong_way"
        return None

    def handoff_ready(self) -> bool:
        """Measurable on the real sub: gate in view, heading turned about 180° since the orbit began."""
        b = self.builder
        if b.start_heading is None or not b.gate.found:
            return False
        return abs(_wrap(self.readings.heading - b.start_heading - 180.0)) <= HANDOFF_HEADING_DEG

    def _step_outcome(self) -> Optional[str]:
        if abs(self.swept) >= SWEEP_DEG and self.handoff_ready():
            return "success"
        return None

    def _terminal_reward(self, outcome: str) -> float:
        rc = self.reward_cfg
        return {"success": rc.success_bonus, "collision": -rc.collision_penalty,
                "wrong_way": -rc.wrong_way_penalty}[outcome]

    def _potential(self) -> float:
        """Sweep around the marker, then face the gate, at mission depth, without drifting far. Ground truth."""
        s, m, g, rc = self.core.state, self.core.marker, self.core.gate, self.reward_cfg
        sweep = min(abs(self.swept), SWEEP_DEG)
        to_gate = math.degrees(math.atan2(g.center_y - s.y, g.x - s.x))
        facing = 1.0 - abs(_wrap(to_gate - s.heading)) / 180.0          # 1 facing the gate, 0 facing away
        phase_in = float(np.clip((abs(self.swept) - 180.0) / (SWEEP_DEG - 180.0), 0.0, 1.0))
        distance = math.hypot(s.x - m.x, s.y - m.y)
        return (rc.sweep * math.radians(sweep) + rc.face_gate * phase_in * facing
                - rc.depth_weight * abs(s.z - rc.target_depth_m)
                - rc.far_weight * max(0.0, distance - rc.far_m))

    def scripted_mission(self) -> List:
        # The scripted mission's orbit, the pause after it, and the return toward the gate
        from main import build_mission
        return build_mission()[2:5]

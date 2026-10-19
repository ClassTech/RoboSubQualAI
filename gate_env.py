#!/usr/bin/env python3
"""
Gymnasium environment for the gate skill: start somewhere in front of the gate
and go through it. Actions, observation and the episode loop are in sub_env.py.

An episode ends when the sub goes forward through the gate (success, +20), hits
something, crosses the gate line anywhere else, or runs out of time (-5 each).
"""
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from sub_env import DEPTH_MAX, DEPTH_MIN, HEADING_CHANGE_DEG, SubTaskEnv  # noqa: F401  (re-exported)


@dataclass
class StartConfig:
    """Where episodes start, relative to the gate. Each value is drawn uniformly from (min, max)."""
    distance_m: tuple = (3.0, 8.0)        # In front of the gate
    lateral_m: tuple = (-1.5, 1.5)        # Sideways from the gate's center line
    heading_deg: tuple = (-30.0, 30.0)    # Relative to facing the gate squarely
    depth_m: tuple = (0.1, 0.5)


@dataclass
class RewardConfig:
    progress: float = 1.0          # Per meter closer to the gate center (horizontal + depth + center line)
    depth_weight: float = 1.0      # How much depth error counts in that distance
    center_line_weight: float = 0.5  # How much distance from the gate's center line counts
    target_depth_m: float = 1.5    # Middle of the usable gate opening
    gate_bonus: float = 20.0
    collision_penalty: float = 5.0
    missed_gate_penalty: float = 5.0
    timeout_penalty: float = 5.0   # So hovering near the gate isn't a safe alternative to committing
    time_penalty: float = 0.01     # Per step
    energy_penalty: float = 0.05   # Per second at full thrust on every thruster


class GateEnv(SubTaskEnv):
    OUTCOMES = ("is_success", "collision", "missed_gate", "surfaced", "timeout")

    def __init__(self, detector_mode: str = "perfect", control_period: float = 0.1,
                 max_seconds: float = 60.0, start: Optional[StartConfig] = None,
                 reward: Optional[RewardConfig] = None):
        self.start_cfg = start or StartConfig()
        self.reward_cfg = rc = reward or RewardConfig()
        super().__init__(detector_mode, control_period, max_seconds, energy_penalty=rc.energy_penalty,
                         time_penalty=rc.time_penalty, progress_weight=rc.progress,
                         timeout_penalty=rc.timeout_penalty)

    def _place_sub(self, rng: random.Random):
        s, g, sc = self.core.state, self.core.gate, self.start_cfg
        s.x = g.x - rng.uniform(*sc.distance_m)
        s.y = g.center_y + rng.uniform(*sc.lateral_m)
        s.heading = rng.uniform(*sc.heading_deg) % 360
        s.z = rng.uniform(*sc.depth_m)

    def _physics_outcome(self) -> Optional[str]:
        for c in self.new_crossings():
            return "success" if c.inside_opening and c.direction > 0 else "missed_gate"
        return None

    def _terminal_reward(self, outcome: str) -> float:
        rc = self.reward_cfg
        return {"success": rc.gate_bonus, "collision": -rc.collision_penalty,
                "missed_gate": -rc.missed_gate_penalty}[outcome]

    def _potential(self) -> float:
        """Minus the distance to the gate center (horizontal, plus weighted depth error and distance
        from the gate's center line, so approaching square-on pays). Ground truth."""
        s, g, rc = self.core.state, self.core.gate, self.reward_cfg
        horizontal = float(np.hypot(g.x - s.x, g.center_y - s.y))
        return -(horizontal + rc.depth_weight * abs(s.z - rc.target_depth_m)
                 + rc.center_line_weight * abs(s.y - g.center_y))

    def scripted_mission(self) -> List:
        from ai.tasks import GateTask
        return [GateTask(target_depth=1.5)]

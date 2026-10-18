#!/usr/bin/env python3
"""
Gymnasium environment for the gate task: start somewhere in front of the gate
and go through it.

The policy acts at 10 Hz. Each action sets targets that the sub's own control
loops (controller.py) then hold at the physics rate until the next action:
  action[0]  heading change from the current heading, -1..1 -> -45..+45 deg (+ = left)
  action[1]  target depth, -1..1 -> 0.3..1.9 m
  action[2]  surge effort (open-loop forward/back thrust), -1..1
The policy sees observation.py's observation, built only from what the real
sub measures.

An episode ends when the sub goes forward through the gate (success, +20), hits
something, crosses the gate line anywhere else, or runs out of time (-5 each).
At the end, info carries the outcome for logging: is_success, collision,
missed_gate, timeout, time_s and energy.
"""
import random
from dataclasses import dataclass
from typing import Callable, Optional

import gymnasium as gym
import numpy as np

from controller import SetpointController
from core_types import ThrusterCommands
from observation import ObservationBuilder
from sensor_model import SensorReadings
from sim_core import SubSim

HEADING_CHANGE_DEG = 45.0
DEPTH_MIN, DEPTH_MAX = 0.3, 1.9


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


Driver = Callable[[SensorReadings, object, float], ThrusterCommands]


class GateEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, detector_mode: str = "perfect", control_period: float = 0.1,
                 max_seconds: float = 60.0, start: Optional[StartConfig] = None,
                 reward: Optional[RewardConfig] = None):
        assert detector_mode in ("perfect", "noisy"), detector_mode
        self.detector_mode = detector_mode
        self.control_period = control_period
        self.max_seconds = max_seconds
        self.start_cfg = start or StartConfig()
        self.reward_cfg = reward or RewardConfig()
        self.controller = SetpointController()
        self.on_physics_step = None   # Optional callback(env, thruster_commands) after each physics step, for viewers
        self.core = SubSim(detector_mode=detector_mode, log=None)
        self.builder = ObservationBuilder(self.core.camera, action_dim=3)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-10.0, 10.0, shape=(self.builder.size,), dtype=np.float32)

    # --- Episode start ---
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        rng = random.Random(int(self.np_random.integers(2 ** 31)))
        self.core = SubSim(detector_mode=self.detector_mode, rng=rng, log=None)
        s, g, sc = self.core.state, self.core.gate, self.start_cfg
        s.x = g.x - rng.uniform(*sc.distance_m)
        s.y = g.center_y + rng.uniform(*sc.lateral_m)
        s.heading = rng.uniform(*sc.heading_deg) % 360
        s.z = rng.uniform(*sc.depth_m)

        self.controller.reset()
        self.builder.reset()
        self.readings = self.core.read_sensors(0.0)
        self.detections = self.core.detect(0.0)
        self.target_heading, self.target_depth = self.readings.heading, self.readings.depth
        self.time, self.energy = 0.0, 0.0
        self.potential = self._potential()
        self.last_action = np.zeros(3, dtype=np.float32)
        return self._observe(), {}

    # --- Acting ---
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.target_heading = (self.readings.heading + HEADING_CHANGE_DEG * float(a[0])) % 360
        self.target_depth = DEPTH_MIN + (float(a[1]) + 1) / 2 * (DEPTH_MAX - DEPTH_MIN)
        surge = float(a[2])

        def driver(r: SensorReadings, detections, dt: float) -> ThrusterCommands:
            return self.controller.update(dt, r.depth, r.depth_rate, r.heading, r.pitch, r.imu.gyro_y,
                                          r.imu.gyro_z, self.target_heading, self.target_depth, surge)
        self.last_action = a
        return self.step_with_driver(driver)

    def step_with_driver(self, driver: Driver):
        """
        Runs one control period with `driver` choosing thruster commands at every
        physics step from the latest readings and detections. step() uses the
        policy's setpoints; evaluation tools can plug in the scripted tasks instead.
        """
        dt = self.core.config.physicsDt
        n = max(1, round(self.control_period / dt))
        crossings_before = len(self.core.crossings)
        collisions_before = self.core.collision_count
        outcome, step_energy = None, 0.0
        for _ in range(n):
            cmd = driver(self.readings, self.detections, dt)
            self.core.advance(dt, cmd)
            step_energy += dt * (abs(cmd.hfl) + abs(cmd.hfr) + abs(cmd.hal) + abs(cmd.har)
                                 + abs(cmd.vf) + abs(cmd.vr)) / 6
            self.readings = self.core.read_sensors(dt)
            self.detections = self.core.detect(dt)
            self.time += dt
            if self.on_physics_step:
                self.on_physics_step(self, cmd)
            if self.core.collision_count > collisions_before:
                outcome = "collision"
            elif len(self.core.crossings) > crossings_before:
                c = self.core.crossings[-1]
                outcome = "success" if c.inside_opening and c.direction > 0 else "missed_gate"
            if outcome:
                break

        self.energy += step_energy
        rc = self.reward_cfg
        new_potential = self._potential()
        reward = rc.progress * (new_potential - self.potential) - rc.time_penalty
        reward -= rc.energy_penalty * step_energy
        self.potential = new_potential
        if outcome == "success":
            reward += rc.gate_bonus
        elif outcome == "collision":
            reward -= rc.collision_penalty
        elif outcome == "missed_gate":
            reward -= rc.missed_gate_penalty

        terminated = outcome is not None
        truncated = not terminated and self.time >= self.max_seconds - 1e-9
        if truncated:
            reward -= rc.timeout_penalty
        info = {}
        if terminated or truncated:
            info = {"is_success": outcome == "success", "collision": outcome == "collision",
                    "missed_gate": outcome == "missed_gate", "timeout": truncated,
                    "time_s": self.time, "energy": self.energy}
        return self._observe(), float(reward), terminated, truncated, info

    # --- Helpers ---
    def _potential(self) -> float:
        """Minus the distance to the gate center (horizontal, plus weighted depth error and distance
        from the gate's center line, so approaching square-on pays). Ground truth."""
        s, g, rc = self.core.state, self.core.gate, self.reward_cfg
        horizontal = float(np.hypot(g.x - s.x, g.center_y - s.y))
        return -(horizontal + rc.depth_weight * abs(s.z - rc.target_depth_m)
                 + rc.center_line_weight * abs(s.y - g.center_y))

    def _observe(self) -> np.ndarray:
        r = self.readings
        red = self.detections.red_blobs if self.detections else []
        obs = self.builder.update(r.depth, r.depth_rate, r.heading, r.pitch, r.imu.gyro_y, r.imu.gyro_z,
                                  red, self.last_action, self.control_period)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

#!/usr/bin/env python3
"""
Base class for the RL environments (one per skill: gate, orbit, ...).

The policy acts at 10 Hz. Each action sets targets that the sub's own control
loops (controller.py) then hold at the physics rate until the next action:
  action[0]  heading change from the current heading, -1..1 -> -45..+45 deg (+ = left)
  action[1]  target depth, -1..1 -> 0.3..1.9 m
  action[2]  surge effort (open-loop forward/back thrust), -1..1
  action[3]  sway effort (open-loop sideways thrust, + = left), -1..1; only skills with ACTION_DIM = 4
The policy sees observation.py's observation, built only from what the real
sub measures.

Each skill subclass decides where episodes start, how they end, and what
counts as progress:
  _place_sub(rng)          put the sub at a random start
  _physics_outcome()       after each physics step: an outcome name, or None
  _step_outcome()          after each 10 Hz step (once the observation is updated): outcome or None
  _potential()             progress measure (ground truth); reward is its change each step
  _terminal_reward(name)   bonus or penalty for how the episode ended
  scripted_mission()       the scripted tasks that do this skill, for comparison
OUTCOMES lists the possible endings ("is_success", "timeout" and the
skill's own failures); info carries them as booleans when an episode ends,
plus time_s and energy. Collisions always end an episode.
"""
import random
from typing import Callable, List, Optional

import gymnasium as gym
import numpy as np

from controller import SetpointController
from core_types import ThrusterCommands
from observation import ObservationBuilder
from sensor_model import SensorReadings
from sim_core import SubSim

HEADING_CHANGE_DEG = 45.0
DEPTH_MIN, DEPTH_MAX = 0.3, 1.9

Driver = Callable[[SensorReadings, object, float], ThrusterCommands]


class SubTaskEnv(gym.Env):
    metadata = {"render_modes": []}
    OUTCOMES = ("is_success", "collision", "timeout")
    INCLUDE_MARKER = False   # Whether the observation tracks the green marker pole
    ACTION_DIM = 3           # 4 adds sway

    def __init__(self, detector_mode: str = "perfect", control_period: float = 0.1,
                 max_seconds: float = 60.0, energy_penalty: float = 0.05, time_penalty: float = 0.01,
                 progress_weight: float = 1.0, timeout_penalty: float = 5.0):
        assert detector_mode in ("perfect", "noisy"), detector_mode
        self.detector_mode = detector_mode
        self.control_period = control_period
        self.max_seconds = max_seconds
        self.energy_penalty, self.time_penalty = energy_penalty, time_penalty
        self.progress_weight, self.timeout_penalty = progress_weight, timeout_penalty
        self.controller = SetpointController()
        self.on_physics_step = None   # Optional callback(env, thruster_commands) after each physics step, for viewers
        self.core = SubSim(detector_mode=detector_mode, log=None)
        self.builder = ObservationBuilder(self.core.camera, action_dim=self.ACTION_DIM,
                                          include_marker=self.INCLUDE_MARKER)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.ACTION_DIM,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-10.0, 10.0, shape=(self.builder.size,), dtype=np.float32)

    # --- For subclasses ---
    def _place_sub(self, rng: random.Random):
        raise NotImplementedError

    def _physics_outcome(self) -> Optional[str]:
        return None

    def _step_outcome(self) -> Optional[str]:
        return None

    def _potential(self) -> float:
        raise NotImplementedError

    def _terminal_reward(self, outcome: str) -> float:
        raise NotImplementedError

    def _on_reset(self):
        pass

    def scripted_mission(self) -> List:
        raise NotImplementedError

    # --- Episode start ---
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        rng = random.Random(int(self.np_random.integers(2 ** 31)))
        self.core = SubSim(detector_mode=self.detector_mode, rng=rng, log=None)
        self._place_sub(rng)

        self.controller.reset()
        self.builder.reset()
        self.readings = self.core.read_sensors(0.0)
        self.detections = self.core.detect(0.0)
        self.target_heading, self.target_depth = self.readings.heading, self.readings.depth
        self.time, self.energy = 0.0, 0.0
        self._crossings_seen, self._collisions_seen = 0, 0
        self._on_reset()
        self.potential = self._potential()
        self.last_action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        return self._observe(), {}

    # --- Acting ---
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.target_heading = (self.readings.heading + HEADING_CHANGE_DEG * float(a[0])) % 360
        self.target_depth = DEPTH_MIN + (float(a[1]) + 1) / 2 * (DEPTH_MAX - DEPTH_MIN)
        surge = float(a[2])
        sway = float(a[3]) if self.ACTION_DIM > 3 else 0.0

        def driver(r: SensorReadings, detections, dt: float) -> ThrusterCommands:
            return self.controller.update(dt, r.depth, r.depth_rate, r.heading, r.pitch, r.imu.gyro_y,
                                          r.imu.gyro_z, self.target_heading, self.target_depth, surge, sway)
        self.last_action = a
        return self.step_with_driver(driver)

    def new_crossings(self) -> list:
        """Gate-plane crossings since the last call (for _physics_outcome)."""
        new = self.core.crossings[self._crossings_seen:]
        self._crossings_seen = len(self.core.crossings)
        return new

    def step_with_driver(self, driver: Driver):
        """
        Runs one control period with `driver` choosing thruster commands at every
        physics step from the latest readings and detections. step() uses the
        policy's setpoints; evaluation tools can plug in the scripted tasks instead.
        """
        dt = self.core.config.physicsDt
        n = max(1, round(self.control_period / dt))
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
            if self.core.collision_count > self._collisions_seen:
                outcome = "collision"
            else:
                outcome = self._physics_outcome()
            if outcome:
                break

        obs = self._observe()
        if not outcome:
            outcome = self._step_outcome()
        self.energy += step_energy
        new_potential = self._potential()
        reward = self.progress_weight * (new_potential - self.potential) - self.time_penalty
        reward -= self.energy_penalty * step_energy
        self.potential = new_potential

        terminated = outcome is not None
        truncated = not terminated and self.time >= self.max_seconds - 1e-9
        if terminated:
            reward += self._terminal_reward(outcome)
        elif truncated:
            reward -= self.timeout_penalty
            outcome = "timeout"
        info = {}
        if terminated or truncated:
            info = {k: (k == "is_success" and outcome == "success") or k == outcome for k in self.OUTCOMES}
            info.update(time_s=self.time, energy=self.energy)
        return obs, float(reward), terminated, truncated, info

    # --- Helpers ---
    def _observe(self) -> np.ndarray:
        r, d = self.readings, self.detections
        obs = self.builder.update(r.depth, r.depth_rate, r.heading, r.pitch, r.imu.gyro_y, r.imu.gyro_z,
                                  d.red_blobs if d else [], self.last_action, self.control_period,
                                  green_boxes=d.green_blobs if d else [])
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

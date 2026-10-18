#!/usr/bin/env python3
"""
Controllers that can fly the gate environment, shared by tools/eval_gate.py
(scoring) and watch.py (viewing):
  scripted   the scripted GateTask (the baseline)
  random     random actions
  straight   hold the start heading at 1.5 m and drive forward
  <path>     a saved Stable-Baselines3 PPO model (.zip)

Each controller's step(env, obs) advances the environment one control period
and returns what env.step() returns.
"""
import io
from contextlib import redirect_stdout

import numpy as np

from gate_env import DEPTH_MAX, DEPTH_MIN, GateEnv


class ActionController:
    """Chooses an action from the observation each step."""
    def __init__(self, name, choose):
        self.name, self.choose, self.last_action = name, choose, None

    def reset(self, env: GateEnv):
        self.last_action = None

    def step(self, env: GateEnv, obs):
        self.last_action = np.asarray(self.choose(obs), dtype=np.float32)
        return env.step(self.last_action)


class _ImageSize:
    """Stands in for the camera image: the scripted tasks only ask for its size."""
    def __init__(self, w, h): self.w, self.h = w, h
    def get_size(self): return self.w, self.h


class ScriptedController:
    """The scripted GateTask, driving the thrusters directly at the physics rate."""
    name, last_action = "scripted", None

    def reset(self, env: GateEnv):
        from ai.submarine import Submarine
        from ai.tasks import GateTask
        with redirect_stdout(io.StringIO()):
            self.sub = Submarine(mission_plan=[GateTask(target_depth=1.5)])
        self.image = _ImageSize(env.core.camera.width, env.core.camera.height)

    def state_name(self) -> str:
        return self.sub.get_current_state_name()

    def step(self, env: GateEnv, obs):
        from data_structures import SensorSuite

        def driver(r, detections, dt):
            s = env.core.state   # Ground truth: only the scripted hover/damping helpers use it (SIM ONLY)
            sensors = SensorSuite(camera_image=self.image, depth=r.depth, heading=r.heading, pitch=r.pitch,
                                  imu=r.imu, depth_rate=r.depth_rate, x=s.x, y=s.y,
                                  velocity_x=s.velocity_x, velocity_y=s.velocity_y,
                                  angular_velocity_y=s.angular_velocity_y, velocity_z=s.velocity_z,
                                  detections=detections)
            with redirect_stdout(io.StringIO()):
                commands, _ = self.sub.update(dt, sensors)
            return commands
        return env.step_with_driver(driver)


def make_controller(name: str, seed: int = 0):
    if name == "scripted":
        return ScriptedController()
    if name == "random":
        rng = np.random.default_rng(seed)
        return ActionController("random", lambda obs: rng.uniform(-1, 1, 3))
    if name == "straight":
        depth_action = (1.5 - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN) * 2 - 1   # Target 1.5 m
        return ActionController("straight", lambda obs: np.array([0.0, depth_action, 0.6]))
    from stable_baselines3 import PPO
    model = PPO.load(name)
    return ActionController(name, lambda obs: model.predict(obs, deterministic=True)[0])

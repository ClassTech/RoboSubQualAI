#!/usr/bin/env python3
"""
Scores a controller on the gate environment over a fixed set of seeds, so
different controllers face exactly the same starts and noise.

Controllers:
  scripted   the scripted GateTask (the baseline the policy has to beat)
  random     random actions
  straight   hold the start heading at 1.5 m and drive forward
  <path>     a saved Stable-Baselines3 PPO model (.zip), once training exists

Usage: .venv/bin/python tools/eval_gate.py scripted --detector noisy --episodes 50
"""
import argparse
import io
import os
import statistics
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                     # noqa: E402

from gate_env import GateEnv                            # noqa: E402

OUTCOMES = ("is_success", "collision", "missed_gate", "timeout")


class _ImageSize:
    """Stands in for the camera image: the scripted tasks only ask for its size."""
    def __init__(self, w, h): self.w, self.h = w, h
    def get_size(self): return self.w, self.h


def scripted_episode(env: GateEnv, seed: int):
    """Flies the scripted GateTask through the environment. Returns the final info and total reward."""
    from ai.submarine import Submarine
    from ai.tasks import GateTask
    from data_structures import SensorSuite

    env.reset(seed=seed)
    with redirect_stdout(io.StringIO()):
        sub = Submarine(mission_plan=[GateTask(target_depth=1.5)])
    cam = env.core.camera
    image = _ImageSize(cam.width, cam.height)

    def driver(r, detections, dt):
        s = env.core.state   # Ground truth: only the scripted hover/damping helpers use it (SIM ONLY)
        sensors = SensorSuite(camera_image=image, depth=r.depth, heading=r.heading, pitch=r.pitch,
                              imu=r.imu, depth_rate=r.depth_rate, x=s.x, y=s.y,
                              velocity_x=s.velocity_x, velocity_y=s.velocity_y,
                              angular_velocity_y=s.angular_velocity_y, velocity_z=s.velocity_z,
                              detections=detections)
        with redirect_stdout(io.StringIO()):
            commands, _ = sub.update(dt, sensors)
        return commands

    total = 0.0
    while True:
        _, reward, term, trunc, info = env.step_with_driver(driver)
        total += reward
        if term or trunc:
            return info, total


def action_episode(env: GateEnv, seed: int, choose):
    obs, _ = env.reset(seed=seed)
    total = 0.0
    while True:
        obs, reward, term, trunc, info = env.step(choose(obs))
        total += reward
        if term or trunc:
            return info, total


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("controller", help="scripted, random, straight, or a path to a PPO model .zip")
    parser.add_argument("--detector", choices=["perfect", "noisy"], default="noisy")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10_000, help="First seed; evaluation seeds stay apart from training")
    args = parser.parse_args()

    env = GateEnv(detector_mode=args.detector)
    if args.controller == "scripted":
        run = lambda seed: scripted_episode(env, seed)
    elif args.controller == "random":
        rng = np.random.default_rng(0)
        run = lambda seed: action_episode(env, seed, lambda obs: rng.uniform(-1, 1, 3))
    elif args.controller == "straight":
        # Heading change 0, target depth 1.5 m, forward at 0.6
        depth_action = (1.5 - 0.3) / (1.9 - 0.3) * 2 - 1
        run = lambda seed: action_episode(env, seed, lambda obs: np.array([0.0, depth_action, 0.6]))
    else:
        from stable_baselines3 import PPO
        model = PPO.load(args.controller)
        run = lambda seed: action_episode(env, seed, lambda obs: model.predict(obs, deterministic=True)[0])

    counts = dict.fromkeys(OUTCOMES, 0)
    times, rewards, energies = [], [], []
    for i in range(args.episodes):
        info, total = run(args.seed + i)
        for k in OUTCOMES:
            counts[k] += bool(info[k])
        if info["is_success"]:
            times.append(info["time_s"])
        rewards.append(total)
        energies.append(info["energy"])

    n = args.episodes
    print(f"{args.controller} on {n} episodes ({args.detector} detector, seeds {args.seed}-{args.seed + n - 1}):")
    print(f"  success {counts['is_success']}/{n} ({100 * counts['is_success'] / n:.0f}%), "
          f"collision {counts['collision']}, missed gate {counts['missed_gate']}, timeout {counts['timeout']}")
    if times:
        print(f"  time to gate: median {statistics.median(times):.1f} s, max {max(times):.1f} s")
    print(f"  reward: mean {statistics.mean(rewards):.2f}   energy: mean {statistics.mean(energies):.2f}")


if __name__ == "__main__":
    main()

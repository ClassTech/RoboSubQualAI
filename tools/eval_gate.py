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
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gate_controllers import make_controller, run_episode  # noqa: E402
from gate_env import GateEnv                            # noqa: E402

OUTCOMES = ("is_success", "collision", "missed_gate", "timeout")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("controller", help="scripted, random, straight, or a path to a PPO model .zip")
    parser.add_argument("--detector", choices=["perfect", "noisy"], default="noisy")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10_000, help="First seed; evaluation seeds stay apart from training")
    args = parser.parse_args()

    env = GateEnv(detector_mode=args.detector)
    controller = make_controller(args.controller)
    run = lambda seed: run_episode(env, controller, seed)

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

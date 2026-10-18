#!/usr/bin/env python3
"""
Trains a PPO policy on the gate environment.

Everything for a run goes in runs/<name>/:
  tb/                          TensorBoard logs (view with: .venv/bin/tensorboard --logdir runs)
  best_model.zip               the checkpoint with the best scored-test result so far
  latest_model.zip             the most recent checkpoint
  *_vecnormalize.pkl           the observation normalization each checkpoint needs
  eval.csv                     every scored test, one row each
  config.json                  this run's settings

Scored tests: every --eval-every steps the current policy flies --eval-episodes
fixed starts (seeds 10000+, never used for training) with no randomness in its
actions, scored like tools/eval_gate.py. The scripted GateTask's result on the
same starts is logged alongside for comparison.

Watch a checkpoint at any time, even while training runs:
  .venv/bin/python watch.py runs/<name>/best_model.zip

Usage: .venv/bin/python train.py --name gate_perfect --detector perfect --steps 3000000
"""
import argparse
import copy
import csv
import json
import os
import statistics
import time
from collections import deque

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from gate_controllers import make_controller, policy_controller, run_episode, vecnormalize_path
from gate_env import GateEnv

OUTCOME_KEYS = ("is_success", "collision", "missed_gate", "timeout")
EVAL_SEED = 10_000


def make_env(detector: str, seed: int):
    def _init():
        env = Monitor(GateEnv(detector_mode=detector), info_keywords=OUTCOME_KEYS + ("time_s", "energy"))
        env.reset(seed=seed)
        return env
    return _init


def score(env: GateEnv, controller, episodes: int) -> dict:
    """Flies the fixed evaluation starts and summarizes the outcomes."""
    counts = dict.fromkeys(OUTCOME_KEYS, 0)
    times, rewards = [], []
    for i in range(episodes):
        info, total = run_episode(env, controller, EVAL_SEED + i)
        for k in OUTCOME_KEYS:
            counts[k] += bool(info[k])
        if info["is_success"]:
            times.append(info["time_s"])
        rewards.append(total)
    return {"success_rate": counts["is_success"] / episodes,
            "collision_rate": counts["collision"] / episodes,
            "missed_gate_rate": counts["missed_gate"] / episodes,
            "timeout_rate": counts["timeout"] / episodes,
            "time_to_gate": statistics.median(times) if times else float("nan"),
            "mean_reward": statistics.mean(rewards)}


class OutcomeLogger(BaseCallback):
    """Logs how training episodes end (collision, missed gate, timeout, time to gate) to TensorBoard."""
    def __init__(self, window: int = 200):
        super().__init__()
        self.recent = deque(maxlen=window)

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and "collision" in info:
                self.recent.append(info)
        return True

    def _on_rollout_end(self):
        if not self.recent:
            return
        n = len(self.recent)
        for key in ("collision", "missed_gate", "timeout"):
            self.logger.record(f"outcomes/{key}_rate", sum(bool(i[key]) for i in self.recent) / n)
        times = [i["time_s"] for i in self.recent if i["is_success"]]
        if times:
            self.logger.record("outcomes/time_to_gate", statistics.median(times))
        self.logger.record("outcomes/energy", statistics.mean(i["energy"] for i in self.recent))


class ScoredTest(BaseCallback):
    """Every eval_every steps, scores the current policy on the fixed starts and keeps the best checkpoint."""
    def __init__(self, run_dir: str, detector: str, eval_every: int, episodes: int, baseline: dict):
        super().__init__()
        self.run_dir, self.eval_every, self.episodes, self.baseline = run_dir, eval_every, episodes, baseline
        self.env = GateEnv(detector_mode=detector)
        self.next_eval = eval_every
        self.best = (-1.0, float("inf"))   # (success rate, time to gate)
        self.csv_path = os.path.join(run_dir, "eval.csv")
        self.started = time.time()

    def _save(self, name: str):
        path = os.path.join(self.run_dir, f"{name}.zip")
        self.model.save(path)
        self.model.get_vec_normalize_env().save(vecnormalize_path(path))

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_eval:
            return True
        self.next_eval += self.eval_every
        normalizer = copy.deepcopy(self.model.get_vec_normalize_env())
        normalizer.venv = None   # Only its statistics are needed
        result = score(self.env, policy_controller(self.model, normalizer), self.episodes)
        for key, value in result.items():
            self.logger.record(f"eval/{key}", value)
        self.logger.record("eval/scripted_success_rate", self.baseline["success_rate"])
        self.logger.record("eval/scripted_time_to_gate", self.baseline["time_to_gate"])
        self.logger.dump(self.num_timesteps)

        self._save("latest_model")
        t = result["time_to_gate"] if result["success_rate"] > 0 else float("inf")
        is_best = (result["success_rate"], -t) > (self.best[0], -self.best[1])
        if is_best:
            self.best = (result["success_rate"], t)
            self._save("best_model")
        new_file = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["steps", "minutes", *result.keys(), "best"])
            w.writerow([self.num_timesteps, f"{(time.time() - self.started) / 60:.1f}",
                        *[f"{v:.3f}" for v in result.values()], int(is_best)])
        print(f"[{(time.time() - self.started) / 60:5.1f} min] {self.num_timesteps:>9,} steps: "
              f"success {100 * result['success_rate']:3.0f}% (scripted {100 * self.baseline['success_rate']:.0f}%), "
              f"collision {100 * result['collision_rate']:.0f}%, missed {100 * result['missed_gate_rate']:.0f}%, "
              f"timeout {100 * result['timeout_rate']:.0f}%, time to gate {result['time_to_gate']:.1f} s"
              f"{'  <- best' if is_best else ''}", flush=True)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", default="gate", help="Run directory name under runs/")
    parser.add_argument("--detector", choices=["perfect", "noisy"], default="perfect")
    parser.add_argument("--steps", type=float, default=3e6, help="Total environment steps (10 per simulated second)")
    parser.add_argument("--envs", type=int, default=16, help="Parallel environments")
    parser.add_argument("--threads", type=int, default=2, help="PyTorch threads for the PPO update")
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = os.path.join("runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    # Each update uses 8192 steps however many environments collect them
    hyper = dict(n_steps=max(64, 8192 // args.envs), batch_size=256, n_epochs=10, gamma=0.995, gae_lambda=0.95,
                 learning_rate=3e-4, clip_range=0.2, ent_coef=0.005,
                 policy_kwargs=dict(net_arch=[128, 128]))
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "ppo": hyper}, f, indent=2)
    torch.set_num_threads(args.threads)

    print("Scoring the scripted baseline on the evaluation starts...", flush=True)
    baseline = score(GateEnv(detector_mode=args.detector), make_controller("scripted"), args.eval_episodes)
    print(f"Scripted: success {100 * baseline['success_rate']:.0f}%, time to gate {baseline['time_to_gate']:.1f} s", flush=True)

    venv = SubprocVecEnv([make_env(args.detector, args.seed * 1000 + i) for i in range(args.envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=hyper["gamma"])
    model = PPO("MlpPolicy", venv, seed=args.seed, device="cpu", verbose=0,
                tensorboard_log=os.path.join(run_dir, "tb"), **hyper)
    callbacks = [OutcomeLogger(),
                 ScoredTest(run_dir, args.detector, args.eval_every, args.eval_episodes, baseline)]
    print(f"Training {int(args.steps):,} steps with {args.envs} environments; logs in {run_dir}/tb", flush=True)
    model.learn(total_timesteps=int(args.steps), callback=callbacks, tb_log_name="ppo")
    model.save(os.path.join(run_dir, "final_model.zip"))
    venv.save(vecnormalize_path(os.path.join(run_dir, "final_model.zip")))
    venv.close()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()

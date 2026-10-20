#!/usr/bin/env python3
"""
A skill's done detector: a small network that looks at the same observation
as the skill's policy (only what the real sub measures) and says whether the
skill's job is finished. The mission runner moves to the next skill when it
fires, so the switching is learned, not hand-written.

It's trained by supervised learning (tools/train_done_detector.py) on moments
flown by the skill's own policy, each labeled with the simulator's ground
truth (the environment's done_label()): for the gate, through the opening and
1.5 m beyond; for the orbit, 270° around the marker and facing the gate.
Learning this from the policy's reward instead failed: the cost of saying done
too early came every step and the payoff for the right moment rarely, so the
policies learned never to say it.

numpy only, so it runs on the Pi unchanged. A detector file (.npz) carries its
own copy of the observation normalization the policy was trained with.
"""
import numpy as np


class DoneDetector:
    """Fires once the probability has stayed above `threshold` for `confirm` steps in a row (both saved
    with the detector), so one odd moment can't end a skill early. Call reset() when the skill starts."""
    def __init__(self, path: str, threshold: float = None, confirm: int = None):
        d = np.load(path)
        self.mean, self.std, self.clip = d["obs_mean"], np.sqrt(d["obs_var"] + 1e-8), float(d["clip_obs"])
        self.layers = [(d[f"w{i}"], d[f"b{i}"]) for i in range(int(d["n_layers"]))]
        self.threshold = float(d["threshold"]) if threshold is None else threshold
        self.confirm = int(d["confirm"]) if confirm is None else confirm
        self.streak = 0

    def reset(self):
        self.streak = 0

    def probability(self, obs: np.ndarray) -> float:
        """Chance the skill is finished, from one raw (unnormalized) observation."""
        x = np.clip((np.asarray(obs, dtype=np.float32) - self.mean) / self.std, -self.clip, self.clip)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = np.tanh(x)
        return float(1.0 / (1.0 + np.exp(-x[0])))

    def update(self, obs: np.ndarray) -> bool:
        """Feed this step's observation; True once the skill is judged finished."""
        self.streak = self.streak + 1 if self.probability(obs) > self.threshold else 0
        return self.streak >= self.confirm


def detector_path(policy_path: str) -> str:
    """Where a policy's done detector lives: next to it, as done_detector.npz."""
    import os
    return os.path.join(os.path.dirname(policy_path), "done_detector.npz")

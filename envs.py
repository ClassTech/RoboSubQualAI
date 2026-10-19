#!/usr/bin/env python3
"""The RL skills by name, for the tools' --task option."""
from gate_env import GateEnv
from orbit_env import OrbitEnv

ENVS = {"gate": GateEnv, "orbit": OrbitEnv}


def make(task: str, detector_mode: str = "perfect", **kwargs):
    return ENVS[task](detector_mode=detector_mode, **kwargs)

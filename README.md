# RoboSubQualAI

A simulator for a RoboSub pre-qualification run, and a reinforcement-learning (RL) setup that trains neural-network policies to fly it: dive, go through the gate, circle the marker, and come back through the gate.

**How everything works**, from the big picture down to the details, is in the design document, [docs/design.html](docs/design.html) (open it in a browser; GitHub shows HTML as source). The step-by-step plan and progress log are in [PLAN.md](PLAN.md).

## What it does

- **Simulates the pool and the sub realistically.** The simulation covers physics at 60 Hz, the sub's real size and thruster layout, and thrusters that ramp from zero to full in 0.5 s. Sensors (a 1280×960 camera, pressure, IMU with magnetometer) have noise and per-run errors. Gate poles and the marker are drawn at their real size.
- **Flies the course two ways:**
  - **Scripted:** the original hand-written mission, kept as the baseline.
  - **Learned:** two small neural-network skills, *gate* and *orbit*. A short hand-written runner switches between them, using only what the real sub can measure.
- **Trains the skills with PPO** (Stable-Baselines3), using 16 simulators in parallel at about 290× real time. Training varies the sub's gains, inertia, drag and thrust between runs so the policies don't depend on exact numbers.

The policies never drive the thrusters directly. Ten times a second they pick targets (turn, depth, forward thrust, and sideways thrust for the orbit). The sub's PID loops fly those targets.

## Where it stands

On fresh test starts, with the noisy detector:

| Full mission | Success, nominal | Success, varied dynamics | Typical time |
|---|---|---|---|
| Scripted mission | 57/60 | 58/60 | ~110 s |
| **Learned skills + runner** | **59/60** | **57/60** | **~57 s** |

The learned mission is as reliable as the scripted one and takes about half the time. Training each skill from the exact states the runner hands it (recorded from real missions) was the last big step. The remaining failures are a few missed gates and one skipped orbit.

Running the policies on the sub's Raspberry Pi is planned but deferred.

## Setup

PyTorch's last build for Intel Macs (2.2.2) supports Python up to 3.12, so the project uses Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Things to run

```bash
# The scripted mission in the pygame window
.venv/bin/python main.py                        # camera image + OpenCV, like the real sub
.venv/bin/python main.py --detector noisy       # simulated detector with realistic noise

# Watch the learned mission, the scripted one, or a single skill
.venv/bin/python watch.py mission --task mission
.venv/bin/python watch.py scripted --task mission
.venv/bin/python watch.py runs/gate_mstart_ft/final_model.zip --task gate
.venv/bin/python watch.py runs/orbit_mstart_ft/final_model.zip --task orbit

# Train a skill (--task gate | orbit; --randomize varies the dynamics; --resume fine-tunes)
.venv/bin/python train.py --name my_gate --task gate --detector noisy --wide-starts --randomize --steps 2000000

# Follow training at http://localhost:6006
.venv/bin/tensorboard --logdir runs

# Score a controller on fixed test starts (--task gate | gate_return | orbit | mission)
.venv/bin/python tools/eval_gate.py mission --task mission --episodes 60 --seed 30000
```

`runs/` (trained models) is not in git. In a fresh clone, train a skill before `watch.py` or `eval_gate.py` have a model to load. The `scripted`, `straight` and `random` controllers work without one.

## Where things are

| Area | Files |
|---|---|
| Settings | `config.py` (pool, course, sub properties, noise ranges) |
| Simulation | `sim_core.py` (headless physics, sensors, detector), `collision.py`, `sensor_model.py`, `camera_geometry.py`, `detector_model.py`, `world.py` |
| Runs on the sub (numpy only) | `perception.py`, `controller.py`, `observation.py` |
| RL | `sub_env.py`, `gate_env.py`, `orbit_env.py`, `mission.py`, `envs.py`, `train.py`, `gate_controllers.py` |
| Scripted mission | `main.py`, `simulator.py`, `ai/` |
| Viewing | `sim_view.py`, `watch.py` |
| Tools | `tools/eval_gate.py`, `run_headless.py`, `check_detector.py`, `check_observation.py`, `record_handoffs.py` |

**Conventions:** z is depth (positive down). Heading and bearing are in degrees, with positive meaning left. Training uses seeds 0 and up; scored tests use 10000 and up.

## History

This repository was split from `QualSimSub` (kept as is), which is based on [ClassTech/RoboSim](https://github.com/ClassTech/RoboSim). The scripted task framework, the rendering and the original physics come from there.

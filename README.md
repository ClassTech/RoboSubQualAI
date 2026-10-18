# RoboSubQualAI

A simulator for a RoboSub pre-qualification run, and a reinforcement-learning (RL) setup that trains a neural-network policy to fly the sub through the gate.

The simulator started as a hand-written ("scripted") mission: dive, go through the gate, circle the marker pole, come back through the gate, surface. This repository keeps that mission working, makes the simulation realistic enough to train on, and adds the RL environment, training, evaluation and viewing tools.

**Where it stands:** on the gate task, a policy trained for about 3 minutes gets through the gate as reliably as the scripted task and more than twice as fast:

| Gate task, 100 fixed test starts | Success | Time to gate (median / worst) |
|---|---|---|
| Scripted `GateTask` | 100% | 21.7 s |
| Trained policy, perfect detector | **100%** | **8.8 s** / 11.3 s |
| Trained policy, noisy detector (never trained with noise) | **99%** | **10.0 s** / 16.6 s |

The plan, with what's done and what's next, is in [PLAN.md](PLAN.md). Running the policy on the real sub's Raspberry Pi is planned but deferred.

---

## Contents

1. [Quick start](#1-quick-start)
2. [The big picture](#2-the-big-picture)
3. [Repository map](#3-repository-map)
4. [How each piece works](#4-how-each-piece-works)
5. [Training in depth](#5-training-in-depth)
6. [Design decisions and lessons learned](#6-design-decisions-and-lessons-learned)
7. [Conventions](#7-conventions)
8. [Limitations and next steps](#8-limitations-and-next-steps)

---

## 1. Quick start

### Setup

PyTorch's last build for Intel Macs (2.2.2) only supports Python up to 3.12, so the project uses Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Things to run

```bash
# The original scripted mission in the pygame window
.venv/bin/python main.py                        # camera image + OpenCV, like the real sub
.venv/bin/python main.py --detector noisy       # simulated detector with realistic noise

# Watch any controller fly the gate task (same starts as the scored tests)
.venv/bin/python watch.py scripted
.venv/bin/python watch.py runs/r3_perfect_s0/best_model.zip --detector noisy

# Train a policy (about 3 minutes to 100% with the perfect detector)
.venv/bin/python train.py --name my_run --detector perfect --steps 1500000

# Follow training in the browser at http://localhost:6006
.venv/bin/tensorboard --logdir runs

# Score a controller on 100 fixed test starts
.venv/bin/python tools/eval_gate.py runs/my_run/best_model.zip --detector noisy --episodes 100
```

`runs/` is not in git. A fresh clone needs a training run before `watch.py` or `eval_gate.py` have a model to load; `scripted`, `straight` and `random` work without one.

---

## 2. The big picture

### On the real sub (and in the simulator)

```
  camera ──► detector ──► boxes ──► perception ──► observation ──► POLICY ──► targets ──► controller ──► thrusters
  (1280×960)  (OpenCV)    (pixels)  (bearing,       (108 numbers)  (neural    (heading,   (PID loops,   (6 motors)
                                     range, gate                    network,    depth,      60 Hz)
  pressure sensor ──────────────────► angle)                        10 Hz)      thrust)
  IMU + magnetometer ───────────────────────────────────────────────────────────────────────────► (also feeds the PID loops)
```

The policy doesn't drive thrusters directly. Ten times a second it chooses three targets: a heading change, a target depth and a forward thrust. The sub's ordinary PID loops then hold those targets between decisions. That split keeps the policy small and lets it focus on *where to go*, not on balancing the sub.

Everything to the left of the policy uses only what the real sub can measure: the camera, the pressure sensor, and the IMU with its magnetometer. Nothing in the policy's path uses the simulator's ground truth.

### In training

```
   ┌──────────────────────── 16 copies in parallel ────────────────────────┐
   │  SubSim (physics, 60 Hz) ─► simulated sensors + simulated detector    │
   │        ▲                                   │                          │
   │        │ thruster commands                 ▼                          │
   │   controller ◄── policy targets ◄── observation ◄── perception        │
   └───────────────────────────────────────────────────────────────────────┘
                          │ rewards, observations
                          ▼
                 PPO (Stable-Baselines3) updates the policy, ~2,900 steps/s
```

Training never draws anything. Instead of rendering a camera image and running OpenCV on it, the simulated detector computes the boxes OpenCV *would* find directly from the sub's position, then adds realistic noise. That's what makes it fast: about 290 simulated seconds per real second.

### Two ways to fly the same course

- **The scripted mission** (`main.py`, `ai/`) is a sequence of hand-written tasks and subtasks. It's the baseline the policy has to beat, and it still works end to end.
- **The policy** (`gate_env.py`, `train.py`) currently learns only the gate part. The orbit and return trip are future work.

Both use the same physics, sensors, detector, perception and PID loops, so comparisons between them are fair.

---

## 3. Repository map

Grouped by layer, from the bottom up. Files marked **(Pi)** use only numpy and the standard library and are meant to run unchanged on the real sub.

| Layer | File | What it does |
|---|---|---|
| Settings | `config.py` | Pool and course layout, the sub's measured and estimated properties, sensor and detector noise ranges |
| Shared types | `core_types.py` | `ThrusterCommands`, `MPU6050Readings` |
| World | `world.py` | Data classes for the gate, marker and the sub's physical state |
| Simulation | `sim_core.py` | `SubSim`: physics, collisions, gate passes, sensors and detector. No pygame or OpenCV |
| | `collision.py` | Box-versus-pole collision geometry and gate-plane crossing detection |
| | `sensor_model.py` | Simulated pressure sensor, magnetometer and IMU, with noise and per-run errors |
| | `camera_geometry.py` | Projects poles into the camera image. Shared by the renderer and the simulated detector |
| | `detector_model.py` | Simulated detector: boxes from the true pose, then noise |
| Perception **(Pi)** | `perception.py` | Boxes → bearing and range; gate pairing; gate angle and center-line offset |
| Control **(Pi)** | `controller.py` | Depth, pitch and heading PID loops; thruster mixer |
| Policy input **(Pi)** | `observation.py` | Builds the policy's 108-number observation |
| RL | `gate_env.py` | Gymnasium environment for the gate task: actions, reward, episode rules |
| | `gate_controllers.py` | Scripted, straight, random or trained controllers, shared by tools |
| | `train.py` | PPO training with TensorBoard logging and scored tests |
| Scripted mission | `main.py` | Builds and runs the full scripted mission |
| | `simulator.py` | Runs the scripted AI in the window |
| | `ai/submarine.py` | The scripted "brain": runs tasks, holds PID gains |
| | `ai/tasks/` | Tasks (`GateTask`, `OrbitTurnTask`, …) and reusable subtasks |
| | `ai/vision.py`, `data_structures.py` | OpenCV color blob detection; `Vision` picks out the gate and marker |
| Display | `sim_view.py` | The pygame window: top view, camera view, panel, thruster bars |
| | `watch.py` | Watch any controller fly the gate task |
| Tools | `tools/eval_gate.py` | Score a controller on fixed test starts |
| | `tools/run_headless.py` | Run the full scripted mission many times without a window |
| | `tools/check_detector.py` | Compare simulated boxes with OpenCV on the rendered image |
| | `tools/check_observation.py` | Compare the observation with ground truth |

---

## 4. How each piece works

### 4.1 The pool, the course and the sub

The pool is 40 m long (x), 15 m wide (y) and 2.1 m deep (z, positive down).

- **The gate:** at x = 10 m, centered across the pool. Two red poles 2 m apart, made of 1" PVC (33.4 mm across), run from a gray top bar 1 m below the surface to the floor. The opening is below the bar.
- **The marker:** a green pole 0.3 m across at x = 20 m. It runs from the floor to 0.6 m above the surface.
- **The scripted mission's start:** 3 m in front of the gate, near the surface.

The sub's numbers live in `SimulationConfig`:

| Property | Value | Source |
|---|---|---|
| Footprint | 0.46 m × 0.457 m | measured |
| Height | 0.25 m | measured |
| Mass | 4.0 kg, slightly buoyant (4.1 L) | assumed |
| Thruster lever arm | 0.229 m (9") | measured |
| Thrust | 0.8 N per thruster | assumed |
| Yaw / pitch inertia | 0.13 / 0.09 kg·m² | estimated from the layout: ¾" PVC frame, 4"×9" electronics tube, 4"×6"×4" battery box on top |
| Drag | various | assumed |

### 4.2 Physics (`sim_core.py`)

`SubSim` advances the world in fixed 1/60 s steps. The core math is in `_physics_step()`.

- **Six thrusters:** four horizontal ones at the corners, angled 45°, mixed for forward, sideways and yaw; two vertical ones, front and back, for heave and pitch.
- **Forces:** thrust, quadratic drag per axis (forward, sideways, vertical, yaw, pitch), and net buoyancy. Torques use the 9" lever arm.
- **Integration:** velocities and position integrate in world coordinates; heading and pitch integrate from their rates.
- **Collisions** (`collision.py`): the sub is a box turned to its heading; poles and the top bar are segments with a radius. On contact, the sub returns to its last clear position and stops. Repeated bumps within half a second count as one collision.
- **Gate passes:** every time the sub's center crosses the gate's plane, it's recorded with its direction and whether it was inside the opening.

`advance(dt, commands)` runs as many fixed steps as `dt` covers. So the graphics window, which runs at whatever frame rate it gets, and training, at exactly 0.1 s per decision, use identical physics.

### 4.3 Sensors (`sensor_model.py`)

The real sub has a camera, a pressure sensor and an IMU with a magnetometer. Nothing measures horizontal speed or position. The simulator reports what each real sensor would:

| Sensor | Simulated as |
|---|---|
| Pressure → depth | true depth + 3 mm noise + a fixed zero error of up to ±2 cm per run |
| Depth rate | an alpha-beta filter on the noisy depth (tracks a change in speed in ~0.3 s) |
| Magnetometer → heading | true heading + 0.5° noise + a fixed offset of up to ±15° per run (steel in the pool) |
| IMU → pitch, rates, acceleration | true values + small noise; gyro bias up to ±0.005 rad/s per run |

The per-run errors are drawn fresh at every reset, so a policy can't learn to depend on one calibration.

The scripted tasks still use ground-truth position and velocity in a few helpers: holding position, and damping at the end of a task. Those are marked **SIM ONLY**, and so are the matching fields in `SensorSuite`. They give the scripted baseline an advantage the policy doesn't get.

### 4.4 Camera and detection

The camera is 1280×960 with a 70° horizontal field of view. There are two ways to get detections:

1. **Render and run OpenCV**, like the real sub: `main.py`'s default. `sim_view.py` draws the camera image, with each pole projected at its real size by `camera_geometry.py`. Then `ai/vision.py` does HSV color thresholding and finds contours. This is slow at 1280×960, about 20 frames per second.
2. **The simulated detector** (`detector_model.py`), used for training. It projects the same pole geometry with the same code and takes the bounding box directly. It returns boxes in exactly OpenCV's format, so everything downstream is unchanged.

`tools/check_detector.py` confirms the two agree to within 1–2 pixels over a whole mission.

**Detector noise.** These are drawn per run from the ranges in `DetectorNoiseConfig`:
- **Box position and size:** the center shifts by a few pixels, and the size scales by 2–10%. Thin, distant poles stay thin but don't vanish.
- **Dropouts:** an object goes missing for about 0.3 s at a time, up to 0.5 times per second.
- **Range limits:** detection is reliable within 25–35 m and impossible beyond 35–50 m. The pool is very clear.
- **False positives:** one-frame phantom boxes of either color, up to 0.5 per second.
- **Latency:** detections arrive 30–150 ms late.

### 4.5 Perception (`perception.py`, Pi)

Perception turns boxes into geometry the policy can use:

- **Bearing:** from the box's horizontal position and the camera's field of view. Positive is to the left.
- **Range to a pole:** the best available of three methods.
  1. **Bottom:** the pole's foot is in view, and the pool depth and the sub's depth are known, so range comes from the angle down to the foot. This is the most accurate, within 0.3% with clean boxes.
  2. **Height:** both ends are in view and the pole's length is known.
  3. **Width:** the pole's apparent diameter. This always works but is noisy for thin poles.
- **Gate pairing:** `find_gate_pair()` picks the two tall red boxes that look like a pair. The scripted `Vision` and the policy's observation share it.
- **Gate geometry:** `gate_geometry()` uses the two poles' positions to work out:
  - the gate's center
  - the turn needed to face straight through it
  - how far the sub is from the gate's center line

  The angle depends on the small difference between the two poles' ranges over the 2 m gate width, so per-frame noise makes it jumpy. The observation smooths it over about a second.

### 4.6 Low-level control (`controller.py`, Pi)

These are the sub's PID loops, used by both the scripted tasks and the policy:

- **Depth:** PID on the measured depth, with the derivative from the filtered depth rate.
- **Pitch:** PD toward level.
- **Heading:** PD toward the target heading, using the gyro for the derivative.
- **Mixer:** turns forward, sideways, yaw, heave and pitch efforts into six thruster commands. If any command exceeds 1, all six are scaled down together.

The gains are the scripted sub's tuned values. They'll be retuned in the pool, so training should vary them (see [next steps](#8-limitations-and-next-steps)).

### 4.7 The scripted mission (`main.py`, `ai/`)

A mission is a list of **tasks**. Each task is a sequence of **subtasks** that pass information through a shared `context` dictionary.

1. **`GateTask`:**
   - dive (`DiveToDepth`)
   - turn until the gate is in view (`WaitForTargetVisible`)
   - center on it (`AlignToObjectX`)
   - steady (`Stabilize`)
   - drive at it, steering at its center (`DriveUntilTargetLost`)
   - drive straight 4 s, then steady
2. **`StabilizeTask`:** hold position for 2 s.
3. **`OrbitTurnTask`:**
   - find and approach the marker
   - circle it, keeping it centered at about 1.7 m, until the gate appears to its right
   - sway sideways until the marker leaves the view
4. **Back through the gate** with another `GateTask`.
5. **Surface** with a `StabilizeTask` at 0.1 m, then shut down.

`Vision` holds a target for 0.5 s after it disappears, and only accepts a new one once it has been seen in two consecutive frames. Tasks that wait for something to leave the view only count it as gone when it was last seen at the edge of the image. So a mid-view dropout doesn't trigger them, but genuinely moving past an object does.

With the noisy detector, the full mission completes 40 of 40 runs (`tools/run_headless.py`).

### 4.8 The gate environment (`gate_env.py`)

This is a [Gymnasium](https://gymnasium.farama.org/) environment: the standard interface between a simulator and an RL algorithm.

**Each episode starts at a random position:**
- 3–8 m in front of the gate
- up to ±1.5 m off its center line
- up to ±30° off facing it squarely
- 0.1–0.5 m deep

**Actions** (three numbers from −1 to 1, ten times a second):

| Action | Range | Held by |
|---|---|---|
| Heading change | ±45° from the current heading | heading PD loop |
| Target depth | 0.3 to 1.9 m | depth PID loop |
| Surge effort | full reverse to full forward thrust | open loop (nothing measures speed) |

**The episode ends** when the sub goes forward through the gate opening (success), hits a pole or the bar, crosses the gate line anywhere else, or reaches 60 s.

**Reward:**

| When | Reward |
|---|---|
| Every step | +1 per meter closer to the gate, −0.01 time cost, −0.05 per second of full thrust (energy) |
| Through the gate | +20 |
| Collision, missed gate, or timeout | −5 |

"Closer" combines three distances:
- the horizontal distance to the gate's center
- the error from 1.5 m depth
- half the distance from the gate's center line

So approaching square-on, at the right depth, is what earns the most. The shaping uses ground truth, which is fine: the reward is only used in training and never reaches the sub.

At the end of each episode, `info` reports how it ended (`is_success`, `collision`, `missed_gate`, `timeout`), the time and the energy used. That's what the training curves plot.

### 4.9 The observation (`observation.py`, Pi)

Each decision, the policy sees four frames of 27 numbers: the current one and the three before it, 108 in total. Stacking frames lets it sense its own motion from how things change.

| Group | Values |
|---|---|
| Own state (7) | depth, depth rate, sin/cos of heading *relative to the start*, pitch, pitch rate, yaw rate |
| Left pole (5) | found now, bearing, range, range confidence, time since seen |
| Right pole (5) | same |
| Gate (7) | same five, plus the turn to face straight through it and the distance from its center line |
| Last action (3) | the previous decision |

`observation.INDEX` names every position, for example `INDEX["gate_offset"]`.

**Memory.** When something drops out of view, its last values are kept, with found = 0 and the time since seen counting up. Bearings and the gate angle are corrected for how much the sub has turned since, so "the gate was 10° left" becomes "the gate is 40° left" after a 30° right turn. They are *not* corrected for how far the sub has moved.

**Pole identity.** Close to the gate, the two poles no longer fit in the view. A lone pole is labeled left or right by whichever pole was last seen nearest to it. With no memory yet, the label comes from which side of the image it's on: a pole near the left edge is probably the *right* pole, with its partner out of view beyond that edge. Checked against ground truth, the label is correct 100% of the time with the perfect detector and 91% with noise. The noisy misses are dropouts, where there's genuinely no way to tell.

**Relative heading.** Heading is measured from the start of the run, not from magnetic north, so the policy can't memorize which compass direction the gate happens to be in.

---

## 5. Training in depth

### 5.1 PPO in one page

[PPO](https://arxiv.org/abs/1707.06347) (Proximal Policy Optimization) trains two small networks:

- **The policy (actor):** observation → actions. During training it outputs a mean and a spread for each action and samples within that range, which is how it explores. When deployed, it uses the mean.
- **The critic:** observation → how much reward is coming from here. It's only used in training, to judge each action.

The loop:
1. **Collect experience.** The current policy flies 16 simulators for 512 steps each: 8,192 steps, about 14 minutes of simulated swimming.
2. **Score each action.** Compare what actually followed with what the critic predicted. "Better than expected" gives a positive *advantage*, "worse" a negative one.
3. **Update.** Make positive-advantage actions more likely and negative ones less, in small batches over several passes. PPO caps how much any action's probability can change per update, at about 20%. That cap is what keeps it stable.

Both networks are two layers of 128. The policy has about 30,000 numbers, which is trivial for a Raspberry Pi.

### 5.2 Settings (`train.py`)

| Setting | Value | Why |
|---|---|---|
| Parallel environments | 16 (`SubprocVecEnv`) | ~15% faster than 8. The bottleneck is round trips to the workers, not computing |
| Steps per update | 8,192 (512 per environment) | |
| Batch / passes | 256 / 10 | |
| γ (discount) | 0.995 | Looks about 20 s ahead, about the length of a gate run |
| GAE λ | 0.95 | |
| Learning rate | 3×10⁻⁴, falling linearly to 0 | Gentle late updates |
| `target_kl` | 0.03 | Stops an update once the policy has moved this far |
| Entropy bonus | 0.005 | Keeps some exploration |
| Normalization | `VecNormalize` on observations and rewards | Saved with every checkpoint |

### 5.3 What a run produces

`runs/<name>/` contains:

| File | What it is |
|---|---|
| `tb/` | TensorBoard logs |
| `best_model.zip` (+ `_vecnormalize.pkl`) | The checkpoint with the best scored-test result |
| `latest_model.zip` (+ `_vecnormalize.pkl`) | The most recent checkpoint |
| `final_model.zip` (+ `_vecnormalize.pkl`) | The checkpoint at the end of the run |
| `eval.csv` | Every scored test |
| `config.json` | The run's settings |

The `_vecnormalize.pkl` next to each checkpoint holds the input normalization it was trained with. `watch.py` and `eval_gate.py` apply it automatically, and the Pi will need it too.

### 5.4 Reading TensorBoard

| Section | Meaning |
|---|---|
| `eval/success_rate` against `eval/scripted_success_rate` | **The headline.** Every 100k steps, the current policy flies 30 fixed test starts (seeds 10000+, never used in training) using its mean action. The scripted task's score on the same starts is plotted alongside |
| `eval/time_to_gate` against `eval/scripted_time_to_gate` | Median time of the successful runs |
| `eval/collision_rate`, `missed_gate_rate`, `timeout_rate` | How the failures happen |
| `rollout/ep_rew_mean`, `rollout/success_rate` | Training episodes, with exploration noise, so they trail the eval numbers |
| `outcomes/*` | How training episodes end, plus time to gate and energy |
| `train/approx_kl` | How far each update moved the policy. About 0.01–0.03 is healthy; larger and climbing means instability |
| `train/clip_fraction` | How often PPO's 20% cap kicked in |
| `train/std` | How much the policy still explores. It should fall gradually, not collapse early |
| `train/explained_variance` | How well the critic predicts. Close to 1 is good |

A typical successful run: timeouts vanish first (it learns to move), then misses (it learns to aim). Collisions rise briefly as it reaches the gate more often, then fall as it learns to line up.

### 5.5 Results so far

| Run | What changed | Best scored test |
|---|---|---|
| `gate_perfect` | First observation, perfect detector | 70%, bouncing 35–70% late |
| `gate_perfect_v2`, `gate_noisy_v2` | Separate pole tracks, gate angle, center-line reward | 23% / 10% (unstable training; see 6.4) |
| `stable_perfect_s1` | + `target_kl` and learning-rate decay | 80%, holding |
| `r3_perfect_s0`, `r3_perfect_s1` | + timeout penalty, gate bonus +20 | **100% / 97–100%, from 400k steps** |

---

## 6. Design decisions and lessons learned

### 6.1 Train on what the real sub can sense

Early on, the scripted PID loops damped sideways motion using the simulator's true velocity, which the real sub can't measure. Those inputs were removed from every loop the policy uses. Heading comes from a magnetometer with a per-run offset; depth rate is filtered from a noisy pressure reading. A policy trained on ground truth would fly differently in the water.

### 6.2 Bounding boxes as the interface

The real vision pipeline wasn't available, so the interface is the bounding box. Almost every OpenCV color detector produces one. The simulated detector produces boxes; perception turns boxes into bearing and range with code shared with the Pi. Plugging in the real pipeline should only need the camera's actual field of view and resolution.

### 6.3 Draw things at their real size

The original renderer drew gate poles 5 pixels wide at any distance, and drew the marker too narrow when seen at an angle. The scripted mission had been tuned around those shortcuts and failed once objects were drawn correctly. Fixing the drawing, then retuning the tasks as fractions of the image instead of pixels, made the scripted mission and the simulated detector agree with reality and with each other.

### 6.4 One change at a time, and more than one seed

Observation v2 went in together with a reward change and a training-setup change, and the result got much worse. Testing each change on its own showed that none of them was the cause. Training itself was unstable: every run's late updates were 10–20× larger than healthy, so runs learned the gate, then forgot it. The very first run had simply been lucky. Two lessons:
- change one thing at a time
- check a change on at least two random seeds before trusting it

### 6.5 Make "don't try" expensive

With the stability fix, one seed still settled on hovering near the gate until time ran out. Timing out cost nothing, while trying and missing cost −5. Making timeouts cost the same as other failures, and raising the gate bonus to +20, removed that safe-but-useless option. Both seeds then reached 97–100%.

### 6.6 Sim-only shortcuts are labeled

Wherever the simulator uses information the real sub doesn't have, it's labeled **SIM ONLY**:
- ground truth in the scripted hover and damping helpers
- the reward shaping

None of it reaches the policy's observation.

---

## 7. Conventions

| Quantity | Convention |
|---|---|
| World axes | x along the pool (start → gate → marker), y across, z depth (positive **down**) |
| Heading | degrees, 0 = facing +x, increasing counterclockwise seen from above (turning **left**) |
| Bearing | degrees from the camera's forward axis, positive = **left**. An object's heading = sub heading + bearing |
| Pitch | degrees, positive = nose **up** |
| Sub frame | x forward, y left (in `perception.gate_geometry`) |
| Gate offset | positive = the sub is left of the gate's center line, as seen approaching |
| Image | pixels, origin top-left; boxes as `find_blobs_hsv()` returns them |
| Seeds | training uses 0+; scored tests use 10000+ |

---

## 8. Limitations and next steps

**Not modeled yet:**
- **Thrusters:** no dead zone, forward/reverse difference, battery sag or spin-up delay. Thrust is linear and instant.
- **Robustness:** PID gains and inertia are fixed rather than varied between runs.
- **Occlusion:** one object hiding another isn't modeled in the simulated detector.
- **Pitch:** collision boxes ignore it.
- **The policy only does the gate.** The marker orbit and return trip are scripted only.

**Next** (details in [PLAN.md](PLAN.md)), one layer at a time, retraining after each:
1. Train with the noisy detector. The current policy already scores 99% with it untrained.
2. Add a thruster model.
3. Vary PID gains and inertia (about ±30%) each run.

After that, once it's needed: export the policy and its normalization to numpy, and check that the Pi's output matches PyTorch's.

### History

This repository was split from `QualSimSub` (kept as is), itself based on [ClassTech/RoboSim](https://github.com/ClassTech/RoboSim). The scripted task and subtask framework, the rendering and the original physics come from there.

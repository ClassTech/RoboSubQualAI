# RoboSubQualAI Plan

Train an RL policy in this simulator to fly the RoboSub gate task, then run it on the Raspberry Pi 5.

**Architecture:** camera → OpenCV bounding boxes → bearing/range → RL policy → existing PID loops and mixer → thrusters. The policy runs at 5–10 Hz.

**Settled decisions**
- Real sensors: camera, pressure (depth), IMU with magnetometer. Nothing measures horizontal speed.
- Policy actions: heading setpoint, depth setpoint, open-loop surge power.
- PID gains get tuned in the pool, so training randomizes them.
- Development and training run on the iMac: Python 3.12 `.venv`, torch 2.2.2 (see `requirements.txt`).

## Phase 1: Simulator that runs without the display
- [ ] Move the physics out of `SubmarineSimulator.applyPhysics()` into its own class with a fixed time step and no pygame. `main.py` keeps working on top of it and behaves the same.
- [x] Fix yaw torque so it uses the thruster lever arm (it was in N, not N·m).
- [x] Add pole collisions and a gate-pass check: the sub crosses the gate plane between the poles, at a valid depth. (`collision.py`)

## Phase 2: Sensors the real sub actually has
- [ ] Depth comes from the pressure sensor, and depth rate from a smoothed derivative of it.
- [ ] Heading comes from a simulated magnetometer, with noise and a random constant offset per episode. Pitch and rates come from the IMU.
- [ ] Remove true-velocity inputs from the loops. Depth damping switches to the smoothed depth rate. Sway damping in `get_heading_commands()` goes. Position hold (`_get_pid_hover_commands`) is kept out of training.

## Phase 3: Simulated detector
- [ ] Compute each object's bounding box from the sub's true pose and the camera geometry (70° FOV, 320×240), in the same format as `find_blobs_hsv`.
- [ ] Add noise in this order: box jitter, missed detections, false positives, latency.
- [ ] Write one shared function that turns a box into bearing and range. The same code runs on the Pi, so only the real camera's FOV and resolution need checking.

## Phase 4: Gymnasium environment for the gate task
- [ ] **Observations:** depth, depth rate, heading, pitch, pitch rate, yaw rate, and per object found/bearing/range/confidence plus time since last seen. Add the last action and a few stacked past steps, because the gate drops out of view about 1.4 m before you pass it.
- [ ] **Actions:** heading setpoint, depth setpoint and open-loop surge power. There's no speed setpoint, because the sub can't measure speed.
- [ ] **Reward:** progress toward the gate, a bonus for passing through, and penalties for collisions, timeouts and energy use.
- [ ] Use a perfect detector at first, and run the scripted `GateTask` through the environment as a baseline before any training.

## Phase 5: Training on the iMac
- [ ] PPO in Stable-Baselines3 with about 8–10 parallel environments and `VecNormalize`.
- [ ] Turn on realism one layer at a time, retraining after each: detector noise, then a thruster model (deadband, forward/reverse asymmetry, battery sag, spin-up lag), then randomized PID gains.
- [ ] Measure the policy's success rate against the scripted baseline in the same randomized environment.

## Phase 6: Export to the Pi 5
- [ ] Export the weights plus the `VecNormalize` statistics, and write a plain-numpy forward pass.
- [ ] Check that the numpy policy and the PyTorch policy give the same outputs on recorded observations.

## Open questions
- Should the policy control sway as well? The orbit task uses it, but the gate task doesn't need it. Leaning toward leaving it out for the gate task.
- `subInertia_Z = 0.35` looks high. A solid 4 kg box of the sub's size would be about 0.14 kg·m², so the sim may turn slower than the real sub.
- `submarineHeight` (0.25 m) and `gatePoleRadius` (0.025 m) in `config.py` are guesses. Measure the real sub and gate.

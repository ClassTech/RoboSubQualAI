# RoboSubQualAI Plan

Train an RL policy in this simulator to fly the RoboSub gate task, then run it on the Raspberry Pi 5.

**Architecture:** camera → OpenCV bounding boxes → bearing/range → RL policy → existing PID loops and mixer → thrusters. The policy runs at 5–10 Hz.

**Settled decisions**
- Real sensors: camera, pressure (depth), IMU with magnetometer. Nothing measures horizontal speed.
- Policy actions: heading setpoint, depth setpoint, open-loop surge power.
- PID gains get tuned in the pool, so training randomizes them.
- Development and training run on the iMac: Python 3.12 `.venv`, torch 2.2.2 (see `requirements.txt`).
- The pool is very clear: detection is easy at 25 m (`DetectorNoiseConfig.clearRange` 25–35 m, `maxRange` 35–50 m).
- Measured: gate poles are 1" PVC (33.4 mm OD), the sub is 0.25 m tall, and the thrusters are 9" (0.229 m) from center. Scripted noisy baseline with these values: 36/40.
- Camera: 1280×960, 70° horizontal FOV. Objects are drawn at their real size (`camera_geometry.py`, shared by the renderer and the simulated detector).

## Phase 1: Simulator that runs without the display
- [x] Move the physics out of `SubmarineSimulator.applyPhysics()` into its own class with a fixed time step and no pygame. `main.py` keeps working on top of it and behaves the same.
  - `sim_core.SubSim` holds the course, physics (fixed `physicsDt`, 1/60 s), collisions, gate passes, sensors and simulated detector. It imports no pygame or OpenCV and runs about 3,000 10-Hz control steps per second on one core (~300× real time).
  - Scripted mission with the noisy detector over 40 seeds: 35/40 and 37/40 for the old and new code, so behavior is unchanged. The earlier 20/20 was a lucky streak; the noisy baseline is about 90%.
- [x] Fix yaw torque so it uses the thruster lever arm (it was in N, not N·m).
- [x] Add pole collisions and a gate-pass check: the sub crosses the gate plane between the poles, at a valid depth. (`collision.py`)

## Phase 2: Sensors the real sub actually has
- [x] Depth comes from the pressure sensor, and depth rate from a smoothed derivative of it. (`sensor_model.py`, alpha-beta filter)
- [x] Heading comes from a simulated magnetometer, with noise and a random constant offset per episode. Pitch and rates come from the IMU, with noise and per-episode gyro bias. Noise levels are in `SensorNoiseConfig` and are guesses to tune against the real sensors.
- [x] Remove true-velocity inputs from the loops. Depth damping switches to the smoothed depth rate. Sway damping in `get_heading_commands()` goes. Position hold (`_get_pid_hover_commands`) is kept out of training.
  - The scripted tasks still use ground truth for position hold and end-of-task damping (`_get_pid_hover_commands`, `_get_damping_commands`, `get_spin_damping_commands`). These are marked SIM ONLY, as are the ground-truth fields in `SensorSuite`, so the scripted baseline has an advantage the policy won't.

## Phase 3: Simulated detector
- [x] Compute each object's bounding box from the sub's true pose and the camera geometry (70° FOV, now 1280×960), in the same format as `find_blobs_hsv`. (`detector_model.py`; `main.py --detector opencv|perfect|noisy`)
- [x] Add noise in this order: box jitter, missed detections, false positives, latency. Levels are in `DetectorNoiseConfig` and are guesses to tune against the real camera.
- [x] Write one shared function that turns a box into bearing and range. The same code runs on the Pi, so only the real camera's FOV and resolution need checking. (`perception.py`: `estimate_pole`, `estimate_gate`)
- [x] Draw objects at their real size, move to a 1280×960 camera, and retune the scripted tasks.
  - `tools/check_detector.py`: simulated boxes match OpenCV on the rendered image within 1–2 px. Range from the pole's bottom end is within 0.3%, gate range within ~0.5%. Range from pole width alone is within ~5%.
  - Task thresholds are now fractions of the image instead of pixels.
  - `Vision` holds a target for 0.5 s after a miss and only accepts one seen in 2 consecutive frames at about the same place.
  - "Until lost" subtasks only count a target as gone when it was close (gate) or at the image edge (sway), not after a mid-view dropout.
  - `DriveUntilTargetLost` steers at the gate while it's visible.
  - Scripted mission, `tools/run_headless.py`: perfect detector 8/8. Noisy detector with clear-pool ranges about 90% over 40 runs (see Phase 1). With the earlier murky-water guess (6–10 m) it was about 4/12, because the gate is ~10–12 m from the marker.

## Phase 4: Gymnasium environment for the gate task
- [ ] **Observations:** depth, depth rate, heading, pitch, pitch rate, yaw rate, and per object found/bearing/range/confidence plus time since last seen. Add the last action and a few stacked past steps, because the gate drops out of view about 1.4 m before you pass it.
- [ ] **Actions:** heading setpoint, depth setpoint and open-loop surge power. There's no speed setpoint, because the sub can't measure speed.
- [ ] **Reward:** progress toward the gate, a bonus for passing through, and penalties for collisions, timeouts and energy use.
- [ ] Use a perfect detector at first, and run the scripted `GateTask` through the environment as a baseline before any training.

## Phase 5: Training on the iMac
- [ ] PPO in Stable-Baselines3 with about 8–10 parallel environments and `VecNormalize`.
- [ ] Turn on realism one layer at a time, retraining after each: detector noise, then a thruster model (deadband, forward/reverse asymmetry, battery sag, spin-up lag), then randomized PID gains.
- [ ] Measure the policy's success rate against the scripted baseline in the same randomized environment.

## Seeing what's going on
The display stays optional: training runs without a window, and the same physics draws in the pygame window when you want to watch.
- [ ] **Watch live:** `main.py` keeps running the scripted mission as now. A `watch.py` loads a trained policy and flies it in the pygame window in real time (`render_mode="human"`).
- [ ] **Training curves:** log reward, gate-pass rate, collision rate and episode length to TensorBoard (add `tensorboard` to `requirements.txt`).
- [ ] **Evaluation videos (later):** during training, periodically record a few runs flown by the current policy as short videos.
- [ ] **Replays (later):** save each run's path (pose, detections, actions) and play it back in the pygame window with pause and single-step.

## Phase 6: Export to the Pi 5
- [ ] Export the weights plus the `VecNormalize` statistics, and write a plain-numpy forward pass.
- [ ] Check that the numpy policy and the PyTorch policy give the same outputs on recorded observations.

## Open questions
- The simulated detector doesn't model occlusion (e.g. a gate pole in front of the marker).
- OpenCV mode at 1280×960 runs at roughly 20 fps on the iMac, so the graphics window is slower than before. The simulated detector is unaffected.
- Should the policy control sway as well? The orbit task uses it, but the gate task doesn't need it. Leaning toward leaving it out for the gate task.
- `subInertia_Z = 0.35` looks high. A solid 4 kg box of the sub's size would be about 0.14 kg·m², so the sim may turn slower than the real sub.

# RoboSubQualAI Plan

Train an RL policy in this simulator to fly the RoboSub gate task, then run it on the Raspberry Pi 5.

**Architecture:** camera → OpenCV bounding boxes → bearing/range → RL policy → existing PID loops and mixer → thrusters. The policy runs at 5–10 Hz.

**Settled decisions**
- Real sensors: camera, pressure (depth), IMU with magnetometer. Nothing measures horizontal speed.
- Policy actions: heading setpoint, depth setpoint, open-loop surge power.
- PID gains get tuned in the pool, so training randomizes them.
- Development and training run on the iMac: Python 3.12 `.venv`, torch 2.2.2 (see `requirements.txt`).
- The pool is very clear: detection is easy at 25 m (`DetectorNoiseConfig.clearRange` 25–35 m, `maxRange` 35–50 m).
- Measured: gate poles are 1" PVC (33.4 mm OD), the sub is 0.25 m tall, and the thrusters are 9" (0.229 m) from center. Inertia is estimated from the layout (¾" PVC frame, 4"×9" electronics tube, 4"×6"×4" battery box on top): yaw 0.13, pitch 0.09 kg·m². Scripted noisy baseline: 40/40 on two sets of 40 seeds, after fixing the gate-pass test (below).
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
  - "Until lost" subtasks only count a target as gone when it was last seen at the image edge, not after a mid-view dropout. `DriveUntilTargetLost` also accepts a gate lost while big if it stays gone for 3 s.
  - `DriveUntilTargetLost` steers at the gate while it's visible.
  - Scripted mission, `tools/run_headless.py`: perfect detector 8/8. Noisy detector with clear-pool ranges about 90% over 40 runs (see Phase 1). With the earlier murky-water guess (6–10 m) it was about 4/12, because the gate is ~10–12 m from the marker.

## Phase 4: Gymnasium environment for the gate task
- [x] **Observations:** depth, depth rate, heading, pitch, pitch rate, yaw rate, and per object found/bearing/range/confidence plus time since last seen. Add the last action and a few stacked past steps, because the gate drops out of view about 1.4 m before you pass it.
  - `observation.py` (numpy + stdlib, shared with the Pi): 20 values per frame × 4 stacked frames = 80. Own state: depth, depth rate, sin/cos of heading relative to the start heading (so the policy can't memorize the course's compass direction), pitch, pitch rate, yaw rate. Objects: the gate (red pole pair) and the most prominent single red pole, each with found, bearing, range, range confidence (from the estimation method) and time since seen. Unseen objects keep their last bearing, corrected for turning, and range. Then the last action.
  - Gate pairing moved into `perception.find_gate_pair()`, used by both `Vision` and the observation.
  - `tools/check_observation.py`: with the noisy detector, gate bearing is within 0.3° median and range within 4.5% median. Range error comes mostly from the pressure sensor's ±2 cm zero error.
  - Fixed the detector's box noise. It moved each edge independently, which made thin, far poles vanish (the gate pair was found in only 39% of in-view frames). It now shifts the center and scales the size: 81%, with the rest from dropouts. Baseline still 40/40.
- [x] **Actions:** heading setpoint, depth setpoint and open-loop surge power. There's no speed setpoint, because the sub can't measure speed.
  - `gate_env.py`, 10 Hz: heading change of up to ±45° from the current heading, target depth 0.3–1.9 m, surge effort −1..1. The sub's own loops hold the targets at the physics rate in between.
  - The PID loops and thruster mixer moved into `controller.py` (shared with the Pi). The scripted tasks use it too, unchanged (still 40/40).
- [x] **Reward:** progress toward the gate, a bonus for passing through, and penalties for collisions, timeouts and energy use.
  - Per step: +1 per meter closer to the gate center (horizontal distance plus depth error from 1.5 m), −0.01 time cost, −0.05 per second of full thrust. On ending: +10 through the gate, −5 for a collision, −5 for crossing the gate line outside the opening. Timeout at 60 s.
  - Episodes start 3–8 m in front of the gate, ±1.5 m off center, ±30° off heading, 0.1–0.5 m deep.
  - At the end of each episode, `info` reports is_success, collision, missed_gate, timeout, time_s and energy, for TensorBoard.
- [x] Use a perfect detector at first, and run the scripted `GateTask` through the environment as a baseline before any training.
  - `tools/eval_gate.py <scripted|straight|random|model.zip>` scores a controller on fixed seeds (10000+, kept apart from training). On 50 episodes:

    | Controller | Perfect detector | Noisy detector |
    |---|---|---|
    | Scripted `GateTask` | 100% (median 21.5 s) | 100% (median 21.7 s) |
    | Straight ahead at 1.5 m | 26% | 26% |
    | Random | 0% | 0% |
  - `WaitForTargetVisible` now turns until its target is visible, toward a single visible red pole when waiting for the gate. Before, the scripted task scored 84% / 96%. Its search turn was a special case in `Task.execute` that seeing the distant green marker cancelled, so it sat still when a start left only one gate pole in view.
  - `GateEnv` passes Stable-Baselines3's `check_env`, imports no pygame or OpenCV, and runs ~1,200 steps/s on one core (~120× real time).

## Phase 5: Training on the iMac
- [ ] PPO in Stable-Baselines3 with about 8–10 parallel environments and `VecNormalize`.
- [ ] Turn on realism one layer at a time, retraining after each: detector noise, then a thruster model (deadband, forward/reverse asymmetry, battery sag, spin-up lag), then randomized PID gains.
- [ ] Measure the policy's success rate against the scripted baseline in the same randomized environment.

## Seeing what's going on
The display stays optional: training runs without a window, and the same physics draws in the pygame window when you want to watch.
- [x] **Watch live:** `main.py` keeps running the scripted mission as now. A `watch.py` loads a trained policy and flies it in the pygame window in real time (`render_mode="human"`).
  - `watch.py <scripted|straight|random|model.zip>` flies gate-environment episodes (same seeds as `tools/eval_gate.py`) in the pygame window. The panel shows the action, the reward, and the gate bearing/range the sub believes next to the truth. The top view shows the sub's path. Keys: pause, single-step, restart, next episode, quit; `--speed` and `--render-every` for faster playback.
  - Drawing moved into `sim_view.SimView`, shared by `main.py` and `watch.py`. Controllers moved into `gate_controllers.py`, shared by `watch.py` and `tools/eval_gate.py`.
- [ ] **Training curves:** log reward, gate-pass rate, collision rate and episode length to TensorBoard (add `tensorboard` to `requirements.txt`). The env already reports the outcomes in `info`.
- [ ] **Scored tests during training:** every N steps, fly the current policy deterministically on fixed seeds with `tools/eval_gate.py`'s scoring, plot its success rate against the scripted baseline, and keep the best checkpoint.
- [ ] **Evaluation videos (later):** during training, periodically record a few runs flown by the current policy as short videos.
- [ ] **Replays (later):** save each run's path (pose, detections, actions) and play it back in the pygame window with pause and single-step.

## Phase 6: Export to the Pi 5
- [ ] Export the weights plus the `VecNormalize` statistics, and write a plain-numpy forward pass.
- [ ] Check that the numpy policy and the PyTorch policy give the same outputs on recorded observations.

## Open questions
- The simulated detector doesn't model occlusion (e.g. a gate pole in front of the marker).
- OpenCV mode at 1280×960 runs at roughly 20 fps on the iMac, so the graphics window is slower than before. The simulated detector is unaffected.
- Should the policy control sway as well? The orbit task uses it, but the gate task doesn't need it. Leaning toward leaving it out for the gate task.
- The sub's real mass and how its 4 kg splits between frame, thrusters, tube and battery box. The inertias (yaw 0.13, pitch 0.09 kg·m²) are estimated from the layout and should be randomized ±30% in training.

#!/usr/bin/env python3
"""
The full qualification run with the learned skills: through the gate, around
the marker, and back through the gate.

MissionEnv is the whole course from the scripted mission's start (3 m before
the gate, at the surface). It judges the run on ground truth: success needs a
forward pass through the gate opening, at least 270° swept around the marker,
then a pass back through the opening. Failures: a collision, crossing the gate
line outside the opening, coming back through the gate without having gone
around the marker (skipped_orbit), surfacing, or running out of time.

MissionRunner is a small hand-written state machine that runs the learned
skills in turn, switching only on things the real sub can measure:
  DIVE   (scripted)      -> GATE   once the measured depth is past DIVE_DONE_M and
                                    steady (the run starts at the surface 3 m from
                                    the gate, too close for the gate policy to dive)
  GATE   (gate policy)   -> CLEAR  once the gate was last seen close (< PASS_RANGE_M),
                                    then not seen for PASS_UNSEEN_S
  CLEAR  (scripted)      -> ORBIT  after CLEAR_S of driving straight at 1.5 m, so the
                                    sub is well past the gate's top bar before the orbit
                                    starts maneuvering (and carried through if the pass
                                    trigger fired a little early)
  ORBIT  (orbit policy)  -> RETURN once the gate is in view, the heading has turned
                                    ~180° since the orbit began (and through at least
                                    ORBIT_TURN_DEG in total), and the marker has left the
                                    view to the side (last seen beyond the edge of the
                                    view, or not seen for MARKER_GONE_S). That last check
                                    separates the real finish (~270° around) from facing
                                    the gate with the marker still in between. And the
                                    marker must be on the side the sub has been turning
                                    toward: at ~100° around, facing the gate, it's on the
                                    other side. (Requiring the marker 45-60° off the nose
                                    instead delayed the handoff and made results worse.)
  RETURN (gate policy)   -> DONE   same pass check as GATE
Each skill gets its own observation builder (the two policies see different
layouts), reset when the skill takes over, as in training.

Safety limit: the gate and orbit skills never command a depth shallower than
MIN_DEPTH_M, matching the limit their environments train with (the opening
is below 1.15 m, and both policies tended to rise and surface otherwise).
"""
import math
import random
from typing import List, Optional

import numpy as np

from observation import ObservationBuilder
from sub_env import SubTaskEnv

GATE_POLICY = "runs/gate_rand_ft2/final_model.zip"
DIVE_DEPTH_M = 1.5
DIVE_DONE_M = 1.3           # Dive is done below this...
DIVE_STEADY_MPS = 0.1       # ...once the depth rate has settled under this
MIN_DEPTH_M = 1.2
ORBIT_POLICY = "runs/orbit_tight_ft/final_model.zip"

PASS_RANGE_M = 4.0          # The gate counts as passed if last seen closer than this (2.5 m missed passes
                            # where the pair dropped out of view a little farther out)...
PASS_UNSEEN_S = 3.0         # ...and not seen for this long since (it leaves the view ~1.5-2 m out)
CLEAR_S = 3.0               # Then drive straight this long before the orbit takes over
CLEAR_SURGE = 0.6
CLEAR_DEPTH_M = 1.5
ORBIT_TURN_DEG = 135.0      # Total heading turn needed before handing back to the gate skill
MARKER_GONE_S = 2.0         # The marker counts as out of view after this long unseen (longer than a dropout)...
HALF_FOV_DEG = 35.0         # ...or if last seen beyond the edge of the view
HANDOFF_HEADING_DEG = 45.0  # Heading within this of reversed (as in orbit_env)
SWEEP_DEG = 270.0           # Ground truth: what counts as having gone around the marker
MIN_MARKER_DISTANCE_M = 1.5


def _wrap(deg: float) -> float:
    return (deg + 180) % 360 - 180


class MissionEnv(SubTaskEnv):
    OUTCOMES = ("is_success", "collision", "missed_gate", "skipped_orbit", "surfaced", "timeout")
    EXTRA_INFO = ("inside_margin_s", "closest_m")
    INCLUDE_MARKER = True
    ACTION_DIM = 4

    def __init__(self, detector_mode: str = "noisy", max_seconds: float = 300.0, randomize: bool = False):
        super().__init__(detector_mode, max_seconds=max_seconds, timeout_penalty=0.0, randomize=randomize)

    def _place_sub(self, rng: random.Random):
        s, g = self.core.state, self.core.gate
        s.x = self.core.prequal.START_X_POS
        s.y = g.center_y + rng.uniform(-0.5, 0.5)
        s.z = self.core.prequal.START_Z_POS
        s.heading = rng.uniform(-10.0, 10.0) % 360

    def _marker_angle(self) -> float:
        s, m = self.core.state, self.core.marker
        return math.degrees(math.atan2(s.y - m.y, s.x - m.x))

    def _on_reset(self):
        self.through_gate = False
        self.swept = 0.0
        self._last_angle = self._marker_angle()
        self._inside_margin_s, self._closest = 0.0, float("inf")

    def _physics_outcome(self) -> Optional[str]:
        s, m = self.core.state, self.core.marker
        if self.through_gate:
            angle = self._marker_angle()
            self.swept += _wrap(angle - self._last_angle)
            self._last_angle = angle
        d = math.hypot(s.x - m.x, s.y - m.y)
        self._closest = min(self._closest, d)
        if d < MIN_MARKER_DISTANCE_M:
            self._inside_margin_s += self.core.config.physicsDt
        for c in self.new_crossings():
            if not c.inside_opening:
                return "missed_gate"
            if not self.through_gate and c.direction > 0:
                self.through_gate = True
                self._last_angle = self._marker_angle()
            elif self.through_gate and c.direction < 0:
                return "success" if abs(self.swept) >= SWEEP_DEG else "skipped_orbit"
            else:
                return "missed_gate"
        return None

    def _potential(self) -> float:
        return 0.0   # Only for scoring, not training

    def _terminal_reward(self, outcome: str) -> float:
        return 0.0

    def _episode_info(self) -> dict:
        return {"inside_margin_s": self._inside_margin_s, "closest_m": self._closest}

    def scripted_mission(self) -> List:
        from main import build_mission
        return build_mission()


class MissionRunner:
    """Runs the learned gate and orbit skills in turn, switching on measurable triggers."""
    name = "mission"

    def __init__(self, gate_path: str = GATE_POLICY, orbit_path: str = ORBIT_POLICY):
        from gate_controllers import load_policy
        self.gate_model, self.gate_norm = load_policy(gate_path)
        self.orbit_model, self.orbit_norm = load_policy(orbit_path)
        self.last_action = None

    def reset(self, env):
        cam = env.core.camera
        self.gate_builder = ObservationBuilder(cam, action_dim=3)
        self.orbit_builder = ObservationBuilder(cam, action_dim=4, include_marker=True)
        self.phase = "DIVE"
        self.dive_heading = env.readings.heading
        self.gate_was_close = False
        self.turned, self._prev_heading = 0.0, None
        self.skill_action = np.zeros(3, dtype=np.float32)
        self.last_action = None
        self.switch_times = {}

    def state_name(self) -> str:
        extra = f", turned {self.turned:+.0f} deg" if self.phase == "ORBIT" else ""
        return f"{self.phase}{extra}"

    def _update(self, env, builder) -> np.ndarray:
        r, d = env.readings, env.detections
        return builder.update(r.depth, r.depth_rate, r.heading, r.pitch, r.imu.gyro_y, r.imu.gyro_z,
                              d.red_blobs if d else [], self.skill_action, env.control_period,
                              green_boxes=d.green_blobs if d else [])

    def _passed_gate(self) -> bool:
        gate = self.gate_builder.gate
        if gate.found:
            self.gate_was_close = gate.range_m < PASS_RANGE_M
            return False
        return self.gate_was_close and gate.unseen >= PASS_UNSEEN_S

    def _switch(self, env, phase: str):
        self.phase = phase
        self.switch_times[phase] = env.time
        self.skill_action = np.zeros(4 if phase == "ORBIT" else 3, dtype=np.float32)
        if phase == "CLEAR":
            self.clear_heading, self.clear_until = env.readings.heading, env.time + CLEAR_S
        elif phase == "ORBIT":
            self.orbit_builder.reset()
            self.turned, self._prev_heading = 0.0, None
        elif phase == "RETURN":
            self.gate_builder.reset()
            self.gate_was_close = False

    @staticmethod
    def _depth_action(depth_m: float) -> float:
        return (depth_m - 0.3) / (1.9 - 0.3) * 2 - 1

    def step(self, env, obs):
        heading = env.readings.heading
        r = env.readings
        if self.phase == "DIVE" and r.depth > DIVE_DONE_M and abs(r.depth_rate) < DIVE_STEADY_MPS:
            self._switch(env, "GATE")
        if self.phase in ("GATE", "RETURN"):
            o = self._update(env, self.gate_builder)
            if self._passed_gate():
                self._switch(env, "CLEAR" if self.phase == "GATE" else "DONE")
        if self.phase == "CLEAR" and env.time >= self.clear_until:
            self._switch(env, "ORBIT")
        if self.phase == "ORBIT":
            if self._prev_heading is not None:
                self.turned += _wrap(heading - self._prev_heading)
            self._prev_heading = heading
            o = self._update(env, self.orbit_builder)
            b = self.orbit_builder
            reversed_ = b.start_heading is not None and abs(_wrap(heading - b.start_heading - 180)) <= HANDOFF_HEADING_DEG
            marker_gone = (b.marker.seen_ever and (b.marker.unseen >= MARKER_GONE_S
                                                  or abs(b.marker.bearing(heading)) > HALF_FOV_DEG))
            right_side = b.marker.seen_ever and b.marker.bearing(heading) * self.turned > 0
            if (b.gate.found and reversed_ and marker_gone and right_side
                    and abs(self.turned) >= ORBIT_TURN_DEG):
                self._switch(env, "RETURN")
                o = self._update(env, self.gate_builder)

        if self.phase == "ORBIT":
            a = self.orbit_model.predict(self.orbit_norm.normalize_obs(o), deterministic=True)[0]
            self.skill_action = np.asarray(a, dtype=np.float32)
            action = self.skill_action.copy()
            action[1] = max(action[1], self._depth_action(MIN_DEPTH_M))   # Safety limit on depth
        elif self.phase in ("GATE", "RETURN"):
            a = self.gate_model.predict(self.gate_norm.normalize_obs(o), deterministic=True)[0]
            self.skill_action = np.asarray(a, dtype=np.float32)
            action = np.append(self.skill_action, 0.0)
            action[1] = max(action[1], self._depth_action(MIN_DEPTH_M))   # Safety limit on depth
        else:
            # DIVE: hold the start heading, no thrust forward, dive to 1.5 m.
            # CLEAR: straight ahead at 1.5 m, holding the heading from the pass. DONE: hold position.
            depth_action = self._depth_action(DIVE_DEPTH_M if self.phase == "DIVE" else CLEAR_DEPTH_M)
            turn = 0.0
            hold = {"CLEAR": getattr(self, "clear_heading", None), "DIVE": self.dive_heading}.get(self.phase)
            if hold is not None:
                turn = float(np.clip(_wrap(hold - env.readings.heading) / 45.0, -1, 1))
            surge = CLEAR_SURGE if self.phase == "CLEAR" else 0.0
            action = np.array([turn, depth_action, surge, 0.0], dtype=np.float32)
        self.last_action = action
        return env.step(action)

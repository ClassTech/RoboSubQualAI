#!/usr/bin/env python3
"""
Watch a controller fly the gate environment in the pygame window: the same
episodes, starts and noise that training and tools/eval_gate.py use.

The panel shows the controller's action, the reward, and what the policy
believes about the gate (from its observation) next to the truth.

Usage:
  .venv/bin/python watch.py scripted
  .venv/bin/python watch.py straight --detector perfect
  .venv/bin/python watch.py runs/gate/best_model.zip --speed 2

Keys: SPACE pause, N single physics step while paused, R restart the episode,
ENTER next episode, Q or ESC quit.
"""
import argparse
import math

import pygame

from gate_controllers import ScriptedController, make_controller
from gate_env import DEPTH_MAX, DEPTH_MIN, HEADING_CHANGE_DEG, GateEnv
from perception import find_gate_pair
from sim_view import SimView

OUTCOME_NAMES = {"is_success": "SUCCESS", "collision": "COLLISION", "missed_gate": "MISSED GATE", "timeout": "TIMEOUT"}


class _Restart(Exception): pass
class _Next(Exception): pass
class _Quit(Exception): pass


class Viewer:
    def __init__(self, args):
        self.args = args
        self.env = GateEnv(detector_mode=args.detector)
        self.view = SimView(self.env.core.config, title="Gate task viewer")
        self.controller = make_controller(args.controller)
        self.env.on_physics_step = self._on_physics_step
        self.paused = False
        self.frame = 0
        self.tally = dict.fromkeys(OUTCOME_NAMES, 0)
        self.last_outcome = ""

    # --- Input ---
    def _handle_events(self) -> bool:
        """Processes keys. Returns True if a single step was requested while paused."""
        step = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise _Quit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE): raise _Quit()
                if event.key == pygame.K_SPACE: self.paused = not self.paused
                elif event.key == pygame.K_n: step = True
                elif event.key == pygame.K_r: raise _Restart()
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER): raise _Next()
        return step

    # --- Drawing ---
    def _lines(self):
        env, a = self.env, self.args
        r, s, g = env.readings, env.core.state, env.core.gate
        lines = [f"Controller: {self.controller.name}   Detector: {a.detector}",
                 f"Episode {self.episode + 1}, seed {self.seed}   Time {env.time:4.1f} / {env.max_seconds:.0f} s",
                 f"Reward: last step {self.step_reward:+.2f}, total {self.total_reward:+.2f}"]
        if isinstance(self.controller, ScriptedController):
            lines.append(f"Scripted state: {self.controller.state_name()}")
        act = self.controller.last_action
        if act is not None:
            depth = DEPTH_MIN + (act[1] + 1) / 2 * (DEPTH_MAX - DEPTH_MIN)
            lines.append(f"Action: turn {HEADING_CHANGE_DEG * act[0]:+5.1f} deg, depth {depth:.2f} m, surge {act[2]:+.2f}")
        rel_heading = (r.heading - env.builder.start_heading + 180) % 360 - 180 if env.builder.start_heading is not None else 0.0
        lines += [f"Sub: depth {r.depth:.2f} m, heading {rel_heading:+.0f} deg from start, pitch {r.pitch:+.1f} deg",
                  ""]
        track = env.builder.tracks["gate"]
        dx, dy = g.x - s.x, g.center_y - s.y
        true_bearing = (math.degrees(math.atan2(dy, dx)) - s.heading + 180) % 360 - 180
        if not track.seen_ever:
            lines.append("Gate: not seen yet")
        else:
            seen = "in view" if track.found else f"remembered, last seen {track.unseen:.1f} s ago"
            believed = (track.abs_bearing - r.heading + 180) % 360 - 180
            lines += [f"Gate: {seen}",
                      f"  believes  bearing {believed:+6.1f} deg, range {track.range_m:5.2f} m (confidence {track.confidence:.1f})"]
        lines += [f"  truth     bearing {true_bearing:+6.1f} deg, range {math.hypot(dx, dy):5.2f} m",
                  "",
                  f"Last episode: {self.last_outcome or '-'}",
                  "Totals: " + ", ".join(f"{OUTCOME_NAMES[k].lower()} {v}" for k, v in self.tally.items())]
        return lines

    def _draw(self, thrusters):
        env = self.env
        self.view.draw_camera(env.core)
        red = env.detections.red_blobs if env.detections else []
        self.view.draw_boxes(red, find_gate_pair(red, env.core.camera.height))
        self.view.render(env.core, self._lines(), thrusters, title="Gate task viewer", trail=self.trail,
                         controls=("SPACE - Pause", "N - Step (paused)", "R - Restart episode",
                                   "ENTER - Next episode", "Q - Quit"))

    def _on_physics_step(self, env, thrusters):
        self.trail.append((env.core.state.x, env.core.state.y))
        self.thrusters = thrusters
        self.frame += 1
        if self.frame % self.args.render_every == 0 or self.paused:
            self._draw(thrusters)
            self.view.clock.tick(60 * self.args.speed / self.args.render_every)
        stepping = self._handle_events()
        while self.paused and not stepping:
            self._draw(thrusters)
            self.view.clock.tick(30)
            stepping = self._handle_events()

    # --- Episodes ---
    def run_episode(self):
        env = self.env
        obs, _ = env.reset(seed=self.seed)
        self.controller.reset(env)
        self.trail, self.step_reward, self.total_reward = [], 0.0, 0.0
        while True:
            obs, reward, term, trunc, info = self.controller.step(env, obs)
            self.step_reward, self.total_reward = reward, self.total_reward + reward
            if term or trunc:
                outcome = next(k for k in OUTCOME_NAMES if info[k])
                self.tally[outcome] += 1
                self.last_outcome = f"{OUTCOME_NAMES[outcome]} after {info['time_s']:.1f} s, reward {self.total_reward:+.1f}"
                print(f"Episode {self.episode + 1} (seed {self.seed}): {self.last_outcome}")
                end = pygame.time.get_ticks() + 1500   # Show the end for a moment
                while pygame.time.get_ticks() < end:
                    self._draw(self.thrusters)
                    self._handle_events()
                    self.view.clock.tick(30)
                return

    def run(self):
        self.episode, self.seed = 0, self.args.seed
        try:
            while self.args.episodes == 0 or self.episode < self.args.episodes:
                try:
                    self.run_episode()
                except _Restart:
                    continue
                except _Next:
                    pass
                self.episode += 1
                self.seed += 1
        except _Quit:
            pass
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("controller", help="scripted, straight, random, or a path to a PPO model .zip")
    parser.add_argument("--detector", choices=["perfect", "noisy"], default="noisy")
    parser.add_argument("--seed", type=int, default=10_000, help="First episode's seed (eval seeds start at 10000)")
    parser.add_argument("--episodes", type=int, default=0, help="Stop after this many episodes (0 = keep going)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed; above 1 is faster than real time")
    parser.add_argument("--render-every", type=int, default=2,
                        help="Draw every Nth physics step (the 1280x960 camera is slow to draw every step)")
    Viewer(parser.parse_args()).run()


if __name__ == "__main__":
    main()

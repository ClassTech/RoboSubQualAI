#!/usr/bin/env python3
"""
Contains the "brain" of the sub: the Submarine class that manages the mission plan and AI logic.
"""
import math
import numpy as np
from typing import Tuple, List, Optional

# --- Import Vision class ---
from data_structures import ThrusterCommands, SensorSuite, Vision
# ---
from config import SimulationConfig
from utils import angle_diff
from ai.tasks import TaskStatus # Corrected import
from ai.tasks.task_base import Task

class Submarine:
    def __init__(self, mission_plan: List[Task]):
        # === PID and Control Gains ===
        
        # --- Horizontal (XY) Gains ---
        self.ALIGN_YAW_P_GAIN = 0.6
        self.YAW_D_GAIN = 4.0 
        self.HOVER_YAW_P_GAIN = 0.1
        self.HOVER_XY_P_GAIN = 1.0
        self.HOVER_XY_I_GAIN = 0.0 
        self.HOVER_XY_D_GAIN = 4.0
        self.DAMPING_GAIN = 1.5
        self.MANEUVER_DAMPING_GAIN = 3.0 

       # --- Vertical (Z/Pitch) Gains (TUNE ME) ---
        # --- Adjusted P and I gains ---
        self.DEPTH_P_GAIN = 2.5  # Was 2.8. Reduced slightly.
        self.DEPTH_I_GAIN = 0.4  # Was 0.15. Increased significantly to fight steady-state error.
        self.DEPTH_D_GAIN = 5.0  # Keep D gain high for now.
        self.PITCH_P_GAIN = 1.0
        self.PITCH_D_GAIN = 1.5
        
        # Integrals
        self.integral_x_err = 0.0
        self.integral_y_err = 0.0
        self.integral_z_err = 0.0 
        self.integral_clamp = 2.0 
        self.integral_clamp_z = 2.0 
        # ---

        self.MIN_PIXELS_FOR_DETECTION = 20

        self.mission_plan = mission_plan
        self.config = SimulationConfig()
        
        self._latest_sensors: Optional[SensorSuite] = None
        
        self.vision = Vision(image_provider=lambda: self._latest_sensors.camera_image if self._latest_sensors else None,
                             min_pole_pixels=self.MIN_PIXELS_FOR_DETECTION,
                             min_gate_pixels=50) 
        
        self.reset()

    def reset(self):
        self.current_task_index = 0
        for task in self.mission_plan:
            task.reset()
        # --- 3D Targets ---
        self.target_x, self.target_y = 0.0, 0.0
        self.target_heading = 0.0
        self.target_depth = 0.0 
        self.target_pitch = 0.0
        # ---
        self.integral_x_err, self.integral_y_err = 0.0, 0.0 
        self.integral_z_err = 0.0 
        self._latest_sensors = None 

    def update(self, dt: float, sensors: SensorSuite) -> Tuple[ThrusterCommands, Vision]: 
        
        self._latest_sensors = sensors

        if self.current_task_index >= len(self.mission_plan):
            cmds = self._get_damping_commands(sensors)
            return cmds, self.vision 

        current_task = self.mission_plan[self.current_task_index]
        self.target_depth = getattr(current_task, 'target_depth', self.target_depth)
        self.vision.update()
        
        # --- UPDATE INTEGRAL TERMS ---
        depth_error = self.target_depth - sensors.depth
        self.integral_z_err = np.clip(self.integral_z_err + depth_error * dt, -self.integral_clamp_z, self.integral_clamp_z)
        # ---

        status, commands = current_task.execute(self, dt, sensors, self.vision, self.config)

        # --- Task Transition Logic ---
        if status == TaskStatus.COMPLETED and self.current_task_index < len(self.mission_plan) - 1:
            
            self.current_task_index += 1
            next_task = self.mission_plan[self.current_task_index]
            
            self.target_depth = getattr(next_task, 'target_depth', self.target_depth)
            
            # Reset integrals before tasks that need precise stabilization
            if hasattr(next_task, 'SPEED_THRESHOLD') or next_task.__class__.__name__ == 'DiveToDepth': 
                 print(f"INFO: Resetting PID integrals for task: {next_task.__class__.__name__}")
                 self.integral_x_err, self.integral_y_err, self.integral_z_err = 0.0, 0.0, 0.0
            if hasattr(next_task, 'on_start'): # Should use on_enter
                next_task.on_start(self, sensors) 

        return commands, self.vision
    
    # --- ADDED METHODS BACK ---
    def get_current_task_name(self) -> str:
        """Returns the class name of the currently running task."""
        if self.current_task_index < len(self.mission_plan):
             return self.mission_plan[self.current_task_index].__class__.__name__
        return "MISSION_COMPLETE"

    def get_current_state_name(self) -> str:
        """Returns the specific state name within the current task."""
        if self.current_task_index < len(self.mission_plan):
             return self.mission_plan[self.current_task_index].state_name
        return ""
    # --- END ADDED METHODS ---

    # ---
    # === NEW/UPDATED CONTROL HELPERS ===
    # ---

    def get_depth_pitch_commands(self, sensors: SensorSuite, 
                                 target_depth: float, 
                                 target_pitch: float = 0.0) -> Tuple[float, float]:
        """
        Calculates heave (vertical) and pitch commands using PID for depth.
        Returns: (heave_command, pitch_command)
        """
        depth_error = target_depth - sensors.depth
        
        heave_p = depth_error * self.DEPTH_P_GAIN
        heave_i = self.integral_z_err * self.DEPTH_I_GAIN 
        heave_d = -sensors.velocity_z * self.DEPTH_D_GAIN
        heave_cmd = np.clip(heave_p + heave_i + heave_d, -1.0, 1.0)
        
        pitch_error = angle_diff(target_pitch, sensors.pitch)
        pitch_p = pitch_error * self.PITCH_P_GAIN
        pitch_d = -sensors.imu.gyro_y * self.PITCH_D_GAIN 
        pitch_cmd = np.clip(pitch_p + pitch_d, -1.0, 1.0)
        
        return heave_cmd, pitch_cmd

    def _get_damping_commands(self, sensors: SensorSuite) -> ThrusterCommands:
        """Fully damps all 6 axes of motion."""
        wdx, wdy = -sensors.velocity_x * self.DAMPING_GAIN, -sensors.velocity_y * self.DAMPING_GAIN
        h_rad = math.radians(sensors.heading)
        c, s = math.cos(h_rad), math.sin(h_rad)
        surge, sway = wdx*c+wdy*s, -wdx*s+wdy*c
        
        yaw = -sensors.imu.gyro_z * self.YAW_D_GAIN 
        
        # Hold current depth
        heave, pitch = self.get_depth_pitch_commands(sensors, sensors.depth, 0.0)
        
        return self._mix_and_normalize_commands(surge, sway, yaw, heave, pitch)

    def get_spin_damping_commands(self, sensors: SensorSuite, target_depth: Optional[float] = None):
        """Damps everything *except* yaw, while holding depth."""
        target_depth = target_depth if target_depth is not None else self.target_depth

        wdx, wdy = -sensors.velocity_x * self.DAMPING_GAIN, -sensors.velocity_y * self.DAMPING_GAIN
        h_rad, cos_h, sin_h = math.radians(sensors.heading), math.cos(math.radians(sensors.heading)), math.sin(math.radians(sensors.heading))
        surge, sway = wdx * cos_h + wdy * sin_h, -wdx * sin_h + wdy * cos_h
        
        heave, pitch = self.get_depth_pitch_commands(sensors, target_depth, 0.0)
        
        return self._mix_and_normalize_commands(surge, sway, 0.0, heave, pitch)

    def get_heading_commands(self, sensors, heading, 
                             surge_power=0.0, 
                             sway_power=0.0, 
                             target_depth: Optional[float] = None,
                             target_pitch: float = 0.0):
        """Holds heading and depth/pitch (now with PID), applies surge/sway power."""
        target_depth = target_depth if target_depth is not None else self.target_depth
        
        h_rad = math.radians(sensors.heading)
        cos_h, sin_h = math.cos(h_rad), math.sin(h_rad)

        # Yaw Control
        yaw_err = angle_diff(heading, sensors.heading)
        yaw_p = yaw_err * self.HOVER_YAW_P_GAIN
        yaw_d = -sensors.imu.gyro_z * self.YAW_D_GAIN 
        yaw = np.clip(yaw_p + yaw_d, -1.0, 1.0)
        
        # --- MODIFIED Sway Control ---
        sway_vel = -sensors.velocity_x * sin_h + sensors.velocity_y * cos_h
        sway_damp = -sway_vel * self.MANEUVER_DAMPING_GAIN
        sway = np.clip(sway_power + sway_damp, -1.0, 1.0) 
        # ---
        
        # Vertical Control
        heave, pitch = self.get_depth_pitch_commands(sensors, target_depth, target_pitch)
        
        return self._mix_and_normalize_commands(surge_power, sway, yaw, heave, pitch)

    def _get_pid_hover_commands(self, sensors: SensorSuite, dt: float, 
                                tx: float, ty: float, 
                                tz: float, 
                                yaw_p_gain_override: Optional[float] = None) -> ThrusterCommands:
        
        # --- Horizontal XY Control (PID) ---
        yaw_p_gain = yaw_p_gain_override if yaw_p_gain_override is not None else self.HOVER_YAW_P_GAIN
        ex, ey = tx - sensors.x, ty - sensors.y
        self.integral_x_err = np.clip(self.integral_x_err + ex * dt, -self.integral_clamp, self.integral_clamp)
        self.integral_y_err = np.clip(self.integral_y_err + ey * dt, -self.integral_clamp, self.integral_clamp)
        wtx = (ex * self.HOVER_XY_P_GAIN) + (self.integral_x_err * self.HOVER_XY_I_GAIN) - (sensors.velocity_x * self.HOVER_XY_D_GAIN)
        wty = (ey * self.HOVER_XY_P_GAIN) + (self.integral_y_err * self.HOVER_XY_I_GAIN) - (sensors.velocity_y * self.HOVER_XY_D_GAIN)
        h_rad, c, s = math.radians(sensors.heading), math.cos(math.radians(sensors.heading)), math.sin(math.radians(sensors.heading))
        surge, sway = wtx * c + wty * s, -wtx * s + wty * c
        
        # --- Yaw Control (PD) ---
        yaw_err = angle_diff(self.target_heading, sensors.heading)
        yaw_p = yaw_err * yaw_p_gain
        yaw_d = -sensors.imu.gyro_z * self.YAW_D_GAIN
        yaw_cmd = np.clip(yaw_p + yaw_d, -1.0, 1.0)
        
        # --- Vertical Z/Pitch Control (PID for Z, PD for Pitch) ---
        heave, pitch = self.get_depth_pitch_commands(sensors, tz, 0.0)
        
        return self._mix_and_normalize_commands(surge, sway, yaw_cmd, heave, pitch)

    def get_go_to_visual_target_commands(self, sensors: SensorSuite, 
                                         nav_target_x: float, 
                                         surge_power: float,
                                         target_depth: Optional[float] = None):
        """Holds depth (PID), applies surge, and steers yaw to a visual target."""
        target_depth = target_depth if target_depth is not None else self.target_depth

        cam_w, cam_h = sensors.camera_image.get_size()
        pixel_error_x = nav_target_x - (cam_w / 2)
        
        yaw_p = -(pixel_error_x / (cam_w / 2)) * self.ALIGN_YAW_P_GAIN 
        yaw_d = -sensors.imu.gyro_z * self.YAW_D_GAIN
        yaw = np.clip(yaw_p + yaw_d, -1.0, 1.0)
        
        h_rad = math.radians(sensors.heading)
        sway_vel = -sensors.velocity_x * math.sin(h_rad) + sensors.velocity_y * math.cos(h_rad)
        sway = -sway_vel * self.MANEUVER_DAMPING_GAIN
        sway = np.clip(sway, -0.5, 0.5) 
        
        heave, pitch = self.get_depth_pitch_commands(sensors, target_depth, 0.0)
        
        return self._mix_and_normalize_commands(surge_power, sway, yaw, heave, pitch)

    def _mix_and_normalize_commands(self, surge, sway, yaw, heave, pitch) -> ThrusterCommands:
        """
        Mixes 5-axis commands into 6 thruster commands.
        """
        commands = ThrusterCommands()
        cos_45 = 0.7071 
        surge_cmd = surge * cos_45
        sway_cmd  = sway * cos_45
        
        commands.hfl = surge_cmd + sway_cmd + yaw
        commands.hfr = surge_cmd - sway_cmd - yaw
        commands.hal = surge_cmd - sway_cmd + yaw
        commands.har = surge_cmd + sway_cmd - yaw

        commands.vf = heave + pitch
        commands.vr = heave - pitch

        max_abs = max(1.0, 
                      abs(commands.hfl), abs(commands.hfr),
                      abs(commands.hal), abs(commands.har),
                      abs(commands.vf), abs(commands.vr))

        if max_abs > 1.0:
            commands.hfl /= max_abs
            commands.hfr /= max_abs
            commands.hal /= max_abs
            commands.har /= max_abs
            commands.vf /= max_abs
            commands.vr /= max_abs

        return commands
#!/usr/bin/env python3
"""
Orbit Turn Task: Navigates around the marker using a dynamic orbit
(sway + yaw + surge) until gate+pole are centered, then sways left until pole is lost.
"""
import math
from typing import Tuple, List, Dict, Any
import pygame
import numpy as np

# Absolute imports
from ai.tasks.task_base import Task, TaskStatus
from ai.tasks.subtask_base import Subtask, SubtaskStatus
# Import all the building blocks
# --- Make sure SwayUntilTargetLost is imported ---
from ai.tasks.common_subtasks import (WaitForTargetVisible, AlignToObjectX,
                                      ApproachAndCenterObject, Stabilize,
                                      DynamicOrbitPole, SwayUntilTargetLost) # <-- Added SwayUntilTargetLost
# ---

from data_structures import SensorSuite, Vision, ThrusterCommands
from config import SimulationConfig
from utils import angle_diff

class OrbitTurnTask(Task):
    """
    Finds, approaches (centered), orbits (centered) until gate+pole centered,
    then sways left until pole is lost.
    """
    def __init__(self,
                 target_depth: float = 1.0,
                 approach_height_px: int = 180,
                 orbit_target_fraction: float = 0.5,
                 orbit_sway_power: float = -0.1,     # Negative = Left for orbit
                 orbit_yaw_gain: float = 2.5,
                 orbit_target_height_px: int = 180,
                 final_sway_power: float = -0.3):    # Sway power for final move

        self.target_depth = target_depth

        super().__init__() # Call base constructor

        # --- CORRECTED SUBTASK LIST ---
        self.subtasks = [
            # 1. Find the pole
            WaitForTargetVisible(target_type='pole'),

            # 2. Align to center (50%)
            AlignToObjectX(
                target_x_fraction=0.5,
                tolerance_px=15,
                yaw_gain=orbit_yaw_gain, # Use consistent gain
                yaw_rate_tolerance=0.05
            ),

            # 3. Get close to it while keeping it CENTERED
            ApproachAndCenterObject(
                height_threshold_px=approach_height_px,
                # Default target_x_fraction is 0.5 (center)
                surge_p_gain=0.1,
                height_tolerance_px=15,
                yaw_gain=orbit_yaw_gain, # Use consistent gain
                align_tolerance_px=20
            ),

            # 4. Stop briefly
            Stabilize(duration=1.0),

            # 5. Execute the dynamic orbit keeping pole CENTERED until gate+pole centered
            DynamicOrbitPole(
                target_x_fraction=orbit_target_fraction, # Use 0.5 here
                sway_power=orbit_sway_power,
                yaw_gain=orbit_yaw_gain,
                target_height_px=orbit_target_height_px,
                orbit_surge_p_gain=0.1,
                orbit_surge_i_gain=0.02,
                orbit_surge_d_gain=0.1,
                orbit_surge_i_clamp=0.5,
                center_tolerance_fraction=0.15 # +/- 15% tolerance
            ),

            # 6. Stabilize after alignment is achieved
            Stabilize(duration=1.0),

            # --- ADDED STEPS BACK ---
            # 7. Sway left until the POLE is lost
            SwayUntilTargetLost(
                sway_power=final_sway_power, # Use potentially different sway power
                target_type='pole'
            ),

            # 8. Stabilize after pole is lost (ready for GateTask)
            Stabilize(duration=1.0)
            # ---
        ]
        # --- END CORRECTED LIST ---
        self.reset()

    def reset(self, search_direction: int = 1):
        """Resets the task and adds target_depth to the context."""
        super().reset(search_direction)
        self.context['target_depth'] = self.target_depth

    # No special execute() logic needed.
#!/usr/bin/env python3
"""
Makes the tasks in this directory importable as a package.
Imports base classes first to potentially resolve circular dependencies.
"""
# Import Base Classes FIRST
from .subtask_base import Subtask, SubtaskStatus
from .task_base import Task, TaskStatus

# Import Common Subtasks needed by other tasks
# (List the specific classes needed - adjust if more are used by tasks imported below)
from .common_subtasks import (
    DiveToDepth, SwayStraight, DriveUntilTargetLostForward, DynamicOrbitPole,
    TurnToHeading, DriveStraight, Stabilize, DriveUntilTargetLost,
    WaitForTargetVisible, AlignToObjectX, DriveUntilTargetLostRight,
    SpinUntilTargetReappearsLeft, ApproachAndCenterObject, SwayUntilTargetLost
)

# Import Specific Task Implementations LAST
from .gate_task import GateTask
from .stabilize_task import StabilizeTask
from .orbit_turn_task import OrbitTurnTask
# Add any other specific task files here (e.g., marker_turn_task if re-enabled)
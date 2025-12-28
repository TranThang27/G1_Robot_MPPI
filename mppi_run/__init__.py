"""
MPPI Robot Control System

Modules:
  - utils: Utility functions for robot control
  - constants: Global configuration constants
  - sim_utils: Simulation setup and management
  - mppi_controller: MPPI-based motion planning
  - A_star: A* pathfinding algorithm
  - map_config: Scene map configurations
  - path_smoother: Path interpolation and smoothing
  
Tests:
  - test_avoid_collision: Navigate through obstacles
  - test_room: Multi-target room navigation
"""

__version__ = "1.0.0"
__author__ = "TranThang"

from .utils import (
    get_gravity_orientation,
    pd_control,
    get_next_path_point,
    extract_robot_state
)

from .constants import (
    GOAL_REACHED_THRESHOLD,
    MPPI_HORIZON,
    MPPI_NUM_SAMPLES,
    MPPI_DT
)

from .sim_utils import (
    SimulationConfig,
    MuJoCoSimulator,
    ControlLoop
)

from .mppi_controller import G1MPPIController
from .A_star import AStarPlanner
from .map_config import get_map_config, plan_global_path

__all__ = [
    'get_gravity_orientation',
    'pd_control',
    'get_next_path_point',
    'extract_robot_state',
    'SimulationConfig',
    'MuJoCoSimulator',
    'ControlLoop',
    'G1MPPIController',
    'AStarPlanner',
    'get_map_config',
    'plan_global_path',
]

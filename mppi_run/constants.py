"""
Global constants for robot control and simulation
"""

# ============================================
# CONTROL PARAMETERS
# ============================================
GOAL_REACHED_THRESHOLD = 0.4  # meters - robot reaches goal when within this distance
WAYPOINT_THRESHOLD = 0.3      # meters - move to next waypoint when within this distance
SIMULATION_SPEED = 3.0        # Speed multiplier for visualization (3x faster)

# ============================================
# VIEWER PARAMETERS
# ============================================
VIEWER_SYNC_SKIP = 5          # Update viewer every N physics steps

# ============================================
# MPPI CONTROLLER PARAMETERS
# ============================================
MPPI_HORIZON = 20             # Number of steps to look ahead (reduced for immediate path following)
MPPI_NUM_SAMPLES = 1500       # Number of trajectory samples (reduced for speed)
MPPI_LAMBDA = 0.3             # Temperature parameter (lower = more deterministic, follow path)
MPPI_DT = 0.04                # Control timestep (40ms = 50Hz)

# Control limits
MPPI_U_MIN = [0.0, -0.35, -0.5]    # [vx_min, vy_min, omega_min] (slightly increased)
MPPI_U_MAX = [1.2, 0.35, 0.5]      # [vx_max, vy_max, omega_max] (slightly increased)

# Noise sigma for MPPI exploration
MPPI_NOISE_SIGMA = [0.30, 0.25, 0.15]  # [vx, vy, omega]

# ============================================
# OBSTACLE AVOIDANCE PARAMETERS
# ============================================
OBSTACLE_COLLISION_DIST = 0.20      # meters - hard collision threshold (reduced)
OBSTACLE_SAFE_DIST = 0.40           # meters - soft avoidance threshold (reduced from 0.50)
OBSTACLE_HARD_PENALTY = 1e4         # Hard collision penalty weight (reduced from 5e4)
OBSTACLE_SOFT_PENALTY = 50.0        # Soft avoidance penalty weight (reduced from 100.0)

# ============================================
# COST FUNCTION WEIGHTS
# ============================================
COST_WEIGHT_DISTANCE = 1.0          # Distance to target weight
COST_WEIGHT_HEADING = 3.0           # Heading error weight (INCREASED - prioritize facing goal)
COST_WEIGHT_SPEED_REWARD = -2.0     # Speed forward reward
COST_WEIGHT_BACKWARD = 1.0          # Backward motion penalty
COST_WEIGHT_ROTATION = 0.2          # Angular velocity penalty
COST_WEIGHT_PATH_TRACKING = 10.0    # Weight for staying on A* global path

# ============================================
# PATH PLANNING PARAMETERS
# ============================================
ASTAR_RESOLUTION = 0.1              # Grid resolution (10cm)
ASTAR_ROBOT_RADIUS = 0.10           # Robot collision radius (10cm)
PATH_SMOOTH_RESOLUTION = 0.1        # Path interpolation resolution (10cm)
PATH_SMOOTH_FACTOR = 0.3            # Spline smoothing factor

# ============================================
# AVOID_COLLISION SCENARIO
# ============================================
AVOID_COLLISION_GOAL = (8.5, 0.0)  # Goal position for avoid collision test (past all cylinders)

# ============================================
# ROOM_SCENE SCENARIO
# ============================================
ROOM_SAFE_POINTS = [
    [-0.5, 2.0],    # Near left wall
    [7.5, 2.0],     # Near right wall
    [4.5, -3.5],    # Near bottom wall
    [4.5, 3.5],     # Near top wall
]

DEBUG_MODE = True
DEBUG_PRINT_INTERVAL = 50  # Print debug info every N control steps

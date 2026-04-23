"""
Global constants for robot control and simulation
"""

# ============================================
# CONTROL PARAMETERS
# ============================================
GOAL_REACHED_THRESHOLD = 0.4  # meters - robot reaches goal when within this distance
WAYPOINT_THRESHOLD = 0.5      # meters - INCREASED from 0.3 to easily advance to next waypoint
SIMULATION_SPEED = 10.0       # Speed multiplier for visualization (10x faster)

# Velocity smoothing (exponential filter for smooth motion)
VELOCITY_SMOOTH_ALPHA = 0.3   # Smoothing factor (0-1): higher = more smooth, lower = more responsive

# ============================================
# VIEWER PARAMETERS
# ============================================
VIEWER_SYNC_SKIP = 12         # Update viewer every N physics steps

# ============================================
# MPPI CONTROLLER PARAMETERS
# ============================================
MPPI_HORIZON = 12             # Number of steps to look ahead
MPPI_NUM_SAMPLES = 800        # Number of trajectory samples
MPPI_LAMBDA = 0.15            # REDUCED from 0.3 - follow path more strictly
MPPI_DT = 0.04                # Control timestep (40ms = 50Hz)

# Control limits - 2 m/s max speed, faster rotation
MPPI_U_MIN = [0.0, -0.65, -1.2]    # [vx_min, vy_min, omega_min] - faster rotation
MPPI_U_MAX = [2.0, 0.65, 1.2]      # [vx_max, vy_max, omega_max] - faster rotation

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
# COST FUNCTION WEIGHTS
# ============================================
COST_WEIGHT_DISTANCE = 1.0          # Distance to target weight
COST_WEIGHT_HEADING = 3.0           # Heading error weight
COST_WEIGHT_SPEED_REWARD = -2.0     # Speed forward reward
COST_WEIGHT_BACKWARD = 1.0          # Backward motion penalty
COST_WEIGHT_ROTATION = 0.2          # Angular velocity penalty
COST_WEIGHT_PATH_TRACKING = 18.0    

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
AVOID_COLLISION_GOAL = (12.0, -0.2)  # Goal position beyond all obstacles - must navigate through to reach

# ============================================
# ROOM_SCENE SCENARIO
# ============================================
ROOM_SAFE_POINTS = [
    [7.5, 2.0],
    [-0.5, 2.0],    
    [4.5, -3.5],   
    [-0.5, 2.0],     
]

DEBUG_MODE = True
DEBUG_PRINT_INTERVAL = 50  

"""
Simulation setup and management utilities
"""
import yaml
import numpy as np
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
import mujoco
from config.camera_config import CAMERA_CONFIGS


class SimulationConfig:
    """Load and manage simulation configuration"""
    
    def __init__(self, config_file):
        """
        Load simulation config from YAML file
        
        Args:
            config_file: Path to config YAML file (relative to deploy/deploy_mujoco/configs/)
        """
        config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}"
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        # Parse paths
        self.policy_path = self.config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        self.xml_path = self.config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        
        # Simulation parameters
        self.simulation_duration = self.config["simulation_duration"]
        self.simulation_dt = self.config["simulation_dt"]
        self.control_decimation = self.config["control_decimation"]
        
        # Control parameters
        self.kps = np.array(self.config["kps"], dtype=np.float32)
        self.kds = np.array(self.config["kds"], dtype=np.float32)
        self.default_angles = np.array(self.config["default_angles"], dtype=np.float32)
        
        # Scaling factors
        self.ang_vel_scale = self.config["ang_vel_scale"]
        self.dof_pos_scale = self.config["dof_pos_scale"]
        self.dof_vel_scale = self.config["dof_vel_scale"]
        self.action_scale = self.config["action_scale"]
        self.cmd_scale = np.array(self.config["cmd_scale"], dtype=np.float32)
        
        # Dimensions
        self.num_actions = self.config["num_actions"]
        self.num_obs = self.config["num_obs"]
        
        # Initial command
        self.cmd_init = np.array(self.config["cmd_init"], dtype=np.float32)


class MuJoCoSimulator:
    """MuJoCo simulation environment"""
    
    def __init__(self, xml_path, dt=0.004):
        """
        Initialize MuJoCo simulator
        
        Args:
            xml_path: Path to robot XML file
            dt: Simulation timestep
        """
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = dt
    
    def step(self, ctrl):
        """
        Execute one simulation step
        
        Args:
            ctrl: Control command for the robot
        """
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)
    
    def get_state(self):
        """
        Get current robot state
        
        Returns:
            qpos: Position vector
            qvel: Velocity vector
        """
        return self.data.qpos.copy(), self.data.qvel.copy()
    
    def set_dof_torque(self, torque):
        """
        Set joint torques (PD control output)
        
        Args:
            torque: Joint torques
        """
        self.data.ctrl[:] = torque


class ControlLoop:
    """High-level control loop manager"""
    
    def __init__(self, sim_config, policy_path, mppi_controller, astar_planner, map_config):
        """
        Initialize control loop
        
        Args:
            sim_config: SimulationConfig object
            policy_path: Path to neural network policy
            mppi_controller: MPPI controller instance
            astar_planner: A* path planner
            map_config: Map configuration
        """
        self.config = sim_config
        self.mppi = mppi_controller
        self.planner = astar_planner
        self.map_cfg = map_config
        
        # Load policy
        self.policy = torch.jit.load(policy_path)
        
        # State tracking
        self.counter = 0
        self.global_path = []
        self.path_idx = 0
    
    def reset(self):
        """Reset control loop state"""
        self.counter = 0
        self.global_path = []
        self.path_idx = 0
    
    def increment_counter(self):
        """Increment simulation counter"""
        self.counter += 1
    
    def should_plan_path(self):
        """Check if path planning is needed"""
        return len(self.global_path) == 0
    
    def plan_path(self, start_x, start_y, goal_x, goal_y):
        """
        Plan path from start to goal
        
        Args:
            start_x, start_y: Start position
            goal_x, goal_y: Goal position
        
        Returns:
            path: List of (x, y) waypoints or None
        """
        try:
            from .map_config import plan_global_path
            path = plan_global_path(self.planner, start_x, start_y, goal_x, goal_y)
            if path:
                self.global_path = path
                self.path_idx = 0
            return path
        except Exception as e:
            print(f"Planning Error: {e}")
            return None
    
    def get_policy_action(self, obs):
        """
        Get action from neural network policy
        
        Args:
            obs: Observation vector
        
        Returns:
            action: Joint angles
        """
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        action = self.policy(obs_tensor).detach().cpu().numpy().squeeze()
        return action


# ============ SIMULATION SETUP FUNCTIONS ============

def setup_simulation(config_file, map_name):
    """
    Complete simulation setup for a scenario.
    
    Args:
        config_file: Simulation config filename (e.g., "scene_1.yaml")
        map_name: Map configuration name (e.g., "avoid_collision" or "room_scene")
    
    Returns:
        dict: Contains simulator, sim_config, policy, astar_planner, obstacles_array
    """
    from core import G1MPPIController
    from .map_config import get_map_config
    
    # Load simulation config
    sim_config = SimulationConfig(config_file)
    
    # Initialize MuJoCo simulator
    simulator = MuJoCoSimulator(sim_config.xml_path, dt=sim_config.simulation_dt)
    
    # Load map and create path planner
    map_cfg = get_map_config(map_name)
    obstacles_array = map_cfg.get_obstacles_array()
    print(f"Loaded '{map_cfg.name}' map")
    
    astar_planner = map_cfg.create_planner()
    print("A* Initialized")
    
    # Initialize MPPI controller
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mppi_controller = G1MPPIController(
        device=device, 
        local_target=torch.zeros(2, device=device),
        obstacles=obstacles_array,
        global_path=np.array([])
    )
    print("MPPI Controller Initialized")
    
    # Load neural network policy
    policy = torch.jit.load(sim_config.policy_path)
    print("Neural Network Policy Loaded")
    
    return {
        'simulator': simulator,
        'sim_config': sim_config,
        'policy': policy,
        'astar_planner': astar_planner,
        'mppi_controller': mppi_controller,
        'obstacles_array': obstacles_array,
        'device': device,
        'map_config': map_cfg
    }


# ============ CAMERA SETUP ============

def setup_camera(viewer, camera_config):
    cam = viewer.cam
    cam.azimuth = camera_config['azimuth']
    cam.elevation = camera_config['elevation']
    cam.distance = camera_config['distance']
    cam.lookat[:] = camera_config['lookat']


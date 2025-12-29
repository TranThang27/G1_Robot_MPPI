import torch
import numpy as np
from pytorch_mppi import MPPI
from utils import constants as C


class G1MPPIController:
    """MPPI Controller for G1 robot with unified cost function for all scenarios"""
    
    def __init__(self, device="cuda", local_target=None, obstacles=None, global_path=None, dt=C.MPPI_DT):
        """
        Args:
            device: torch device 
            local_target: tensor [x, y] for current target (shared reference)
            obstacles: obstacle positions from environtment (sensor data)
            global_path: A* global path as numpy array (N, 2) for path tracking cost
            dt: control timestep in seconds (default: from constants)
        """
        self.device = device
        self.local_target = local_target
        self.dt = dt
        
        # Store obstacles from environment
        if obstacles is not None:
            self.obstacles = torch.tensor(obstacles, dtype=torch.float32, device=device)
        else:
            self.obstacles = None
        
        # Store A* global path for path tracking
        if global_path is not None:
            self.global_path = torch.tensor(global_path, dtype=torch.float32, device=device)
        else:
            self.global_path = None
        
        # Initialize MPPI with cost function 
        self.mppi = MPPI(
            self.dynamics, #movement model
            self.cost,  #cost function
            nx=3, #state dimension [x, y, yaw] for G1 robot
            horizon=C.MPPI_HORIZON,
            num_samples=C.MPPI_NUM_SAMPLES,
            lambda_=C.MPPI_LAMBDA,
            noise_sigma=torch.diag(torch.tensor(C.MPPI_NOISE_SIGMA, device=device)),
            u_min=torch.tensor(C.MPPI_U_MIN, device=device),
            u_max=torch.tensor(C.MPPI_U_MAX, device=device),
            device=device,
        )
    
    def dynamics(self, state, action):
        """
        G1 robot kinematic model
        state: [x, y, yaw]
        action: [vx, vy, omega]
        """
        x, y, yaw = state[:, 0], state[:, 1], state[:, 2] #current state
        vx, vy, omega = action[:, 0], action[:, 1], action[:, 2] #velocities in body frame
        
        # From body frame to world frame
        new_x = x + (vx * torch.cos(yaw) - vy * torch.sin(yaw)) * self.dt
        new_y = y + (vx * torch.sin(yaw) + vy * torch.cos(yaw)) * self.dt
        new_yaw = yaw + omega * self.dt
        
        return torch.stack([new_x, new_y, new_yaw], dim=1)
    
    def cost(self, state, action):
        """
        - Distance to target
        - Heading error
        - Obstacle avoidance
        - Control effort
        """
        # 1. Target Tracking
        dx = self.local_target[0] - state[:, 0]
        dy = self.local_target[1] - state[:, 1]
        dist_to_target = torch.sqrt(dx**2 + dy**2)
        
        target_heading = torch.atan2(dy, dx)
        heading_error = torch.atan2( torch.sin(target_heading - state[:, 2]), torch.cos(target_heading - state[:, 2]))
        
        # 2. Obstacles avoidance 
        if self.obstacles is not None:
            dist_obs = torch.cdist(state[:, :2], self.obstacles) #compute distances to obstacles
            min_dist = torch.min(dist_obs, dim=1).values # minimum distance to closest obstacle
        else:
            # Fallback: no obstacle avoidance if obstacles not provided
            min_dist = torch.ones(state.shape[0], device=self.device) * 10.0
        
        obstacle_cost = torch.zeros_like(min_dist)
        
        # Hard collision penalty
        obstacle_cost += torch.where(
            min_dist < C.OBSTACLE_COLLISION_DIST,
            C.OBSTACLE_HARD_PENALTY * (C.OBSTACLE_COLLISION_DIST - min_dist) ** 2,
            torch.zeros_like(min_dist)
        )
        
        # Soft avoidance penalty
        obstacle_cost += torch.where(
            (min_dist >= C.OBSTACLE_COLLISION_DIST) & (min_dist < C.OBSTACLE_SAFE_DIST),
            C.OBSTACLE_SOFT_PENALTY * (C.OBSTACLE_SAFE_DIST - min_dist) ** 2,  
            torch.zeros_like(min_dist)
        )
        
        # 3. Control Costs 
        speed_reward = C.COST_WEIGHT_SPEED_REWARD * action[:, 0]
        backward_cost = C.COST_WEIGHT_BACKWARD * torch.relu(-action[:, 0]) ** 2
        
        omega_cost = C.COST_WEIGHT_ROTATION * action[:, 2] ** 2
        
        # 4. Path tracking cost - penalize deviation from A* global path
        path_cost = torch.zeros_like(dist_to_target)
        if self.global_path is not None and self.global_path.shape[0] > 0:
            # Find minimum distance to any point on the global path
            dist_to_path = torch.cdist(state[:, :2], self.global_path)  # (batch, num_path_points)
            min_dist_to_path = torch.min(dist_to_path, dim=1).values     # (batch,)
            path_cost = C.COST_WEIGHT_PATH_TRACKING * min_dist_to_path ** 2
        
        # Total cost - balanced weights for all objectives
        return (
            C.COST_WEIGHT_DISTANCE * dist_to_target +
            C.COST_WEIGHT_HEADING * torch.abs(heading_error) +
            obstacle_cost +
            path_cost +
            speed_reward +
            backward_cost + 
            omega_cost 
        )
    
    def command(self, state_tensor):
        """    
        current state [x, y, yaw]
        Return:
        control command [vx, vy, omega]
        """
        return self.mppi.command(state_tensor)

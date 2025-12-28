import torch
import numpy as np
from pytorch_mppi import MPPI


class G1MPPIController:
    """MPPI Controller for G1 robot with custom dynamics and cost function"""
    
    def __init__(self, device="cuda", local_target=None, scenario="room"):
        """
        Initialize MPPI controller
        
        Args:
            device: torch device (cuda or cpu)
            local_target: tensor [x, y] for current target (shared reference)
            scenario: "room" or "avoid" for different cost functions
        """
        self.device = device
        self.local_target = local_target
        self.scenario = scenario
        
        # Initialize MPPI with appropriate cost function
        if scenario == "avoid":
            # More aggressive parameters for obstacle avoidance
            mppi = MPPI(
                self.dynamics,
                self.cost_avoid,
                nx=3,
                horizon=20,
                num_samples=1500,
                lambda_=0.6,
                noise_sigma=torch.diag(torch.tensor([0.30, 0.25, 0.15], device=device)),
                u_min=torch.tensor([0.0, -0.45, -0.6], device=device),
                u_max=torch.tensor([1.2, 0.45, 0.6], device=device),
                device=device,
            )
        else:  # "room" scenario
            mppi = MPPI(
                self.dynamics,
                self.cost,
                nx=3,
                horizon=25,
                num_samples=1500,
                lambda_=0.7,
                noise_sigma=torch.diag(torch.tensor([0.5, 0.5, 0.2], device=device)),
                u_min=torch.tensor([0.0, -0.5, -0.8], device=device),
                u_max=torch.tensor([1.2, 0.5, 0.8], device=device),
                device=device,
            )
        
        self.mppi = mppi
    
    @staticmethod
    def dynamics(state, action):
        """
        G1 robot kinematic model
        state: [x, y, yaw]
        action: [vx, vy, omega]
        """
        dt = 0.05
        x, y, yaw = state[:, 0], state[:, 1], state[:, 2]
        vx, vy, omega = action[:, 0], action[:, 1], action[:, 2]
        
        new_x = x + (vx * torch.cos(yaw) - vy * torch.sin(yaw)) * dt
        new_y = y + (vx * torch.sin(yaw) + vy * torch.cos(yaw)) * dt
        new_yaw = yaw + omega * dt
        
        return torch.stack([new_x, new_y, new_yaw], dim=1)
    
    def cost(self, state, action):
        """
        Cost function for MPPI optimization (room scenario)
        
        Considers:
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
        heading_error = torch.atan2(
            torch.sin(target_heading - state[:, 2]),
            torch.cos(target_heading - state[:, 2])
        )
        
        # 2. Obstacles
        obs_list = []
        obs_list.append(self._generate_rect(4.0, 0.0, 0.6, 0.6))  # Column
        obs_list.append(self._generate_rect(2.0, 3.0, 0.6, 1.0))  # Table
        obs_list.append(torch.tensor([[2.0, 4.0], [6.0, 4.0], [2.0, -4.0], [6.0, -4.0]], device=self.device))  # Walls
        
        obstacles = torch.cat(obs_list, dim=0).to(self.device)
        dist_obs = torch.cdist(state[:, :2], obstacles)
        min_dist = torch.min(dist_obs, dim=1).values
        
        d_collision = 0.4
        d_safe = 0.8
        obstacle_cost = torch.zeros_like(min_dist)
        obstacle_cost += torch.where(min_dist < d_collision, 1e6 * (d_collision - min_dist)**2, torch.zeros_like(min_dist))
        obstacle_cost += torch.where((min_dist >= d_collision) & (min_dist < d_safe), 200.0 * (d_safe - min_dist)**2, torch.zeros_like(min_dist))
        
        # 3. Control Costs
        speed_reward = -3.0 * action[:, 0]
        backward_cost = torch.relu(-action[:, 0]) ** 2
        
        return (
            1.5 * dist_to_target +
            0.8 * torch.abs(heading_error) +
            obstacle_cost +
            speed_reward +
            5.0 * backward_cost
        )
    
    def cost_avoid(self, state, action):
        """
        Cost function for MPPI optimization (avoid collision scenario)
        
        Optimized for obstacle-rich environments
        """
        device = state.device

        # ===== TARGET =====
        target = torch.tensor([8.0, 0.0], device=device)
        dx = target[0] - state[:, 0]
        dy = target[1] - state[:, 1]
        dist_to_target = torch.sqrt(dx**2 + dy**2)

        target_heading = torch.atan2(dy, dx)
        heading_error = torch.atan2(
            torch.sin(target_heading - state[:, 2]),
            torch.cos(target_heading - state[:, 2])
        )

        backward_cost = torch.relu(-action[:, 0]) ** 2
        lateral_cost = 0.15 * action[:, 1] ** 2
        yaw_cost = 2.5 * action[:, 2] ** 2

        # ===== OBSTACLES (match scene.xml cylinders) =====
        obstacles = torch.tensor([
            [2.0, 0.9],
            [2.0, -0.9],
            [3.5, 0.0],
            [5.0, 0.8],
            [5.0, -0.8],
            [6.5, -0.2],
            [8.0, 0.6],
            [8.0, -0.6],
            [9.5, 0.0],
            [2.0, 0.0],
        ], device=device)

        dist_obs = torch.cdist(state[:, :2], obstacles)
        min_dist = torch.min(dist_obs, dim=1).values

        # ===== COLLISION ZONES =====
        d_collision = 0.24
        d_safe = 0.55

        obstacle_cost = torch.zeros_like(min_dist)

        # Hard collision penalty
        obstacle_cost += torch.where(
            min_dist < d_collision,
            2e5 * (d_collision - min_dist) ** 2,
            torch.zeros_like(min_dist)
        )

        # Soft avoidance cost
        obstacle_cost += torch.where(
            (min_dist >= d_collision) & (min_dist < d_safe),
            40.0 * (d_safe - min_dist) ** 2,
            torch.zeros_like(min_dist)
        )

        # ===== SPEED REWARD =====
        speed_reward = -3.0 * action[:, 0]

        return (
            1.0 * dist_to_target +
            0.6 * torch.abs(heading_error) +
            6.0 * backward_cost +
            lateral_cost +
            yaw_cost +
            obstacle_cost +
            speed_reward
        )
    
    @staticmethod
    def _generate_rect(center_x, center_y, w, h, density=5, device="cuda"):
        """Generate rectangular obstacle points"""
        x_range = torch.linspace(center_x - w/2, center_x + w/2, density, device=device)
        y_range = torch.linspace(center_y - h/2, center_y + h/2, density, device=device)
        grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing='xy')
        return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    
    def command(self, state_tensor):
        """
        Generate control command using MPPI
        
        Args:
            state_tensor: current state [x, y, yaw]
            
        Returns:
            control command [vx, vy, omega]
        """
        return self.mppi.command(state_tensor)

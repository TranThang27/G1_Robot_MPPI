"""
Test script for avoid_collision scenario
Navigate through cylinder obstacles to reach goal (6.0, -0.27)
"""
import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import sys
import os

sys.path.append(os.path.dirname(__file__))

# Import utilities and constants
from utils import (
    get_gravity_orientation, pd_control, get_next_path_point, 
    extract_robot_state
)
from constants import (
    GOAL_REACHED_THRESHOLD, VIEWER_SYNC_SKIP, SIMULATION_SPEED,
    AVOID_COLLISION_GOAL, WAYPOINT_THRESHOLD
)
from sim_utils import SimulationConfig, MuJoCoSimulator
from mppi_controller import G1MPPIController
from map_config import get_map_config, plan_global_path


def run_avoid_collision_test(config_file="g1_avoid.yaml"):
    """
    Main test function for avoid_collision scenario
    
    Args:
        config_file: Simulation config filename
    """
    # Load simulation config
    sim_config = SimulationConfig(config_file)
    
    # Initialize MuJoCo simulator
    simulator = MuJoCoSimulator(sim_config.xml_path, dt=sim_config.simulation_dt)
    
    # Load map and create path planner
    map_cfg = get_map_config("avoid_collision")
    obstacles_array = map_cfg.get_obstacles_array()
    print(f"✅ Loaded '{map_cfg.name}' map with {len(map_cfg.ox)} obstacle points")
    
    astar_planner = map_cfg.create_planner()
    print("✅ A* Pathfinder Initialized")
    
    # Initialize MPPI controller
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mppi_controller = G1MPPIController(
        device=device, 
        local_target=torch.tensor(AVOID_COLLISION_GOAL, device=device, dtype=torch.float32),
        obstacles=obstacles_array,
        global_path=np.array([])  # Will be updated after A* planning
    )
    print("✅ MPPI Controller Initialized")
    
    # Load neural network policy
    policy = torch.jit.load(sim_config.policy_path)
    
    # Initialize state variables
    action = np.zeros(sim_config.num_actions, dtype=np.float32)
    target_dof_pos = sim_config.default_angles.copy()
    obs = np.zeros(sim_config.num_obs, dtype=np.float32)
    cmd = sim_config.cmd_init.copy()
    
    global_path = []
    path_idx = 0
    counter = 0
    path_planned = False  # Flag: A* chỉ tìm 1 lần lúc bắt đầu
    prev_pos = np.array([0.0, 0.0])  # Track robot position
    
    # Launch viewer
    with mujoco.viewer.launch_passive(simulator.model, simulator.data) as viewer:
        # Setup camera
        cam = viewer.cam
        cam.azimuth = 0
        cam.elevation = -75
        cam.distance = 12.0
        cam.lookat[:] = [4.0, 0.0, 0.5]
        
        start_time = time.time()
        
        # Main simulation loop
        while viewer.is_running() and time.time() - start_time < sim_config.simulation_duration:
            step_start = time.time()
            
            # PD control for joint tracking
            tau = pd_control(
                target_dof_pos,
                simulator.data.qpos[7:],
                sim_config.kps,
                np.zeros_like(sim_config.kds),
                simulator.data.qvel[6:],
                sim_config.kds,
            )
            simulator.set_dof_torque(tau)
            simulator.step(tau)
            counter += 1
            
            # High-level planning and control
            if counter % sim_config.control_decimation == 0:
                # Extract robot state
                curr_x, curr_y, yaw = extract_robot_state(
                    simulator.data.qpos, 
                    simulator.data.qvel
                )
                
                # Check if robot is stuck (not moving)
                curr_pos = np.array([curr_x, curr_y])
                movement = np.linalg.norm(curr_pos - prev_pos)
                prev_pos = curr_pos.copy()
                
                # Plan path ONLY once at the beginning
                if not path_planned:
                    goal_x, goal_y = AVOID_COLLISION_GOAL
                    path = plan_global_path(astar_planner, curr_x, curr_y, goal_x, goal_y)
                    if path:
                        global_path = path
                        path_idx = 0
                        path_planned = True
                        # Update MPPI controller with A* global path
                        global_path_array = np.array(global_path)
                        mppi_controller.global_path = torch.tensor(
                            global_path_array, dtype=torch.float32, device=device
                        )
                        print(f"✅ Global Path (A*) Found! {len(path)} waypoints")
                        print(f"   Will follow this global path to reach goal")
                        print(f"   MPPI will handle local obstacle avoidance and stay on path")
                    else:
                        print(f"❌ A* pathfinding failed!")
                        path_planned = True
                
                # Execute control - follow A* path with MPPI local adjustment
                if not global_path:
                    cmd[:] = 0.0
                else:
                    goal_x, goal_y = AVOID_COLLISION_GOAL
                    dist_to_goal = np.hypot(goal_x - curr_x, goal_y - curr_y)
                    
                    if dist_to_goal < GOAL_REACHED_THRESHOLD:
                        cmd[:] = 0.0
                        print(f"✅ REACHED GOAL at ({curr_x:.2f}, {curr_y:.2f})")
                        global_path = []
                        path_idx = 0
                    else:
                        # Follow A* global path, MPPI handles local deviations
                        if path_idx < len(global_path):
                            wp_x, wp_y = global_path[path_idx]
                            dist_to_wp = np.hypot(wp_x - curr_x, wp_y - curr_y)
                            
                            # Advance to next waypoint if close enough
                            if dist_to_wp < WAYPOINT_THRESHOLD and path_idx < len(global_path) - 1:
                                path_idx += 1
                                wp_x, wp_y = global_path[path_idx]
                        else:
                            wp_x, wp_y = global_path[-1]
                        
                        # MPPI control toward current waypoint (local obstacle avoidance)
                        state_tensor = torch.tensor(
                            [[curr_x, curr_y, yaw]],
                            device=device,
                            dtype=torch.float32,
                        )
                        mppi_controller.local_target = torch.tensor(
                            [wp_x, wp_y], device=device, dtype=torch.float32
                        )
                        mppi_cmd = mppi_controller.command(state_tensor)
                        mppi_cmd = mppi_cmd.squeeze().cpu().numpy()
                        
                        # Clip to control limits
                        mppi_cmd = np.clip(mppi_cmd, [0.0, -0.35, -0.5], [1.0, 0.35, 0.5])
                        cmd[:] = mppi_cmd
                        
                        # Debug output - show A* path progress
                        if counter % int(2.0 / sim_config.simulation_dt) == 0:
                            print(f"[{counter*sim_config.simulation_dt:.1f}s] Pos: ({curr_x:.2f}, {curr_y:.2f}), "
                                  f"A*WP[{path_idx}]: ({wp_x:.2f}, {wp_y:.2f}), "
                                  f"Progress: {path_idx+1}/{len(global_path)}, Dist2Goal: {dist_to_goal:.2f}m")
                
                # Build observation
                qj = (simulator.data.qpos[7:] - sim_config.default_angles) * sim_config.dof_pos_scale
                dqj = simulator.data.qvel[6:] * sim_config.dof_vel_scale
                gravity_orientation = get_gravity_orientation(simulator.data.qpos[3:7])
                omega = simulator.data.qvel[3:6] * sim_config.ang_vel_scale
                
                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * sim_config.cmd_scale
                obs[9:9+sim_config.num_actions] = qj
                obs[9+sim_config.num_actions:9+2*sim_config.num_actions] = dqj
                obs[9+2*sim_config.num_actions:9+3*sim_config.num_actions] = action
                
                # Get policy action
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().cpu().numpy().squeeze()
                target_dof_pos = action * sim_config.action_scale + sim_config.default_angles
            
            # Update viewer
            if counter % VIEWER_SYNC_SKIP == 0:
                viewer.sync()
            
            # Frame rate control
            target_dt = simulator.model.opt.timestep / SIMULATION_SPEED
            process_time = time.time() - step_start
            if target_dt > process_time:
                time.sleep(target_dt - process_time)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, default="g1_avoid.yaml")
    args = parser.parse_args()
    
    run_avoid_collision_test(args.config_file)
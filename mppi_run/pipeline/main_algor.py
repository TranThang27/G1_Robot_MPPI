"""
Main Algorithm: Unified Pipeline for Robot Control
Combines A* pathfinding and MPPI control into a single execution pipeline.
Scenario scripts only differ in environment initialization and scenario goals.
"""

import time
import numpy as np
import torch
import mujoco.viewer
import sys
import os

# Add parent directory to path to import from root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import core algorithms
from core import AStarPlanner, G1MPPIController

# Import utilities
from utils import (
    utils,
    constants,
    setup_camera,
    map_config
)

# Extract specific functions
get_gravity_orientation = utils.get_gravity_orientation
pd_control = utils.pd_control
extract_robot_state = utils.extract_robot_state
plan_global_path = map_config.plan_global_path


def run_simulation_pipeline(
    simulator,
    sim_config,
    policy,
    astar_planner,
    mppi_controller,
    obstacles_array,
    target_getter_fn,
    viewer_config,
    device="cuda"
):
    """
    Main simulation pipeline executed by both scene1 and scene2.
    This contains the complete control loop logic.
    
    Args:
        simulator: MuJoCo simulator instance
        sim_config: Simulation configuration
        policy: Neural network policy (torch model)
        astar_planner: A* pathfinder
        mppi_controller: MPPI controller
        obstacles_array: Obstacle positions
        target_getter_fn: Function that returns current goal given robot position and state.
                         Should return (goal_x, goal_y) or None if scenario complete
        viewer_config: Dict with camera settings {azimuth, elevation, distance, lookat}
        device: torch device
    """
    # Initialize state variables
    action = np.zeros(sim_config.num_actions, dtype=np.float32)
    target_dof_pos = sim_config.default_angles.copy()
    obs = np.zeros(sim_config.num_obs, dtype=np.float32)
    cmd = sim_config.cmd_init.copy()
    
    global_path = []
    path_idx = 0
    counter = 0
    path_planned = False
    
    # Launch viewer
    with mujoco.viewer.launch_passive(simulator.model, simulator.data) as viewer:
        # Setup camera from viewer_config
        setup_camera(viewer, viewer_config)
        
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
                
                # Get current goal from scenario-specific function
                goal = target_getter_fn(curr_x, curr_y)
                
                if goal is None:
                    # Scenario complete
                    cmd[:] = 0.0
                else:
                    goal_x, goal_y = goal
                    
                    # Plan path ONLY once for current goal
                    if not path_planned:
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
                            print(f" Global Path (A*) Found! {len(path)} waypoints")
                            print(f"   MPPI will handle local obstacle avoidance")
                        else:
                            print(f"A* pathfinding failed!")
                            path_planned = True
                    
                    # Execute control
                    if not global_path:
                        cmd[:] = 0.0
                    else:
                        dist_to_goal = np.hypot(goal_x - curr_x, goal_y - curr_y)
                        
                        if dist_to_goal < constants.GOAL_REACHED_THRESHOLD:
                            cmd[:] = 0.0
                            print(f"REACHED GOAL at ({curr_x:.2f}, {curr_y:.2f})")
                            global_path = []
                            path_idx = 0
                            path_planned = False
                        else:
                            # Follow A* global path, MPPI handles local deviations
                            if path_idx < len(global_path):
                                wp_x, wp_y = global_path[path_idx]
                                dist_to_wp = np.hypot(wp_x - curr_x, wp_y - curr_y)
                                
                                # Advance to next waypoint if close enough
                                if dist_to_wp < constants.WAYPOINT_THRESHOLD and path_idx < len(global_path) - 1:
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
                            
                            # Debug output
                            if counter % int(2.0 / sim_config.simulation_dt) == 0:
                                print(f"[{counter*sim_config.simulation_dt:.1f}s] Pos: ({curr_x:.2f}, {curr_y:.2f}), "
                                      f"A*WP[{path_idx}]: ({wp_x:.2f}, {wp_y:.2f}), "
                                      f"Progress: {path_idx+1}/{len(global_path)}, Dist2Goal: {dist_to_goal:.2f}m")
                
                # Build observation for neural network policy
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
            if counter % constants.VIEWER_SYNC_SKIP == 0:
                viewer.sync()
            
            # Frame rate control
            target_dt = simulator.model.opt.timestep / constants.SIMULATION_SPEED
            process_time = time.time() - step_start
            if target_dt > process_time:
                time.sleep(target_dt - process_time)

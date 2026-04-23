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
    map_config,
    BEVVisualizer
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
    device="cuda",
    show_bev=True
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
    
    # Dynamic obstacle management (chair 11 and chair 9)
    chair_11_enabled = False
    chair_11_pos = [10.6, 0.02]
    
    chair_9_enabled = False
    chair_9_pos = [4.6, 0.02]
    resume_time = 0.0

    
    # Initialize BEV visualizer
    bev_viz = BEVVisualizer(map_config.AVOID_COLLISION_MAP) if show_bev else None
    
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
                
                elapsed_time = time.time() - start_time
                
                # Check if 3 seconds elapsed and enable chair 9
                if elapsed_time >= 3.0 and not chair_9_enabled:
                    print(f"\n{'='*60}")
                    print(f"[{elapsed_time:.2f}s] CHAIR 9 SUDDENLY APPEARED AT ({chair_9_pos[0]}, {chair_9_pos[1]})")
                    print(f"ROBOT POSITION: ({curr_x:.2f}, {curr_y:.2f})")
                    print(f"Enabling chair 9 in obstacle map, recalculating path, and PAUSING FOR 1.5s...")
                    print(f"{'='*60}\n")
                    
                    # Add chair 9 to obstacle map
                    map_config.AVOID_COLLISION_MAP.cylinder_centers.append(chair_9_pos)
                    map_config.AVOID_COLLISION_MAP.ox, map_config.AVOID_COLLISION_MAP.oy = map_config.AVOID_COLLISION_MAP._build_grid()
                    
                    # Update AStar Planner instance
                    astar_planner = map_config.AVOID_COLLISION_MAP.create_planner()
                    
                    # Update MPPI local obstacles array
                    obstacles_array = map_config.AVOID_COLLISION_MAP.get_obstacles_array()
                    mppi_controller.obstacles = torch.tensor(obstacles_array, dtype=torch.float32, device=device)
                    
                    # Force path replanning
                    path_planned = False
                    global_path = []
                    path_idx = 0
                    cmd[:] = 0.0  # STOP ROBOT
                    
                    # Move chair 9 in MuJoCo scene using mocap
                    body_id_9 = mujoco.mj_name2id(simulator.model, mujoco.mjtObj.mjOBJ_BODY, "chair_body_9")
                    mocap_idx_9 = simulator.model.body_mocapid[body_id_9]
                    if mocap_idx_9 >= 0:
                        simulator.data.mocap_pos[mocap_idx_9] = [4.6, 0.02, 0.0]
                    
                    chair_9_enabled = True
                    resume_time = elapsed_time + 1.5  # Pause for 1.5 seconds
                
                # Check if 8 seconds elapsed and enable chair 11

                if elapsed_time >= 8.0 and not chair_11_enabled:
                    print(f"\n{'='*60}")
                    print(f"[{elapsed_time:.2f}s] CHAIR 11 SUDDENLY APPEARED AT ({chair_11_pos[0]}, {chair_11_pos[1]})")
                    print(f"ROBOT POSITION: ({curr_x:.2f}, {curr_y:.2f})")
                    print(f"Enabling chair 11 in obstacle map and recalculating path...")
                    print(f"{'='*60}\n")
                    
                    # Add chair 11 to obstacle map
                    map_config.AVOID_COLLISION_MAP.cylinder_centers.append(chair_11_pos)
                    map_config.AVOID_COLLISION_MAP.ox, map_config.AVOID_COLLISION_MAP.oy = map_config.AVOID_COLLISION_MAP._build_grid()
                    
                    # Update AStar Planner instance
                    astar_planner = map_config.AVOID_COLLISION_MAP.create_planner()
                    
                    # Update MPPI local obstacles array
                    obstacles_array = map_config.AVOID_COLLISION_MAP.get_obstacles_array()
                    mppi_controller.obstacles = torch.tensor(obstacles_array, dtype=torch.float32, device=device)
                    
                    # Force path replanning
                    path_planned = False
                    global_path = []
                    path_idx = 0
                    cmd[:] = 0.0  # STOP ROBOT - it must recalculate
                    
                    # Move chair 11 in MuJoCo scene using mocap
                    body_id = mujoco.mj_name2id(simulator.model, mujoco.mjtObj.mjOBJ_BODY, "chair_body_11")
                    mocap_idx = simulator.model.body_mocapid[body_id]
                    if mocap_idx >= 0:
                        simulator.data.mocap_pos[mocap_idx] = [10.6, 0.02, 0.0]
                    
                    chair_11_enabled = True
                
                # If in pause interval, force stop and skip execution (except A* replanning)
                if elapsed_time < resume_time:
                    cmd[:] = 0.0
                
                # Get current goal from scenario-specific function
                goal = target_getter_fn(curr_x, curr_y)
                
                if goal is None:
                    # Scenario complete
                    cmd[:] = 0.0
                    resume_time = 0.0  # Allow stopping cleanly
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
                    
                    # Execute control ONLY if we are past the pause time
                    if elapsed_time < resume_time:
                        cmd[:] = 0.0
                        if counter % int(2.0 / sim_config.simulation_dt) == 0:
                            print(f"[{elapsed_time:.1f}s] PAUSED. Resuming in {resume_time - elapsed_time:.1f}s...")
                    elif not global_path:
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
                
                # Update BEV visualization
                if bev_viz is not None and counter % int(0.1 / sim_config.simulation_dt) == 0:
                    curr_x, curr_y, yaw = extract_robot_state(simulator.data.qpos, simulator.data.qvel)
                    goal = target_getter_fn(curr_x, curr_y)
                    goal_pos = goal if goal is not None else None
                    bev_viz.update(curr_x, curr_y, yaw, 
                                  goal_x=goal_pos[0] if goal_pos else None,
                                  goal_y=goal_pos[1] if goal_pos else None)
            
            # Frame rate control
            target_dt = simulator.model.opt.timestep / constants.SIMULATION_SPEED
            process_time = time.time() - step_start
            if target_dt > process_time:
                time.sleep(target_dt - process_time)
    
    # Cleanup
    if bev_viz is not None:
        bev_viz.save_trajectory('/tmp/robot_trajectory_bev.png')
        bev_viz.show()
        print("BEV visualization saved. Close figure window to continue.")

#!/usr/bin/env python3
"""
Scene 3: Dynamic Moving Obstacles Test
Robot navigates to a goal while avoiding moving cylinder obstacles.
"""

import sys
import os
import argparse
import numpy as np

# Add parent directory to import mppi_run modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils import setup_simulation, constants
from config.camera_config import CAMERA_CONFIGS
from pipeline.main_algor import run_simulation_pipeline


def run_moving_obstacles_test(config_file):
    """
    Scene 3: Robot navigates with moving obstacle avoidance.
    Goal: Navigate from (0, 0) to (8.5, 0) while avoiding moving obstacles.
    """
    
    # Initialize simulation with scene_3 config
    sim_dict = setup_simulation(config_file, "moving_obstacles")
    
    # Goal for this scenario
    goal = np.array([8.5, 0.0])
    goal_reached = False
    
    def get_current_goal():
        """Simple: single goal until reached"""
        nonlocal goal_reached
        if goal_reached:
            return None
        return goal
    
    def on_goal_reached():
        """Called when robot reaches the goal"""
        nonlocal goal_reached
        # print("\nGoal Reached!")
        goal_reached = True
    
    # Update moving obstacles dynamically during simulation
    def update_obstacles_callback(step):
        """
        Update obstacle positions to simulate movement.
        Called each control step.
        
        Simple pattern: obstacles oscillate in Y direction
        """
        if hasattr(update_obstacles_callback, 'step_count'):
            update_obstacles_callback.step_count += 1
        else:
            update_obstacles_callback.step_count = 0
        
        # Oscillation frequency (period in steps)
        period = 100  # ~4 seconds at 25 Hz control
        amplitude = 0.3  # ±0.3m oscillation
        
        # Get simulation and data
        simulator = sim_dict['simulator']
        data = simulator.data
        
        # Update each obstacle position (obs_1 to obs_9)
        for i in range(1, 10):
            body_name = f"obs_{i}"
            try:
                body_id = simulator.model.body(body_name).id
                # Oscillate in Y direction
                y_offset = amplitude * np.sin(2 * np.pi * update_obstacles_callback.step_count / period)
                data.xpos[body_id, 1] = simulator.initial_obs_y[i-1] + y_offset
            except:
                pass  # Body not found, skip
    
    # Store initial obstacle positions for oscillation
    simulator = sim_dict['simulator']
    simulator.initial_obs_y = []
    for i in range(1, 10):
        body_name = f"obs_{i}"
        try:
            body_id = simulator.model.body(body_name).id
            simulator.initial_obs_y.append(simulator.data.xpos[body_id, 1])
        except:
            simulator.initial_obs_y.append(0.0)
    
    # Run the main control pipeline
    run_simulation_pipeline(
        simulator=sim_dict['simulator'],
        sim_config=sim_dict['config'],
        policy=sim_dict['policy'],
        astar_planner=sim_dict['planner'],
        mppi_controller=sim_dict['controller'],
        obstacles_array=sim_dict['obstacles'],
        target_getter_fn=get_current_goal,
        viewer_config=sim_dict['viewer_config'],
        device=sim_dict['device'],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scene 3: Moving Obstacles Navigation Test")
    parser.add_argument(
        "config_file",
        type=str,
        default="scene_3.yaml",
        nargs="?",
        help="Path to scene config YAML file (default: scene_3.yaml)"
    )
    args = parser.parse_args()
    
    # print(f"Starting Scene 3: Moving Obstacles Test")
    # print(f"   Config: {args.config_file}")
    
    run_moving_obstacles_test(args.config_file)

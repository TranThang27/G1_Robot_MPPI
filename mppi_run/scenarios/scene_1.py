"""
Scenario: Obstacle Avoidance
Navigate through obstacles to reach a single goal.
Uses the unified simulation pipeline from main_algor.
"""
import numpy as np
import torch
import sys
import os
import argparse


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils import setup_simulation, constants
from config.camera_config import CAMERA_CONFIGS
from pipeline.main_algor import run_simulation_pipeline


def run_avoid_collision_test(config_file="scene_1.yaml"):
    """
    Main test function for obstacle avoidance scenario.
    Single goal navigation with obstacle avoidance.
    
    Args:
        config_file: Simulation config filename
    """
    # SIMULATION SETUP 
    setup_data = setup_simulation(config_file, "avoid_collision")
    
    simulator = setup_data['simulator']
    sim_config = setup_data['sim_config']
    policy = setup_data['policy']
    astar_planner = setup_data['astar_planner']
    mppi_controller = setup_data['mppi_controller']
    obstacles_array = setup_data['obstacles_array']
    device = setup_data['device']
    
    # Set initial target for MPPI
    mppi_controller.local_target = torch.tensor(
        constants.AVOID_COLLISION_GOAL, device=device, dtype=torch.float32
    )
    
   
    
    # Define target getter for single goal scenario
    goal_reached = False
    def get_current_goal(curr_x, curr_y):
        """Return current goal position. None if scenario complete."""
        nonlocal goal_reached
        
        if goal_reached:
            return None
        
        goal_x, goal_y = constants.AVOID_COLLISION_GOAL
        dist_to_goal = np.hypot(goal_x - curr_x, goal_y - curr_y)
        
        if dist_to_goal < 0.3:  # Goal reached threshold
            goal_reached = True
            print(f"SCENARIO COMPLETE: Reached obstacle avoidance goal!")
            return None
        
        return (goal_x, goal_y)
    
   
    run_simulation_pipeline(
        simulator=simulator,
        sim_config=sim_config,
        policy=policy,
        astar_planner=astar_planner,
        mppi_controller=mppi_controller,
        obstacles_array=obstacles_array,
        target_getter_fn=get_current_goal,
        viewer_config=CAMERA_CONFIGS['avoid_collision'],
        device=device
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run G1 Robot obstacle avoidance scenario"
    )
    parser.add_argument(
        "config_file", 
        nargs="?",
        default="scene_1.yaml",
        help="Config file name (e.g., scene_1.yaml, g1.yaml)"
    )
    args = parser.parse_args()
    
    # Pass just the filename - setup_simulation will handle the path
    run_avoid_collision_test(args.config_file)

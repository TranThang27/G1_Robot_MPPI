"""
Scenario: Room Navigation
Navigate between multiple safe points in a room.
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


def run_room_test(config_file="scene_2.yaml"):
    """
    Main test function for room navigation scenario.
    Multi-target navigation visiting safe points.
    
    Args:
        config_file: Simulation config filename
    """
    # SIMULATION SETUP
    setup_data = setup_simulation(config_file, "room_scene")
    
    simulator = setup_data['simulator']
    sim_config = setup_data['sim_config']
    policy = setup_data['policy']
    astar_planner = setup_data['astar_planner']
    mppi_controller = setup_data['mppi_controller']
    obstacles_array = setup_data['obstacles_array']
    device = setup_data['device']
    
    # SCENARIO-SPECIFIC CONFIGURATION 
    
    # Multi-target scenario state
    current_target_idx = 0
    visited_targets = set()
    
    def get_current_goal(curr_x, curr_y):
        """Return current goal position. Cycles through safe points. None if all visited."""
        nonlocal current_target_idx, visited_targets
        
        if len(visited_targets) >= len(constants.ROOM_SAFE_POINTS):
            print(f"SCENARIO COMPLETE")
            return None
        
        # Get current target
        target_x, target_y = constants.ROOM_SAFE_POINTS[current_target_idx]
        dist_to_target = np.hypot(target_x - curr_x, target_y - curr_y)
        
        # Check if reached current target
        if dist_to_target < 0.3:  # Target reached threshold
            visited_targets.add(current_target_idx)
            print(f"REACHED TARGET {current_target_idx}: ({target_x:.2f}, {target_y:.2f})")
            
            # Move to next unvisited target
            current_target_idx = (current_target_idx + 1) % len(constants.ROOM_SAFE_POINTS)
            
            if len(visited_targets) >= len(constants.ROOM_SAFE_POINTS):
                return None
            
            target_x, target_y = constants.ROOM_SAFE_POINTS[current_target_idx]
            print(f" Moving to next target {current_target_idx}: ({target_x:.2f}, {target_y:.2f})")
            return (target_x, target_y)
        
        return (target_x, target_y)
    
  
    run_simulation_pipeline(
        simulator=simulator,
        sim_config=sim_config,
        policy=policy,
        astar_planner=astar_planner,
        mppi_controller=mppi_controller,
        obstacles_array=obstacles_array,
        target_getter_fn=get_current_goal,
        viewer_config=CAMERA_CONFIGS['room_scene'],
        device=device
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run G1 Robot room navigation scenario"
    )
    parser.add_argument(
        "config_file",
        nargs="?",
        default="scene_2.yaml",
        help="Config file name (e.g., scene_2.yaml, g1.yaml)"
    )
    args = parser.parse_args()
    
    # Pass just the filename - setup_simulation will handle the path
    run_room_test(args.config_file)

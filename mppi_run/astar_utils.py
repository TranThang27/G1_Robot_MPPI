"""
A* Pathfinding utilities for different scenarios
"""

import numpy as np
from A_star import AStarPlanner


def setup_astar_avoid_collision():
    """Setup A* pathfinding for avoid collision scenario (scene.xml with cylinders)"""
    ox, oy = [], []
    # Walls
    for x in np.arange(-0.5, 10.5, 0.2):
        ox.append(x); oy.append(4.2)
        ox.append(x); oy.append(-4.2)
    for y in np.arange(-4.2, 4.2, 0.2):
        ox.append(-0.5); oy.append(y)
        ox.append(10.5); oy.append(y)
    # Obstacles (match scene.xml cylinders)
    for x in np.arange(1.7, 2.3, 0.15):
        for y in np.arange(0.6, 1.2, 0.15):
            ox.append(x); oy.append(y)
    for x in np.arange(1.7, 2.3, 0.15):
        for y in np.arange(-1.2, -0.6, 0.15):
            ox.append(x); oy.append(y)
    for x in np.arange(3.2, 3.8, 0.15):
        for y in np.arange(-0.3, 0.3, 0.15):
            ox.append(x); oy.append(y)
    for x in np.arange(4.7, 5.3, 0.15):
        for y in np.arange(0.5, 1.1, 0.15):
            ox.append(x); oy.append(y)
    for x in np.arange(4.7, 5.3, 0.15):
        for y in np.arange(-1.1, -0.5, 0.15):
            ox.append(x); oy.append(y)
    for x in np.arange(6.2, 6.8, 0.15):
        for y in np.arange(-0.5, 0.1, 0.15):
            ox.append(x); oy.append(y)
    for x in np.arange(7.7, 8.3, 0.15):
        for y in np.arange(0.3, 0.9, 0.15):
            ox.append(x); oy.append(y)
    for x in np.arange(7.7, 8.3, 0.15):
        for y in np.arange(-0.9, -0.3, 0.15):
            ox.append(x); oy.append(y)
    for x in np.arange(9.2, 9.8, 0.15):
        for y in np.arange(-0.3, 0.3, 0.15):
            ox.append(x); oy.append(y)
    for x in np.arange(1.7, 2.3, 0.15):
        for y in np.arange(-0.3, 0.3, 0.15):
            ox.append(x); oy.append(y)
    
    return AStarPlanner(ox, oy, resolution=0.3, rr=0.5)


def setup_astar_room():
    """Setup A* pathfinding for room scenario with furniture"""
    ox, oy = [], []
    # Walls
    for x in np.arange(-2.0, 9.0, 0.2):
        ox.append(x); oy.append(4.2)
        ox.append(x); oy.append(-4.2)
    for y in np.arange(-4.2, 4.2, 0.2):
        ox.append(-1.5); oy.append(y)
        ox.append(8.5); oy.append(y)
    # Column (4.0, 0.0)
    for x in np.arange(3.5, 4.6, 0.15):
        for y in np.arange(-0.6, 0.6, 0.15):
            ox.append(x); oy.append(y)
    # Table (2.0, 3.0)
    for x in np.arange(1.5, 2.6, 0.15):
        for y in np.arange(2.4, 3.6, 0.15):
            ox.append(x); oy.append(y)
    return AStarPlanner(ox, oy, resolution=0.25, rr=0.6)


def plan_global_path(astar, start_x, start_y, goal_x, goal_y):
    """Plan path using A* algorithm"""
    try:
        rx, ry = astar.planning(start_x, start_y, goal_x, goal_y)
        if rx is None or len(rx) == 0:
            return None
        return list(zip(list(reversed(rx)), list(reversed(ry))))
    except:
        return None

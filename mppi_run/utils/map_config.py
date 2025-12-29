"""
Unified map configuration with hard-coded rules for each scene
Combines scene_maps.py and astar_utils.py into one clean interface
"""

import numpy as np
from core import AStarPlanner


class MapConfig:
    """Base class for map configurations"""
    def __init__(self, name, cylinder_centers, grid_bounds, cylinder_radius, 
                 astar_resolution, astar_rr):
        self.name = name
        self.cylinder_centers = cylinder_centers
        self.grid_bounds = grid_bounds  # (min_x, max_x, min_y, max_y)
        self.cylinder_radius = cylinder_radius
        self.astar_resolution = astar_resolution
        self.astar_rr = astar_rr
        
        # Build grid
        self.ox, self.oy = self._build_grid()
        self.planner = None
    
    def _build_grid(self):
        """Build obstacle grid from cylinder centers"""
        ox, oy = [], []
        grid_resolution = 0.1  # 10cm grid
        min_x, max_x, min_y, max_y = self.grid_bounds
        
        # Sample cylinder obstacles
        for x in np.arange(min_x, max_x + grid_resolution, grid_resolution):
            for y in np.arange(min_y, max_y + grid_resolution, grid_resolution):
                is_obstacle = False
                for cx, cy in self.cylinder_centers:
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    if dist <= self.cylinder_radius:
                        is_obstacle = True
                        break
                
                if is_obstacle:
                    ox.append(x)
                    oy.append(y)
        
        # Add boundary walls
        for x in np.arange(min_x, max_x + grid_resolution, grid_resolution):
            ox.extend([x, x])
            oy.extend([min_y, max_y])
        
        for y in np.arange(min_y, max_y + grid_resolution, grid_resolution):
            ox.extend([min_x, max_x])
            oy.extend([y, y])
        
        return ox, oy
    
    def create_planner(self):
        """Create A* planner with map-specific settings"""
        self.planner = AStarPlanner(
            self.ox, self.oy, 
            resolution=self.astar_resolution, 
            rr=self.astar_rr
        )
        return self.planner
    
    def get_obstacles_array(self):
        """Get obstacles as numpy array for MPPI"""
        return np.column_stack([self.ox, self.oy]).astype(np.float32)
    
    def plan_path(self, sx, sy, gx, gy, smooth=True):
        """Plan path using A*"""
        if self.planner is None:
            self.create_planner()
        return self.planner.planning(sx, sy, gx, gy, smooth=smooth)


# ============================================
# SCENE 1: avoid_collision - Cylinder course
# ============================================
AVOID_COLLISION_MAP = MapConfig(
    name="avoid_collision",
    cylinder_centers=[
        [2.0, 0.9],     # cyl_1
        [2.0, -0.9],    # cyl_2
        [3.5, 0.0],     # cyl_3
        [5.0, 0.8],     # cyl_4
        [5.0, -0.8],    # cyl_5
        [6.5, -0.2],    # cyl_6
        [8.0, 0.6],     # cyl_7
        [8.0, -0.6],    # cyl_8
        [9.5, 0.0],     # cyl_9
        [2.0, 0.0],     # cyl_10
    ],
    grid_bounds=(-1.0, 10.0, -2.0, 2.0),
    cylinder_radius=0.12,
    astar_resolution=0.1,
    astar_rr=0.10  # Tight fit for narrow passages
)


# ============================================
# SCENE 2: room_scene - Office with furniture
# ============================================
ROOM_MAP = MapConfig(
    name="room_scene",
    cylinder_centers=[
        # Center table (3.5 to 4.5 x in range, -0.7 to 0.7 y range) - legs + surface
        [3.5, -0.7],    # center_table_leg1
        [4.5, -0.7],    # center_table_leg2
        [3.5, 0.7],     # center_table_leg3
        [4.5, 0.7],     # center_table_leg4
        [4.0, 0.0],     # center table surface
        
        # Table 1 legs (1.6-2.4 x, 2.1-3.9 y) - legs + surface
        [1.6, 2.1],     # table_leg1
        [2.4, 2.1],     # table_leg2
        [1.6, 3.9],     # table_leg3
        [2.4, 3.9],     # table_leg4
        [2.0, 3.0],     # table 1 surface center
        
        # Table 2 legs (1.6-2.4 x, -3.9 to -2.1 y) - legs + surface
        [1.6, -3.9],    # table2_leg1
        [2.4, -3.9],    # table2_leg2
        [1.6, -2.1],    # table2_leg3
        [2.4, -2.1],    # table2_leg4
        [2.0, -3.0],    # table 2 surface center
    ],
    grid_bounds=(-2.0, 10.0, -5.0, 6.0),  # INCREASED from (-2.0, 9.0, -5.0, 5.0) - wider room
    cylinder_radius=0.2,  # INCREASED further to cover table surfaces + robot size
    astar_resolution=0.1,
    astar_rr=0.7  # Even wider clearance for furniture
)


# Map registry
_MAP_REGISTRY = {
    "avoid_collision": AVOID_COLLISION_MAP,
    "room_scene": ROOM_MAP,
}


def get_map_config(scene_name):
    """Get map configuration for a scene"""
    if scene_name not in _MAP_REGISTRY:
        raise ValueError(f"Unknown scene: {scene_name}. Available: {list(_MAP_REGISTRY.keys())}")
    return _MAP_REGISTRY[scene_name]


def plan_global_path(astar, start_x, start_y, goal_x, goal_y):
    """Plan path from start to goal using A*"""
    try:
        rx, ry = astar.planning(start_x, start_y, goal_x, goal_y, smooth=True)
        if rx is None or len(rx) == 0:
            return None
        path = list(zip(rx, ry))
        return path
    except Exception as e:
        print(f"Planning Error: {e}")
        return None

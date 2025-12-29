import math
import numpy as np
from .path_smoother import smooth_path_spline
from scipy.spatial import KDTree

class AStarPlanner:
    def __init__(self, ox, oy, resolution, rr =0.3):
        self.resolution = resolution # grid resolution
        self.rr = rr # robot radius

        #Find min and max grid coordinates (Map input)
        self.min_x, self.min_y = round(min(ox)), round(min(oy))
        self.max_x, self.max_y = round(max(ox)), round(max(oy))

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy) # create obstacle map

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  
            self.y = y 
            self.cost = cost
            self.parent_index = parent_index

    def planning(self, sx, sy, gx, gy, smooth=True):
        """    
        Inputs:
            sx, sy: Start position
            gx, gy: Goal position
            smooth: Apply spline smoothing
        
        Outputs:
            rx, ry: Path waypoints
        """
        goal_ix = self.calc_xy_index(gx, self.min_x)
        goal_iy = self.calc_xy_index(gy, self.min_y)

        if not self.verify_node(self.Node(goal_ix, goal_iy, 0, -1)):
            print("❌ Error: Goal point is inside an obstacle or out of bounds!")
            return [], []
        # Initial node
        nstart = self.Node(self.calc_xy_index(sx, self.min_x),
                           self.calc_xy_index(sy, self.min_y), 0.0, -1)
        ngoal = self.Node(self.calc_xy_index(gx, self.min_x),
                          self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict() 
        open_set[self.calc_grid_index(nstart)] = nstart

        while len(open_set) > 0:
            # Get node with lowest f = g + h
            c_id = min(open_set, key=lambda o: open_set[o].cost + self.calc_heuristic(ngoal, open_set[o]))
            current = open_set[c_id]

            # Check goal
            if current.x == ngoal.x and current.y == ngoal.y:
                ngoal.parent_index = current.parent_index
                ngoal.cost = current.cost
                break

            del open_set[c_id] #remove from open set
            closed_set[c_id] = current #add to closed set

            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # Skip if not valid in map
                if not self.verify_node(node) or n_id in closed_set:
                    continue
                
                # Update open set
                if n_id not in open_set or open_set[n_id].cost > node.cost:
                    open_set[n_id] = node

        rx, ry = self.calc_final_path(ngoal, closed_set)
        
        # Debug: print raw path length and coordinates
        print(f"  [A*] Raw path: {len(rx)} waypoints")
        print(f"  [A*] Path coordinates (start to goal):")
        for i, (x, y) in enumerate(zip(rx, ry)):
            if i % max(1, len(rx)//10) == 0 or i == len(rx)-1:  # Print every 10% + last point
                print(f"       [{i:3d}] ({x:6.2f}, {y:6.2f})")
        print(f"       Start: ({rx[0]:.2f}, {ry[0]:.2f}) → Goal: ({rx[-1]:.2f}, {ry[-1]:.2f})")
        
        # Smoothing with appropriate resolution (0.1m = 10cm waypoint spacing)
        if smooth and len(rx) > 2:
            rx, ry = smooth_path_spline(rx, ry, smoothing_factor=0.3, resolution=0.1)
            print(f"  [A*] After smoothing: {len(rx)} waypoints")
            print(f"  [A*] Smoothed path sample (every ~{max(1, len(rx)//15)} waypoints):")
            for i, (x, y) in enumerate(zip(rx, ry)):
                if i % max(1, len(rx)//15) == 0 or i == len(rx)-1:
                    print(f"       [{i:4d}] ({x:6.2f}, {y:6.2f})")
        elif smooth and len(rx) <= 2:
            # Even for short paths, add intermediate points
            rx, ry = smooth_path_spline(rx, ry, smoothing_factor=0.1, resolution=0.1)
            print(f"  [A*] After smoothing (short path): {len(rx)} waypoints")
        
        return rx, ry

    # Calculate final path
    def calc_final_path(self, goal_node, closed_set):
        rx = [self.calc_grid_position(goal_node.x, self.min_x)]
        ry = [self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        
        count = 0
        while parent_index != -1 and count < 1000:  # Add safety limit
            if parent_index not in closed_set:
                break
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index
            count += 1
        
        return rx[::-1], ry[::-1] # Interval reverse
    
    #Eclidean distance
    def calc_heuristic(self, n1, n2):
        return math.hypot(n1.x - n2.x, n1.y - n2.y)

    def calc_grid_position(self, index, min_pos):
        return index * self.resolution + min_pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)
        if px < self.min_x or py < self.min_y or px >= self.max_x or py >= self.max_y:
            return False
        if self.obstacle_map[node.x][node.y]:
            return False
        return True

    # Create obstacle map
    def calc_obstacle_map(self, ox, oy):
            self.obstacle_map = [[False for _ in range(self.y_width)] for _ in range(self.x_width)]
            
            # Nếu không có vật cản thì return luôn
            if not ox:
                return

            # Tạo cây tìm kiếm không gian (nhanh hơn rất nhiều so với vòng lặp for lồng nhau)
            self.tree = KDTree(list(zip(ox, oy)))
            
            for ix in range(self.x_width):
                x = self.calc_grid_position(ix, self.min_x)
                for iy in range(self.y_width):
                    y = self.calc_grid_position(iy, self.min_y)
                    
                    # Tìm điểm vật cản gần nhất với (x, y)
                    dist, _ = self.tree.query([x, y], k=1)
                    
                    # Nếu khoảng cách < bán kính robot -> Đánh dấu là vật cản
                    if dist <= self.rr:
                        self.obstacle_map[ix][iy] = True

# Motion model run 8 directions 
    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        return [[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1],
                [-1, -1, math.sqrt(2)], [-1, 1, math.sqrt(2)],
                [1, -1, math.sqrt(2)], [1, 1, math.sqrt(2)]]

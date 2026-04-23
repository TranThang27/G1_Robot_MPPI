"""
Bird's Eye View (BEV) Visualizer for obstacle avoidance scenarios.
Displays obstacles, robot position/heading, and trajectory in real-time.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow
from collections import deque


class BEVVisualizer:
    """Real-time Bird's Eye View visualization for robot navigation"""
    
    def __init__(self, map_config, max_trajectory_length=1000):
        """
        Initialize BEV visualizer.
        
        Args:
            map_config: MapConfig object containing obstacle positions
            max_trajectory_length: Maximum number of trajectory points to store
        """
        self.map_config = map_config
        self.max_trajectory_length = max_trajectory_length
        self.trajectory = deque(maxlen=max_trajectory_length)
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.suptitle("Robot BEV Navigation")
        
        # Setup plot
        self._setup_plot()
        
        # Matplotlib interactive mode
        plt.ion()
    
    def _setup_plot(self):
        """Setup the plot structure"""
        min_x, max_x, min_y, max_y = self.map_config.grid_bounds
        
        # Set axis limits with padding
        padding = 1.0
        self.ax.set_xlim(min_x - padding, max_x + padding)
        self.ax.set_ylim(min_y - padding, max_y + padding)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')

    
    def update(self, robot_x, robot_y, robot_yaw, goal_x=None, goal_y=None):
        """
        Update visualization with current robot state.
        
        Args:
            robot_x: Robot X position
            robot_y: Robot Y position
            robot_yaw: Robot heading angle (radians)
            goal_x: Goal X position (optional)
            goal_y: Goal Y position (optional)
        """
        # Add to trajectory
        self.trajectory.append((robot_x, robot_y))
        
        # Clear ALL patches and lines
        for patch in list(self.ax.patches):
            patch.remove()
        for line in list(self.ax.lines):
            line.remove()
        
        # Draw obstacles (dynamically reflects newly added ones)
        for i, (cx, cy) in enumerate(self.map_config.cylinder_centers):
            circle = Circle((cx, cy), self.map_config.cylinder_radius, 
                          color='red', alpha=0.6, label='Obstacles' if i == 0 else '')
            self.ax.add_patch(circle)
        
        # Draw trajectory
        if len(self.trajectory) > 1:
            traj_array = np.array(list(self.trajectory))
            self.ax.plot(traj_array[:, 0], traj_array[:, 1], 'b-', linewidth=1.5, 
                        label='Trajectory', alpha=0.7)
        
        # Draw robot as circle with heading arrow
        robot_circle = Circle((robot_x, robot_y), 0.3, color='blue', alpha=0.8, label='Robot')
        self.ax.add_patch(robot_circle)
        
        # Draw heading arrow
        arrow_length = 0.5
        arrow_dx = arrow_length * np.cos(robot_yaw)
        arrow_dy = arrow_length * np.sin(robot_yaw)
        self.ax.arrow(robot_x, robot_y, arrow_dx, arrow_dy, 
                     head_width=0.2, head_length=0.15, fc='darkblue', ec='darkblue')
        
        # Draw goal if provided
        if goal_x is not None and goal_y is not None:
            goal_circle = Circle((goal_x, goal_y), 0.25, color='green', alpha=0.8, label='Goal')
            self.ax.add_patch(goal_circle)
        
        # Update labels and legend
        self.ax.set_title(f"Robot Position: ({robot_x:.2f}, {robot_y:.2f}) | Heading: {np.degrees(robot_yaw):.1f}°")
        
        # Update legend (avoid duplicates)
        handles, labels = self.ax.get_legend_handles_labels()
        unique_labels = {}
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels[label] = handle
        
        if unique_labels:
            self.ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
        
        # Redraw
        plt.pause(0.001)
    
    def save_trajectory(self, filename):
        """
        Save trajectory plot to file.
        
        Args:
            filename: Output image filename
        """
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Trajectory saved to {filename}")
    
    def show(self):
        """Display the figure (blocking call)"""
        plt.show(block=True)
    
    def close(self):
        """Close the figure window"""
        plt.close(self.fig)


def create_bev_visualizer(map_config, max_trajectory_length=1000):
    """
    Factory function to create a BEV visualizer.
    
    Args:
        map_config: MapConfig object
        max_trajectory_length: Maximum trajectory points to keep
    
    Returns:
        BEVVisualizer instance
    """
    return BEVVisualizer(map_config, max_trajectory_length)

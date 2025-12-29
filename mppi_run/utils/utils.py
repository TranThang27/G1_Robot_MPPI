"""
Utility functions for robot control
"""
import numpy as np
import math


def get_gravity_orientation(quaternion):
    """
    Convert quaternion to gravity orientation vector
    
    Args:
        quaternion: [qw, qx, qy, qz]
    
    Returns:
        gravity_orientation: 3D vector
    """
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """
    Proportional-Derivative control
    
    Args:
        target_q: Target joint positions
        q: Current joint positions
        kp: Proportional gains
        target_dq: Target joint velocities
        dq: Current joint velocities
        kd: Derivative gains
    
    Returns:
        Control torques
    """
    return (target_q - q) * kp + (target_dq - dq) * kd


def get_next_path_point(curr_x, curr_y, path, path_idx):
    """
    Get next point from interpolated path
    
    Args:
        curr_x, curr_y: Current robot position
        path: List of (x, y) waypoints
        path_idx: Current waypoint index
    
    Returns:
        target: Next target waypoint as numpy array
        path_idx: Updated waypoint index
    """
    if not path or path_idx >= len(path):
        return np.array([curr_x, curr_y]), path_idx
    
    current_pt = path[path_idx]
    target = np.array(current_pt)
    
    # Move to next point if close enough (match waypoint resolution)
    dist_to_current = math.hypot(current_pt[0] - curr_x, current_pt[1] - curr_y)
    if dist_to_current < 0.2:  # 2x the 0.1m resolution
        path_idx += 1
        if path_idx < len(path):
            target = np.array(path[path_idx])
    
    return target, path_idx


def extract_quaternion_from_qpos(qpos):
    """
    Extract quaternion from MuJoCo qpos array
    
    Args:
        qpos: Full position vector [x, y, z, qw, qx, qy, qz, ...]
    
    Returns:
        quaternion: [qw, qx, qy, qz]
    """
    return qpos[3:7]


def extract_robot_state(qpos, qvel):
    """
    Extract robot's x, y position and yaw angle
    
    Args:
        qpos: Position vector from MuJoCo
        qvel: Velocity vector from MuJoCo
    
    Returns:
        x, y, yaw: Robot position and orientation
    """
    x, y = qpos[0], qpos[1]
    q = qpos[3:7]
    yaw = np.arctan2(
        2 * (q[0] * q[3] + q[1] * q[2]),
        1 - 2 * (q[2] ** 2 + q[3] ** 2),
    )
    return x, y, yaw

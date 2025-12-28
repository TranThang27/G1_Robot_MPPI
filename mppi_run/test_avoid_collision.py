import time
import mujoco.viewer
import math
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import sys
import os

sys.path.append(os.path.dirname(__file__))
from A_star import AStarPlanner
from mppi_controller import G1MPPIController
from astar_utils import setup_astar_avoid_collision, plan_global_path

SIMULATION_SPEED = 5.0 
VIEWER_SYNC_SKIP = 5


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def debug_distances(x, y):
    target = np.array([10.0, 0.0])
    obstacles = np.array([
        [2.0,  0.9],
        [2.0, -0.9],
        [3.5,  0.0],
        [5.0,  0.8],
        [5.0, -0.8],
        [6.5, -0.2],
        [8.0,  0.6],
        [8.0, -0.6],
        [9.5,  0.0],
        [2.0, 0.0],
    ])

    dist_target = np.linalg.norm([x - target[0], y - target[1]])
    dist_obs = np.linalg.norm(obstacles - np.array([x, y]), axis=1)

    return dist_target, dist_obs.min()


def get_lookahead_point(curr_x, curr_y, path, lookahead_dist=1.5):
    """Get carrot following target point on path"""
    if not path:
        return np.array([curr_x, curr_y])
    
    target_point = path[-1]
    found = False
    
    for pt in reversed(path):
        dist = math.hypot(pt[0] - curr_x, pt[1] - curr_y)
        if dist <= lookahead_dist:
            target_point = pt
            found = True
            break
    
    if not found:
        target_point = path[0]
    
    return np.array(target_point)

# =========================
# 2. MPPI CONTROLLER
# =========================
# (Encapsulated in G1MPPIController class - see mppi_controller.py)




# =========================
# 3. MAIN
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()

    # ---------- LOAD CONFIG ----------
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{args.config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

    simulation_duration = config["simulation_duration"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]

    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)

    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]

    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

    num_actions = config["num_actions"]
    num_obs = config["num_obs"]

    cmd = np.array(config["cmd_init"], dtype=np.float32)

    # ---------- MUJOCO ----------
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # ---------- A* PLANNER ----------
    astar_planner = setup_astar_avoid_collision()
    print("✅ A* Pathfinder (avoid collision) Initialized.")
    GLOBAL_PATH = []

    # ---------- POLICY ----------
    policy = torch.jit.load(policy_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- MPPI CONTROLLER ----------
    mppi_controller = G1MPPIController(device=device, local_target=torch.tensor([8.0, 0.0], device=device), scenario="avoid")
    print("✅ MPPI Controller (avoid scenario) Initialized.")


    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    # ---------- SIM LOOP ----------
    with mujoco.viewer.launch_passive(m, d) as viewer:

        cam = viewer.cam
        cam.azimuth = 0     # nhìn thẳng trục x
        cam.elevation = -75   # top-down
        cam.distance = 7.0    # ZOOM OUT (cái bạn cần)
        cam.lookat[:] = [1.0, 0.0, 0.5]  # tâm mê cung
        start = time.time()

        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            # ===== PD CONTROL =====
            tau = pd_control(
                target_dof_pos,
                d.qpos[7:],
                kps,
                np.zeros_like(kds),
                d.qvel[6:],
                kds,
            )
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

            counter += 1

            # ===== HIGH LEVEL =====
            if counter % control_decimation == 0:
                curr_x, curr_y = d.qpos[0], d.qpos[1]
                q = d.qpos[3:7]

                yaw = np.arctan2(
                    2 * (q[0] * q[3] + q[1] * q[2]),
                    1 - 2 * (q[2] ** 2 + q[3] ** 2),
                )

                # ===== A* PATHFINDING =====
                # Plan path once at start or if needed
                if not GLOBAL_PATH:
                    goal_x, goal_y = 8.0, 0.0  # Target from g1_cost
                    path = plan_global_path(astar_planner, curr_x, curr_y, goal_x, goal_y)

                
                if GLOBAL_PATH:
                    lookahead_pt = get_lookahead_point(curr_x, curr_y, GLOBAL_PATH, lookahead_dist=1.5)
                else:
                    lookahead_pt = np.array([8.0, 0.0])  # Default target

                state_tensor = torch.tensor(
                    [[curr_x, curr_y, yaw]],
                    device=device,
                    dtype=torch.float32,
                )

                # ===== MPPI =====
                mppi_cmd = mppi_controller.command(state_tensor)
                mppi_cmd = mppi_cmd.squeeze().cpu().numpy()

                cmd[:] = mppi_cmd

                # ===== OBS =====
                qj = (d.qpos[7:] - default_angles) * dof_pos_scale
                dqj = d.qvel[6:] * dof_vel_scale

                gravity_orientation = get_gravity_orientation(q)
                omega = d.qvel[3:6] * ang_vel_scale

                period = 0.8
                phase = (counter * simulation_dt) % period / period

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = qj
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = [
                    np.sin(2 * np.pi * phase),
                    np.cos(2 * np.pi * phase),
                ]

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().cpu().numpy().squeeze()
                target_dof_pos = action * action_scale + default_angles

            if counter % VIEWER_SYNC_SKIP == 0:
                viewer.sync()

            time_until_next_step = m.opt.timestep / SIMULATION_SPEED - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
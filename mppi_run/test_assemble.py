import time
import mujoco.viewer

import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
from pytorch_mppi import MPPI

# =========================
# DEBUG CONFIG
# =========================
DEBUG = True
DEBUG_EVERY = 20   # in mỗi 20 lần control_decimation

# =========================
# 1. UTILS
# =========================
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

# =========================
# 2. MPPI DYNAMICS & COST
# =========================
def g1_dynamics(state, action):
    dt = 0.05
    x, y, yaw = state[:, 0], state[:, 1], state[:, 2]
    vx, vy, omega = action[:, 0], action[:, 1], action[:, 2]

    new_x = x + (vx * torch.cos(yaw) - vy * torch.sin(yaw)) * dt
    new_y = y + (vx * torch.sin(yaw) + vy * torch.cos(yaw)) * dt
    new_yaw = yaw + omega * dt

    return torch.stack([new_x, new_y, new_yaw], dim=1)




def g1_cost(state, action):
    device = state.device

    # ===== TARGET =====
    target = torch.tensor([8.0, 0.0], device=device)
    dx = target[0] - state[:, 0]
    dy = target[1] - state[:, 1]
    dist_to_target = torch.sqrt(dx**2 + dy**2)

    target_heading = torch.atan2(dy, dx)
    heading_error = torch.atan2(
        torch.sin(target_heading - state[:, 2]),
        torch.cos(target_heading - state[:, 2])
    )

    
    backward_cost = torch.relu(-action[:, 0]) ** 2

   
    lateral_cost = 0.15 * action[:, 1] ** 2
    yaw_cost     = 2.5  * action[:, 2] ** 2

# ===== OBSTACLES (match Mujoco scene) =====
    obstacles = torch.tensor([
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
], device=device)


    dist_obs = torch.cdist(state[:, :2], obstacles)
    min_dist = torch.min(dist_obs, dim=1).values

    # ===== ZONES (G1 thật) =====
    d_collision = 0.24
    d_safe      = 0.55

    obstacle_cost = torch.zeros_like(min_dist)

    # cấm tuyệt đối
    obstacle_cost += torch.where(
        min_dist < d_collision,
        2e5 * (d_collision - min_dist) ** 2,
        torch.zeros_like(min_dist)
    )

    # chỉ né khi rất sát
    obstacle_cost += torch.where(
        (min_dist >= d_collision) & (min_dist < d_safe),
        40.0 * (d_safe - min_dist) ** 2,
        torch.zeros_like(min_dist)
    )

    # ===== SPEED REWARD =====
    speed_reward = -3.0 * action[:, 0]

    return (
        1.0 * dist_to_target +
        0.6 * torch.abs(heading_error) +
        6.0 * backward_cost +
        lateral_cost +
        yaw_cost +
        obstacle_cost +
        speed_reward
    )




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

    # ---------- POLICY ----------
    policy = torch.jit.load(policy_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- MPPI ----------
    mppi = MPPI(
    g1_dynamics,
    g1_cost,
    nx=3,
    horizon=20,                 # G1 quyết rất nhanh
    num_samples=1500,
    lambda_=0.6,                # rất aggressive
    noise_sigma=torch.diag(
        torch.tensor([0.30, 0.25, 0.15], device=device)
    ),
    u_min=torch.tensor([0.0, -0.45, -0.6], device=device),
    u_max=torch.tensor([1.2,  0.45,  0.6], device=device),
    device=device,
)


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

                state_tensor = torch.tensor(
                    [[curr_x, curr_y, yaw]],
                    device=device,
                    dtype=torch.float32,
                )

                # ===== MPPI =====
                mppi_cmd = mppi.command(state_tensor)
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

                # ===== DEBUG PRINT =====
                if DEBUG and counter % (control_decimation * DEBUG_EVERY) == 0:
                    dist_t, dist_o = debug_distances(curr_x, curr_y)
                    print("\n================ DEBUG =================")
                    print(f"pos=({curr_x:.2f},{curr_y:.2f}) yaw={yaw:.2f}")
                    print(f"cmd(vx,vy,w)=({cmd[0]:.2f},{cmd[1]:.2f},{cmd[2]:.2f})")
                    print(f"dist_target={dist_t:.2f}  min_obs={dist_o:.2f}")
                    print(f"tau mean={np.mean(tau):.3f} max={np.max(np.abs(tau)):.3f}")
                    print(f"obs cmd scaled={obs[6:9]}")
                    print(f"qj mean={np.mean(qj):.3f} dqj mean={np.mean(dqj):.3f}")
                    print("=======================================\n")

            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

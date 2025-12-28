import time
import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import random
import sys
import os
import math

sys.path.append(os.path.dirname(__file__))
from A_star import AStarPlanner
from mppi_controller import G1MPPIController
from astar_utils import setup_astar_room, plan_global_path


DEBUG = True
DEBUG_EVERY = 50


SIMULATION_SPEED = 5.0 
VIEWER_SYNC_SKIP = 5


SAFE_POINTS = [
    [4.5 , 0.0],   
    [6.0, 3.0],   
    [6.0, -3.0],  
]

# =========================
# GLOBAL VARIABLES
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- KHỞI TẠO QUAN TRỌNG: TRÁNH SỐ 0 ---
# Chọn ngẫu nhiên một điểm an toàn ngay từ đầu
_start_pt = random.choice(SAFE_POINTS)
CURRENT_TARGET = torch.tensor(_start_pt, device=device) # Đích cuối
LOCAL_TARGET   = torch.tensor(_start_pt, device=device) # Đích tạm thời (MPPI Follow)
GLOBAL_PATH = []  

def get_new_target():
    
    # Lọc ra các điểm khác với đích hiện tại để robot không đứng yên mãi
    curr_list = CURRENT_TARGET.cpu().numpy().tolist()
    candidates = [p for p in SAFE_POINTS if np.linalg.norm(np.array(p) - np.array(curr_list)) > 1.0]
    
    if not candidates: candidates = SAFE_POINTS # Fallback
    new_pt = random.choice(candidates)
    
    print(f"\n>>> NEW MISSION: Go to ({new_pt[0]:.2f}, {new_pt[1]:.2f}) <<<")
    return torch.tensor(new_pt, device=device)

def get_lookahead_point(curr_x, curr_y, path, lookahead_dist=1.2):
    """Tìm điểm dẫn đường (Carrot) trên Path"""
    if not path: return np.array([curr_x, curr_y]) # Nếu không có path, trả về chính nó (đứng yên)
    
    target_point = path[-1]
    found = False
    
    # Logic Pure Pursuit đơn giản
    for pt in reversed(path):
        dist = math.hypot(pt[0] - curr_x, pt[1] - curr_y)
        if dist <= lookahead_dist:
            target_point = pt
            found = True
            break
            
   
    if not found: target_point = path[0]
        
    return np.array(target_point)


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

# =========================
# MAIN LOOP
# =========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()

    # --- Load Config ---
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
    
    cmd = np.zeros(3, dtype=np.float32) 

    # --- Init MuJoCo ---
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    astar_planner = setup_astar_room()
    print("✅ A* Map (room scenario) Initialized.")

    # Initialize MPPI controller with shared LOCAL_TARGET reference
    mppi_controller = G1MPPIController(device=device, local_target=LOCAL_TARGET)
    print("✅ MPPI Controller Initialized.")

    policy = torch.jit.load(policy_path)

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    

    with mujoco.viewer.launch_passive(m, d) as viewer:
        cam = viewer.cam
        cam.azimuth = 0; cam.elevation = -50; cam.distance = 9.0
        cam.lookat[:] = [4.0, 0.0, 0.0]
        
        start_time = time.time()
        
        while viewer.is_running() and time.time() - start_time < simulation_duration:
            step_start = time.time()

            # 1. Control
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            counter += 1

            # 2. Planning Logic
            if counter % control_decimation == 0:
                curr_x, curr_y = d.qpos[0], d.qpos[1]
                q = d.qpos[3:7]
                yaw = np.arctan2(2*(q[0]*q[3] + q[1]*q[2]), 1-2*(q[2]**2 + q[3]**2))

                # --- NẾU CHƯA CÓ ĐƯỜNG, TÍNH TOÁN ---
                if not GLOBAL_PATH:
                    if counter % 50 == 0: print("⏳ Waiting for A*...")
                    
                    # Cập nhật LOCAL_TARGET bằng vị trí hiện tại để cost function không hút về 0
                    LOCAL_TARGET[0] = curr_x
                    LOCAL_TARGET[1] = curr_y

                    # Tính A* từ vị trí hiện tại đến CURRENT_TARGET
                    path = plan_global_path(astar_planner, curr_x, curr_y, CURRENT_TARGET[0].item(), CURRENT_TARGET[1].item())
                    
                    if path:
                        GLOBAL_PATH = path
                        print(f"✅ Path Found! Len: {len(path)}")

                # --- ĐIỀU KHIỂN ---
                if not GLOBAL_PATH:
                    # Chưa có đường -> ĐỨNG IM
                    cmd[:] = 0.0 
                else:
                    # Có đường -> ĐI THEO
                    tgt_cpu = CURRENT_TARGET.cpu().numpy()
                    dist_to_final = np.linalg.norm([curr_x - tgt_cpu[0], curr_y - tgt_cpu[1]])
                    
                    if dist_to_final < 0.5:
                        print(f"✅ ARRIVED AT SAFE POINT ({tgt_cpu[0]:.1f}, {tgt_cpu[1]:.1f})")
                        CURRENT_TARGET = get_new_target() # Lấy điểm Safe Point mới
                        GLOBAL_PATH = [] # Xóa đường cũ để vòng sau tính lại
                        cmd[:] = 0.0 # Dừng lại nghỉ tí
                    else:
                        # Tìm Carrot Point trên đường A*
                        lookahead_pt = get_lookahead_point(curr_x, curr_y, GLOBAL_PATH, lookahead_dist=1.2)
                        
                        # Cập nhật Target cho MPPI Cost
                        LOCAL_TARGET[0] = lookahead_pt[0]
                        LOCAL_TARGET[1] = lookahead_pt[1]

                        # Chạy MPPI
                        state_tensor = torch.tensor([[curr_x, curr_y, yaw]], device=device, dtype=torch.float32)
                        mppi_cmd = mppi_controller.command(state_tensor)
                        cmd[:] = mppi_cmd.squeeze().cpu().numpy()

                        if DEBUG and counter % (control_decimation * DEBUG_EVERY) == 0:
                            print(f"Go -> ({lookahead_pt[0]:.1f}, {lookahead_pt[1]:.1f})")

                # --- Observation ---
                qj = (d.qpos[7:] - default_angles) * dof_pos_scale
                dqj = d.qvel[6:] * dof_vel_scale
                gravity_orientation = get_gravity_orientation(q)
                omega = d.qvel[3:6] * ang_vel_scale
                
                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9:9+num_actions] = qj
                obs[9+num_actions:9+2*num_actions] = dqj
                obs[9+2*num_actions:9+3*num_actions] = action
                
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().cpu().numpy().squeeze()
                target_dof_pos = action * action_scale + default_angles

            if counter % VIEWER_SYNC_SKIP == 0:
                viewer.sync()
            
            target_dt = m.opt.timestep / SIMULATION_SPEED
            process_time = time.time() - step_start
            if target_dt > process_time:
                time.sleep(target_dt - process_time)
import torch
import mujoco
import mujoco.viewer
import numpy as np
from pytorch_mppi import MPPI
import time
import os
import yaml
import argparse

# ============================================================================
# PHẦN 1: SETUP UNITREE G1 (từ code 2)
# ============================================================================

def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD controller for joint torques"""
    return (target_q - q) * kp + (target_dq - dq) * kd

# ============================================================================
# PHẦN 2: MPPI DYNAMICS & COST (từ code 1, adapted)
# ============================================================================

def unitree_dynamics(state, action, dt=0.1):
    """
    Simplified kinematic model for Unitree base motion
    state: [x, y, theta] - base position and orientation
    action: [vx, vy, omega] - base velocity commands
    """
    x, y, theta = state[:, 0], state[:, 1], state[:, 2]
    vx, vy, omega = action[:, 0], action[:, 1], action[:, 2]
    
    # Transform body-frame velocity to world frame
    new_x = x + (vx * torch.cos(theta) - vy * torch.sin(theta)) * dt
    new_y = y + (vx * torch.sin(theta) + vy * torch.cos(theta)) * dt
    new_theta = theta + omega * dt
    
    # Normalize theta to [-pi, pi]
    new_theta = torch.atan2(torch.sin(new_theta), torch.cos(new_theta))
    
    return torch.stack([new_x, new_y, new_theta], dim=1)

def navigation_cost(state, action):
    """Cost function cho MPPI - đồng bộ với XML environment"""
    target = torch.tensor([4.5, 4.5], device=state.device)
    
    # Obstacles từ XML (12 vật cản)
    obstacles = torch.tensor([
        [1.0, 1.0], [1.3, 1.2],
        [2.0, 1.5], [2.3, 1.8],
        [1.5, 2.5], [1.8, 2.7],
        [2.8, 3.2], [3.1, 3.4],
        [2.2, 3.9], [2.5, 4.1],
        [2.6, 3.2], [3.9, 4.4]
    ], device=state.device)
    
    # Cost 1: Distance to target
    dist_to_target = torch.norm(state[:, :2] - target, dim=1)
    target_cost = dist_to_target * 15.0
    
    # Cost 2: Obstacle avoidance
    dist_to_obs = torch.cdist(state[:, :2], obstacles)
    obs_radius = 0.1
    safety_margin = 0.3  # Tăng margin vì Unitree lớn hơn
    
    soft_collision = torch.sum(
        torch.exp(-3.0 * (dist_to_obs - (obs_radius + safety_margin))),
        dim=1
    ) * 5.0
    
    hard_collision = torch.any(
        dist_to_obs < (obs_radius + safety_margin), dim=1
    ).float() * 200.0
    
    # Cost 3: Heading alignment
    direction_to_target = target - state[:, :2]
    direction_norm = torch.norm(direction_to_target, dim=1, keepdim=True) + 1e-6
    direction_to_target = direction_to_target / direction_norm
    
    heading = torch.stack([torch.cos(state[:, 2]), torch.sin(state[:, 2])], dim=1)
    alignment = torch.sum(direction_to_target * heading, dim=1)
    alignment_cost = (1.0 - alignment) * 2.0
    
    # Cost 4: Encourage forward motion
    speed_cost = torch.exp(-action[:, 0]) * 2.0
    
    # Cost 5: Smooth control
    action_penalty = 0.1 * torch.sum(action**2, dim=1)
    
    return target_cost + soft_collision + hard_collision + alignment_cost + speed_cost + action_penalty

# ============================================================================
# PHẦN 3: MAIN - TÍCH HỢP MPPI + UNITREE
# ============================================================================

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="unitree_g1.yaml", 
                        help="Config file for Unitree G1")
    parser.add_argument("--xml", type=str, default="g1_scene_with_obstacles.xml",
                        help="MuJoCo XML environment file")
    args = parser.parse_args()
    
    # Load Unitree configuration (nếu có config file)
    try:
        with open(args.config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            kps = np.array(config["kps"], dtype=np.float32)
            kds = np.array(config["kds"], dtype=np.float32)
            default_angles = np.array(config["default_angles"], dtype=np.float32)
            simulation_dt = config["simulation_dt"]
            control_decimation = config["control_decimation"]
            policy_path = config.get("policy_path", None)
            
            # Load policy nếu có
            policy = torch.jit.load(policy_path) if policy_path else None
    except:
        print("⚠️ Không tìm thấy config file, sử dụng default parameters")
        kps = np.ones(12) * 100.0  # Increased from 40
        kds = np.ones(12) * 5.0    # Increased from 2
        # Default standing angles for G1 12DOF: [L_hip_pitch, L_hip_roll, L_hip_yaw, L_knee, L_ankle_pitch, L_ankle_roll,
        #                                         R_hip_pitch, R_hip_roll, R_hip_yaw, R_knee, R_ankle_pitch, R_ankle_roll]
        default_angles = np.array([
            0.0, 0.0, 0.0, -0.6, 0.3, 0.0,  # Left leg
            0.0, 0.0, 0.0, -0.6, 0.3, 0.0   # Right leg
        ], dtype=np.float32)
        simulation_dt = 0.01
        control_decimation = 10
        policy = None
    
    # Load MuJoCo environment
    xml_path = os.path.join(os.path.dirname(__file__), args.xml)
    if not os.path.exists(xml_path):
        # Try without directory prefix if file is in current directory
        xml_path = args.xml
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = simulation_dt
    
    # Setup MPPI
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")
    
    mppi = MPPI(
        unitree_dynamics,
        navigation_cost,
        nx=3,  # [x, y, theta]
        num_samples=1500,
        horizon=30,
        noise_sigma=torch.diag(torch.tensor([0.3, 0.2, 0.8], device=device)),
        lambda_=1.0,
        device=device,
        u_min=torch.tensor([-0.3, -0.3, -1.5], device=device),  # [vx, vy, omega]
        u_max=torch.tensor([0.8, 0.3, 1.5], device=device)
    )
    
    # Initialize state
    curr_state = torch.tensor([[0.5, 0.5, 0.0]], device=device)  # Start position
    target_pos = np.array([4.5, 4.5])
    reached_target = False
    
    # Control variables
    cmd_vel = np.zeros(3)  # [vx, vy, omega] commands
    counter = 0
    
    print("\n" + "="*60)
    print("🤖 UNITREE G1 MPPI NAVIGATION")
    print("="*60)
    print(f"📍 Start: (0.5, 0.5) → Target: (4.5, 4.5)")
    print(f"🚧 Obstacles: 12 (zig-zag pattern)")
    print(f"🧠 MPPI samples: 1500, horizon: 30")
    print("="*60 + "\n")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Setup camera
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [2.5, 2.5, 0]
        viewer.cam.distance = 10.0
        viewer.cam.elevation = -60
        viewer.cam.azimuth = 0
        
        step = 0
        stuck_counter = 0
        prev_pos = curr_state[0, :2].cpu().numpy()
        total_distance = 0.0
        start_time = time.time()
        
        while viewer.is_running() and not reached_target and step < 3000:
            step_start = time.time()
            
            # ============================================================
            # MPPI: Tính toán velocity command
            # ============================================================
            if step % 5 == 0:  # Update MPPI every 5 steps
                mppi_action = mppi.command(curr_state)
                cmd_vel = mppi_action.cpu().numpy()
            
            # ============================================================
            # Unitree: Low-level control (nếu có policy)
            # ============================================================
            if policy is not None and counter % control_decimation == 0:
                # Create observation (simplified)
                quat = data.qpos[3:7]
                omega = data.qvel[3:6]
                qj = data.qpos[7:]
                dqj = data.qvel[6:]
                
                gravity_orientation = get_gravity_orientation(quat)
                
                # Build observation vector (cần match với config)
                obs = np.zeros(48)  # Adjust size based on your policy
                obs[:3] = omega * 0.25
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd_vel  # Inject MPPI commands here!
                obs[9:21] = qj
                obs[21:33] = dqj
                
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()
                target_dof_pos = action * 0.25 + default_angles
            else:
                target_dof_pos = default_angles
            
            # Apply PD control
            tau = pd_control(target_dof_pos, data.qpos[7:], kps, 
                             np.zeros_like(kds), data.qvel[6:], kds)
            data.ctrl[:] = tau
            
            # Step physics
            mujoco.mj_step(model, data)
            counter += 1
            
            # ============================================================
            # Update MPPI state from MuJoCo
            # ============================================================
            pos = data.qpos[0:2]
            quat = data.qpos[3:7]
            yaw = np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 
                             1 - 2*(quat[2]**2 + quat[3]**2))
            
            curr_state = torch.tensor([[pos[0], pos[1], yaw]], device=device)
            
            # Calculate metrics
            step_distance = np.linalg.norm(pos - prev_pos)
            total_distance += step_distance
            
            # Anti-stuck mechanism
            if step_distance < 0.005:
                stuck_counter += 1
                if stuck_counter > 100:
                    print(f"⚠️ Robot stuck at ({pos[0]:.2f}, {pos[1]:.2f}), adding noise...")
                    # Add random yaw perturbation
                    curr_state[0, 2] += (torch.rand(1, device=device).item() - 0.5) * 0.8
                    stuck_counter = 0
            else:
                stuck_counter = 0
            
            prev_pos = pos.copy()
            
            # Check collision
            if data.ncon > 0:
                print(f"⚠️ Collision detected at step {step}! Contacts: {data.ncon}")
            
            # Check target reached
            dist_to_target = np.linalg.norm(pos - target_pos)
            if dist_to_target < 0.3:
                reached_target = True
                elapsed_time = time.time() - start_time
                print(f"\n🎉 TARGET REACHED!")
                print(f"⏱️  Time: {elapsed_time:.1f}s")
                print(f"📏 Distance traveled: {total_distance:.2f}m")
                print(f"📍 Final position: ({pos[0]:.2f}, {pos[1]:.2f})")
            
            # Logging
            if step % 100 == 0:
                print(f"Step {step:4d} | Pos: ({pos[0]:4.2f}, {pos[1]:4.2f}) | "
                      f"Yaw: {np.degrees(yaw):5.1f}° | "
                      f"Dist: {dist_to_target:4.2f}m | "
                      f"Cmd: ({cmd_vel[0]:4.2f}, {cmd_vel[1]:4.2f}, {cmd_vel[2]:4.2f})")
            
            viewer.sync()
            
            # Time keeping
            time_until_next_step = simulation_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            
            step += 1
        
        if reached_target:
            print("\n✅ Mission accomplished! Holding view for 5 seconds...")
            time.sleep(5)
        elif step >= 3000:
            print(f"\n❌ Timeout after {step} steps")
            print(f"Final distance to target: {dist_to_target:.2f}m")
    
    print("\n" + "="*60)
    print("Simulation ended.")
    print("="*60)
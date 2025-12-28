# G1 Robot MPPI Navigation System


## Cấu Trúc Thư Mục

```
mppi_run/
├── A_star.py              # A* Path Planning Algorithm
├── constants.py           # Cấu hình chung (parameters, weights)
├── map_config.py          # Cấu hình scene (obstacles, bounds)
├── mppi_controller.py     # MPPI Controller logic
├── path_smoother.py       # B-spline path smoothing
├── sim_utils.py           # Simulation utilities
├── utils.py               # Helper functions
├── test_avoid_collision.py # Test 1: Tránh cylinder obstacles
├── test_room.py           # Test 2: Multi-target trong phòng
├── scene_1.py             # Scene 1 definition
├── scene_2.py             # Scene 2 definition
└── __init__.py            # Python package init
```

## Cài Đặt

### Hướng dẫn cài đặt chi tiết

Để cài đặt toàn bộ môi trường Unitree RL Gym, vui lòng tham khảo:
- **[Setup Guide (English)](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/doc/setup_en.md)**

### 1. Cài đặt mppi_controller

```bash
pip install pytorch_mppi 
```

### 2. Cài đặt MuJoCo 

```bash
pip install mujoco==3.2.3
```

### Test 1: Avoid Cylinder Obstacles

```bash
python scene_1.py scene_1.yaml
```


### Test 2: Multi-Target Navigation

```bash
python scene_2.py scene_2.yaml
```

---



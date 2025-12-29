# G1 Robot MPPI Navigation System


## Cấu Trúc Thư Mục

```
mppi_run/
├── core/
│   ├── mppi_controller.py     # MPPI Controller logic
│   ├── astar.py               # A* Path Planning Algorithm
│   └── path_smoother.py       # B-spline path smoothing
├── utils/
│   ├── constants.py           # Cấu hình chung (parameters, weights)
│   ├── sim_utils.py           # Simulation utilities
│   ├── map_config.py          # Cấu hình scene (obstacles, bounds)
│   └── utils.py               # Helper functions
├── pipeline/
│   └── main_algor.py          # Unified control pipeline
├── scenarios/
│   ├── scene_1.py             # Test 1: Tránh cylinder obstacles
│   ├── scene_2.py             # Test 2: Multi-target trong phòng
│   └── scene_3.py             # Test 3: Dynamic moving obstacles
└── config/
    └── __init__.py            # Configuration package init
```

## Cài Đặt

### Setup guide

Install Unitree RL Gym, Issac Gym:
- **[Setup Guide](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/doc/setup_en.md)**

- **[Install IssacGym](https://medium.com/@piliwilliam0306/install-isaac-gym-on-ubuntu-22-04-8ebf4b86e6f7) **

### 1. Cài đặt mppi_controller

```bash
pip install pytorch_mppi 
```

### 2. Cài đặt MuJoCo 

```bash
pip install mujoco==3.2.3
```

### Test 1: Avoid Cylinder Obstacles (Vao file mppi_run/scenarios )

```bash
python scene_1 scene_1.yaml
```


### Test 2: Multi-Target Navigation (Vao file mppi_run/scenarios )

```bash
python scene_2 scene_2.yaml
```

---



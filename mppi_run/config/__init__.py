"""
Configuration module
Contains camera and viewer configurations for different scenarios
"""

# Camera configurations for different scenarios
CAMERA_CONFIGS = {
    'avoid_collision': {
        'azimuth': 0,
        'elevation': -75,
        'distance': 12.0,
        'lookat': [4.0, 0.0, 0.5]
    },
    'room_scene': {
        'azimuth': 0,
        'elevation': -50,
        'distance': 9.0,
        'lookat': [4.0, 0.0, 0.0]
    }
}

__all__ = ['CAMERA_CONFIGS']

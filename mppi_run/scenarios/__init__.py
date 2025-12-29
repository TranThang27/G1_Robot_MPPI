"""
Scenarios module
Contains different test scenarios and use cases
"""

from .obstacle_avoidance import run_avoid_collision_test
from .room_navigation import run_room_test

__all__ = [
    'run_avoid_collision_test',
    'run_room_test'
]

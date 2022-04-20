from helper_functions import Particle, PoseTuple, normalize_angle
from collections import namedtuple
from itertools import islice

import matplotlib.pyplot as plt

from pathlib import Path
import math
import time
from typing import Iterable, Optional, Tuple, List
import rospy
import random
from nav_msgs.msg import Odometry, OccupancyGrid
from nav_msgs.srv import GetMap
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, PoseStamped, Pose
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion

import numpy as np
from numpy.random import default_rng, Generator

from helper_functions import TFHelper, print_time, sample_normal, sample_normal_error
from occupancy_field import OccupancyField


class SensorModel:
    lidar_actual: Optional[np.array] = None
    map_obstacles: np.array

    def __init__(self, map_obstacles: np.array):
        self.map_obstacles = map_obstacles

    def set_lidar_ranges(self, ranges: list):
        self.lidar_actual = np.array(ranges[0:360])

    # KLUDGE: All local state is stored as instance variables to easily separate graphing
    def calculate_weight(self, particle: Particle) -> float:
        self.particle = particle

        # Take map data as cartesian coords
        # Shift to center at particle
        # NB: Both the map and all particles are in the `map` frame
        self.obstacles_shifted = self.map_obstacles - \
            [self.particle.x, self.particle.y]

        # Convert to polar coordinates
        self.obstacle_rs = np.sqrt(
            (self.obstacles_shifted[:, 0] ** 2) +
            (self.obstacles_shifted[:, 1] ** 2)
        )
        self.obstacle_thetas_rad = normalize_angle(
            np.arctan2(
                self.obstacles_shifted[:, 1],
                self.obstacles_shifted[:, 0]
            ) + self.particle.theta  # Rotate by particle's heading
        )

        # Convert to degrees, and descretize to whole-degree increments (like LIDAR data)
        # This is the only place we use degrees, but it's helpful since LIDAR is indexed by degree
        self.obstacle_thetas = np.rad2deg(self.obstacle_thetas_rad).round()

        # Take the minimum at each angle
        # Indexed like lidar data, where each index is the degree
        self.lidar_expected = np.zeros(360)

        for theta, r in zip(self.obstacle_thetas, self.obstacle_rs):
            # theta is already a whole number, just make it the right type
            idx = int(theta)

            # Assume the LIDAR can't see anything beyond 3m
            # TODO: refine this estimate (I think it's roughly correct)
            if r > 3.0:
                continue

            if r < self.lidar_expected[idx]:
                self.lidar_expected[idx] = r

        # Compare to LIDAR data
        self.lidar_diff = np.abs(self.lidar_actual - self.lidar_expected)

        # Calculate weight
        self.weight = np.sum((1 / self.lidar_diff[self.lidar_diff > 0.0]) ** 3)

        return self.weight

    def save_debug_plot(self, name: str):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        # https://stackoverflow.com/a/18486470
        ax.set_theta_offset(math.pi/2.0)
        ax.grid(True)
        ax.arrow(0, 0, 0, 1)
        ax.plot(self.obstacle_thetas_rad, self.obstacle_rs, 'b,')
        ax.plot(np.deg2rad(np.arange(0, 360)), self.lidar_expected, 'c.')
        ax.plot(np.deg2rad(np.arange(0, 360)), self.lidar_actual, 'r.')
        ax.set_title(
            f"({self.particle.x:.2f}, {self.particle.y:.2f}; {self.particle.theta:.2f}; w: {self.weight:.6f})"
        )
        data_dir = Path(__file__).parent.parent / \
            'particle_sensor_data'
        data_dir.mkdir(exist_ok=True)
        fig.savefig(data_dir / f"{name}_{self.weight:010.6f}.png")
        plt.close(fig)

    @classmethod
    def from_map(cls, map: OccupancyGrid) -> 'SensorModel':
        return cls(cls.preprocess_map(map))

    @staticmethod
    def preprocess_map(map: OccupancyGrid) -> np.array:
        if map.info.origin.orientation.w != 1.0:
            raise ValueError("Unsupported map with rotated origin.")

        # The coordinates of each occupied grid cell in the map
        total_occupied = np.sum(np.array(map.data) > 0)
        occupied = np.zeros((total_occupied, 2))
        curr = 0
        for x in range(map.info.width):
            for y in range(map.info.height):
                # occupancy grids are stored in row major order
                ind = x + y*map.info.width
                if map.data[ind] > 0:
                    occupied[curr, 0] = (float(x) * map.info.resolution) \
                        + map.info.origin.position.x
                    occupied[curr, 1] = (float(y) * map.info.resolution) \
                        + map.info.origin.position.y
                    curr += 1
        return occupied

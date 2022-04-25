import math
import rospy
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod

from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from occupancy_field import OccupancyField
from visualization_msgs.msg import Marker
from helper_functions import Particle, normalize_angle, make_marker


class SensorModel(ABC):
    last_lidar: Optional[np.array] = None
    """ Most recent LIDAR data. """

    def set_lidar(self, ranges: list):
        """ Notify the model of new LIDAR data. """
        self.last_lidar = np.array(ranges[0:360])

    @abstractmethod
    def set_map(self, map: OccupancyGrid):
        pass

    @abstractmethod
    def calculate_weight(self, particle: Particle) -> float:
        pass

    @abstractmethod
    def save_debug_plot(self, name: str):
        pass


class OccupancyFieldSensorModel(SensorModel):
    """ A sensor model based on an occupancy field. """

    occupancy_field: OccupancyField

    closest_obstacle: float = 0.0
    """ Distance to closest obstacle in most recent LIDAR data. """

    def __init__(self, display_pub: Optional[rospy.Publisher] = None, debug_data_dir: Path = Path(__file__).parent.parent / 'particle_sensor_data'):
        self.debug_data_dir = debug_data_dir
        self.debug_data_dir.mkdir(exist_ok=True)
        self.occupancy_field = OccupancyField()
        self.display_pub = display_pub

    def set_lidar(self, ranges: list):
        """ Notify the model of new LIDAR data. """
        super().set_lidar(ranges)

        ranges = np.array(ranges)
        self.closest_obstacle = np.min(ranges[ranges > 0])

    def calculate_weight(self, particle: Particle) -> float:
        self.particle = particle
        self.thetas = np.deg2rad(np.arange(0, 360)) + particle.theta
        self.rs = self.last_lidar
        self.xs = (self.rs * np.cos(self.thetas)) + particle.x
        self.ys = (self.rs * np.sin(self.thetas)) + particle.y

        self.weight = 0
        self.num_valid = 0
        self.actual_distances = np.zeros_like(self.rs)
        self.expected_distances = np.zeros_like(self.rs)

        for i in range(len(self.rs)):
            if self.rs[i] == 0.0:  # Ignore angles where we don't have any LIDAR data
                continue

            x = self.xs[i]
            y = self.ys[i]
            # actual_distance = self.rs[i]  # np.sqrt((x**2) + (y**2))
            # self.actual_distances[i] = actual_distance
            expected_distance = self.occupancy_field.get_closest_obstacle_distance(
                x, y)
            if np.isnan(expected_distance):  # or expected_distance > 0.1:
                self.weight = max(self.weight - 1, 0)
            self.expected_distances[i] = expected_distance
            # diff = abs(actual_distance - expected_distance)
            self.num_valid += 1
            self.weight += (10 *
                            (np.exp(-(expected_distance ** 2) / 0.0005) ** 3)) ** 2

        # print(self.expected_distances)

        # marker = make_marker(
        #     point=[Point(x, y, 0) for x, y in zip(self.xs, self.ys)],
        #     shape=Marker.CUBE_LIST,
        #     scale=(0.1, 0.1, 0.1),
        #     frame_id='map'
        # )

        # if self.display_pub is not None:
        #     self.display_pub.publish(marker)

        # print("Distances:", distances, "weight:", weight)

        # print("NUM VALID:", num_valid, "WEIGHT:", weight)

        return self.weight

    def set_map(self, map: OccupancyGrid):
        # OccupancyField gets the map itself, do nothing
        pass

    def save_debug_plot(self, name: str):
        # Not supported
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        # https://stackoverflow.com/a/18486470
        ax.set_theta_offset(math.pi/2.0)
        ax.grid(True)
        ax.arrow(0, 0, 0, 1)
        ax.plot(np.deg2rad(np.arange(0, 360)), self.rs, 'r.')
        ax.plot(np.deg2rad(np.arange(0, 360)), self.expected_distances, 'b,')
        # ax.plot(np.deg2rad(np.arange(0, 360)), self.actual_distances, 'c.')
        # ax.plot(self.obstacle_thetas_rad, self.obstacle_rs, 'b,')
        # ax.plot(np.deg2rad(np.arange(0, 360)), self.lidar_expected, 'c,')
        ax.set_title(
            f"OF: ({self.particle.x:.2f}, {self.particle.y:.2f}; {self.particle.theta:.2f}; w: {self.weight:.6f})"
        )
        fig.savefig(self.debug_data_dir / f"{name}_{self.weight:010.6f}.png")
        plt.close(fig)


class RayTracingSensorModel(SensorModel):
    """ A sensor model based on pseudo-ray tracing. """
    map_obstacles: Optional[np.array] = None
    """ (n, 2)-sized matrix of x, y coordinates of occupied squares (ie. obstacles) on the map. """

    debug_data_dir: Path
    """ Folder to store debugging images (see save_debug_plot). Defaults to __file__/../particle_sensor_data. """

    def set_map(self, map: OccupancyGrid):
        """ Set the map. """
        self.map_obstacles = self.preprocess_map(map)

    def __init__(self, debug_data_dir: Path = Path(__file__).parent.parent / 'particle_sensor_data'):
        self.debug_data_dir = debug_data_dir
        self.debug_data_dir.mkdir(exist_ok=True)

    def calculate_weight(self, particle: Particle) -> float:
        """
        Use the sensor model to figure out how likely it is that the robot was at the particle given
        the most recent LIDAR data.

        Think of this as a pure method, although it isn't actually: it stores all internal state as
        instance variables on the model class. This enables you to call save_debug_plot immediately
        afterwards to get a pretty graph showing the internal state of the model.
        """
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
            ) - self.particle.theta  # Rotate by particle's heading
        )

        # Convert to degrees, and descretize to whole-degree increments (like LIDAR data)
        # This is the only place we use degrees, but it's helpful since LIDAR is indexed by degree
        self.obstacle_thetas = (np.rad2deg(
            self.obstacle_thetas_rad).round())

        self.obstacle_thetas[self.obstacle_thetas < 0.0] += 360.0

        # Take the minimum at each angle
        # Indexed like lidar data, where each index is the degree
        self.lidar_expected = np.zeros(360)

        for theta, r in zip(self.obstacle_thetas, self.obstacle_rs):
            # theta is already a whole number, just make it the right type
            idx = int(theta)
            if idx < 0 or idx >= 360:
                print("ERROR: INVALID IDX:", idx)

            # Assume the LIDAR can't see anything beyond 3m
            # TODO: refine this estimate (I think it's roughly correct)
            if r > 3.0:
                continue

            if self.lidar_expected[idx] == 0.0 or r < self.lidar_expected[idx]:
                self.lidar_expected[idx] = r

        # Compare to LIDAR data
        self.lidar_diff = np.abs(self.last_lidar - self.lidar_expected)

        # Calculate weight
        self.weight = np.sum(
            0.5 *
            (np.exp(-(self.lidar_diff[self.lidar_diff >
             0.0] ** 2) / 0.01) ** 3)
        )

        return self.weight

    def save_debug_plot(self, name: str):
        """
        Call this method right after calculate_weight to save (to disk) a graph showing the internal
        state of the model, for debugging purposes. This is quite slow.
        """
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        # https://stackoverflow.com/a/18486470
        ax.set_theta_offset(math.pi/2.0)
        ax.grid(True)
        ax.arrow(0, 0, 0, 1)
        # ax.plot(self.obstacle_thetas_rad, self.obstacle_rs, 'b,')
        ax.plot(np.deg2rad(np.arange(0, 360)), self.lidar_expected, 'c,')
        ax.plot(np.deg2rad(np.arange(0, 360)), self.last_lidar, 'r.')
        ax.set_title(
            f"({self.particle.x:.2f}, {self.particle.y:.2f}; {self.particle.theta:.2f}; w: {self.weight:.6f})"
        )
        fig.savefig(self.debug_data_dir / f"{name}_{self.weight:010.6f}.png")
        plt.close(fig)

    @staticmethod
    def preprocess_map(map: OccupancyGrid) -> np.array:
        """
        Convert a map from a ROS OccupancyGrid to the x/y coordinate format this model uses.

        Only do this once per map, then pass it to set_map.

        OccupancyGrids store their data in a giant array in row-major order. Each point is in the
        range [0, 100], where 100 is "very occupied" and 0 is "unoccupied." -1 represents unknown.
        All points in our testing maps are either 0, 100, or -1.
        """

        if map.info.origin.orientation.w != 1.0:
            raise ValueError("Unsupported map with rotated origin.")

        # Number of obstacle points to deal with
        total_occupied = np.sum(np.array(map.data) > 0)

        # The coordinates of each occupied grid cell in the map
        occupied = np.zeros((total_occupied, 2))

        curr = 0
        for x in range(map.info.width):
            for y in range(map.info.height):
                # Occupancy grids are stored in row major order
                ind = x + (y * map.info.width)
                if map.data[ind] > 0:
                    occupied[curr, 0] = (float(x) * map.info.resolution) \
                        + map.info.origin.position.x
                    occupied[curr, 1] = (float(y) * map.info.resolution) \
                        + map.info.origin.position.y
                    curr += 1

        return occupied

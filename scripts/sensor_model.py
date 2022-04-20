import math
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Optional

from nav_msgs.msg import OccupancyGrid
from helper_functions import Particle, normalize_angle


class SensorModel:
    last_lidar: Optional[np.array] = None
    """ Most recent LIDAR data. """

    map_obstacles: Optional[np.array] = None
    """ (n, 2)-sized matrix of x, y coordinates of occupied squares (ie. obstacles) on the map. """

    debug_data_dir: Path
    """ Folder to store debugging images (see save_debug_plot). Defaults to __file__/../particle_sensor_data. """

    def set_map(self, map: OccupancyGrid):
        """ Set the map. """
        self.map_obstacles = self.preprocess_map(map)

    def set_lidar(self, ranges: list):
        """ Notify the model of new LIDAR data. """
        self.last_lidar = np.array(ranges[0:360])

    def __init__(self, debug_data_dir: Path = Path(__file__).parent.parent / 'particle_sensor_data'):
        self.debug_data_dir = debug_data_dir
        self.debug_data_dir.mkdir(exist_ok=True)

    # KLUDGE: All local state is stored as instance variables to easily separate graphing
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

            if self.lidar_expected[idx] == 0.0 or r < self.lidar_expected[idx]:
                self.lidar_expected[idx] = r

        # Compare to LIDAR data
        self.lidar_diff = np.abs(self.last_lidar - self.lidar_expected)

        # Calculate weight
        self.weight = np.sum((1 / self.lidar_diff[self.lidar_diff > 0.0]) ** 3)

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
        ax.plot(self.obstacle_thetas_rad, self.obstacle_rs, 'b,')
        ax.plot(np.deg2rad(np.arange(0, 360)), self.lidar_expected, 'c.')
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

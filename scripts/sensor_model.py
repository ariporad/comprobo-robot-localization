import math
import rospy
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from threading import Lock
from abc import ABC, abstractmethod
from typing import List, Optional, Iterable
from concurrent.futures import Executor, ProcessPoolExecutor

from nav_msgs.msg import OccupancyGrid
from helper_functions import Particle, normalize_angle


class SensorModel(ABC):
    """ Base class for sensor models. """

    last_lidar: Optional[np.array] = None
    """ Most recent LIDAR data. """

    def set_lidar(self, ranges: list):
        """ Notify the model of new LIDAR data. """
        self.last_lidar = np.array(ranges[0:360])

    @abstractmethod
    def weight_particles(self, particles: Iterable[Particle]) -> List[Particle]:
        pass

    @abstractmethod
    def calculate_weight(self, particle: Particle) -> float:
        pass

    @abstractmethod
    def save_debug_plot(self, name: str):
        pass


class RayTracingSensorModel(SensorModel):
    """
    A sensor model based on ray tracing.

    Also consider ParallelRayTracingSensorModel, which is much faster because it uses multiple cores.
    """

    MAX_DISTANCE: float = 3.0
    """
    Ignore any obstacles beyond this distance (instead assume the LIDAR wouldn't see anything,
    and would return 0).
    """

    map_obstacles: np.array = None
    """ (n, 2)-sized matrix of x, y coordinates of occupied squares (ie. obstacles) on the map. """

    debug_data_dir: Path
    """ Folder to store debugging images (see save_debug_plot). Defaults to __file__/../particle_sensor_data. """

    # Data for debug plots
    weight: float = 0.0
    particle: Particle = Particle(0, 0, 0, 0)
    obstacle_rs: np.array = np.array()
    obstacle_thetas: np.array = np.array()
    lidar_expected: np.array = np.array()

    def __init__(self, map: OccupancyGrid, debug_data_dir: Path = Path(__file__).parent.parent / 'particle_sensor_data'):
        self.map_obstacles = self.preprocess_map(map)

        self.debug_data_dir = debug_data_dir
        self.debug_data_dir.mkdir(exist_ok=True)

    def weight_particles(self, particles: Iterable[Particle]) -> List[Particle]:
        """ Re-weight a set of particles using the sensor model. Does not mutate its input. """
        return [
            Particle(p.x, p.y, p.theta, self.calculate_weight(p))
            for p in particles
        ]

    def calculate_weight(self, particle: Particle) -> float:
        """
        Use the sensor model to figure out how likely it is that the robot was at the particle given
        the most recent LIDAR data.

        Think of this as a pure method, although it isn't actually: it stores some internal state as
        instance variables on the model class. This enables you to call save_debug_plot immediately
        afterwards to get a pretty graph showing the internal state of the model.

        This method is thread-safe, although save_debug_plot is unsupported with any form of
        parallelism.
        """
        # Take map data as cartesian coords, and shift to center at particle
        # NB: Both the map and all particles are in the `map` frame
        obstacles_shifted = self.map_obstacles - \
            [particle.x, particle.y]

        # Convert to polar coordinates
        obstacle_rs = np.linalg.norm(obstacles_shifted, axis=1)
        mask = obstacle_rs < self.MAX_DISTANCE  # ignore any obstacles too far away
        obstacle_rs = np.concatenate((
            obstacle_rs[mask],
            # Add some an obstacle very far away for all possible angles, so there's something at
            # every angle.
            np.full(360, 10.0)
        ))

        # Calculate the angle of each obstacle, rotating by the particle's heading
        obstacle_thetas_rad = normalize_angle(
            np.arctan2(
                obstacles_shifted[mask][:, 1],
                obstacles_shifted[mask][:, 0]
            ) - particle.theta  # Rotate by particle's heading
        )

        # Convert to degrees, and descretize to whole-degree increments (like LIDAR data)
        # This is the only place we use degrees, but it's helpful since LIDAR is indexed by degree
        obstacle_thetas = np.concatenate((
            np.rad2deg(obstacle_thetas_rad).round(),
            # Make sure there's something for every angle here too
            np.arange(360)
        ))

        # We've normalized angels to [-180, 180], so shift to [0, 360]
        obstacle_thetas[obstacle_thetas < 0.0] += 360.0

        # Find the closest obstacle at each angle

        order = obstacle_thetas.argsort()  # returns indexes in order by sorted value
        # NOTE: obstacles_by_angle is a *normal list* of arrays, where the array at index i is the
        # (expected) LIDAR distance values at angle i.
        splits = np.unique(obstacle_thetas[order], return_index=True)[1][1:]
        obstacles_by_angle = np.split(obstacle_rs[order], splits)

        # Find the nearest obstacle at each angle
        lidar_expected = np.array([
            np.minimum.reduce(
                a,
                initial=10.0,
                axis=0
            ) for a in obstacles_by_angle  # can't figure out how to replace this loop with numpy magic
        ])

        # Account for LIDAR's max range
        lidar_expected[lidar_expected > self.MAX_DISTANCE] = 0.0

        # Compare to LIDAR data
        lidar_diff = np.abs(self.last_lidar - lidar_expected)

        # Calculate weight
        weight = np.sum(
            (
                0.5 * (
                    np.exp(-(lidar_diff[lidar_diff > 0.0] ** 2) / 0.01)
                )
            ) ** 3
        )

        # Save data for generating debug plots
        self.weight = weight
        self.particle = particle
        self.obstacle_rs = obstacle_rs
        self.obstacle_thetas = obstacle_thetas
        self.lidar_expected = lidar_expected

        return weight

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
        ax.plot(np.deg2rad(self.obstacle_thetas), self.obstacle_rs, 'b,')
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

        print("Num Map Obstacles:", len(occupied), '!\n\n')

        return occupied

##
# Helpers for multi-process raytracing.
##


_worker_ray_tracer: Optional[RayTracingSensorModel] = None
# this is intentionally *not* a multiprocessing Lock, it's a threading lock
# ie. there's one per process
_lock: Lock = Lock()


def _setup_worker_process(map):
    with _lock:
        global _worker_ray_tracer
        # XXX: It would be much more efficient to process the map only once,
        # but we only need to process the map at the start of the program so
        # it's not terribly important.
        _worker_ray_tracer = RayTracingSensorModel(map)


def _ray_trace_particle(data):
    with _lock:
        if _worker_ray_tracer is None:
            raise ValueError("Worker process hasn't been setup!")

        p, lidar = data
        _worker_ray_tracer.set_lidar(lidar)
        return Particle(p.x, p.y, p.theta, _worker_ray_tracer.calculate_weight(p))


class ParallelRayTracingSensorModel(SensorModel):
    """
    Version of RayTracingSensorModel that ray-traces in parallel, using the multiprocessing module.

    Because ray tracing is embarassingly parallel, this is *much* faster.

    Do not use more than one ParallelRayTracingSensorModel at once.
    """

    executor: Executor

    def __init__(self, map: OccupancyGrid):
        self.executor = ProcessPoolExecutor(
            initializer=_setup_worker_process,
            initargs=(map,)
        )

    def __del__(self):
        self.executor.shutdown()

    def weight_particles(self, particles: Iterable[Particle]) -> List[Particle]:
        # XXX: It's inefficient to pass the LIDAR data through once for each particle,
        # but there isn't a good way with the Executor API to provide contextual data.
        return list(self.executor.map(
            _ray_trace_particle,
            ((p, self.last_lidar) for p in particles)
        ))

    def calculate_weight(self, particle: Particle) -> float:
        raise NotImplementedError(
            "Don't call calculate_weight on ParallelRayTracingSensorModel!"
        )

    _has_done_debug_plot_warning = False

    def save_debug_plot(self, name: str):
        if not self._has_done_debug_plot_warning:
            print("WARNING: can't print debug plot in parallel ray tracing mode")
            self._has_done_debug_plot_warning = True

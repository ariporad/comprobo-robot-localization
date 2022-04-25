import numpy as np

from typing import List
from helper_functions import PoseTuple, Particle, rotation_matrix, RelativeRandomSampler


class MotionModel:
    """
    A fairly simple motion model, which simply takes odometry data and adds some random error.
    """
    error_sampler: RelativeRandomSampler

    def __init__(self, stddev: float):
        self.error_sampler = RelativeRandomSampler(stddev)

    def apply(self, particle: Particle, delta_pose: PoseTuple) -> Particle:
        dx_robot = self.error_sampler.sample(delta_pose.x)
        dy_robot = self.error_sampler.sample(delta_pose.y)
        dtheta = self.error_sampler.sample(delta_pose.theta)

        dx, dy = np.matmul(rotation_matrix(
            particle.theta), [dx_robot, dy_robot])

        return Particle(
            x=particle.x - dx,
            y=particle.y - dy,
            theta=particle.theta - dtheta,
            weight=particle.weight
        )

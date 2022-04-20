
from collections import namedtuple
from itertools import islice

import matplotlib.pyplot as plt

from pathlib import Path
import math
import time
from typing import Iterable, Optional, Tuple, List
import rospy
import random
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, PoseStamped, Pose
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion

import numpy as np
from numpy.random import default_rng, Generator

from helper_functions import TFHelper,  PoseTuple, Particle, RandomSampler, rotation_matrix, RelativeRandomSampler, print_time
from sensor_model import SensorModel
from occupancy_field import OccupancyField


class MotionModel:
    error_sampler: RelativeRandomSampler

    def __init__(self, stddev: float):
        self.error_sampler = RelativeRandomSampler(stddev)

    def apply(self, particles: List[Particle], delta_pose: PoseTuple) -> List[Particle]:
        dx_robot = self.error_sampler.sample(delta_pose.x)
        dy_robot = self.error_sampler.sample(delta_pose.y)
        dtheta = self.error_sampler.sample(delta_pose.theta)

        dx, dy = np.matmul(rotation_matrix(dtheta), [dx_robot, dy_robot])

        return [
            Particle(
                x=p.x + dx,
                y=p.y + dy,
                theta=p.theta - dtheta,
                weight=p.weight
            )
            for p in particles
        ]

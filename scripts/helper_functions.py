""" Some convenience functions for translating between various representations
    of a robot pose. """

from time import perf_counter
import rospy
from typing import Optional, Tuple

from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion

import tf.transformations as t
import tf2_ros
from tf import TransformListener
from tf import TransformBroadcaster

import numpy as np
from numpy.random import default_rng, Generator

import math
from contextlib import contextmanager
from collections import namedtuple

rng = default_rng()

# NB: All particles are in the `map` frame


class Particle(namedtuple('Particle', ['x', 'y', 'theta', 'weight'])):
    def __repr__(self):
        return f"Particle(x={self.x:.3f}, y={self.y:.3f}, theta={self.theta:.3f}, w={self.weight:.6f})"


class PoseTuple(namedtuple('PoseTuple', ['x', 'y', 'theta'])):
    def __repr__(self):
        return f"PoseTuple(x={self.x:.3f}, y={self.y:.3f}, theta={self.theta:.3f})"


def normalize_angle(angle):
    """ Normalize angle to [-pi, pi]. Works on floats or numpy arrays."""
    return np.arctan2(np.sin(angle), np.cos(angle))


class RandomSampler:
    stddev: float
    noise: float
    noise_range: Optional[Tuple[float, float]]

    def __init__(self, stddev: float, noise: float = 0.0, noise_range: Optional[Tuple[float, float]] = None):
        self.stddev = stddev
        self.noise = noise

        if self.noisy and noise_range is None:
            raise ValueError("Must provide noise_range!")

        self.noise_range = noise_range

    def sample(self, value: float) -> float:
        if self.noisy and rng.random() < self.noise:
            return rng.uniform(*self.noise_range)
        else:
            return rng.normal(value, self.stddev)

    @property
    def noisy(self):
        self.noise != 0.0


class RelativeRandomSampler:
    """ Just like RandomSampler, but stddev, noise, and noise_range are all proportions of the value being sampled. """
    stddev: float
    noise: float
    noise_range: Optional[Tuple[float, float]]

    def __init__(self, stddev: float, noise: float = 0.0, noise_range: Optional[Tuple[float, float]] = None):
        self.stddev = stddev
        self.noise = noise

        if self.noisy and noise_range is None:
            raise ValueError("Must provide noise_range!")

        self.noise_range = noise_range

    def sample(self, value: float) -> float:
        if self.noisy and rng.random() < self.noise:
            return rng.uniform(self.noise_range[0] * value, self.noise_range[1] * value)
        else:
            sign = signum(value)
            value_abs = abs(value)
            return sign * rng.normal(value_abs, self.stddev * value_abs)

    @property
    def noisy(self):
        self.noise != 0.0


@contextmanager
def print_time(name: str = "Timer"):
    start = perf_counter()
    yield
    duration = perf_counter() - start
    print(f"Timer '{name}' took {duration:.2f}s   ")


def signum(a: float) -> float:
    if a > 0.0:
        return 1.0
    elif a < 0.0:
        return -1.0
    else:
        return 0.0


class TFHelper(object):
    """ TFHelper Provides functionality to convert poses between various
        forms, compare angles in a suitable way, and publish needed
        transforms to ROS """

    def __init__(self):
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener2 = tf2_ros.TransformListener(self.tf_buf)
        self.tf_listener = TransformListener()
        self.tf_broadcaster = TransformBroadcaster()

    def convert_xy_and_theta_to_pose(self, xytheta):
        """ Convert a (x, y, theta) tuple to a geometry_msgs/Pose message. """
        x, y, theta = xytheta
        return self.convert_translation_rotation_to_pose((x, y, 0), t.quaternion_from_euler(0, 0, theta))

    def convert_translation_rotation_to_pose(self, translation, rotation):
        """ Convert from representation of a pose as translation and rotation
            (Quaternion) tuples to a geometry_msgs/Pose message """
        return Pose(position=Point(x=translation[0],
                                   y=translation[1],
                                   z=translation[2]),
                    orientation=Quaternion(x=rotation[0],
                                           y=rotation[1],
                                           z=rotation[2],
                                           w=rotation[3]))

    def convert_pose_inverse_transform(self, pose):
        """ This is a helper method to invert a transform (this is built into
            the tf C++ classes, but ommitted from Python) """
        transform = t.concatenate_matrices(
            t.translation_matrix([pose.position.x,
                                  pose.position.y,
                                  pose.position.z]),
            t.quaternion_matrix([pose.orientation.x,
                                 pose.orientation.y,
                                 pose.orientation.z,
                                 pose.orientation.w]))
        inverse_transform_matrix = t.inverse_matrix(transform)
        return (t.translation_from_matrix(inverse_transform_matrix),
                t.quaternion_from_matrix(inverse_transform_matrix))

    def convert_pose_to_xy_and_theta(self, pose):
        """ Convert pose (geometry_msgs.Pose) to a (x,y,yaw) tuple """
        orientation_tuple = (pose.orientation.x,
                             pose.orientation.y,
                             pose.orientation.z,
                             pose.orientation.w)
        angles = t.euler_from_quaternion(orientation_tuple)
        return (pose.position.x, pose.position.y, angles[2])

    def angle_normalize(self, z):
        """ convenience function to map an angle to the range [-pi,pi] """
        return math.atan2(math.sin(z), math.cos(z))

    def angle_diff(self, a, b):
        """ Calculates the difference between angle a and angle b (both should
            be in radians) the difference is always based on the closest
            rotation from angle a to angle b.
            examples:
                angle_diff(.1,.2) -> -.1
                angle_diff(.1, 2*math.pi - .1) -> .2
                angle_diff(.1, .2+2*math.pi) -> -.1
        """
        a = self.angle_normalize(a)
        b = self.angle_normalize(b)
        d1 = a-b
        d2 = 2*math.pi - math.fabs(d1)
        if d1 > 0:
            d2 *= -1.0
        if math.fabs(d1) < math.fabs(d2):
            return d1
        else:
            return d2

    def fix_map_to_odom_transform(self, stamp, robot_pose):
        """ This method constantly updates the offset of the map and
            odometry coordinate systems based on the latest results from
            the localizer.

            robot_pose should be of type geometry_msgs/Pose in the base_link frame, 
            and timestamp is of type rospy.Time and represents the time at which the
            robot's pose corresponds.
            """
        (translation_bl2map, rotation_bl2map) = \
            self.convert_pose_inverse_transform(robot_pose)
        pose_map_in_bl = PoseStamped(
            pose=self.convert_translation_rotation_to_pose(translation_bl2map,
                                                           rotation_bl2map),
            header=Header(stamp=stamp, frame_id='base_link'))
        self.tf_listener.waitForTransform('base_link',
                                          'odom',
                                          stamp,
                                          rospy.Duration(0.2))
        self.odom_to_map = self.tf_listener.transformPose(
            'odom', pose_map_in_bl)
        # self.odom_to_map = self.tf_buf.transform(
        #     pose_map_in_bl, 'odom', rospy.Duration(0.25))
        # self.odom_to_map = self.tf_listener.transformPose(
        #     'odom', pose_map_in_bl)
        (self.translation, self.rotation) = \
            self.convert_pose_inverse_transform(self.odom_to_map.pose)

        # self.translation = [
        #     self.translation[0],
        #     self.translation[1] * -1,
        #     self.translation[2],
        # ]

        # print("updated map ref frame:", self.translation, self.rotation)

    def send_last_map_to_odom_transform(self):
        if not(hasattr(self, 'translation') and hasattr(self, 'rotation')):
            return
        self.tf_broadcaster.sendTransform(self.translation,
                                          self.rotation,
                                          rospy.get_rostime(),
                                          'odom',
                                          'map')

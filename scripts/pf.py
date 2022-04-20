#!/usr/bin/env python3

""" This is the starter code for the robot localization project """

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
from motion_model import MotionModel
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, PoseStamped, Pose
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion

import numpy as np
from numpy.random import default_rng, Generator

from helper_functions import TFHelper,  PoseTuple, Particle, RandomSampler, RelativeRandomSampler, print_time
from sensor_model import SensorModel
from occupancy_field import OccupancyField

rng: Generator = default_rng()


class ParticleFilter:
    """
    The class that represents a Particle Filter ROS Node
    """

    particle_sampler_xy = RandomSampler(0.25, 0.1, (-5, 5))
    particle_sampler_theta = RandomSampler(0.15 * math.pi, 0)

    motion_model = MotionModel(stddev=.15)
    sensor_model = SensorModel()

    NUM_PARTICLES = 300

    sensor_model: SensorModel

    particles: List[Particle] = None
    robot_pose: PoseStamped = None

    last_pose: PoseTuple = None

    map_obstacles: np.array
    tf_listener: tf2_ros.TransformListener

    is_updating: bool = False

    def __init__(self):
        rospy.init_node('pf')
        self.last_update = rospy.Time.now()
        self.particles_stamp = rospy.Time.now()

        self.update_count = 0

        # create instances of two helper objects that are provided to you
        # as part of the project
        self.occupancy_field = OccupancyField()  # NOTE: hangs if a map isn't published
        self.transform_helper = TFHelper()
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

        # publisher for the particle cloud for visualizing in rviz.
        self.map_pub = rospy.Publisher("parsed_map",
                                       PoseArray,
                                       queue_size=10)
        self.particle_pub = rospy.Publisher("particlecloud",
                                            PoseArray,
                                            queue_size=10)

        rospy.wait_for_service("static_map")
        get_static_map = rospy.ServiceProxy("static_map", GetMap)
        self.sensor_model.set_map(get_static_map().map)

        # IMPORTANT: Register subscribers last, so callbacks can't happen before ready
        # pose_listener responds to selection of a new approximate robot
        # location (for instance using rviz)
        rospy.Subscriber("initialpose",
                         PoseWithCovarianceStamped,
                         self.on_initial_pose)

        rospy.Subscriber("odom", Odometry, self.on_odom)
        rospy.Subscriber("stable_scan", LaserScan, self.on_lidar)

    def on_initial_pose(self, msg: PoseWithCovarianceStamped):
        """ Callback function to handle re-initializing the particle filter
            based on a pose estimate.  These pose estimates could be generated
            by another ROS Node or could come from the rviz GUI """
        x, y, theta = self.transform_helper.convert_pose_to_xy_and_theta(
            msg.pose.pose)

        particles = self.resample_particles([Particle(x, y, theta, 1)])

        self.set_particles(msg.header.stamp, particles)

    def on_lidar(self, msg: LaserScan):
        self.sensor_model.set_lidar_ranges(msg.ranges)

    def on_odom(self, msg: Odometry):
        pose = PoseTuple(
            *self.transform_helper.convert_pose_to_xy_and_theta(msg.pose.pose))

        if self.last_pose is None:
            self.last_pose = pose
            return

        delta_pose = PoseTuple(
            self.last_pose[0] - pose[0],
            self.last_pose[1] - pose[1],
            self.transform_helper.angle_diff(
                self.last_pose[2], pose[2])
        )

        # Make sure we've moved at least a bit
        if math.sqrt((delta_pose[0] ** 2) + (delta_pose[1] ** 2)) < 0.01 and delta_pose[2] < 0.05:
            return

        if self.update(msg.header.stamp, delta_pose):
            self.last_pose = pose

    def update(self, stamp: rospy.Time, delta_pose: PoseTuple) -> bool:
        # Ignore any updates that happened while we were working on the last update (allows queue to drain)
        if stamp < self.last_update:
            return False

        # Require previous particles (initialized by initial pose)
        if self.particles is None:
            return False

        # We don't do multiple updates in parallel
        if self.is_updating:
            return False

        self.is_updating = True
        try:
            with print_time('Updating'):
                # Resample Particles
                particles = self.resample_particles(self.particles)

                # Apply Motion Model
                particles = self.motion_model.apply(particles, delta_pose)

                # Update Weights Based on Sensor Model
                particles = [
                    Particle(p.x, p.y, p.theta,
                             self.sensor_model.calculate_weight(p))
                    for p in particles
                ]

                # Set Particles
                self.set_particles(stamp, particles)

                self.last_update = rospy.Time.now()
        finally:
            self.is_updating = False

        return True

    def resample_particles(self, particles: List[Particle], k: int = None) -> List[Particle]:
        if k is None:
            k = self.NUM_PARTICLES

        choices = random.choices(
            particles,
            weights=[p.weight for p in particles],
            k=k
        )

        return [
            Particle(
                x=self.particle_sampler_xy.sample(choice.x),
                y=self.particle_sampler_xy.sample(choice.y),
                theta=self.particle_sampler_theta.sample(choice.theta),
                weight=1
            )
            for choice in choices
        ]

    def set_particles(self, stamp: rospy.Time, particles: List[Particle]):
        self.particles = self.normalize_weights(particles)
        self.particles_stamp = stamp

        # NB: Particles are always in the map reference frame
        robot_pose = np.average([  # TODO: should this be median?
            (particle.x, particle.y, particle.theta)
            for particle in self.particles
        ], axis=0, weights=[p.weight for p in particles])

        self.transform_helper.fix_map_to_odom_transform(
            stamp,
            self.transform_helper.convert_xy_and_theta_to_pose(robot_pose)
        )

    def visualize_particles(self):
        # Publish particles
        poses = PoseArray()
        poses.header.stamp = self.particles_stamp
        poses.header.frame_id = 'map'
        poses.poses = [
            self.transform_helper.convert_xy_and_theta_to_pose(
                (particle.x, particle.y, particle.theta)
            )
            for particle in self.particles
        ]
        self.particle_pub.publish(poses)

        # particles = list(
        #     sorted(list(random.choices(self.particles, k=30)), key=lambda p: p.weight))
        # for i, particle in enumerate(particles):
        #     # self.sensor_model.calculate_weight(particle)
        #     # self.sensor_model.save_debug_plot(
        #     #     f"particle_{self.update_count:03d}")
        #     self.update_count += 1

    def normalize_weights(self, particles: List[Particle]):
        total = sum(p.weight for p in particles)
        return [
            Particle(p.x, p.y, p.theta, p.weight / total)
            for p in particles
        ]

    def run(self):
        r = rospy.Rate(5)

        while not rospy.is_shutdown():
            # in the main loop all we do is continuously broadcast the latest
            # map to odom transform
            self.transform_helper.send_last_map_to_odom_transform()
            r.sleep()


if __name__ == '__main__':
    n = ParticleFilter()
    n.run()

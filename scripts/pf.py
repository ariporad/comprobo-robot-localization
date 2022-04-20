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
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, PoseStamped, Pose
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion

import numpy as np
from numpy.random import default_rng, Generator

from helper_functions import TFHelper,  PoseTuple, Particle, RandomSampler, RelativeRandomSampler
from sensor_model import SensorModel
from occupancy_field import OccupancyField

rng: Generator = default_rng()


"""
# The Plan

We need:
- Initial State Model: P(x0)
    - initialpose topic + uncertainty
- Motion Model (odometry): P(Xt | X(t-1), Ut)
    - This needs to be odometry + uncertainty
- Sensor Model: P(Zt | xt)
    - Based on occupancy model + flat noise uncertainty + normal distribution

Setup:
1. Create initial particles:
    - Weighted random sample from p(x0)
       - How to do 2d weighted random sample properly?
       - We're going to do this wrong to start, by doing X and Y seperately

Repeat:
2. Resample particles, using weights as the distribution
3. Update each particle with the motion model (odometry)
    - Figure out where odom is now (convert (0,0,0):base_link -> odom)
    - Compare that with last cycle to get delta_odom
    - For each particle, update x/y/theta by a random sample of delta_odom
4. Compute weights: likelyhood that we would have gotten the laser data if we were at each particle
    - Use the occupancy field for this
    - Normalize weights to 1
5. Goto Step 2

Notes:
- Convention for normal distributions: sigma is stddev, noise the proportion of time to pick a random value

"""


class ParticleFilter:
    """
    The class that represents a Particle Filter ROS Node
    """

    particle_sampler_xy = RandomSampler(0.25, 0.1, (-5, 5))
    particle_sampler_theta = RandomSampler(0.15 * math.pi, 0)

    motion_error_sampler = RelativeRandomSampler(.15)

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
        self.sensor_model = SensorModel.from_map(get_static_map().map)

        # IMPORTANT: Register subscribers last, so callbacks can't happen before ready
        # pose_listener responds to selection of a new approximate robot
        # location (for instance using rviz)
        rospy.Subscriber("initialpose",
                         PoseWithCovarianceStamped,
                         self.update_initial_pose)

        rospy.Subscriber("odom", Odometry, self.on_odom)
        rospy.Subscriber("stable_scan", LaserScan, self.on_lidar)

    def update_initial_pose(self, msg: PoseWithCovarianceStamped):
        """ Callback function to handle re-initializing the particle filter
            based on a pose estimate.  These pose estimates could be generated
            by another ROS Node or could come from the rviz GUI """
        x, y, theta = self.transform_helper.convert_pose_to_xy_and_theta(
            msg.pose.pose)

        particles = self.sample_particles([Particle(x, y, theta, 1)])

        self.set_particles(msg.header.stamp, particles)

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

        print("Delta pose:", delta_pose)

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
        start_time = time.perf_counter()
        try:
            # Resample Particles
            particles = self.sample_particles(self.particles)
            # particles = list(self.particles)

            # Apply Motion
            particles = self.apply_motion(particles, delta_pose)

            particles = [
                Particle(p.x, p.y, p.theta,
                         self.sensor_model.calculate_weight(p))
                for p in particles
            ]

            self.set_particles(stamp, particles)
            self.last_update = rospy.Time.now()
        finally:
            self.is_updating = False

            duration_ms = (time.perf_counter() - start_time) * 1000
            print(f"Update took {duration_ms:.2f}ms.\n")

        return True

    def apply_motion(self, particles: List[Particle], delta_pose: PoseTuple, sigma: float) -> List[Particle]:
        dx_robot = self.motion_error_sampler.sample(delta_pose.x)
        dy_robot = self.motion_error_sampler.sample(delta_pose.y)
        dtheta = self.motion_error_sampler.sample(delta_pose.theta)

        rot_dtheta = np.array([
            [np.cos(dtheta), -np.sin(dtheta)],
            [np.sin(dtheta), np.cos(dtheta)]
        ])

        dx, dy = np.matmul(rot_dtheta, [dx_robot, dy_robot])

        return [
            Particle(
                x=p.x + dx,
                y=p.y + dy,
                theta=p.theta - dtheta,
                weight=p.weight
            )
            for p in particles
        ]

    def sample_particles(self, particles: List[Particle], k: int = None) -> List[Particle]:
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
        # self.particles = particles
        self.particles = self.normalize_weights(particles)

        # if self.tf_buf.can_transform('base_link', 'odom', stamp, rospy.Duration(1)) or True:
        # Calculate robot pose / map frame
        # NB: Particles are always in the map reference frame
        robot_pose = self.transform_helper.convert_xy_and_theta_to_pose(np.average([  # TODO: should this be median?
            (particle.x, particle.y, particle.theta)
            for particle in self.particles
        ], axis=0, weights=[p.weight for p in particles]))
        # robot_pose is where the robot is in map
        # By definition, that's also (0, 0, 0) in base_link
        # So the inverse of robot_pose is the transformation between base_link and map
        # Subtract whatever the tranformation from base_link to odom is, and you get odom->map

        self.transform_helper.fix_map_to_odom_transform(stamp, robot_pose)

        # Publish particles
        poses = PoseArray()
        poses.header.stamp = stamp
        poses.header.frame_id = 'map'

        particles = list(
            sorted(list(random.choices(self.particles, k=30)), key=lambda p: p.weight))
        for i, particle in enumerate(particles):
            # for particle in self.particles:
            poses.poses.append(
                self.transform_helper.convert_xy_and_theta_to_pose(
                    (particle.x, particle.y, particle.theta)
                ))
            # self.sensor_model.calculate_weight(particle)
            # self.sensor_model.save_debug_plot(
            #     f"particle_{self.update_count:03d}")
        self.update_count += 1

        self.particle_pub.publish(poses)

    def normalize_weights(self, particles: List[Particle]):
        total = sum(p.weight for p in particles)
        return [
            Particle(p.x, p.y, p.theta, p.weight / total)
            for p in particles
        ]

    def on_lidar(self, msg: LaserScan):
        self.sensor_model.set_lidar_ranges(msg.ranges)

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

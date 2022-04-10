#!/usr/bin/env python3

""" This is the starter code for the robot localization project """

from collections import namedtuple

import math
from secrets import choice
from typing import Iterable, Optional, Tuple, List
import rospy
import random
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, PoseStamped
import tf2_ros
import tf2_geometry_msgs

import numpy as np
from numpy.random import default_rng, Generator

from helper_functions import TFHelper, sample_normal_error
from occupancy_field import OccupancyField

PoseTuple = namedtuple('PoseTuple', ['x', 'y', 'theta'])
Particle = namedtuple('Particle', ['x', 'y', 'theta', 'weight'])
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

    INITIAL_STATE_SIGMA = 0.5
    INITIAL_STATE_NOISE = 0.25

    NUM_PARTICLES = 100

    particles: List[Particle] = None
    robot_pose: PoseStamped = None

    last_odom: PoseTuple = None
    last_lidar: Optional[LaserScan] = None

    def __init__(self):
        rospy.init_node('pf')
        self.last_update = rospy.Time.now()

        # create instances of two helper objects that are provided to you
        # as part of the project
        self.occupancy_field = OccupancyField()  # NOTE: hangs if a map isn't published
        self.transform_helper = TFHelper()
        self.tf_buf = tf2_ros.Buffer()

        # pose_listener responds to selection of a new approximate robot
        # location (for instance using rviz)
        rospy.Subscriber("initialpose",
                         PoseWithCovarianceStamped,
                         self.update_initial_pose)

        rospy.Subscriber("odom", Odometry, self.update)
        rospy.Subscriber("stable_scan", LaserScan, self.on_lidar)

        # publisher for the particle cloud for visualizing in rviz.
        self.particle_pub = rospy.Publisher("particlecloud",
                                            PoseArray,
                                            queue_size=10)

    def update_initial_pose(self, msg: PoseWithCovarianceStamped):
        """ Callback function to handle re-initializing the particle filter
            based on a pose estimate.  These pose estimates could be generated
            by another ROS Node or could come from the rviz GUI """
        x, y, theta = self.transform_helper.convert_pose_to_xy_and_theta(
            msg.pose.pose)

        particles = self.sample_particles(
            [Particle(x, y, theta, 1)],
            self.INITIAL_STATE_SIGMA, self.INITIAL_STATE_NOISE, self.NUM_PARTICLES)

        self.set_particles(rospy.Time.now(), particles)

    def update(self, msg: Odometry):
        last_odom = self.last_odom
        odom = self.transform_helper.convert_pose_to_xy_and_theta(
            msg.pose.pose)
        self.last_odom = odom

        if last_odom is None or self.particles is None:
            return

        now = rospy.Time.now()

        # Resample Particles
        particles = self.sample_particles(
            self.particles,
            self.INITIAL_STATE_SIGMA, self.INITIAL_STATE_NOISE, self.NUM_PARTICLES)

        # Apply Motion
        delta_pose = PoseTuple(
            odom[0] - last_odom[0],
            odom[1] - last_odom[1],
            self.transform_helper.angle_diff(odom[2], last_odom[2])
        )
        particles = self.apply_motion(particles, delta_pose, 0.05)
        particles = [
            Particle(p.x, p.y, p.theta, self.calculate_sensor_weight(p))
            for p in particles
        ]

        self.set_particles(now, particles)

    def calculate_sensor_weight(self, particle: Particle) -> float:
        if self.last_lidar is None:
            print("No LIDAR data!")
            return 1.0

        closest_actual = min(r for r in self.last_lidar.ranges if r > 0)
        closest_expected = self.occupancy_field.get_closest_obstacle_distance(
            particle.x, particle.y)

        if math.isnan(closest_expected):
            return 0.0

        return (1.0 / (closest_actual - closest_expected)) ** 3

    def apply_motion(self, particles: List[Particle], delta_pose: PoseTuple, sigma: float) -> List[Particle]:
        return [
            Particle(
                x=p.x + sample_normal_error(delta_pose.x, sigma),
                y=p.y + sample_normal_error(delta_pose.y, sigma),
                theta=p.theta + sample_normal_error(delta_pose.theta, sigma),
                weight=p.weight
            )
            for p in particles
        ]

    def sample_particles(self, particles: List[Particle], sigma: float, noise: float, k: int) -> List[Particle]:
        choices = random.choices(
            particles,
            weights=[p.weight for p in particles],
            k=k
        )

        return [
            Particle(
                x=rng.normal(choice.x, sigma),
                y=rng.normal(choice.y, sigma),
                theta=rng.normal(choice.theta, sigma),
                weight=1
            )
            for choice in choices
        ]

    def set_particles(self, stamp: rospy.Time, particles: List[Particle]):
        self.last_update = stamp
        self.particles = self.normalize_weights(particles)

        if self.tf_buf.can_transform('base_link', 'odom', stamp, rospy.Duration(1)) or True:
            # Calculate robot pose / map frame
            robot_pose = self.transform_helper.convert_xy_and_theta_to_pose(np.mean([  # TODO: should this be median?
                (particle.x, particle.y, particle.theta)
                for particle in self.particles
            ], axis=0))
            self.transform_helper.fix_map_to_odom_transform(stamp, robot_pose)

        # Publish particles
        poses = PoseArray()
        poses.header.stamp = stamp
        poses.header.frame_id = 'map'
        poses.poses = [
            self.transform_helper.convert_xy_and_theta_to_pose(
                (particle.x, particle.y, particle.theta))
            for particle in self.particles
        ]
        self.particle_pub.publish(poses)

    def normalize_weights(self, particles: List[Particle]):
        total = sum(p.weight for p in particles)
        return [
            Particle(p.x, p.y, p.theta, p.weight / total)
            for p in particles
        ]

    def on_lidar(self, msg: LaserScan):
        self.last_lidar = msg

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

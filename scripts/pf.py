#!/usr/bin/env python3

""" This is the starter code for the robot localization project """

from collections import namedtuple
from math import pi
from typing import Iterable, Optional, Tuple, List
import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, PoseStamped
import tf2_ros
import tf2_geometry_msgs

import numpy as np
from numpy.random import default_rng, Generator

from helper_functions import TFHelper
from occupancy_field import OccupancyField

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
2. Update each particle with the motion model (odometry)
3. Compute weights: likelyhood that we would have gotten the laser data if we were at each particle
    - Use the occupancy field for this
    - Normalize weights to 1
4. Resample particles, using weights as the distribution
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

    particles: Optional[List[Particle]] = None
    robot_pose: Optional[PoseStamped] = None

    def __init__(self):
        rospy.init_node('pf')

        # create instances of two helper objects that are provided to you
        # as part of the project
        # self.occupancy_field = OccupancyField()  # NOTE: hangs if a map isn't published
        self.transform_helper = TFHelper()
        self.tf_buf = tf2_ros.Buffer()

        # pose_listener responds to selection of a new approximate robot
        # location (for instance using rviz)
        rospy.Subscriber("initialpose",
                         PoseWithCovarianceStamped,
                         self.update_initial_pose)

        # publisher for the particle cloud for visualizing in rviz.
        self.particle_pub = rospy.Publisher("particlecloud",
                                            PoseArray,
                                            queue_size=10)

    def update_initial_pose(self, msg: PoseWithCovarianceStamped):
        """ Callback function to handle re-initializing the particle filter
            based on a pose estimate.  These pose estimates could be generated
            by another ROS Node or could come from the rviz GUI """
        x, y, theta = \
            self.transform_helper.convert_pose_to_xy_and_theta(msg.pose.pose)

        particles = list(self.sample_particles(
            (x, y, theta),
            self.INITIAL_STATE_SIGMA, self.INITIAL_STATE_NOISE, self.NUM_PARTICLES))

        # For some reason, passing through the time prevents anything from working
        self.set_particles(rospy.Time.now(), particles)

        # Use the helper functions to fix the transform

    def sample_particles(self, pose: Tuple[float, float, float], sigma: float, noise: float, num: int) -> Iterable[Particle]:
        x, y, theta = pose

        for _ in range(num):
            if rng.random() < noise:
                pass  # noise isn't implemented yet

            yield Particle(
                x=rng.normal(x, sigma),
                y=rng.normal(y, sigma),
                theta=rng.normal(theta, sigma),
                weight=1
            )

    def set_particles(self, stamp: rospy.Time, particles: Iterable[Particle]):
        self.particles = list(particles)

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

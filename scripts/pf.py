#!/usr/bin/env python3

import math
import rospy
import random

import numpy as np
from typing import List

import tf2_ros
import tf2_geometry_msgs  # Importing for side-effects
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray

from sensor_model import SensorModel
from motion_model import MotionModel

from helper_functions import TFHelper,  PoseTuple, Particle, RandomSampler, print_time


class ParticleFilter:
    NUM_PARTICLES = 300

    particle_sampler_xy = RandomSampler(0.25, 0.1, (-5, 5))
    particle_sampler_theta = RandomSampler(0.15 * math.pi, 0)

    motion_model = MotionModel(stddev=.15)
    sensor_model = SensorModel()

    # Don't update unless we've moved a bit
    UPDATE_MIN_DISTANCE: float = 0.01
    UPDATE_MIN_ROTATION: float = 0.05

    last_pose: PoseTuple = None

    particles: List[Particle] = None
    particles_stamp: rospy.Time
    """ Timestamp of the odometry update that generated the current particles. """

    update_count: int = 0
    is_updating: bool = False
    last_update: rospy.Time
    """ Timestamp at which the last update *finished.* """

    particle_pub: rospy.Publisher

    tf_listener: tf2_ros.TransformListener
    tf_buf: tf2_ros.Buffer
    transform_helper: TFHelper = TFHelper()

    def __init__(self):
        rospy.init_node('pf')

        self.last_update = rospy.Time.now()
        self.particles_stamp = rospy.Time.now()

        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

        # publisher for the particle cloud for visualizing in rviz.
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
        """ Callback to (re-)initialize the particle filter whenever an initial pose is set. """
        x, y, theta = \
            self.transform_helper.convert_pose_to_xy_and_theta(msg.pose.pose)

        particles = self.resample_particles([Particle(x, y, theta, 1)])

        self.set_particles(msg.header.stamp, particles)

    def on_lidar(self, msg: LaserScan):
        """ Callback whenever new LIDAR data is available. """
        self.sensor_model.set_lidar_ranges(msg.ranges)

    def on_odom(self, msg: Odometry):
        """ Callback whenever new odometry data is available. """
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
        if math.sqrt((delta_pose[0] ** 2) + (delta_pose[1] ** 2)) < self.update_min_distance \
                and delta_pose[2] < self.update_min_rotation:
            return

        if self.update(msg.header.stamp, delta_pose):
            self.last_pose = pose

    def update(self, stamp: rospy.Time, delta_pose: PoseTuple) -> bool:
        """
        Re-run the particle filter with new odometry data. Uses the most recent LIDAR data.

        Arguments:
            stamp: the time stamp of the most recent odometry data
            delta_pose: the change between the most recent odometry data and the odometry data
                        _at the time of the last successful update._

        Note: The caller of this method needs to keep track of the change in odometry pose over time,
              and reset it whenever this method returns True.

        Steps:
        1. Resample particles
        2. Apply motion model
        3. Re-weight based on sensor model

        Calling this method may or may not trigger an update (depending on: if an update is already
        in progress, if the initial pose/particles haven't been set yet, and if this odometry data
        is out of date). Returns True if an update actually happened, or false otherwise.
        """
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
        """
        Resample particles using a weighted random sample.

        All returned particles have an equal weight (1). `k` particles are returned, which defaults
        to self.NUM_PARTICLES.

        This is a pure method (doesn't mutate anything, returns new particles).
        """

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
        """ Save a new set of particles, including updating the computed reference frame. """
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

        self.visualize_particles()

    def visualize_particles(self):
        """ Publish particles for viewing in Rviz. """
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

    def normalize_weights(self, particles: List[Particle]) -> List[Particle]:
        """
        Normalize the weights of the particles (so the all add to 1).

        This is a pure method (doesn't mutate anything, returns new particles).
        """
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

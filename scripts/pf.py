#!/usr/bin/env python3

import math
import rospy
import random

import numpy as np
from typing import List, Optional

import tf2_ros
import tf2_geometry_msgs  # Importing for side-effects
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseWithCovarianceStamped

from sensor_model import ParallelRayTracingSensorModel, SensorModel, RayTracingSensorModel
from motion_model import MotionModel

from helper_functions import TFHelper,  PoseTuple, Particle, RandomSampler, make_marker, print_time


class ParticleFilter:
    USE_MULTIPROCESS_SENSOR_MODEL = True
    DEBUG_SAVE_SENSOR_STATE_PLOTS = 0

    NUM_PARTICLES = 200

    particle_sampler_xy = RandomSampler(0.10, 0)
    particle_sampler_theta = RandomSampler(0.10 * math.pi, 0)

    motion_model = MotionModel(stddev=.05)
    sensor_model: SensorModel

    # Don't update unless we've moved a bit
    # Currently set to 0 because more updates imperically seems to lead to more precision
    UPDATE_MIN_DISTANCE: float = 0
    UPDATE_MIN_ROTATION: float = 0

    last_pose: PoseTuple = None
    last_lidar: Optional[np.array] = None

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
    tf_helper: TFHelper

    def __init__(self):
        rospy.init_node('pf')

        self.last_update = rospy.Time.now()
        self.particles_stamp = rospy.Time.now()

        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)
        self.tf_helper = TFHelper()

        # publisher for the particle cloud for visualizing in rviz.
        self.particle_pub = rospy.Publisher("particlecloud",
                                            MarkerArray,
                                            queue_size=10)

        rospy.wait_for_service("static_map")
        get_static_map = rospy.ServiceProxy("static_map", GetMap)
        self.sensor_model = ParallelRayTracingSensorModel(get_static_map().map)

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
            self.tf_helper.convert_pose_to_xy_and_theta(msg.pose.pose)

        particles = self.resample_particles([Particle(x, y, theta, 1)])

        self.set_particles(msg.header.stamp, particles)

    def on_lidar(self, msg: LaserScan):
        """ Callback whenever new LIDAR data is available. """
        self.last_lidar = msg.ranges

    def on_odom(self, msg: Odometry):
        """ Callback whenever new odometry data is available. """
        pose = PoseTuple(
            *self.tf_helper.convert_pose_to_xy_and_theta(msg.pose.pose))

        if self.last_pose is None:
            self.last_pose = pose
            return

        delta_pose = PoseTuple(
            self.last_pose[0] - pose[0],
            self.last_pose[1] - pose[1],
            self.tf_helper.angle_diff(
                self.last_pose[2], pose[2])
        )

        # Make sure we've moved at least a bit
        if math.sqrt((delta_pose[0] ** 2) + (delta_pose[1] ** 2)) < self.UPDATE_MIN_DISTANCE \
                and delta_pose[2] < self.UPDATE_MIN_ROTATION:
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

        # Make sure we have some LIDAR data
        if self.last_lidar is None:
            return False

        # We don't do multiple updates in parallel
        if self.is_updating:
            return False

        self.is_updating = True
        try:
            with print_time('Update'):
                # Use a consistent LIDAR scan for the entire update
                self.sensor_model.set_lidar(self.last_lidar)

                # Resample Particles
                particles = self.resample_particles(self.particles)

                # Apply Motion Model
                # NOTE: Doing this in parallel has a small performance benefit, but it's not worth the complexity
                particles = [
                    self.motion_model.apply(p, delta_pose)
                    for p in particles
                ]

                # Apply Sensor Model
                particles = self.sensor_model.weight_particles(particles)

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
        robot_pose = np.average([
            (particle.x, particle.y, particle.theta)
            for particle in self.particles
        ], axis=0, weights=[p.weight for p in self.particles])

        # TF explodes if it ever sees a NaN
        if np.isnan(robot_pose).any():
            print("WARNING: Robot pose is NaN!", robot_pose)
            return

        self.tf_helper.fix_map_to_odom_transform(
            stamp,
            self.tf_helper.convert_xy_and_theta_to_pose(robot_pose)
        )

        self.visualize_particles()

    def visualize_particles(self):
        """ Publish particles for viewing in RViz. """
        # Publish particles
        markers = MarkerArray()
        for i, particle in enumerate(self.normalize_weights(self.particles)):
            pose = self.tf_helper.convert_xy_and_theta_to_pose(
                (particle.x, particle.y, particle.theta)
            )

            # Heuristic to produce decently-sized particle arrows
            scale_factor = max(
                0 if np.isnan(particle.weight) else (
                    particle.weight * (self.NUM_PARTICLES / 3)),
                0.1)

            scale = (scale_factor, scale_factor * 0.1, scale_factor * 0.1)

            marker = make_marker(pose, shape=Marker.ARROW,
                                 frame_id='map', ns="particle", id=i, lifetime=60, scale=scale)

            markers.markers.append(marker)

        self.particle_pub.publish(markers)

        # Save some images of sensor model internal state
        # To enable or disable this, change DEBUG_SAVE_SENSOR_STATE_PLOTS to a number (15 is good),
        # or 0 (to disable).
        for particle in random.choices(self.particles, k=self.DEBUG_SAVE_SENSOR_STATE_PLOTS):
            self.sensor_model.calculate_weight(particle)
            self.sensor_model.save_debug_plot(
                f"particle_{self.update_count:03d}")
        self.update_count += 1

    def normalize_weights(self, particles: List[Particle]) -> List[Particle]:
        """
        Normalize the weights of the particles (so they all add to 1).

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
            self.tf_helper.send_last_map_to_odom_transform()
            r.sleep()


if __name__ == '__main__':
    n = ParticleFilter()
    n.run()

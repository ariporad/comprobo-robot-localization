#!/usr/bin/env python3

""" This is the starter code for the robot localization project """

from collections import namedtuple

import math
from secrets import choice
from typing import Iterable, Optional, Tuple, List
import rospy
import random
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, PoseStamped
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion

import numpy as np
from numpy.random import default_rng, Generator

from helper_functions import TFHelper, print_time, sample_normal_error
from occupancy_field import OccupancyField

# NB: All particles are in the `map` frame
Particle = namedtuple('Particle', ['x', 'y', 'theta', 'weight'])
PoseTuple = namedtuple('PoseTuple', ['x', 'y', 'theta'])
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

    INITIAL_STATE_XY_SIGMA = 0.15
    INITIAL_STATE_XY_NOISE = 0.15

    INITIAL_STATE_THETA_SIGMA = math.pi / 10
    INITIAL_STATE_THETA_NOISE = 0.05

    NUM_PARTICLES = 100

    particles: List[Particle] = None
    robot_pose: PoseStamped = None

    last_pose: PoseTuple = None
    last_lidar: Optional[LaserScan] = None

    map_obstacles: np.array
    tf_listener: tf2_ros.TransformListener

    def __init__(self):
        rospy.init_node('pf')
        self.last_update = rospy.Time.now()

        # create instances of two helper objects that are provided to you
        # as part of the project
        self.occupancy_field = OccupancyField()  # NOTE: hangs if a map isn't published
        self.transform_helper = TFHelper()
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

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

        self.preprocess_map()

    def update_initial_pose(self, msg: PoseWithCovarianceStamped):
        """ Callback function to handle re-initializing the particle filter
            based on a pose estimate.  These pose estimates could be generated
            by another ROS Node or could come from the rviz GUI """
        x, y, theta = self.transform_helper.convert_pose_to_xy_and_theta(
            msg.pose.pose)

        particles = self.sample_particles(
            [Particle(x, y, theta, 1)],
            self.INITIAL_STATE_XY_SIGMA, self.INITIAL_STATE_XY_NOISE, self.INITIAL_STATE_THETA_SIGMA, self.INITIAL_STATE_THETA_NOISE, self.NUM_PARTICLES)

        self.set_particles(rospy.Time.now(), particles)

    def update(self, msg: Odometry):
        # last_odom = self.last_odom
        # translation, orientation_q = self.transform_helper.convert_pose_inverse_transform(
        #     msg.pose.pose)
        # orientation = euler_from_quaternion(orientation_q)[2]
        # odom = (translation[0], translation[1], orientation)

        cur_pose_bl = PoseStamped()
        cur_pose_bl.pose.orientation.w = 1.0
        cur_pose_bl.header.frame_id = 'base_link'
        cur_pose_bl.header.stamp = msg.header.stamp
        pose_odom = self.tf_buf.transform(
            cur_pose_bl, 'odom', rospy.Duration(0.5))
        cur_pose = PoseTuple(*self.transform_helper.convert_pose_to_xy_and_theta(
            pose_odom.pose))

        if self.last_pose is None or self.particles is None:
            self.last_pose = cur_pose
            return

        delta_pose = PoseTuple(
            self.last_pose[0] - cur_pose[0],
            self.last_pose[1] - cur_pose[1],
            self.transform_helper.angle_diff(self.last_pose[2], cur_pose[2])
        )

        cur_pose_bl = PoseStamped()
        cur_pose_bl.pose.orientation.w = 1.0
        cur_pose_bl.header.frame_id = 'base_link'
        cur_pose_bl.header.stamp = self.last_odom.header.stamp  # msg.header.stamp

        delta_pose = self.tf_buf.transform_full(
            cur_pose_bl, 'base_link', msg.header.stamp, 'odom', rospy.Duration(0.5))

        delta_pose = PoseTuple(*self.transform_helper.convert_pose_to_xy_and_theta(
            delta_pose.pose))

        print("Delta Pose:", delta_pose)

        # Make sure we've moved at least a bit
        if math.sqrt((delta_pose[0] ** 2) + (delta_pose[1] ** 2)) < 0.05 and delta_pose[2] < 0.1:
            return
        self.last_odom = msg

        now = rospy.Time.now()

        # Resample Particles
        # particles = self.sample_particles(
        #     self.particles,
        #     self.INITIAL_STATE_XY_SIGMA, self.INITIAL_STATE_XY_NOISE, self.INITIAL_STATE_THETA_SIGMA, self.INITIAL_STATE_THETA_NOISE, self.NUM_PARTICLES)
        particles = list(self.particles)

        # Apply Motion
        particles = self.apply_motion(particles, delta_pose, 0.05)

        particles = [
            Particle(p.x, p.y, p.theta, self.calculate_sensor_weight(p))
            for p in particles
        ]

        particles = self.normalize_weights(particles)

        print("Particle weights:", sorted([p.weight for p in particles]))

        self.set_particles(now, particles)

    def calculate_sensor_weight(self, particle: Particle) -> float:
        # I think this is broken
        # Try debugging by visualizing markers with weight
        if self.last_lidar is None:
            return 1.0

        actual_lidar = np.array(self.last_lidar.ranges[:-1])

        # Take map data as cartesian coords
        # Shift to center at particle
        # NB: Both the map and all particles are in the `map` frame
        obstacles_shifted = self.map_obstacles - [particle.x, particle.y]

        # Convert to polar, and descritize to whole angle increments [0-359]
        rho = np.sqrt(
            (obstacles_shifted[:, 0] ** 2) + (obstacles_shifted[:, 1] ** 2)
        )
        phi_rad = np.arctan2(obstacles_shifted[:, 1], obstacles_shifted[:, 0])

        # Rotate by the particle's heading
        phi_rad += particle.theta

        # Now convert to degrees
        # This is the only place we use degrees, but it's helpful since LIDAR is indexed by degree
        # arctan2(sin(), cos()) normalizes to [-pi, pi]
        phi = np.rad2deg(np.arctan2(np.sin(phi_rad), np.cos(phi_rad))).round()

        # Take the minimum at each angle
        # Indexed like lidar data, where each index is the degree
        expected_lidar = np.zeros(360)

        for phi, rho in zip(phi, rho):
            # phi is already an integer, just make the type right
            idx = int(phi)

            # Don't care if we don't have any LIDAR data
            if actual_lidar[idx] == 0.0:
                continue

            if expected_lidar[idx] == 0.0 or rho < expected_lidar[idx]:
                expected_lidar[idx] = rho

        # print("Calculated Map Polar Data:", expected_lidar)

        # Compare to LIDAR data (don't forget to drop the extra point #360)
        mask = actual_lidar > 0.0
        diff_lidar = np.abs(actual_lidar - expected_lidar)[mask]
        total_diff = np.sum(diff_lidar)

        weight = np.sum((diff_lidar / 10) ** 10)

        # _debug = np.zeros((360, 3))

        # _debug[:, 0] = expected_lidar
        # _debug[:, 1] = actual_lidar
        # _debug[:, 2] = diff_lidar

        # print(_debug)

        return weight

    def preprocess_map(self):
        rospy.wait_for_service("static_map")
        static_map = rospy.ServiceProxy("static_map", GetMap)
        map = static_map().map

        if map.info.origin.orientation.w != 1.0:
            print("WARNING: Unsupported map with rotated origin.")

        # The coordinates of each occupied grid cell in the map
        total_occupied = np.sum(np.array(map.data) > 0)
        occupied = np.zeros((total_occupied, 2))
        curr = 0
        for x in range(map.info.width):
            for y in range(map.info.height):
                # occupancy grids are stored in row major order
                ind = x + y*map.info.width
                if map.data[ind] > 0:
                    occupied[curr, 0] = (float(x) * map.info.resolution) \
                        + map.info.origin.position.x
                    occupied[curr, 1] = (float(y) * map.info.resolution) \
                        + map.info.origin.position.y
                    curr += 1
        self.map_obstacles = occupied

    def apply_motion(self, particles: List[Particle], delta_pose: PoseTuple, sigma: float) -> List[Particle]:
        # If a particle has a heading of theta
        # ihat(t-1) = [cos(theta), sin(theta)]
        # jhat(t-1) = [-sin(theta), cos(theta)]
        # x(t) = ihat(t) * delta.x + jhat(t)

        dx_robot = sample_normal_error(delta_pose.x, sigma),
        dy_robot = sample_normal_error(delta_pose.y, sigma),
        dtheta = sample_normal_error(delta_pose.theta, sigma)

        rot_dtheta = np.array([
            [np.cos(dtheta), -np.sin(dtheta)],
            [np.sin(dtheta), np.cos(dtheta)]
        ])

        dx, dy = np.matmul(rot_dtheta, [dx_robot, dy_robot])

        return [
            Particle(
                x=p.x + dx,
                y=p.y + dy,
                theta=p.theta + dtheta,
                weight=p.weight
            )
            for p in particles
        ]

    def sample_particles(self, particles: List[Particle], xy_sigma: float, xy_noise: float, theta_sigma: float, theta_noise: float, k: int) -> List[Particle]:
        choices = random.choices(
            particles,
            weights=[p.weight * 1000 for p in particles],
            k=k
        )

        return [
            Particle(
                x=rng.normal(choice.x, xy_sigma),
                y=rng.normal(choice.y, xy_sigma),
                theta=rng.normal(choice.theta, theta_sigma),
                weight=1
            )
            for choice in choices
        ]

    def set_particles(self, stamp: rospy.Time, particles: List[Particle]):
        self.last_update = stamp
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
        poses.poses = [
            self.transform_helper.convert_xy_and_theta_to_pose(
                (particle.x, particle.y, particle.theta))
            for particle in self.particles
            # if particle.weight >= 0.0001
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

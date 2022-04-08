#!/usr/bin/env python3

""" This is the starter code for the robot localization project """

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose

from helper_functions import TFHelper
from occupancy_field import OccupancyField

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

"""


class ParticleFilter:
    """
    The class that represents a Particle Filter ROS Node
    """

    def __init__(self):
        rospy.init_node('pf')

        # pose_listener responds to selection of a new approximate robot
        # location (for instance using rviz)
        rospy.Subscriber("initialpose",
                         PoseWithCovarianceStamped,
                         self.update_initial_pose)

        # publisher for the particle cloud for visualizing in rviz.
        self.particle_pub = rospy.Publisher("particlecloud",
                                            PoseArray,
                                            queue_size=10)

        # create instances of two helper objects that are provided to you
        # as part of the project
        self.occupancy_field = OccupancyField()
        self.transform_helper = TFHelper()

    def update_initial_pose(self, msg):
        """ Callback function to handle re-initializing the particle filter
            based on a pose estimate.  These pose estimates could be generated
            by another ROS Node or could come from the rviz GUI """
        xy_theta = \
            self.transform_helper.convert_pose_to_xy_and_theta(msg.pose.pose)

        # initialize your particle filter based on the xy_theta tuple

        # Use the helper functions to fix the transform

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

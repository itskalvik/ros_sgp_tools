#! /usr/bin/env python3
from tf.transformations import quaternion_matrix, quaternion_from_matrix
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import message_filters
import numpy as np
import rospy
import tf


class OptiTrackFrame:
    """
    Class to map vicon vrpn pose to odom
    """
    def __init__(self):
        super().__init__()

        # Setup the ROS node
        rospy.init_node('OptiTrackFrame', anonymous=True)                      

        # Setup the subscribers
        ns = rospy.get_namespace()
        if len(ns) > 1:
            self.robot_index = int(ns[:-1].split('_')[-1])+1
        else:
            self.robot_index = 1

        rospy.Subscriber('/vrpn_client_node/tb3/pose', 
                         PoseStamped,
                         self.callback)
        self.listener = tf.TransformListener()

        # Setup the frame broadcaster
        self.br = tf.TransformBroadcaster()
        self.pos = (0., 0., 0.)
        self.rot = (0., 0., 0., 1.)

        rate = rospy.Rate(10)
        rospy.loginfo('OptiTrackFrame Publisher initialized')

        while not rospy.is_shutdown():
            self.publish_frame()
            rate.sleep()

    def callback(self, msg):
        try:
            (trans,rot) = self.listener.lookupTransform('/odom', '/base_footprint', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        # Odom to Base
        Tob = quaternion_matrix(rot)
        Tob[:3,3] = trans

        # World to Base
        Twb = PoseStamped_2_mat(msg)
        # Base to World
        Tbw = T_inv(Twb)

        # Odom to World
        Tow = np.matmul(Tob, Tbw)
        Two = T_inv(Tow)

        q = quaternion_from_matrix(Two) 
        self.pos = (Two[0,3], Two[1,3], Two[2,3])
        self.rot = (q[0], q[1], q[2], q[3])

    def publish_frame(self):
        self.br.sendTransform(self.pos, self.rot,
                              rospy.Time.now(),
                              "odom", "world")
        # transform from frame "optitrack" to frame "base_footprint"

def PoseStamped_2_mat(p):
    q = p.pose.orientation
    pos = p.pose.position
    T = quaternion_matrix([q.x,q.y,q.z,q.w])
    T[:3,3] = np.array([pos.x,pos.y,pos.z])
    return T

def T_inv(T_in):
    R_in = T_in[:3,:3]
    t_in = T_in[:3,[-1]]
    R_out = R_in.T
    t_out = -np.matmul(R_out,t_in)
    return np.vstack((np.hstack((R_out,t_out)),np.array([0, 0, 0, 1])))


if __name__ == '__main__':
    try:
        OptiTrackFrame()
    except Exception as e:
        print(e)

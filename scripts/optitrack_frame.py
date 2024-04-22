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

        vrpn_sub = message_filters.Subscriber('/vrpn_client_node/pose', 
                                               PoseStamped)
        odom_sub = message_filters.Subscriber('/odom', Odometry)
        ts = message_filters.ApproximateTimeSynchronizer([vrpn_sub, odom_sub], 
                                                         10, 0.1, allow_headerless=True)
        ts.registerCallback(self.callback)

        # Setup the frame broadcaster
        self.br = tf.TransformBroadcaster()
        self.pos = (0., 0., 0.)
        self.rot = (0., 0., 0., 1.)

        rate = rospy.Rate(10)
        rospy.loginfo('OptiTrackFrame Publisher initialized')

        while not rospy.is_shutdown():
            self.publish_frame()
            rate.sleep()

    def callback(self, vrpn_msg, odom_msg):
        Twv = PoseStamped_2_mat(vrpn_msg)
        Two = PoseStamped_2_mat(odom_msg.pose)

        Tvw = T_inv(Twv)
        Tvo = np.matmul(Tvw, Two)
        q = quaternion_from_matrix(Tvo) 

        self.pos = (Tvo[0,3], Tvo[1,3], Tvo[2,3])
        self.rot = (q[0], q[1], q[2], q[3])

    def publish_frame(self):
        self.br.sendTransform(self.pos, self.rot,
                              rospy.Time.now(),
                              "odom", "optitrack")
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

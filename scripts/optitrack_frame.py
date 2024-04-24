#! /usr/bin/env python3
from tf.transformations import quaternion_matrix, quaternion_from_matrix
from geometry_msgs.msg import PoseStamped
import numpy as np
import rospy
import tf


class OptiTrackFrame:
    """
    Class to map vrpn pose to odom
    """
    def __init__(self):
        super().__init__()

        # Setup the ROS node
        rospy.init_node('OptiTrackFrame', anonymous=True)                      

        # Setup the subscriber
        rospy.Subscriber('/vrpn_client_node/tb3/pose', 
                         PoseStamped,
                         self.callback)

        # Setup the frame listener and broadcaster
        self.listener = tf.TransformListener()
        self.br = tf.TransformBroadcaster()

        rospy.loginfo('OptiTrackFrame Publisher initialized')
        rospy.spin()

    def callback(self, msg):
        try:
            (trans,rot) = self.listener.lookupTransform('/odom', '/base_link', rospy.Time(0))
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

        # TB3 to odom
        Tsol = np.matmul(Tbw, Two)

        q = quaternion_from_matrix(Tsol) 
        pos = (Tsol[0,3], Tsol[1,3], Tsol[2,3])
        rot = (q[0], q[1], q[2], q[3])
        self.br.sendTransform(pos, rot,
                              rospy.Time.now(),
                              "odom", "tb3")
        # transform from frame "tb3" to frame "odom"

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

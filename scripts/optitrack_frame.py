#! /usr/bin/env python3
from geometry_msgs.msg import PoseStamped
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

        vrpn_sub = rospy.Subscriber('/vrpn_client_node/pose',
                                    PoseStamped, 
                                    self.callback)
        
        # Setup the frame broadcaster
        self.br = tf.TransformBroadcaster()

        rospy.loginfo('OptiTrackFrame Publisher initialized')
        rospy.spin()

    def callback(self, msg): 
        self.br.sendTransform((msg.pose.position.x,
                               msg.pose.position.y,
                               msg.pose.position.z),
                              (msg.pose.orientation.x,
                               msg.pose.orientation.y,
                               msg.pose.orientation.z,
                               msg.pose.orientation.w),
                              rospy.Time.now(),
                              "base_footprint", "optitrack")
        # transform from frame "optitrack" to frame "base_footprint"

if __name__ == '__main__':
    try:
        OptiTrackFrame()
    except Exception as e:
        print(e)

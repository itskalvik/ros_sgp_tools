#! /usr/bin/env python3
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import message_filters
import rospy


class OptiTrackOdom:
    """
    Class to map vicon vrpn pose to odom
    """
    def __init__(self):
        super().__init__()

        # Setup the ROS node
        rospy.init_node('optitrack2odom', anonymous=True)                      

        # Setup the subscribers
        ns = rospy.get_namespace()
        if len(ns) > 1:
            self.robot_index = int(ns[:-1].split('_')[-1])+1
        else:
            self.robot_index = 1

        vrpn_sub = message_filters.Subscriber('/vrpn_client_node'+ns+'pose', 
                                              PoseStamped)
        odom_sub = message_filters.Subscriber(ns+'odom', 
                                              Odometry)
        ts = message_filters.TimeSynchronizer([vrpn_sub, odom_sub], 10)
        ts.registerCallback(self.callback)

        # Setup the publisher
        self.pose_publisher = rospy.Publisher('vrpn_odom', 
                                              Odometry, 
                                              queue_size=10)
        self.current_pose = Odometry()

        rospy.loginfo('OptiTrackOdom Publisher initialized')
        rospy.spin()

    def callback(self, vrpn_msg, odom_msg):      
        self.current_pose = odom_msg 
        self.current_pose.pose.pose = vrpn_msg.pose
        self.pose_publisher.publish(self.current_pose)


if __name__ == '__main__':
    try:
        OptiTrackOdom()
    except Exception as e:
        print(e)

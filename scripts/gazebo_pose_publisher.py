#! /usr/bin/env python3

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped
import rospy


class GazeboPosePublisher:
    """
    Class to map Gazebo Pose list to single robot pose similar to Vicon
    """
    def __init__(self):
        super().__init__()

        # Setup the ROS node
        rospy.init_node('gazebo_pose_publisher', anonymous=True)                      
        
        # Setup the subscriber
        ns = rospy.get_namespace()
        if len(ns) > 1:
            self.robot_index = int(ns[:-1].split('_')[-1])+1
        else:
            self.robot_index = 1

        self.pose_subscriber = rospy.Subscriber('/gazebo/model_states', 
                                                ModelStates, 
                                                self.position_callback)

        # Setup the publisher
        self.pose_publisher = rospy.Publisher('/vrpn_client_node'+ns+'pose', 
                                              PoseStamped, 
                                              queue_size=10)
        
        self.current_pose = PoseStamped()

        rospy.loginfo('Gazebo Pose Publisher initialized')
        rospy.spin()

    def position_callback(self, msg):       
        self.current_pose.header.stamp = rospy.Time.now()
        self.current_pose.pose.orientation.x = msg.pose[self.robot_index].orientation.x
        self.current_pose.pose.orientation.y = msg.pose[self.robot_index].orientation.y
        self.current_pose.pose.orientation.z = msg.pose[self.robot_index].orientation.z
        self.current_pose.pose.orientation.w = msg.pose[self.robot_index].orientation.w

        self.current_pose.pose.position.x = msg.pose[self.robot_index].position.x
        self.current_pose.pose.position.y = msg.pose[self.robot_index].position.y
        self.current_pose.pose.position.z = msg.pose[self.robot_index].position.z

        self.pose_publisher.publish(self.current_pose)


if __name__ == '__main__':
    try:
        GazeboPosePublisher()
    except Exception as e:
        print(e)

#! /usr/bin/env python3

from geometry_msgs.msg import PoseStamped
from ros_sgp_ipp.msg import RSSI
from std_srvs.srv import SetBool
import message_filters
import numpy as np
import rospy


class RSSIMapper:
    """
    Class to map RSSI.
    """
    def __init__(self,
                 buffer_size=100):
        super().__init__()

        # Setup the data buffers and the current waypoint
        self.data_X = []
        self.data_y = []
        self.buffer_size = buffer_size
        self.dump_data = False
        self.counter = 0

        # Setup the ROS node
        rospy.init_node('rssi_mapper', anonymous=True)                      
        self.ns = rospy.get_namespace()

        # Setup the subscribers
        pose_subscriber = message_filters.Subscriber('/vrpn_client_node'+self.ns+'pose', 
                                                     PoseStamped)
        rssi_subscriber = message_filters.Subscriber(self.ns+'rssi', 
                                                     RSSI)
        data_subscriber = message_filters.ApproximateTimeSynchronizer([pose_subscriber, 
                                                                       rssi_subscriber], 
                                                                       10, 0.1, 
                                                                       allow_headerless=True)
        data_subscriber.registerCallback(self.data_callback)

        # Setup the timer to update the parameters and waypoints
        self.timer = rospy.Timer(rospy.Duration(5), self.flush_data)

        # Setup the service
        self.stop_service = rospy.Service('stop_rssi_mapper', 
                                          SetBool,
                                          self.stop_rssi_mapper)
        
        rospy.loginfo(self.ns+'RSSI mapper initialized')

        # Flush data to disk before shutdown
        rospy.on_shutdown(self.stop_rssi_mapper)

        rospy.spin()

    def data_callback(self, pose_msg, rssi_msg):
        # Append the new data to the buffers
        self.data_X.append([pose_msg.pose[1].position.x, 
                            pose_msg.pose[1].position.y])
        self.data_y.append(rssi_msg.rssi)

        if len(self.data_X) >= self.buffer_size:
            self.dump_data = True

    def flush_data(self, timer=None):
        rospy.loginfo('Flushing data to disk')

        # Make local copies of the data
        data_X = np.array(self.data_X).reshape(-1, 2)
        data_y = np.array(self.data_y).reshape(-1, 1)

        # Empty global data buffers
        self.data_X = []
        self.data_y = []

        # Save data to disk
        np.save(f"rssi_data_X_{self.counter}", data_X)
        np.save(f"rssi_data_y_{self.counter}", data_y)

        self.dump_data = False
        self.counter += 1

    def stop_rssi_mapper(self, req=None):
        rospy.loginfo('Shutting down RSSI mapper')
        self.flush_data()
        rospy.signal_shutdown("Received shutdown signal")
    

if __name__ == '__main__':
    try:
        RSSIMapper()
    except Exception as e:
        print(e)

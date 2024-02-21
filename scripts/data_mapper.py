#! /usr/bin/env python3

from ros_sgp_ipp.msg import SensorData
from std_srvs.srv import SetBool
import numpy as np
import rospy


class DataMapper:
    """
    Class to map the sensor data.
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
        rospy.init_node('data_mapper', anonymous=True)                      
        self.ns = rospy.get_namespace()

        # Setup the subscriber
        rospy.Subscriber(self.ns+'sensor_data',
                         SensorData,
                         self.data_callback)

        # Setup the timer to update the parameters and waypoints
        self.timer = rospy.Timer(rospy.Duration(5), self.flush_data)

        # Setup the service
        self.stop_service = rospy.Service('stop_data_mapper', 
                                          SetBool,
                                          self.stop_data_mapper)
        
        rospy.loginfo(self.ns+'Data mapper initialized')

        # Flush data to disk before shutdown
        rospy.on_shutdown(self.stop_data_mapper)

        rospy.spin()

    def data_callback(self, msg):
        # Append the new data to the buffers
        self.data_X.append(msg.x)
        self.data_y.append(msg.y)

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
        np.save(f"{self.ns[1:-1]}_sensor_data_X_{self.counter}", data_X)
        np.save(f"{self.ns[1:-1]}_sensor_data_y_{self.counter}", data_y)

        self.dump_data = False
        self.counter += 1

    def stop_rssi_mapper(self, req=None):
        rospy.loginfo(self.ns+'Shutting down data mapper')
        self.flush_data()
        rospy.signal_shutdown(self.ns+"Received shutdown signal")
    

if __name__ == '__main__':
    try:
        DataMapper()
    except Exception as e:
        print(e)

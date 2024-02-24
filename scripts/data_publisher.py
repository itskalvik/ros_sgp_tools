#!/usr/bin/env python3

import os
import rospy
import numpy as np
import netCDF4 as nc
from pathlib import Path
from scipy.spatial import cKDTree
from ros_sgp_ipp.msg import SensorData
from geometry_msgs.msg import PoseStamped


class DataPublisher:
    def __init__(self):
        rospy.init_node('data_publisher', anonymous=True)
        self.ns = rospy.get_namespace()

        # Setup data
        home = str(Path.home())
        filepath = os.path.join(home, 'sgp-tools/datasets/ROMS.nc')
        self.dataset_X, self.dataset_y = self.prep_salinity_data(filepath)
        self.tree = cKDTree(self.dataset_X)

        rospy.Subscriber('/vrpn_client_node'+self.ns+'pose',
                         PoseStamped,
                         self.data_callback)

        self.publisher = rospy.Publisher('sensor_data', 
                                         SensorData, 
                                         queue_size=10)
        
        # Reduce data publish rate to avoid sending the same data and 
        # collapsing the cholskey decomposition in the parameter SGP
        self.timer = rospy.Timer(rospy.Duration(0.1), self.publish_data)
        self.data = SensorData()

        rospy.loginfo(self.ns+'Sensor data publisher initialized')
        rospy.spin()

    def data_callback(self, msg):
        self.data.header.stamp = rospy.Time.now()
        self.data.x = [msg.pose.position.x, 
                       msg.pose.position.y]
        self.data.y = self.dataset_y[self.tree.query(self.data.x, k=1)[1]]

    def publish_data(self, timer):
        self.publisher.publish(self.data)

    def prep_salinity_data(self, filepath):
        sample_rate=2
        ds = nc.Dataset(filepath)
        y = np.array(ds.variables['salt'])[0, :-1, ::sample_rate, ::sample_rate]
        y = y[0]
        X = np.mgrid[0:y.shape[0], 0:y.shape[1]].transpose(1, 2, 0)

        X = X.reshape(-1, 2)
        X = X*0.1 - [2, 2]
        y = y.reshape(-1)

        # Remove nans
        idx = ~np.isnan(y)
        y = y[idx]
        X = X[idx]

        # Normalize
        y = (y - np.mean(y)) / np.std(y)

        return X, y

if __name__ == '__main__':
    data_publisher = DataPublisher()
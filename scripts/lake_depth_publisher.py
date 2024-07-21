#! /usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import NavSatFix, Range

from ament_index_python.packages import get_package_share_directory
from sgptools.utils.gpflow import get_model_params
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import gpflow
import os


class DepthPublisher(Node):

    def __init__(self):
        super().__init__('depth_publisher')
        # Setup the depth publisher
        self.depth_publisher = self.create_publisher(Range, 
                                'mavros/rangefinder_pub', 
                                10)
        self.depth_msg = Range()

        # Setup the position subscriber
        self.vehicle_pose_subscriber = self.create_subscription(NavSatFix, 
                                            'mavros/global_position/global', 
                                            self.data_callback, 
                                            rclpy.qos.qos_profile_sensor_data)

        # Raw data from path
        plan_fname = os.path.join(get_package_share_directory('ros_sgp_tools'), 
                                                              'launch', 
                                                              'lake_gt.csv')
        df = pd.read_csv(plan_fname)
        df = df.dropna()
        x = df['DPTH.Lat'].to_numpy()[::10]
        y = df['DPTH.Lng'].to_numpy()[::10]
        c = df['DPTH.Depth'].to_numpy()[::10]

        X_train = np.vstack([x, y]).T.astype(float)*1e-7
        y_train = c.reshape(-1, 1).astype(float)

        self.get_logger().info(f'{X_train.shape}')
        
        self.X_scaler = StandardScaler()
        self.X_scaler.fit(X_train)
        X_train = self.X_scaler.transform(X_train)*10.0

        # Fit kernel parameters
        _, noise_variance, kernel = get_model_params(X_train, y_train, 
                                                     lengthscales=0.1,
                                                     variance=1.0,
                                                     noise_variance=1e-4,
                                                     optimizer='scipy')

        self.gpr_gt = gpflow.models.GPR(data=(X_train, y_train), 
                                        kernel=kernel,
                                        noise_variance=noise_variance)

        self.get_logger().info("Initialized depth publisher")

    def data_callback(self, msg):
        self.depth_msg.header.stamp = self.get_clock().now().to_msg()
        location = np.array([[msg.latitude, msg.longitude]])
        location = self.X_scaler.transform(location)*10.0
        mean, _ = self.gpr_gt.predict_f(location)
        self.depth_msg.range = mean.numpy()[0][0]
        self.depth_publisher.publish(self.depth_msg)


def main(args=None):
    rclpy.init(args=args)
    depth_publisher = DepthPublisher()
    rclpy.spin(depth_publisher)
    depth_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

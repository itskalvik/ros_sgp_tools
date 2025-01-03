#! /usr/bin/env python3

from sensor_msgs.msg import PointCloud2
from rclpy.node import Node
import rclpy

import os
import h5py
import numpy as np
from sgptools.utils.gpflow import *
from utils import CustomStandardScaler, point_cloud
from sgptools.utils.misc import ploygon2candidats

tf.random.set_seed(2024)
np.random.seed(2024)


class DataVisualizer(Node):

    def __init__(self):
        super().__init__('DataVisualizer')

        # Create a publisher to publish the point cloud
        self.pcd_publisher = self.create_publisher(PointCloud2, 'pcd', 10)

        # Get the parameters
        self.declare_parameter('data_folder', '')
        data_folder = self.get_parameter('data_folder').get_parameter_value().string_value

        self.declare_parameter('num_samples', 5000)
        self.num_samples = self.get_parameter('num_samples').get_parameter_value().integer_value

        self.declare_parameter('mission_log', '')
        mission_log = self.get_parameter('mission_log').get_parameter_value().string_value

        # Get the latest log folder
        if mission_log == '':
            logs = os.listdir(data_folder)
            mission_log = sorted([log for log in logs if 'IPP-mission' in log])[-1]

        # data file
        self.fname = os.path.join(data_folder, 
                                  mission_log,
                                  "mission-log.hdf5")

        # load data
        with h5py.File(self.fname, "r") as f:
            self.fence_vertices = f["fence_vertices"][:].astype(float)
            self.X = f["X"][:].astype(float)
            self.y = f["y"][:].astype(float)

        self.get_logger().info(f'Data Folder: {data_folder}')
        self.get_logger().info(f'Mission Log: {mission_log}')
        self.get_logger().info(f'Number of data samples: {self.X.shape[0]}')
        self.get_logger().info(f'Number of reconstruction samples: {self.num_samples}')

        # Normalize the candidates
        X_candidates = ploygon2candidats(self.fence_vertices, 
                                         num_samples=self.num_samples)
        self.X_scaler = CustomStandardScaler()
        self.X_scaler.fit(X_candidates)
        self.X_scaler.scale_ *= 0.35
        X_candidates = self.X_scaler.transform(X_candidates)    
        self.X = self.X_scaler.transform(self.X)
        self.num_samples = 0 # reset to force update

        # Train GP
        kernel = gpflow.kernels.SquaredExponential(lengthscales=0.1, 
                                                   variance=0.5)
        self.gpr_gt = gpflow.models.GPR(data=(self.X, self.y), 
                                        kernel=kernel)
        optimize_model(self.gpr_gt)
        
        # Publish point cloud every 10 seconds
        self.create_timer(10, callback=self.timer_callback)
        self.get_logger().info('DataVisualizer node initialized')

    def timer_callback(self):
        num_samples = self.get_parameter('num_samples').get_parameter_value().integer_value
        # If the number of samples has changed, update the point cloud
        if self.num_samples != num_samples:
            self.get_logger().info(f'Updated Number of samples: {num_samples}')
            self.num_samples = num_samples
            X_candidates = ploygon2candidats(self.fence_vertices, 
                                             num_samples=self.num_samples)
            X_candidates = self.X_scaler.transform(X_candidates)
            self.candidates_y = self.gpr_gt.predict_f(X_candidates)[0].numpy()
            self.point_cloud_msg = point_cloud(np.concatenate([X_candidates,
                                                               -self.candidates_y], 
                                                               axis=1),
                                               'map')
            
        self.pcd_publisher.publish(self.point_cloud_msg)
        self.get_logger().info('Published point cloud')


if __name__ == '__main__':
    rclpy.init()
    node = DataVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

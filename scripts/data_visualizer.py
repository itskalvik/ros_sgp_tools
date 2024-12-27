#! /usr/bin/env python3

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from rclpy.node import Node
import rclpy

import os
import h5py
import numpy as np
from utils import CustomStandardScaler
from sgptools.utils.gpflow import *
from sgptools.utils.misc import ploygon2candidats

tf.random.set_seed(2024)
np.random.seed(2024)


class DataVisualizer(Node):

    def __init__(self):
        super().__init__('DataVisualizer')
        self.pcd_publisher = self.create_publisher(PointCloud2, 'pcd', 10)

        # data folder
        try:
            data_folder = os.environ['DATA_FOLDER']
        except:
            data_folder = ''
        
        # Get the log folder from the parameter server if available
        self.declare_parameter('mission_log', '')
        mission_log = self.get_parameter('mission_log').get_parameter_value().string_value

        self.declare_parameter('num_samples', 5000)
        self.num_samples = self.get_parameter('num_samples').get_parameter_value().integer_value

        # Get the latest log folder
        if mission_log == '':
            logs = os.listdir(data_folder)
            mission_log = sorted([log for log in logs if 'IPP-mission' in log])[-1]
        self.get_logger().info(f'Mission Log: {mission_log}')
        self.get_logger().info(f'Number of samples: {self.num_samples}')

        # data file
        self.fname = os.path.join(data_folder, 
                                  mission_log,
                                  "mission-log.hdf5")

        # load data
        with h5py.File(self.fname, "r") as f:
            self.fence_vertices = f["fence_vertices"][:].astype(float)
            self.X = f["X"][:].astype(float)
            self.y = f["y"][:].astype(float)

        # Normalize the candidates
        X_candidates = ploygon2candidats(self.fence_vertices, 
                                         num_samples=self.num_samples)
        self.X_scaler = CustomStandardScaler()
        self.X_scaler.fit(X_candidates)
        self.X_scaler.scale_ *= 0.35
        X_candidates = self.X_scaler.transform(X_candidates)    
        self.X = self.X_scaler.transform(self.X)

        # Train GP
        kernel = gpflow.kernels.SquaredExponential(lengthscales=0.1, 
                                                   variance=0.5)
        self.gpr_gt = gpflow.models.GPR(data=(self.X, self.y), 
                                        kernel=kernel)
        optimize_model(self.gpr_gt)
        
        # Create point cloud
        self.candidates_y = self.gpr_gt.predict_f(X_candidates)[0].numpy()
        self.point_cloud_msg = self.point_cloud(np.concatenate([X_candidates,
                                                                -self.candidates_y], 
                                                               axis=1),
                                                'map')
        
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
            self.point_cloud_msg = self.point_cloud(np.concatenate([X_candidates,
                                                                    -self.candidates_y], 
                                                                   axis=1),
                                                    'map')
            
        self.pcd_publisher.publish(self.point_cloud_msg)
        self.get_logger().info('Published point cloud')

    def point_cloud(self, points, parent_frame):
        """ Creates a point cloud message.
        Args:
            points: Nx3 array of xyz positions.
            parent_frame: frame in which the point cloud is defined
        Returns:
            sensor_msgs/PointCloud2 message

        Code source:
            https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0
        """
        # In a PointCloud2 message, the point cloud is stored as an byte 
        # array. In order to unpack it, we also include some parameters 
        # which desribes the size of each individual point.
        points = np.array(points, dtype=np.float32)
        ros_dtype = PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize # A 32-bit float takes 4 bytes.

        data = points.astype(dtype).tobytes() 

        # The fields specify what the bytes represents. The first 4 bytes 
        # represents the x-coordinate, the next 4 the y-coordinate, etc.
        fields = [PointField(
                name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
                for i, n in enumerate('xyz')]

        # The PointCloud2 message also has a header which specifies which 
        # coordinate frame it is represented in. 
        header = Header(frame_id=parent_frame)

        return PointCloud2(
            header=header,
            height=1, 
            width=points.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=(itemsize * 3), # Every point consists of three float32s.
            row_step=(itemsize * 3 * points.shape[0]),
            data=data
        )


if __name__ == '__main__':
    rclpy.init()
    node = DataVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

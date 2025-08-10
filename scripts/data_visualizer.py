#! /usr/bin/env python3

from sensor_msgs.msg import PointCloud2
from rclpy.node import Node
import rclpy

import os
import h5py
import yaml
import pickle
import numpy as np
from utils import LatLonStandardScaler, StandardScaler, point_cloud

from sgptools.utils.misc import polygon2candidates
from sgptools.kernels import get_kernel
from sgptools.utils.metrics import *
from sgptools.utils.gpflow import *

from gpflow.config import default_float

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

        self.declare_parameter('mission_log', '')
        mission_log = self.get_parameter('mission_log').get_parameter_value().string_value

        self.declare_parameter('num_samples', 5000)
        self.num_samples = self.get_parameter('num_samples').get_parameter_value().integer_value

        # Get the latest log folder
        if mission_log == '':
            logs = os.listdir(data_folder)
            mission_log = sorted([log for log in logs if 'IPP-mission' in log])[-1]

        # Extract hyperparameters from viz_config.yaml if available
        config_fname = os.path.join(data_folder, mission_log, f"viz_config.yaml")
        if os.path.exists(config_fname):
            self.get_logger().info(f'Config File: {config_fname}')
            with open(config_fname, 'r') as file:
                self.config = yaml.safe_load(file)
            force_training = self.config.get('force_training', False)
            hyperparameter_config = self.config.get('hyperparameters', {})
            self.kernel = hyperparameter_config.get('kernel_function', 'RBF')

            # Use float32 and higher jitter for deep learning model based kernel functions
            if self.kernel in ['Attentive', 'NeuralSpectral']:
                gpflow.config.set_default_float(np.float32)
                gpflow.config.set_default_jitter(1e-1)
            else:
                gpflow.config.set_default_float(np.float64)
                gpflow.config.set_default_jitter(1e-6)

            kernel_kwargs = hyperparameter_config.get('kernel', {})
            kernel = get_kernel(self.kernel)(**kernel_kwargs)
            noise_variance = float(hyperparameter_config.get('noise_variance', 1e-4))
            optimizer_kwargs = self.config.get('optimizer', {})
        else:
            force_training = False
            self.kernel = 'RBF'
            kernel =  get_kernel(self.kernel)()
            noise_variance = 0.01
            optimizer_kwargs = {}

        # Load the data file
        self.fname = os.path.join(data_folder, 
                                  mission_log,
                                  "mission-log.hdf5")
        if not os.path.exists(self.fname):
            raise ValueError(f'Data file not found: {self.fname}')
        with h5py.File(self.fname, "r") as f:
            self.fence_vertices = f["fence_vertices"][:]
            self.X = f["X"][:]
            self.y = f["y"][:]

        self.get_logger().info(f'Data Folder: {data_folder}')
        self.get_logger().info(f'Mission Log: {mission_log}')
        self.get_logger().info(f'Number of data samples: {self.X.shape[0]}')
        self.get_logger().info(f'Number of reconstruction samples: {self.num_samples}')
        self.get_logger().info(f'Kernel: {self.kernel}')

        # Normalize the candidates
        X_candidates = polygon2candidates(self.fence_vertices, 
                                          num_samples=self.num_samples)
        self.X_scaler = LatLonStandardScaler()
        self.X_scaler.fit(X_candidates)
        self.X_scaler.scale_ *= 0.35
        X_candidates = self.X_scaler.transform(X_candidates)
        self.X = self.X_scaler.transform(self.X)
        self.num_samples = 0 # reset to force update
        self.y_scaler = StandardScaler()
        self.y = self.y_scaler.fit_transform(self.y)

        # Cast to float32 for compatibility with GPflow
        self.X = self.X.astype(default_float())
        self.y = self.y.astype(default_float())

        # Train GP only if pretrained weights are unavailable
        fname = os.path.join(data_folder, mission_log, f"{self.kernel}Params.pkl")
        if os.path.exists(fname) and not force_training:
            with open(fname, 'rb') as handle:
                params = pickle.load(handle)
            optimizer_kwargs['max_steps'] = 0
            self.get_logger().info('Found pre-trained parameters')
        else:
            params = None
            self.get_logger().info('Training from scratch')
        _, _, _, self.gpr_gt = get_model_params(self.X, self.y,
                                                kernel=kernel,
                                                noise_variance=noise_variance,
                                                return_model=True,
                                                train_inducing_pts=True,
                                                verbose=True,
                                                **optimizer_kwargs)
        
        # Load pre-trained parameters
        if params is not None:
            gpflow.utilities.multiple_assign(self.gpr_gt.kernel, params['kernel'])
            gpflow.utilities.multiple_assign(self.gpr_gt.likelihood, params['likelihood'])

        # Create parameter dict h5py file
        if not os.path.exists(fname):
            params_kernel = gpflow.utilities.parameter_dict(self.gpr_gt.kernel)
            params_likelihood = gpflow.utilities.parameter_dict(self.gpr_gt.likelihood)
            params = {'kernel': params_kernel, 
                      'likelihood': params_likelihood}
            with open(fname, 'wb') as handle:
                pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Print GP hyperparams if using RBF kernel
        if self.kernel == 'RBF':
            self.get_logger().info(f'kernel lengthscales: {self.gpr_gt.kernel.lengthscales.numpy():.4f}')
            self.get_logger().info(f'kernel variance: {self.gpr_gt.kernel.variance.numpy():.4f}')
            self.get_logger().info(f'Likelihood variance: {self.gpr_gt.likelihood.variance.numpy():.4f}')

        # Publish point cloud every 10 seconds
        self.create_timer(10, callback=self.timer_callback)
        self.get_logger().info('DataVisualizer node initialized')

    def timer_callback(self):
        num_samples = self.get_parameter('num_samples').get_parameter_value().integer_value
        # If the number of samples has changed, update the point cloud
        if self.num_samples != num_samples:
            self.get_logger().info(f'Updated Number of samples: {num_samples}')
            self.num_samples = num_samples
            X_candidates = polygon2candidates(self.fence_vertices, 
                                              num_samples=self.num_samples)
            X_candidates = self.X_scaler.transform(X_candidates)
            X_candidates = X_candidates.astype(default_float())
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

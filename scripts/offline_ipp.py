#! /usr/bin/env python3

import os
from utils import plan2data, project_waypoints
from ament_index_python.packages import get_package_share_directory

import gpflow
import numpy as np
from sgptools.utils.tsp import run_tsp
from sgptools.models.continuous_sgp import *
from sgptools.models.core.transformations import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from ros_sgp_tools.srv import IPP
from geometry_msgs.msg import Point

import rclpy
from rclpy.node import Node

import matplotlib.pyplot as plt

import tensorflow as tf
tf.random.set_seed(2024)
np.random.seed(2024)


class offlineIPP(Node):
    """
    Class to create an offline IPP mission.

    Note: Make sure the number of waypoints is small enough so that 
    the parameters update and waypoints updates are fast enough to 
    reach the robot before it reaches the next waypoint.
    """
    def __init__(self):
        super().__init__('OfflineIPP')
        self.get_logger().info('Initializing')

        # Declare parameters
        self.declare_parameter('num_waypoints', 10)
        self.num_waypoints = self.get_parameter('num_waypoints').get_parameter_value().integer_value
        self.get_logger().info(f'Num Waypoints: {self.num_waypoints}')

        self.declare_parameter('num_robots', 1)
        self.num_robots = self.get_parameter('num_robots').get_parameter_value().integer_value
        self.get_logger().info(f'Num Robots: {self.num_robots}')

        plan_fname = os.path.join(get_package_share_directory('ros_sgp_tools'), 
                                                              'launch', 
                                                              'sample.plan')
        self.declare_parameter('geofence_plan', plan_fname)
        plan_fname = self.get_parameter('geofence_plan').get_parameter_value().string_value
        self.get_logger().info(f'GeoFence Plan File: {plan_fname}')

        # Get the data and normalize 
        X_train, home_position = plan2data(plan_fname, num_samples=5000)
        
        self.X_train = np.array(X_train).reshape(-1, 2)
        self.X_scaler = StandardScaler()
        self.X_train = self.X_scaler.fit_transform(self.X_train)

        # Shift home position for each robot to avoid collision with other robots
        home_positions = []
        for i in range(self.num_robots):
            home_positions.append(np.array(home_position[:2]) + (np.array([3/111111, 0.0])*i))
        home_position = np.array(home_positions).reshape(-1, 2)
        self.home_position = self.X_scaler.transform(home_position)

        # Get initial solution paths
        self.compute_init_paths()

        # Sync the waypoints with the trajectory planner
        self.sync_waypoints()
        self.get_logger().info('Initial waypoints synced with the online planner')
        self.get_logger().info('Shutting down offline IPP node')

    def compute_init_paths(self):
        # Initialize random SGP parameters
        likelihood_variance = 1e-4
        kernel = gpflow.kernels.RBF(lengthscales=0.1, 
                                    variance=0.5)

        # Get the initial IPP solution
        transform = IPPTransform(n_dim=2, 
                                 num_robots=self.num_robots)
        # Sample uniform random initial waypoints and compute initial paths
        Xu_init = get_inducing_pts(self.X_train, self.num_waypoints*self.num_robots)

        # Add fixed home position
        for i in range(self.num_robots):
            Xu_init[i] = self.home_position[i]

        Xu_init, _ = run_tsp(Xu_init, 
                             num_vehicles=self.num_robots,
                             resample=self.num_waypoints,
                             start_idx=np.arange(self.num_robots).tolist())
        Xu_fixed = np.copy(Xu_init[:, :1, :])
        Xu_init = np.array(Xu_init).reshape(-1, 2)

        # Initialize the SGP
        IPP_model, _ = continuous_sgp(self.num_waypoints, 
                                      self.X_train,
                                      likelihood_variance,
                                      kernel,
                                      transform,
                                      Xu_init=Xu_init,
                                      max_steps=0)
        IPP_model.transform.update_Xu_fixed(Xu_fixed)

        # Get the new inducing points for the path
        optimize_model(IPP_model, 
                       kernel_grad=False, 
                       optimizer='scipy',
                       method='CG')

        # Generate new paths from optimized waypoints and project them back to the bounds of the environment
        self.waypoints = IPP_model.inducing_variable.Z.numpy().reshape(self.num_robots, 
                                                                       self.num_waypoints, -1)
        for i in range(self.num_robots):
            self.waypoints[i] = project_waypoints(self.waypoints[i], self.X_train)

        # Print path lengths
        self.get_logger().info('OfflineIPP: Initial IPP solution found') 
        path_lengths = transform.distance(np.concatenate(self.waypoints, axis=0)).numpy()
        msg = 'Initial path lengths: '
        for path_length in path_lengths:
            msg += f'{path_length:.2f} '
        self.get_logger().info(msg)

        # Generate unlabeled training data for each robot
        self.get_training_sets()

        # Log generated paths and training sets
        self.plot_paths()

    '''
    Generate unlabeled training data for each robot
    '''
    def get_training_sets(self):
        # Setup KNN training dataset
        X = np.concatenate(self.waypoints, axis=0)
        y = []
        i = 0
        for i in range(self.num_robots):
            y.extend(np.ones(len(self.waypoints[i]))*i)
            i += 1
        y = np.array(y).astype(int)

        # Train KNN and get predictions
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(X, y)
        y_pred = neigh.predict(self.X_train)

        # Format data for transmission
        self.data = []
        for i in range(self.num_robots):
            self.data.append(self.X_train[np.where(y_pred==i)[0]])

    '''
    Log generated paths and training sets
    '''
    def plot_paths(self):
        plt.figure()
        for i, path in enumerate(self.waypoints):
            plt.plot(path[:, 1], path[:, 0], 
                     label='Path', zorder=1, marker='o', c='r')
            plt.scatter(self.data[i][:, 1], self.data[i][:, 0],
                        s=1, label='Candidates', zorder=0)
            np.savetxt(f'OfflineIPP-{i}.csv', path, delimiter=',')
        plt.scatter(self.home_position[:, 1], self.home_position[:, 0],
                    label='Home position', zorder=2, c='g')
        plt.legend()
        plt.savefig(f'OfflineIPP.png')

    '''
    Send the new waypoints to the trajectory planner and 
    update the current waypoint from the service
    '''
    def sync_waypoints(self):
        for robot_idx in range(self.num_robots):
            service = f'robot_{robot_idx}/offlineIPP'
            offline_ipp_service = self.create_client(IPP, service)
            request = IPP.Request()

            try:
                while not offline_ipp_service.wait_for_service(timeout_sec=1.0):
                    self.get_logger().info(f'{service} service not avaliable, waiting...')

                waypoints = self.waypoints[robot_idx]
                for waypoint in waypoints:
                    request.data.waypoints.append(Point(x=waypoint[0],
                                                        y=waypoint[1]))
                    
                train_pts = self.data[robot_idx]
                for point in train_pts:
                    request.data.x_train.append(Point(x=point[0],
                                                      y=point[1]))
                future = offline_ipp_service.call_async(request)
                rclpy.spin_until_future_complete(self, future)
                if future.result() is not None:
                    self.get_logger().info(f'Service call successful')
                else:
                    self.get_logger().info(f'Service call failed: {future.exeception()}')
            except Exception as e:
                print(f'Service call failed: {e}')


if __name__ == '__main__':
    # Start the offline IPP mission
    rclpy.init()
    node = offlineIPP()
    
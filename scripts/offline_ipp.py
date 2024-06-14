#! /usr/bin/env python3

import gpflow
import numpy as np
from sgptools.utils.tsp import run_tsp
from sgptools.models.continuous_sgp import *
from sgptools.models.core.transformations import *
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf

from sgptools.utils.metrics import *
from sgptools.utils.data import *
from sgptools.utils.misc import *
from sgptools.utils.tsp import run_tsp
from sgptools.utils.gpflow import get_model_params

from sgptools.models.cma_es import *
from sgptools.models.continuous_sgp import *
from sgptools.models.greedy_sgp import *
from sgptools.models.greedy_mi import *
from sgptools.models.bo import *
from sgptools.models.core.transformations import IPPTransform

from ros_sgp_ipp.msg import OfflineIPPData
from ros_sgp_ipp.srv import OfflineIPP
from geometry_msgs.msg import Point

import rclpy
from rclpy.node import Node
from rclpy.exceptions import InvalidServiceNameException

import matplotlib.pyplot as plt
np.random.seed(2021)

class offlineIPP(Node):
    """
    Class to create an offline IPP mission.

    Note: Make sure the number of waypoints is small enough so that 
    the parameters update and waypoints updates are fast enough to 
    reach the robot before it reaches the next waypoint.

    Args:
        X_train (np.ndarray): The training data for the IPP model, 
                              used to approximate the bounds of the environment.
        num_waypoints (int): The number of waypoints/inducing points for the IPP model.
        num_param_inducing (int): The number of inducing points for the OSGPR model.
        num_robots (int): The number of robots
    """
    def __init__(self, X_train, 
                 num_waypoints=10, 
                 num_robots=1):
        super().__init__('offline_ipp')

        # Setup the ROS node
        self.get_logger().info('Initializing offline IPP mission')

        self.X_train = np.array(X_train).reshape(-1, 2)
        self.num_waypoints = num_waypoints
        self.num_robots = num_robots

        # Get initial solution paths
        self.compute_init_paths()

        # Sync the waypoints with the trajectory planner
        self.sync_waypoints()
        self.get_logger().info('OfflineIPP: Initial waypoints synced with the trajectory planner')
        self.get_logger().info('Shutting down offline IPP node')

    def compute_init_paths(self):
        # Initialize random SGP parameters
        likelihood_variance = 1e-4
        kernel = gpflow.kernels.RBF(variance=1.0, 
                                    lengthscales=1.0)

        # Get the initial IPP solution
        transformer = IPPTransform(n_dim=2, 
                                     num_robots=self.num_robots)
        # Sample uniform random initial waypoints and compute initial paths
        Xu_init = get_inducing_pts(self.X_train, self.num_waypoints*self.num_robots)

        Xu_init, _ = run_tsp(Xu_init, 
                             num_vehicles=self.num_robots,
                             resample=self.num_waypoints)
        Xu_init = Xu_init.reshape(-1, 2)

        # Optimize the SGP
        IPP_model, _ = continuous_sgp(self.num_waypoints, 
                                      self.X_train,
                                      likelihood_variance,
                                      kernel,
                                      transformer,
                                      Xu_init=Xu_init)

        # Generate new paths from optimized waypoints
        self.waypoints = IPP_model.inducing_variable.Z.numpy().reshape(self.num_robots, 
                                                                       self.num_waypoints, -1)

        # Print path lengths
        self.get_logger().info('OfflineIPP: Initial IPP solution found') 
        path_lengths = transformer.distance(np.concatenate(self.waypoints, axis=0)).numpy()
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
            plt.plot(path[:, 0], path[:, 1], 
                     label='Path', zorder=0, marker='o')
            plt.scatter(self.data[i][:, 0], self.data[i][:, 1],
                        s=1, label='Candidates', zorder=1)
        plt.savefig('/tmp/OfflineIPP.png')

    '''
    Send the new waypoints to the trajectory planner and 
    update the current waypoint from the service
    '''
    def sync_waypoints(self):
        for robot_idx in range(self.num_robots):
            service = f'tb3_{robot_idx}/offlineIPP'
            offline_ipp_service = self.create_client(OfflineIPP, service)
            self.get_logger().info(service)

            try:
                while not offline_ipp_service.wait_for_service(timeout_sec=1.0):
                    self.get_logger().info('service not avaliable, waiting...')

                service_data = OfflineIPPData()
                for waypoint in self.waypoints[robot_idx]:
                    service_data.waypoints.append(Point(x=waypoint[0],
                                                        y=waypoint[1]))
                for point in self.data[robot_idx]:
                    service_data.x_train.append(Point(x=point[0],
                                                      y=point[1]))
                success = offline_ipp_service.call_async(service_data)
                rclpy.sign_until_future_complete(self, success)
                if future.result() is not None:
                    self.get_logger().info(f'Service call successful')
                else:
                    self.get_logger().info(f'Service call failed: {future.exeception()}')
            except Exception as e:
                print(f'Service call failed: {e}')


if __name__ == '__main__':

    rclpy.init()

    X_train_data = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]]

    node = offlineIPP(X_train=X_train_data)

    # Define the extent of the environment
    xx = np.linspace(-1.5, 1.5, 25)
    yy = np.linspace(-1.5, 1.5, 25)
    X_train = np.array(np.meshgrid(xx, yy)).T.reshape(-1, 2)

    # Get model parameters
    if node.has_parameter('/num_waypoints'):
        num_waypoints=self.get_parameter('/num_waypoints')
    else:
        num_waypoints=10

    if node.has_parameter('/num_robots'):
        num_robots=self.get_parameter('/num_robots')
    else:
        num_robots=1

    # Start the offline IPP mission
    offlineIPP(X_train, 
               num_waypoints=num_waypoints, 
               num_robots=num_robots)

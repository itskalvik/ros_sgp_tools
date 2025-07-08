#! /usr/bin/env python3

import os
import time
import h5py
import yaml
import shutil
import importlib
import traceback
from threading import Lock

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, \
                                  ReentrantCallbackGroup
import matplotlib.pyplot as plt

import numpy as np
from time import gmtime, strftime

from sgptools.methods import get_method
from sgptools.kernels import get_kernel
from sgptools.utils.tsp import run_tsp, resample_path
from sgptools.utils.misc import polygon2candidates, project_waypoints
from sgptools.utils.gpflow import *
from sgptools.core.transformations import IPPTransform
from sgptools.core.osgpr import init_osgpr

from ros_sgp_tools.srv import Waypoint
from geometry_msgs.msg import Point
from utils import *

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from message_filters import ApproximateTimeSynchronizer
from ament_index_python.packages import get_package_share_directory

import tensorflow as tf
tf.random.set_seed(2024)
np.random.seed(2024)


class PathPlanner(Node):
    """
    Informative path planner
    """
    def __init__(self):
        super().__init__('path_planner')
        self.get_logger().info('Initializing')

        plan_fname = os.path.join(get_package_share_directory('ros_sgp_tools'), 
                                                              'launch', 'data',
                                                              'mission.plan')
        self.declare_parameter('geofence_plan', plan_fname)
        plan_fname = self.get_parameter('geofence_plan').get_parameter_value().string_value
        self.get_logger().info(f'GeoFence Plan File: {plan_fname}')

        config_fname = os.path.join(get_package_share_directory('ros_sgp_tools'), 
                                                                'launch', 'data',
                                                                'config.yaml')
        self.declare_parameter('config_file', config_fname)
        config_fname = self.get_parameter('config_file').get_parameter_value().string_value
        self.get_logger().info(f'Config File: {config_fname}')
        with open(config_fname, 'r') as file:
            self.config = yaml.safe_load(file)

        self.declare_parameter('data_folder', '')
        self.data_folder = self.get_parameter('data_folder').get_parameter_value().string_value
        self.get_logger().info(f'Data Folder: {self.data_folder}')
        
        # Create h5py file to store sensor data and other mission parameters
        time_stamp = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        self.data_folder = os.path.join(self.data_folder, f'IPP-mission-{time_stamp}')
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        shutil.copy(plan_fname, self.data_folder)
        shutil.copy(config_fname, self.data_folder)
        data_fname = os.path.join(self.data_folder, f'mission-log.hdf5')
        self.data_file = h5py.File(data_fname, "a")
        self.dset_X = self.data_file.create_dataset("X", (0, 2), 
                                                    maxshape=(None, 2), 
                                                    dtype=np.float64,
                                                    chunks=True)
        self.dset_y = self.data_file.create_dataset("y", (0, 1), 
                                                    maxshape=(None, 1), 
                                                    dtype=np.float64,
                                                    chunks=True)

        # Get the mission plan (fence and start location) and normalize
        self.mission_type = self.config.get('robot').get('mission_type')
        if self.mission_type == 'Waypoint':
            self.fence_vertices, self.start_location, waypoints = get_mission_plan(plan_fname, 
                                                                                   get_waypoints=True)
        else:
            self.fence_vertices, self.start_location = get_mission_plan(plan_fname,
                                                                        get_waypoints=False)
            
        self.X_objective = polygon2candidates(self.fence_vertices, num_samples=5000)
        self.X_objective = np.array(self.X_objective).reshape(-1, 2)
        self.X_scaler = LatLonStandardScaler()
        self.X_scaler.fit(self.X_objective)
        self.X_objective = self.X_scaler.transform(self.X_objective)
        self.start_location = self.X_scaler.transform(np.array([self.start_location[:2]]))

        if self.mission_type == 'Waypoint':
            self.waypoints = waypoints[:, :2]
            self.waypoints = self.X_scaler.transform(self.waypoints)
        elif self.mission_type == 'IPP':
            self.init_models(init_param_model=False)
        elif self.mission_type == 'AdaptiveIPP':
            self.init_models()
        else:
            raise ValueError(f'Invalid mission type: {self.mission_type}')
        
        # Compute distances between waypoints for estimating waypoint arrival time
        lat_lon_waypoints = self.X_scaler.inverse_transform(self.waypoints)
        self.distances = haversine(lat_lon_waypoints[1:], 
                                    lat_lon_waypoints[:-1])

        # Save fence_vertices and initial path to the data store
        self.data_file.create_dataset("fence_vertices", 
                                      self.fence_vertices.shape, 
                                      dtype=np.float64,
                                      data=self.fence_vertices)
        fname = f"waypoints_{-1}-{strftime('%H-%M-%S', gmtime())}"
        self.data_file.create_dataset(fname, 
                                      self.waypoints.shape, 
                                      dtype=np.float64,
                                      data=self.X_scaler.inverse_transform(self.waypoints))
        self.plot_paths(fname, self.waypoints, update_waypoint=0)
        
        # setup variables
        self.data_X = []
        self.data_y = []
        self.current_waypoint = -1
        self.data_lock = Lock()
        self.waypoints_lock = Lock()
        self.runtime_est = None
        self.heading_velocity = 1.0
        self.data_buffer_size = self.config.get('robot').get('data_buffer_size')
        self.stats = RunningStats()

        # Setup data subscribers
        sensors_module = importlib.import_module('sensors')
        self.sensors = []
        sensor_subscribers = []
        sensor_group = ReentrantCallbackGroup()

        data_obj = getattr(sensors_module, 'GPS')()
        self.sensors.append(data_obj)
        sensor_subscribers.append(data_obj.get_subscriber(self,
                                                          callback_group=sensor_group))

        sensor = self.config.get('robot').get('sensor')
        if sensor != 'Altitude':
            data_obj = getattr(sensors_module, sensor)()
            self.sensors.append(data_obj)
            sensor_subscribers.append(data_obj.get_subscriber(self,
                                                              callback_group=sensor_group))

        self.time_sync = ApproximateTimeSynchronizer([*sensor_subscribers],
                                                     queue_size=10, slop=0.1,
                                                     sync_arrival_time=True)
        self.time_sync.registerCallback(self.data_callback)

        # Setup the timer to update the parameters and waypoints
        # Makes sure only one instance runs at a time
        timer_group = MutuallyExclusiveCallbackGroup()
        self.timer = self.create_timer(5.0, self.update_with_data,
                                       callback_group=timer_group)
        
        # Setup waypoint service
        self.create_service(Waypoint, 
                            'waypoint',
                            self.waypoint_service_callback)
        self.create_subscription(
            Float32MultiArray, 'mavros/waypoint_eta', 
            self.eta_callback,
            rclpy.qos.qos_profile_sensor_data)
        
    '''
    Callback to get the current waypoint and shutdown the node once the mission ends
    '''
    def eta_callback(self, msg):
        self.heading_velocity = msg.data[2]
        if self.current_waypoint < len(self.waypoints)-1:
            self.distances[self.current_waypoint] = msg.data[1]

    def waypoint_service_callback(self, request, response):
        if not request.ok:
            self.get_logger().error('Path follower failed to reach a waypoint; shutting down online planner!')
            rclpy.shutdown()
        else:
            self.current_waypoint += 1

        if self.current_waypoint >= len(self.waypoints):
            response.new_waypoint = False
            return response
        else:
            self.get_logger().info(f'Current waypoint: {self.current_waypoint}')

        self.waypoints_lock.acquire()
        response.new_waypoint = True
        waypoint = self.waypoints[self.current_waypoint].reshape(1, -1)
        waypoint = self.X_scaler.inverse_transform(waypoint)[0]
        response.waypoint = Point(x=waypoint[0],
                                  y=waypoint[1])
        self.waypoints_lock.release()
        return response

    def init_models(self, init_ipp_model=True, init_param_model=True):
        hyperparameter_config = self.config['hyperparameters']
        self.kernel = hyperparameter_config['kernel_function']
        kernel_kwargs = hyperparameter_config['kernel']
        kernel = get_kernel(self.kernel)(**kernel_kwargs)
        noise_variance = float(hyperparameter_config['noise_variance'])

        if init_ipp_model:
            self.ipp_model_config = self.config['ipp_model']
            self.num_waypoints = self.ipp_model_config['num_waypoints']
            
            # Sample uniform random initial waypoints and compute initial paths
            # Sample one less waypoint per robot and add the home position as the first waypoint
            X_init = get_inducing_pts(self.X_objective, (self.num_waypoints-1))
            self.get_logger().info(f"Running TSP solver to get the initial path...")
            X_init, _ = run_tsp(X_init,
                                 start_nodes=self.start_location,
                                 **self.config.get('tsp'))
            X_init = np.array(X_init)

            transform_kwargs = self.ipp_model_config.get('transform')
            self.distance_budget = None
            # Map distance budget in meters to normalized units
            if transform_kwargs.get('distance_budget') is not None:
                self.distance_budget = transform_kwargs['distance_budget']
                transform_kwargs['distance_budget'] = self.X_scaler.meters2units(self.distance_budget)
            transform = IPPTransform(Xu_fixed=X_init[:, :1, :],
                                     **transform_kwargs)

            ipp_model = get_method(self.ipp_model_config['method'])
            self.ipp_model = ipp_model(self.num_waypoints, 
                                       X_objective=self.X_objective,
                                       kernel=kernel,
                                       noise_variance=noise_variance,
                                       transform=transform,
                                       X_init=X_init[0])

            # Project the waypoints to be within the bounds of the environment
            self.get_logger().info(f"Running IPP solver to update the initial path...")
            self.ipp_model_kwargs = self.ipp_model_config.get('optimizer')
            self.waypoints = self.ipp_model.optimize(**self.ipp_model_kwargs)[0]
            self.waypoints = project_waypoints(self.waypoints, self.X_objective)

            if self.distance_budget is not None:
                distance = self.ipp_model.transform.distance(self.waypoints.reshape(-1, 2)).numpy()
                distance = self.X_scaler.units2meters(distance)
                if distance > self.distance_budget:
                    self.get_logger().warn("Distance budget constraint violated! Consider increasing the transform's constraint_weight!")
                self.get_logger().info(f"Distance Budget: {self.distance_budget:.2f} m")
                self.get_logger().info(f"Path Length: {distance[0]:.2f} m")

        if init_param_model:
            # Initialize the param model
            self.param_model_config = self.config['param_model']
            self.param_model_kwargs = self.param_model_config.get('optimizer')
            self.param_model_method = self.param_model_config['method']
            if self.param_model_method == 'SSGP':
                self.train_param_inducing = self.param_model_config.get('train_inducing')
                self.num_param_inducing = self.param_model_config['num_inducing']
                self.param_model = init_osgpr(self.X_objective, 
                                              num_inducing=self.num_param_inducing, 
                                              kernel=kernel,
                                              noise_variance=noise_variance)

    def data_callback(self, *args):
        # Use data only when the vechicle is moving (avoids failed cholskey decomposition in OSGPR)
        if self.current_waypoint > 0 and self.current_waypoint < len(self.waypoints):
            position = self.sensors[0].process_msg(args[0])
            if len(args) == 1:
                data_X = [position[:2]]
                data_y = [position[2]]
            else:
                # position data is used by only a few sensors
                data_X, data_y = self.sensors[1].process_msg(args[1], 
                                                             position=position)
            # Update running stats
            self.stats.push(data_y, per_dim=True)

            self.data_lock.acquire()
            self.data_X.extend(data_X)
            self.data_y.extend(data_y)
            self.data_lock.release()

    def update_with_data(self, force_update=False):
        # Update the hyperparameters and waypoints if the buffer is full 
        # or if force_update is True and atleast num_param_inducing data points are available
        if len(self.data_X) > self.data_buffer_size or \
            (force_update and len(self.data_X) > self.num_param_inducing) or \
            self.current_waypoint >= len(self.waypoints):

            # Make local copies of the data and clear the data buffers         
            self.data_lock.acquire()
            data_X = np.array(self.data_X).reshape(-1, 2).astype(float)
            data_y = np.array(self.data_y).reshape(-1, 1).astype(float)
            self.data_X = []
            self.data_y = []
            self.data_lock.release()

            # Update the parameters
            if self.mission_type == 'AdaptiveIPP' and self.current_waypoint < len(self.waypoints):
                start_time = self.get_clock().now().to_msg().sec
                self.update_param(data_X, data_y)
                end_time = self.get_clock().now().to_msg().sec
                runtime = end_time-start_time
                self.get_logger().info(f'Param update time: {runtime} secs')
                self.runtime_est = runtime

                # Update the waypoints
                start_time = self.get_clock().now().to_msg().sec
                new_waypoints, update_waypoint = self.update_waypoints()
                end_time = self.get_clock().now().to_msg().sec
                runtime = end_time-start_time
                self.get_logger().info(f'IPP update time: {runtime} secs')
                self.runtime_est += runtime
            else:
                update_waypoint = -1

            # If waypoints were updated, accept waypoints if update waypoint was not already passed
            if update_waypoint != -1:
                self.waypoints_lock.acquire()
                if self.current_waypoint < update_waypoint:
                    self.waypoints = new_waypoints
                self.waypoints_lock.release()
                
            # Dump data to data store
            self.dset_X.resize(self.dset_X.shape[0]+len(data_X), axis=0)   
            self.dset_X[-len(data_X):] = data_X

            self.dset_y.resize(self.dset_y.shape[0]+len(data_y), axis=0)   
            self.dset_y[-len(data_y):] = data_y

            current_waypoint = self.current_waypoint if self.current_waypoint>-1 else 0
            fname = f"waypoints_{current_waypoint}-{strftime('%H-%M-%S', gmtime())}"
            if update_waypoint != -1:
                lat_lon_waypoints = self.X_scaler.inverse_transform(new_waypoints)
                self.distances = haversine(lat_lon_waypoints[1:], 
                                           lat_lon_waypoints[:-1])
                dset = self.data_file.create_dataset(fname,
                                                     self.waypoints.shape, 
                                                     dtype=np.float64,
                                                     data=lat_lon_waypoints)
                dset.attrs['update_waypoint'] = update_waypoint

            self.plot_paths(fname, self.waypoints,
                            self.X_scaler.transform(data_X),
                            update_waypoint=update_waypoint)

            # Shutdown the online planner if the mission is complete
            if self.current_waypoint >= len(self.waypoints):
                # Rerun method to get last batch of data
                if not force_update and len(self.data_X) > 0:
                    self.update_with_data(force_update=True)
                self.get_logger().info('Finished mission, shutting down online planner')
                rclpy.shutdown()

    def update_waypoints(self):
        """Update the IPP solution."""
        self.get_logger().info('Updating IPP solution...')

        # Freeze the visited inducing points
        update_waypoint = self.get_update_waypoint()
        if update_waypoint == -1:
            return self.waypoints, update_waypoint
        
        Xu_visited = self.waypoints[:update_waypoint+1]
        Xu_visited = Xu_visited.reshape(1, -1, 2)
        self.ipp_model.transform.update_Xu_fixed(Xu_visited)

        # Get the new inducing points for the path
        self.ipp_model.update(self.param_model.kernel,
                              self.param_model.likelihood.variance)
        waypoints = self.ipp_model.optimize(**self.ipp_model_kwargs)[0]

        # Might move waypoints before the current waypoint (reset to avoid update rejection)
        waypoints = project_waypoints(waypoints, self.X_objective)
        waypoints[:update_waypoint+1] = self.waypoints[:update_waypoint+1]

        if self.distance_budget is not None:
            distance = self.ipp_model.transform.distance(waypoints.reshape(-1, 2)).numpy()
            distance = self.X_scaler.units2meters(distance)
            if distance > self.distance_budget:
                self.get_logger().warn("Distance budget constraint violated! Consider increasing the transform's constraint_weight!")
            self.get_logger().info(f"Distance Budget: {self.distance_budget:.2f} m")
            self.get_logger().info(f"Path Length: {distance[0]:.2f} m")

        return waypoints, update_waypoint

    def update_param(self, X_new, y_new):
        """Update the SSGP parameters."""
        self.get_logger().info('Updating SSGP parameters...')

        # Normalize the data, use running mean and std for sensor data
        X_new = self.X_scaler.transform(X_new)
        y_new = (y_new - self.stats.mean) / self.stats.std
        self.get_logger().info(f'Data Mean: {self.stats.mean}')
        self.get_logger().info(f'Data Std: {self.stats.std}')

        # Don't update the parameters if the current target is the last waypoint
        if self.current_waypoint >= self.num_waypoints-1:
            return
        
        # Set the incucing points to be along the traversed portion of the planned path
        inducing_variable = np.copy(self.waypoints[:self.current_waypoint+1])
        # Ensure inducing points do not extend beyond the collected data
        inducing_variable[-1] = X_new[-1]
        # Resample the path to the number of inducing points
        inducing_variable = resample_path(inducing_variable, 
                                          self.num_param_inducing)
        
        # Update ssgp with new batch of data
        self.param_model.update((X_new, y_new), 
                                inducing_variable=inducing_variable)
        
        if self.train_param_inducing:
            trainable_variables = None
        else:
            trainable_variables=self.param_model.trainable_variables[1:]

        try:
            optimize_model(self.param_model,
                           trainable_variables=trainable_variables,
                           **self.param_model_kwargs)
        except Exception as e:
            # Failsafe for cholesky decomposition failure
            self.get_logger().error(f"{traceback.format_exc()}")
            self.get_logger().warning(f"Failed to update parameter model! Resetting parameter model...")
            self.init_models(init_ipp_model=False)

        if self.kernel == 'RBF':
            self.get_logger().info(f'SSGP kernel lengthscales: {self.param_model.kernel.lengthscales.numpy():.4f}')
            self.get_logger().info(f'SSGP kernel variance: {self.param_model.kernel.variance.numpy():.4f}')
            self.get_logger().info(f'SSGP likelihood variance: {self.param_model.likelihood.variance.numpy():.4f}')

    def get_update_waypoint(self):
        """Returns the waypoint index that is safe to update."""
        # Do not update the current target waypoint
        for i in range(self.current_waypoint, len(self.distances)):
            if self.distances[i]/self.heading_velocity > self.runtime_est:
                # Map path edge idx to waypoint index
                return i+1
        # Do not update the path if none of waypoints can be 
        # updated before the vehicle reaches them
        return -1

    def plot_paths(self, fname, waypoints, 
                   X_data=None, inducing_pts=None, 
                   update_waypoint=None):
        plt.figure()
        plt.gca().set_aspect('equal')
        plt.xlabel('X')
        plt.xlabel('Y')
        plt.scatter(self.X_objective[:, 0], self.X_objective[:, 1], 
                    marker='.', s=1, label='Candidates')
        plt.plot(waypoints[:, 0], waypoints[:, 1], 
                 label='Path', marker='o', c='r')
        
        if update_waypoint is not None:
            plt.scatter(waypoints[update_waypoint, 0], waypoints[update_waypoint, 1],
                        label='Update Waypoint', zorder=2, c='g')
        
        if X_data is not None:
            plt.scatter(X_data[:, 0], X_data[:, 1], 
                        label='Data', c='b', marker='x', zorder=3, s=1)
            
        if inducing_pts is not None:
            plt.scatter(inducing_pts[:, 0], inducing_pts[:, 1], 
                        label='Inducing Pts', marker='.', c='g', zorder=4, s=2)

        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.savefig(os.path.join(self.data_folder, 
                                 f'{fname}.png'),
                                 bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    # Start the online IPP mission
    rclpy.init()

    online_ipp = PathPlanner()
    executor = MultiThreadedExecutor()
    executor.add_node(online_ipp)
    executor.spin()
#! /usr/bin/env python3

import os
import h5py
import importlib
from threading import Lock

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

import matplotlib.pyplot as plt

import gpflow
import numpy as np
from time import gmtime, strftime
from sgptools.utils.misc import project_waypoints
from sgptools.models.continuous_sgp import *
from sgptools.models.core.transformations import *
from sgptools.models.core.osgpr import *
from sgptools.utils.tsp import resample_path
from utils import CustomStandardScaler

from ros_sgp_tools.srv import Waypoints, IPP
from geometry_msgs.msg import Point
from std_msgs.msg import Int32

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from message_filters import ApproximateTimeSynchronizer

import tensorflow as tf
tf.random.set_seed(2024)
np.random.seed(2024)


class OnlineIPP(Node):
    """
    Class to create an online IPP mission.

    Note: Make sure the number of waypoints is small enough so that 
    the parameters update and waypoints updates are fast enough to 
    reach the robot before it reaches the next waypoint.

    Args:
        num_param_inducing (int): The number of inducing points for the OSGPR model.
        buffer_size (int): The size of the buffers to store the sensor data.
        data_type (str): The type of sensor data to use for kernel parameter updates.
                         Any class name from sensors.py is a valid option.
                         Currently supports: AltitudeData, SonarData, and PressureData.
    """
    def __init__(self):
        super().__init__('OnlineIPP')
        self.get_logger().info('Initializing')

        # Declare parameters
        self.declare_parameter('num_param_inducing', 40)
        self.num_param_inducing = self.get_parameter('num_param_inducing').get_parameter_value().integer_value
        self.get_logger().info(f'Num Param Inducing: {self.num_param_inducing}')

        self.declare_parameter('buffer_size', 100)
        self.buffer_size = self.get_parameter('buffer_size').get_parameter_value().integer_value
        self.get_logger().info(f'Data Buffer Size: {self.buffer_size}')

        self.declare_parameter('data_type', 'Altitude')
        self.data_type = self.get_parameter('data_type').get_parameter_value().string_value
        self.get_logger().info(f'Data Type: {self.data_type}')

        self.declare_parameter('adaptive_ipp', True)
        self.adaptive_ipp = self.get_parameter('adaptive_ipp').get_parameter_value().bool_value
        self.get_logger().info(f'Adaptive IPP: {self.adaptive_ipp}')

        self.declare_parameter('data_folder', '')
        self.data_folder = self.get_parameter('data_folder').get_parameter_value().string_value
        self.get_logger().info(f'Data Folder: {self.data_folder}')

        self.declare_parameter('visualize', False)
        self.visualize = self.get_parameter('visualize').get_parameter_value().bool_value
        self.get_logger().info(f'Visualize: {self.visualize}')

        # Create sensor data h5py file
        time_stamp = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        self.data_folder = os.path.join(self.data_folder, f'IPP-mission-{time_stamp}')
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        data_fname = os.path.join(self.data_folder, f'mission-log.hdf5')
        self.data_file = h5py.File(data_fname, "a")
        self.dset_X = self.data_file.create_dataset("X", (0, 2), 
                                                    maxshape=(None, 2), 
                                                    dtype=np.float32,
                                                    chunks=True)
        self.dset_y = self.data_file.create_dataset("y", (0, 1), 
                                                    maxshape=(None, 1), 
                                                    dtype=np.float32,
                                                    chunks=True)

        # setup variables
        self.waypoints = None
        self.data_X = []
        self.data_y = []
        self.current_waypoint = -1
        self.lock = Lock()
        
        # Setup the service to receive the waypoints and X_train data
        self.srv = self.create_service(IPP, 'offlineIPP', 
                                       self.offlineIPP_service_callback)
        
        # Wait to get the waypoints from the offline IPP planner
        # Stop service after receiving the waypoints from the offline IPP planner
        while rclpy.ok() and self.waypoints is None:
            rclpy.spin_once(self, timeout_sec=1.0)
        del self.srv

        self.num_waypoints = len(self.waypoints)

        # Init the sgp models for online IPP and parameter estimation
        self.init_sgp_models()
        
        # Sync the waypoints with the mission planner
        waypoints = self.X_scaler.inverse_transform(np.array(self.waypoints))
        self.sync_waypoints(waypoints)
        self.get_logger().info('Initial waypoints synced with the mission planner')

        if not self.adaptive_ipp:
            self.get_logger().info('Running non-adaptive IPP, shutting down online planner')
            rclpy.shutdown()

        # Setup the subscribers
        self.create_subscription(Int32, 
                                 'current_waypoint', 
                                 self.current_waypoint_callback, 
                                 QoSProfile(depth=10))

        # Setup data subscribers
        sensors_module = importlib.import_module('sensors')
        self.sensors = []
        sensor_subscribers = []

        data_obj = getattr(sensors_module, 'GPS')()
        self.sensors.append(data_obj)
        sensor_subscribers.append(data_obj.get_subscriber(self))

        if self.data_type != 'Altitude':
            data_obj = getattr(sensors_module, self.data_type)()
            self.sensors.append(data_obj)
            sensor_subscribers.append(data_obj.get_subscriber(self))

        self.time_sync = ApproximateTimeSynchronizer([*sensor_subscribers],
                                                     queue_size=10, slop=0.1)
        self.time_sync.registerCallback(self.data_callback)

        # Setup the timer to update the parameters and waypoints
        # Makes sure only one instance runs at a time
        self.timer = self.create_timer(5.0, self.update_with_data,
                                       callback_group=MutuallyExclusiveCallbackGroup())

    '''
    Service callback to receive the waypoints, X_train, and sampling rate from offlineIPP node

    Args:
        req: Request containing the waypoints and X_train data
    Returns:
        WaypointsResponse: Response containing the success flag
    '''
    def offlineIPP_service_callback(self, request, response):
        self.use_altitude = request.data.use_altitude
        self.n_dim = 3 if self.use_altitude else 2

        data = request.data.waypoints
        self.waypoints = []
        for i in range(len(data)):
            self.waypoints.append([data[i].x, data[i].y, data[i].z])
        self.waypoints = np.array(self.waypoints)[:, :self.n_dim]
        self.num_waypoints = len(self.waypoints)

        data = request.data.x_candidates
        self.X_candidates = []
        for i in range(len(data)):
            self.X_candidates.append([float(data[i].x), float(data[i].y)])
        self.X_candidates = np.array(self.X_candidates)

        fence_vertices = request.data.fence_vertices
        fence_vertices_array = []
        for i in range(len(fence_vertices)):
            fence_vertices_array.append([float(fence_vertices[i].x), float(fence_vertices[i].y)])
        fence_vertices_array = np.array(fence_vertices_array)

        self.sampling_rate = request.data.sampling_rate

        # Save fence_vertices and sampling rate to data store
        dset = self.data_file.create_dataset("fence_vertices", 
                                             fence_vertices_array.shape, 
                                             dtype=np.float32,
                                             data=fence_vertices_array)
        dset.attrs['sampling_rate'] = self.sampling_rate

        # Normalize the train set and waypoints
        self.X_scaler = CustomStandardScaler()
        self.X_scaler.fit(self.X_candidates)
        self.X_candidates = self.X_scaler.transform(self.X_candidates)
        self.waypoints = self.X_scaler.transform(np.array(self.waypoints))
    
        response.success = True
        return response
    
    def init_sgp_models(self):
        # Initialize random SGP parameters
        likelihood_variance = 1e-4
        lengthscales = 0.1
        variance = 0.5

        # Initilize SGP for IPP with path received from offline IPP node
        kernel = gpflow.kernels.RBF(lengthscales=lengthscales, 
                                    variance=variance)
        self.transform = IPPTransform(n_dim=self.n_dim,
                                      sampling_rate=self.sampling_rate,
                                      num_robots=1)
        self.IPP_model, _ = continuous_sgp(self.num_waypoints, 
                                           self.X_candidates,
                                           likelihood_variance,
                                           kernel,
                                           self.transform,
                                           max_steps=0,
                                           Xu_init=self.waypoints)
        
        # Initialize the OSGPR model
        self.param_model = init_osgpr(self.X_candidates, 
                                      num_inducing=self.num_param_inducing, 
                                      lengthscales=lengthscales, 
                                      variance=variance, 
                                      noise_variance=likelihood_variance)

    '''
    Callback to get the current waypoint and shutdown the node once the mission ends
    '''
    def current_waypoint_callback(self, msg):
        if msg.data == self.num_waypoints:
            self.get_logger().info('Mission complete')
            rclpy.shutdown()
        self.current_waypoint = msg.data

    def data_callback(self, *args):
        # Use data only when the vechicle is moving (avoids failed cholskey decomposition in OSGPR)
        if self.current_waypoint > 1 and self.current_waypoint != self.num_waypoints:
            position = self.sensors[0].process_msg(args[0])
            if len(args) == 1:
                data_X = [position[:2]]
                data_y = [position[2]]
            else:
                # position data is used by only a few sensors
                data_X, data_y = self.sensors[1].process_msg(args[1], 
                                                             position=position)

            self.lock.acquire()
            self.data_X.extend(data_X)
            self.data_y.extend(data_y)
            self.lock.release()

    def sync_waypoints(self, waypoints):
        # Send the new waypoints to the mission planner and 
        # update the current waypoint from the service
        
        service = f'waypoints'
        waypoints_service = self.create_client(Waypoints, service)
        request = Waypoints.Request()

        while not waypoints_service.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'{service} service not avaliable, waiting...')

        for waypoint in waypoints:
            z = waypoint[2] if self.use_altitude else 20.0
            request.waypoints.waypoints.append(Point(x=waypoint[0],
                                                     y=waypoint[1],
                                                     z=z))
        
        future = waypoints_service.call_async(request)
        while future.result() is not None:
            rclpy.spin_once(self, timeout_sec=0.5)

    def plot_paths(self, waypoints, X_data, pts_ind):
        current_waypoint = self.current_waypoint if self.current_waypoint>-1 else 0

        fname = f"waypoints_{current_waypoint}-{strftime('%H-%M-%S', gmtime())}"
        dset = self.data_file.create_dataset(fname,
                                             waypoints.shape, 
                                             dtype=np.float32,
                                             data=self.X_scaler.inverse_transform(waypoints))
        dset.attrs['lengthscales'] = self.param_model.kernel.lengthscales.numpy()
        dset.attrs['variance'] = self.param_model.kernel.variance.numpy()
        dset.attrs['likelihood_variance'] = self.param_model.likelihood.variance.numpy()

        plt.figure()
        plt.gca().set_aspect('equal')
        plt.xlabel('X')
        plt.xlabel('Y')
        plt.scatter(self.X_candidates[:, 0], self.X_candidates[:, 1], 
                    marker='.', s=1, label='Candidates')
        plt.plot(waypoints[:, 0], waypoints[:, 1], 
                 label='Path', marker='o', c='r')
        plt.scatter(waypoints[current_waypoint, 0], waypoints[current_waypoint, 1],
                    label='Update Waypoint', zorder=2, c='g')
        
        plt.scatter(X_data[:, 0], X_data[:, 1], 
                    label='Data', c='b', marker='x', zorder=3)
        plt.scatter(pts_ind[:, 0], pts_ind[:, 1], 
                    label='Pts', marker='.', c='r', zorder=4)

        plt.legend()
        plt.savefig(os.path.join(self.data_folder, 
                                 f'{fname}.png'),
                                 bbox_inches='tight')
        plt.close()

    def update_with_data(self, force_update=False):
        # Update the hyperparameters and waypoints if the buffer is full 
        # or if force_update is True and atleast num_param_inducing data points are available
        if len(self.data_X) > self.buffer_size or \
            (force_update and len(self.data_X) > self.num_param_inducing):

            start_time = self.get_clock().now().to_msg().sec


            # Make local copies of the data and clear the data buffers         
            self.lock.acquire()
            data_X = np.array(self.data_X).reshape(-1, 2)
            data_y = np.array(self.data_y).reshape(-1, 1)
            self.data_X = []
            self.data_y = []
            self.lock.release()

            # Update the parameters and the waypoints after 
            # the current waypoint the robot is heading to
            self.update_param(self.X_scaler.transform(data_X), data_y)
            self.update_waypoints()

            # Sync the waypoints with the mission planner
            waypoints = self.X_scaler.inverse_transform(np.array(self.waypoints))
            self.sync_waypoints(waypoints)

            end_time = self.get_clock().now().to_msg().sec
            self.get_logger().info('Updated waypoints synced with the mission planner')
            self.get_logger().info(f'Update time: {end_time-start_time} secs')

            # Dump data to data store
            pts_ind = self.param_model.inducing_variable.Z.numpy()
            self.plot_paths(np.array(self.waypoints), 
                            self.X_scaler.transform(data_X), 
                            pts_ind)

            self.dset_X.resize(self.dset_X.shape[0]+len(data_X), axis=0)   
            self.dset_X[-len(data_X):] = data_X

            self.dset_y.resize(self.dset_y.shape[0]+len(data_y), axis=0)   
            self.dset_y[-len(data_y):] = data_y
            
    def update_waypoints(self):
        """Update the IPP solution."""

        # Freeze the visited inducing points
        Xu_visited = self.waypoints.copy()[:self.current_waypoint]
        Xu_visited = np.array(Xu_visited).reshape(1, -1, self.n_dim)
        self.IPP_model.transform.update_Xu_fixed(Xu_visited)

        # Get the new inducing points for the path
        self.IPP_model.update(self.param_model.likelihood.variance,
                              self.param_model.kernel)
        optimize_model(self.IPP_model, 
                       kernel_grad=False, 
                       optimizer='scipy',
                       method='CG')

        waypoints = self.IPP_model.inducing_variable.Z
        waypoints = self.IPP_model.transform.expand(waypoints,
                                                    expand_sensor_model=False).numpy()
        waypoints = project_waypoints(waypoints, self.X_candidates)
        self.waypoints = waypoints

    def update_param(self, X_new, y_new):
        """Update the OSGPR parameters."""
        # Set the incucing points to be along the traversed path
        inducing_variable = self.waypoints[:self.current_waypoint]
        # Ensure inducing points do not extend beyond the collected data
        inducing_variable[-1] = X_new[-1]
        # Resample the path to the number of inducing points
        inducing_variable = resample_path(inducing_variable, 
                                          self.num_param_inducing)
        self.param_model.update((X_new, y_new), 
                                inducing_variable=inducing_variable)
        optimize_model(self.param_model,
                       trainable_variables=self.param_model.trainable_variables[1:], 
                       optimizer='scipy')

        self.get_logger().info(f'SSGP kernel lengthscales: {self.param_model.kernel.lengthscales.numpy():.4f}')
        self.get_logger().info(f'SSGP kernel variance: {self.param_model.kernel.variance.numpy():.4f}')
        self.get_logger().info(f'SSGP likelihood variance: {self.param_model.likelihood.variance.numpy():.4f}')


if __name__ == '__main__':
    # Start the online IPP mission
    rclpy.init()

    online_ipp = OnlineIPP()
    executor = MultiThreadedExecutor()
    executor.add_node(online_ipp)
    executor.spin()
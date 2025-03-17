#! /usr/bin/env python3

import os
import time
import h5py
import importlib
import traceback
from threading import Lock

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

import matplotlib.pyplot as plt

import gpflow
from gpflow.utilities.traversal import deepcopy

import numpy as np
from time import gmtime, strftime

from sgptools.utils.misc import project_waypoints
from sgptools.models.continuous_sgp import *
from sgptools.models.core.transformations import *
from sgptools.models.core.osgpr import *
from sgptools.utils.tsp import resample_path
from sgptools.kernels.attentive_kernel import AttentiveKernel
from sgptools.kernels.neural_kernel import NeuralSpectralKernel

from ros_sgp_tools.srv import Waypoints, IPP
from ros_sgp_tools.msg import ETA
from geometry_msgs.msg import Point

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from message_filters import ApproximateTimeSynchronizer
from utils import CustomStandardScaler

import tensorflow as tf
tf.random.set_seed(2024)
np.random.seed(2024)


class OnlineIPP(Node):
    """
    Online IPP mission planner
    """
    def __init__(self):
        super().__init__('OnlineIPP')
        self.get_logger().info('Initializing')

        # Declare parameters
        self.declare_parameter('num_param_inducing', 40)
        self.num_param_inducing = self.get_parameter('num_param_inducing').get_parameter_value().integer_value
        self.get_logger().info(f'Num Param Inducing: {self.num_param_inducing}')

        self.declare_parameter('data_buffer_size', 100)
        self.data_buffer_size = self.get_parameter('data_buffer_size').get_parameter_value().integer_value
        self.get_logger().info(f'Data Buffer Size: {self.data_buffer_size}')

        self.declare_parameter('train_param_inducing', False)
        self.train_param_inducing = self.get_parameter('train_param_inducing').get_parameter_value().bool_value
        self.get_logger().info(f'Train Param Inducing: {self.train_param_inducing}')

        self.declare_parameter('data_type', 'Altitude')
        self.data_type = self.get_parameter('data_type').get_parameter_value().string_value
        self.get_logger().info(f'Data Type: {self.data_type}')

        self.declare_parameter('adaptive_ipp', True)
        self.adaptive_ipp = self.get_parameter('adaptive_ipp').get_parameter_value().bool_value
        self.get_logger().info(f'Adaptive IPP: {self.adaptive_ipp}')

        self.declare_parameter('data_folder', '')
        self.data_folder = self.get_parameter('data_folder').get_parameter_value().string_value
        self.get_logger().info(f'Data Folder: {self.data_folder}')
        
        self.declare_parameter('kernel', 'RBF')
        self.kernel = self.get_parameter('kernel').get_parameter_value().string_value
        self.get_logger().info(f'Kernel: {self.kernel}')
        self.kernel = None if self.kernel=='None' else self.kernel
        self.kernel = None if not self.adaptive_ipp else self.kernel

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
        self.runtime_est = None
        
        # Setup the service to receive the waypoints and X_train data
        srv = self.create_service(IPP, 'offlineIPP', 
                                  self.offlineIPP_service_callback)
        # Wait to get the waypoints from the offline IPP planner
        while rclpy.ok() and self.waypoints is None:
            rclpy.spin_once(self, timeout_sec=1.0)
        srv.destroy()

        # Init the sgp models for online IPP and parameter estimation
        self.num_waypoints = len(self.waypoints)
        if self.kernel is not None:
            self.init_sgp_models()
        
        # Sync the waypoints with the mission planner
        self.waypoints_service = self.create_client(Waypoints, 'waypoints')
        waypoints = self.X_scaler.inverse_transform(self.waypoints)
        self.sync_waypoints(waypoints)
        fname = f"waypoints_0-{strftime('%H-%M-%S', gmtime())}"
        self.plot_paths(fname, self.waypoints, update_waypoint=0)
        self.get_logger().info('Initial waypoints synced with the mission planner')

        # Setup the subscribers
        self.create_subscription(ETA, 'eta', 
                                 self.eta_callback, 
                                 QoSProfile(depth=10))

        # Setup data subscribers
        sensors_module = importlib.import_module('sensors')
        self.sensors = []
        sensor_subscribers = []
        sensor_group = ReentrantCallbackGroup()

        data_obj = getattr(sensors_module, 'GPS')()
        self.sensors.append(data_obj)
        sensor_subscribers.append(data_obj.get_subscriber(self,
                                                          callback_group=sensor_group))

        if self.data_type != 'Altitude':
            data_obj = getattr(sensors_module, self.data_type)()
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
        self.waypoints = self.X_scaler.transform(self.waypoints)
    
        response.success = True
        return response
    
    def init_sgp_models(self, IPP_model=True):
        # Initialize random SGP parameters
        likelihood_variance = 1e-4

        # Initilize the kernel
        if self.kernel == 'RBF':
            kernel = gpflow.kernels.RBF(lengthscales=0.1, 
                                        variance=0.5)
        elif self.kernel == 'Attentive':
            kernel = AttentiveKernel(np.linspace(0.1, 2.5, 4), 
                                     dim_hidden=10)
        elif self.kernel == 'Neural':
            kernel = NeuralSpectralKernel(input_dim=2, 
                                          Q=3, 
                                          hidden_sizes=[4, 4])
            
        # Initilize SGP for IPP with path received from offline IPP node
        if IPP_model:
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
                                      kernel=kernel, 
                                      noise_variance=likelihood_variance)

    '''
    Callback to get the current waypoint and shutdown the node once the mission ends
    '''
    def eta_callback(self, msg):
        self.current_waypoint = msg.current_waypoint
        self.eta = msg.eta

    def data_callback(self, *args):
        # Use data only when the vechicle is moving (avoids failed cholskey decomposition in OSGPR)
        if self.current_waypoint > 0 and self.current_waypoint != self.num_waypoints:
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
        request = Waypoints.Request()
        while not self.waypoints_service.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'waypoints service not avaliable, waiting...')

        for waypoint in waypoints:
            z = waypoint[2] if self.use_altitude else 20.0
            request.waypoints.waypoints.append(Point(x=waypoint[0],
                                                     y=waypoint[1],
                                                     z=z))
        self.waypoint_response = None
        future = self.waypoints_service.call_async(request)
        future.add_done_callback(self.save_future_response)
    
    def save_future_response(self, future):
        self.waypoint_response = future.result()

    def update_with_data(self, force_update=False):
        # Update the hyperparameters and waypoints if the buffer is full 
        # or if force_update is True and atleast num_param_inducing data points are available
        if len(self.data_X) > self.data_buffer_size or \
            (force_update and len(self.data_X) > self.num_param_inducing):

            # Make local copies of the data and clear the data buffers         
            self.lock.acquire()
            data_X = np.array(self.data_X).reshape(-1, 2).astype(float)
            data_y = np.array(self.data_y).reshape(-1, 1).astype(float)
            self.data_X = []
            self.data_y = []
            self.lock.release()

            # Update the parameters
            if self.kernel is not None:
                start_time = self.get_clock().now().to_msg().sec
                self.update_param(self.X_scaler.transform(data_X), data_y)
                end_time = self.get_clock().now().to_msg().sec
                self.get_logger().info(f'Param update time: {end_time-start_time} secs')
                # Store the initial IPP runtime estimate
                if self.runtime_est is None: 
                    self.runtime_est = end_time-start_time

                # Update the waypoints
                start_time = self.get_clock().now().to_msg().sec
                new_waypoints, update_waypoint = self.update_waypoints()
                end_time = self.get_clock().now().to_msg().sec
                runtime = end_time-start_time
                self.get_logger().info(f'IPP update time: {runtime} secs')
                # Store the IPP runtime upper bound
                if self.runtime_est < runtime:
                    self.runtime_est = runtime
            else:
                update_waypoint = -1

            # Sync the waypoints with the mission planner
            if update_waypoint != -1:
                lat_lon_waypoints = self.X_scaler.inverse_transform(new_waypoints)
                self.sync_waypoints(lat_lon_waypoints)
                while self.waypoint_response is None:
                    time.sleep(0.1)

                # Update the waypoints only if the mission planner accepts the new waypoints
                if self.waypoint_response.success:
                    self.waypoints = new_waypoints

            # Dump data to data store
            self.dset_X.resize(self.dset_X.shape[0]+len(data_X), axis=0)   
            self.dset_X[-len(data_X):] = data_X

            self.dset_y.resize(self.dset_y.shape[0]+len(data_y), axis=0)   
            self.dset_y[-len(data_y):] = data_y

            current_waypoint = self.current_waypoint if self.current_waypoint>-1 else 0
            fname = f"waypoints_{current_waypoint}-{strftime('%H-%M-%S', gmtime())}"
            if update_waypoint != -1:
                dset = self.data_file.create_dataset(fname,
                                                     self.waypoints.shape, 
                                                     dtype=np.float32,
                                                     data=lat_lon_waypoints)
                dset.attrs['update_waypoint'] = update_waypoint

            self.plot_paths(fname, self.waypoints,
                            self.X_scaler.transform(data_X),
                            update_waypoint=update_waypoint)

            # Shutdown the online planner if the mission planner has shutdown
            if self.current_waypoint >= self.num_waypoints-1 and self.eta[-1] < 3:
                # Rerun method to get last batch of data
                if not force_update:
                    self.update_with_data(force_update=True)
                self.get_logger().info('Finished mission, shutting down online planner')
                rclpy.shutdown()

    def update_waypoints(self):
        """Update the IPP solution."""

        # Freeze the visited inducing points
        update_waypoint = self.get_update_waypoint()
        if update_waypoint == -1:
            return self.waypoints, update_waypoint
        
        Xu_visited = self.waypoints[:update_waypoint+1]
        Xu_visited = Xu_visited.reshape(1, -1, self.n_dim)
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
                                                    expand_sensor_model=False)
        # Might move waypoints before the current waypoint (reset to avoid update rejection)
        waypoints = project_waypoints(waypoints.numpy(), self.X_candidates)
        waypoints[:update_waypoint+1] = self.waypoints[:update_waypoint+1]

        return waypoints, update_waypoint

    def update_param(self, X_new, y_new):
        """Update the OSGPR parameters."""
        # Don't update the parameters if the current target is the last waypoint
        if self.current_waypoint >= self.num_waypoints-1:
            return
        
        # Set the incucing points to be along the traversed path
        inducing_variable = np.copy(self.waypoints[:self.current_waypoint+1])
        # Ensure inducing points do not extend beyond the collected data
        inducing_variable[-1] = X_new[-1]
        # Resample the path to the number of inducing points
        inducing_variable = resample_path(inducing_variable, 
                                          self.num_param_inducing)
        self.param_model.update((X_new, y_new), 
                                inducing_variable=inducing_variable)
        
        if self.train_param_inducing:
            trainable_variables = None
        else:
            trainable_variables=self.param_model.trainable_variables[1:]

        try:
            optimize_model(self.param_model,
                           trainable_variables=trainable_variables,
                           optimizer='scipy',
                           method='CG')
        except Exception as e:
            # Failsafe for cholesky decomposition failure
            self.get_logger().info(f"{traceback.format_exc()}")
            self.get_logger().info(f"Failed to update parameter model")
            self.get_logger().info(f"Resetting parameter model")
            self.init_sgp_models(IPP_model=False)

        if self.kernel == 'RBF':
            self.get_logger().info(f'SSGP kernel lengthscales: {self.param_model.kernel.lengthscales.numpy():.4f}')
            self.get_logger().info(f'SSGP kernel variance: {self.param_model.kernel.variance.numpy():.4f}')
        self.get_logger().info(f'SSGP likelihood variance: {self.param_model.likelihood.variance.numpy():.4f}')

    def get_update_waypoint(self):
        """Returns the waypoint index that is safe to update."""
        # Do not update the current target waypoint
        for i in range(self.current_waypoint, len(self.eta)):
            if self.eta[i] > self.runtime_est:
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
        plt.scatter(self.X_candidates[:, 0], self.X_candidates[:, 1], 
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

    online_ipp = OnlineIPP()
    executor = MultiThreadedExecutor()
    executor.add_node(online_ipp)
    executor.spin()
#! /usr/bin/env python3
import importlib

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

import matplotlib.pyplot as plt
from utils import project_waypoints

import gpflow
import numpy as np
from sgptools.models.continuous_sgp import *
from sgptools.models.core.transformations import *
from sgptools.models.core.osgpr import *
from sklearn.preprocessing import StandardScaler

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

        qos_profile = QoSProfile(depth=10)  
        
        # setup variables
        self.waypoints = None
        self.data_X = []
        self.data_y = []
        self.current_waypoint = -1

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
        self.sync_waypoints()
        self.get_logger().info('Initial waypoints synced with the mission planner')

        # Setup the subscribers
        self.create_subscription(Int32, 
                                 'current_waypoint', 
                                 self.current_waypoint_callback, 
                                 qos_profile)

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
                                                     queue_size=10, slop=0.05)
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
        data = request.data.waypoints

        self.waypoints = []
        for i in range(len(data)):
            self.waypoints.append([data[i].x, data[i].y])
        self.num_waypoints = len(self.waypoints)

        data = request.data.x_train
        self.X_train = []
        for i in range(len(data)):
            self.X_train.append([float(data[i].x), float(data[i].y)])
        self.X_train = np.array(self.X_train)

        self.sampling_rate = request.data.sampling_rate

        # Normalize the train set and waypoints
        self.X_scaler = StandardScaler()
        self.X_train = self.X_scaler.fit_transform(self.X_train)
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
        self.transform = IPPTransform(n_dim=2,
                                      num_robots=1)
        self.IPP_model, _ = continuous_sgp(self.num_waypoints, 
                                           self.X_train,
                                           likelihood_variance,
                                           kernel,
                                           self.transform,
                                           max_steps=0,
                                           Xu_init=self.waypoints)
        
        # Initialize the OSGPR model
        self.param_model = init_osgpr(self.X_train, 
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
                
            self.data_X.extend(data_X)
            self.data_y.extend(data_y)
  
    def sync_waypoints(self):
        # Send the new waypoints to the mission planner and 
        # update the current waypoint from the service
        
        service = f'waypoints'
        waypoints_service = self.create_client(Waypoints, service)
        request = Waypoints.Request()

        while not waypoints_service.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'{service} service not avaliable, waiting...')

        # Undo data normalization to map the coordinates to real world coordinates
        waypoints = self.X_scaler.inverse_transform(np.array(self.waypoints))
        self.plot_paths(waypoints)
        for waypoint in waypoints:
            request.waypoints.waypoints.append(Point(x=waypoint[0],
                                                     y=waypoint[1],
                                                     z=20.0))
        
        future = waypoints_service.call_async(request)
        while future.result() is not None:
            rclpy.spin_once(self, timeout_sec=0.5)

    def plot_paths(self, waypoints):
        plt.figure()
        waypoints = np.array(self.waypoints)
        plt.scatter(self.X_train[:, 1], self.X_train[:, 0], 
                    marker='.', s=1)
        plt.plot(waypoints[:, 1], waypoints[:, 0], 
                 label='Path', marker='o', c='r')
        plt.savefig(f'IPPMission-({self.current_waypoint}).png')
        np.savetxt(f'IPPMission-({self.current_waypoint}).csv', waypoints)

    def update_with_data(self, force_update=False):
        # Update the parameters and waypoints if the buffer is full and 
        # empty the buffer after updating 

        if len(self.data_X) > self.buffer_size or \
            (force_update and len(self.data_X) > self.num_param_inducing):

            start_time = self.get_clock().now().to_msg().sec

            # Make local copies of the data
            data_X = np.array(self.data_X).reshape(-1, 2)
            data_y = np.array(self.data_y).reshape(-1, 1)

            # Normalize X locations
            data_X = self.X_scaler.transform(data_X)

            # Empty global data buffers
            self.data_X = []
            self.data_y = []

            # Update the parameters and the waypoints after 
            # the current waypoint the robot is heading to
            self.update_param(data_X, data_y)
            self.update_waypoints(self.current_waypoint)

            # Sync the waypoints with the mission planner
            self.sync_waypoints()

            end_time = self.get_clock().now().to_msg().sec
            self.get_logger().info('Updated waypoints synced with the mission planner')
            self.get_logger().info(f'Update time: {end_time-start_time} secs')

    def update_waypoints(self, current_waypoint):
        """Update the IPP solution."""

        # Freeze the visited inducing points
        Xu_visited = self.waypoints.copy()[:current_waypoint]
        Xu_visited = np.array(Xu_visited).reshape(1, -1, 2)
        self.IPP_model.transform.update_Xu_fixed(Xu_visited)

        # Get the new inducing points for the path
        self.IPP_model.update(self.param_model.likelihood.variance,
                              self.param_model.kernel)
        optimize_model(self.IPP_model, 
                       kernel_grad=False, 
                       optimizer='scipy',
                       method='CG')

        self.waypoints = self.IPP_model.inducing_variable.Z
        self.waypoints = self.IPP_model.transform.expand(self.waypoints).numpy()
        self.waypoints = project_waypoints(self.waypoints, self.X_train)

    def update_param(self, X_new, y_new):
        """Update the OSGPR parameters."""
        # Get the new inducing points for the path
        self.param_model.update((X_new, y_new))
        optimize_model(self.param_model,
                       trainable_variables=self.param_model.trainable_variables[1:], 
                       optimizer='scipy')

        self.get_logger().info(f'SSGP Kernel lengthscales: {self.param_model.kernel.lengthscales.numpy():.4f}')
        self.get_logger().info(f'SSGP Kernel variance: {self.param_model.kernel.variance.numpy():.4f}')


if __name__ == '__main__':
    # Start the online IPP mission
    rclpy.init()

    online_ipp = OnlineIPP()
    executor = MultiThreadedExecutor()
    executor.add_node(online_ipp)
    executor.spin()
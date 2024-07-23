#! /usr/bin/env python3

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

import os
import matplotlib.pyplot as plt
from utils import plan2data, project_waypoints
from ament_index_python.packages import get_package_share_directory

import gpflow
import numpy as np
from sgptools.models.continuous_sgp import *
from sgptools.models.core.transformations import *
from sgptools.models.core.osgpr import *
from sklearn.preprocessing import StandardScaler

from ros_sgp_tools.srv import Waypoints, IPP
from geometry_msgs.msg import Point
from std_msgs.msg import Int32
from sensor_msgs.msg import NavSatFix, Range, Image
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from message_filters import Subscriber, ApproximateTimeSynchronizer

from cv_bridge import CvBridge

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
        X_train (np.ndarray): The training data for the IPP model, 
                              used to approximate the bounds of the environment.
        num_param_inducing (int): The number of inducing points for the OSGPR model.
        buffer_size (int): The size of the buffers to store the sensor data.
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

        plan_fname = os.path.join(get_package_share_directory('ros_sgp_tools'), 
                                                              'launch', 
                                                              'sample.plan')
        self.declare_parameter('geofence_plan', plan_fname)
        plan_fname = self.get_parameter('geofence_plan').get_parameter_value().string_value
        self.get_logger().info(f'GeoFence Plan File: {plan_fname}')

        # Get the data and normalize
        # X_train is used only to get the normalization factors and then discarded
        X_train, home_position = plan2data(plan_fname, num_samples=5000)

        X_train = np.array(X_train).reshape(-1, 2)
        self.X_scaler = StandardScaler()
        self.X_train = self.X_scaler.fit_transform(X_train)

        # Shift home position for each robot to avoid collision with other robots
        robot_idx = self.get_namespace()
        robot_idx = int(robot_idx.split('_')[-1])
        home_position = np.array(home_position[:2])
        home_position += np.array([3/111111, 0.0])*robot_idx
        home_position = home_position.reshape(-1, 2)
        self.home_position = self.X_scaler.transform(home_position)

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
        self.position_sub = Subscriber(self, NavSatFix, 
                                       "mavros/global_position/global",
                                       qos_profile=rclpy.qos.qos_profile_sensor_data)
        self.data_sub = Subscriber(self, Image, 
                                   "zed/zed_node/depth/depth_registered",
                                   qos_profile=qos_profile)
        self.time_sync = ApproximateTimeSynchronizer([self.position_sub, self.data_sub],
                                                     queue_size=2, slop=0.1)
        self.time_sync.registerCallback(self.data_callback)

        # Setup variables to get data from depth map
        self.bridge = CvBridge()

        # Get data from 3x3 grid
        delta = 1280//6
        mask_x = [delta, delta*3, delta*5]
        delta = 720//6
        mask_y = [delta, delta*3, delta*5]
        mask_x, mask_y = np.meshgrid(mask_x, mask_y)
        self.mask_x = mask_x.reshape(-1)
        self.mask_y = mask_y.reshape(-1)        
        self.bool_mask = np.array([[-1,  1], [0,  1], [1,  1],
                                   [-1,  0], [0,  0], [1,  0],
                                   [-1, -1], [0, -1], [1, -1]])

        # ZED FoV 110° (H) x 70° (V)
        # Mapped to right triangle base for 1 meter height
        self.fov_scale = np.array([1.42815, 0.70021])
        self.dist_scale = 1/111111 # 1 meter offset in lat/long

        # Setup the timer to update the parameters and waypoints
        # Makes sure only one instance runs at a time
        self.timer = self.create_timer(5.0, self.update_with_data,
                                       callback_group=MutuallyExclusiveCallbackGroup())

    '''
    Service callback to receive the waypoints and X_train data from offlineIPP node

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
        self.IPP_model.transform.update_Xu_fixed(self.home_position.reshape(1, 1, 2))
        
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

    def data_callback(self, position_msg, data_msg):
        # Use data only when the vechicle is moving (avoids failed cholskey decomposition in OSGPR)
        if self.current_waypoint > 1 and self.current_waypoint != self.num_waypoints:
            # scale FoV spread in proportion to the height from the ground
            depth = self.bridge.imgmsg_to_cv2(data_msg, desired_encoding='passthrough')
            height = np.mean(depth[np.where(np.isfinite(depth))])
            for i, point in enumerate(zip(self.mask_x, self.mask_y)):
                data = depth[point[1], point[0]]
                if np.isfinite(data):
                    self.data_X.append(np.array([position_msg.latitude, position_msg.longitude]) + \
                                  (self.bool_mask[i]*height*self.fov_scale*self.dist_scale))
                    self.data_y.append(data)

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
                                                     y=waypoint[1]))
        
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

            '''
            # Use only first batch to compute mean and std
            if self.y_scaler is None:
                self.y_scaler = StandardScaler()
                data_y = self.y_scaler.fit_transform(data_y)
            else:
                data_y = self.y_scaler.transform(data_y)
            '''

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
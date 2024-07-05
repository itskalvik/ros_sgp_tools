#! /usr/bin/env python3

import gpflow
import numpy as np
from sgptools.models.continuous_sgp import *
from sgptools.models.core.transformations import *
from sgptools.models.core.osgpr import *
from sklearn.preprocessing import StandardScaler

from sensor_msgs.msg import NavSatFix

from ros_sgp_tools.srv import Waypoints, IPP
from geometry_msgs.msg import Point
from std_msgs.msg import Int32
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile


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
    def __init__(self, 
                 X_train,
                 num_param_inducing=40,
                 buffer_size=100):
        super().__init__('OnlineIPP')
        self.get_logger().info('Initializing')

        # Setup the ROS node
        self.ns = self.get_namespace()

        qos_profile = QoSProfile(depth=10)  
        
        # setup variables
        self.waypoints = None
        self.num_param_inducing = num_param_inducing
        self.data_X = []
        self.data_y = []
        self.buffer_size = buffer_size
        self.current_waypoint = -1

        self.X_train = np.array(X_train).reshape(-1, 2)
        self.X_scaler = StandardScaler()
        self.X_train = self.X_scaler.fit_transform(self.X_train)*10.0

        # Setup the service to receive the waypoints and X_train data
        self.srv = self.create_service(IPP, 
                                       'robot_0/offlineIPP', 
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
        self.vehicle_pose_subscriber = self.create_subscription(NavSatFix, 
                                            '/mavros/global_position/global', 
                                            self.data_callback, 
                                            rclpy.qos.qos_profile_sensor_data)

        self.create_subscription(Int32, 
                                 'current_waypoint', 
                                 self.current_waypoint_callback, 
                                 qos_profile)

        # Setup the timer to update the parameters and waypoints
        self.timer = self.create_timer(5.0, self.update_with_data)

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
        kernel = gpflow.kernels.RBF(variance=1.0, 
                                    lengthscales=1.0)
        # Initilize SGP for IPP with path received from offline IPP node
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
                                      num_inducing=40, 
                                      lengthscales=1.0, 
                                      variance=1.0, 
                                      noise_variance=1e-4)

    '''
    Callback to get the current waypoint. If the robot has reached a waypoint and 
    is heading to the next waypoint, update the parameters and the future waypoints.  
    '''
    def current_waypoint_callback(self, msg):
        if msg.data == self.num_waypoints:
            self.get_logger().info('Mission complete')
            rclpy.shutdown()
        elif msg.data > self.current_waypoint:
            self.update_with_data(force_update=True)
        self.current_waypoint = msg.data

    def data_callback(self, msg):
        # Use data only when the vechicle is moving (avoids failed cholskey decomposition in OSGPR)
        if self.current_waypoint > 0 and self.current_waypoint != self.num_waypoints:
            
            # Append the new data to the buffers
            self.data_X.append([msg.latitude, msg.longitude])
            self.data_y.append(msg.altitude)

    def sync_waypoints(self):
        # Send the new waypoints to the mission planner and 
        # update the current waypoint from the service
        
        service = f'waypoints'
        waypoints_service = self.create_client(Waypoints, service)
        request = Waypoints.Request()

        while not waypoints_service.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'{service} service not avaliable, waiting...')

        # Undo data normalization to map the coordinates to real world coordinates
        waypoints = self.X_scaler.inverse_transform(np.array(self.waypoints)/10.0)
        for waypoint in waypoints:
            request.waypoints.waypoints.append(Point(x=waypoint[0],
                                                     y=waypoint[1]))
        
        future = waypoints_service.call_async(request)
        while future.result() is not None:
            rclpy.spin_once(self, timeout_sec=0.5)
        self.get_logger().info(f'Service call successful')

    def update_with_data(self, force_update=False):
        # Update the parameters and waypoints if the buffer is full and 
        # empty the buffer after updating 

        if len(self.data_X) > self.buffer_size or \
            (force_update and len(self.data_X) > self.num_param_inducing):

            # Make local copies of the data
            data_X = np.array(self.data_X).reshape(-1, 2)
            data_y = np.array(self.data_y).reshape(-1, 1)

            # Empty global data buffers
            self.data_X = []
            self.data_y = []

            # Update the parameters and the waypoints after 
            # the current waypoint the robot is heading to
            self.update_param(data_X, data_y)
            self.update_waypoints(self.current_waypoint)

            # Sync the waypoints with the mission planner
            self.sync_waypoints()
            self.get_logger().info('Updated waypoints synced with the mission planner')

    def update_waypoints(self, current_waypoint):
        """Update the IPP solution."""

        # Freeze the visited inducing points
        Xu_visited = self.waypoints.copy()[:current_waypoint]
        Xu_visited = np.array(Xu_visited).reshape(1, -1, 2)
        self.IPP_model.transform.update_Xu_fixed(Xu_visited)

        # Get the new inducing points for the path
        self.IPP_model.update(self.param_model.likelihood.variance,
                              self.param_model.kernel)
        optimize_model(self.IPP_model, max_steps=100, 
                       kernel_grad=False, 
                       lr=1e-4, 
                       optimizer='adam')

        self.waypoints = self.IPP_model.inducing_variable.Z
        self.waypoints = self.IPP_model.transform.expand(self.waypoints).numpy()

    def update_param(self, X_new, y_new):
        """Update the OSGPR parameters."""
        # Get the new inducing points for the path
        self.param_model.update((X_new, y_new))
        optimize_model(self.param_model, optimizer='scipy')


def main():
    # Start the online IPP mission
    rclpy.init()

    # Define the extent of the environment
    xx = np.linspace(-80.73595662137639, -80.73622611393395, 50)
    yy = np.linspace(35.30684640691298, 35.306729637839894, 50)
    X_train = np.array(np.meshgrid(xx, yy)).T.reshape(-1, 2)

    # Start the online IPP mission
    online_ipp = OnlineIPP(X_train)
    rclpy.spin(online_ipp)


if __name__ == '__main__':
    main()

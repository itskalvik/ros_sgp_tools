#! /usr/bin/env python3

import gpflow
import numpy as np
from sgptools.models.continuous_sgp import *
from sgptools.models.core.transformations import *
from sgptools.models.core.osgpr import *

from ros_sgp_tools.srv import Waypoints, IPP
from ros_sgp_tools.msg import SensorData, WaypointsList, IPPData
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
        num_param_inducing (int): The number of inducing points for the OSGPR model.
        buffer_size (int): The size of the buffers to store the sensor data.
    """
    def __init__(self, 
                 num_param_inducing=40,
                 buffer_size=100):
        super().__init__('online_ipp')

        # Setup the ROS node
        self.ns = self.get_namespace()
        qos_profile = QoSProfile(depth=10)  
        
        self.get_logger().info(self.ns+'OnlineIPP: Initializing')

        # setup variables
        self.waypoints = None
        self.num_param_inducing = num_param_inducing
        print(self.num_param_inducing)

        # Setup the data buffers and the current waypoint
        self.data_X = []
        self.data_y = []
        self.buffer_size = buffer_size
        self.current_waypoint = 0
        self.num_waypoints = 10
        xx = np.linspace(-1.5, 1.5, 25)
        yy = np.linspace(-1.5, 1.5, 25)
        self.X_train = np.array(np.meshgrid(xx, yy)).T.reshape(-1, 2)

        # Setup the service to receive the waypoints and X_train data
        self.srv = self.create_service(IPP, 
                                                'tb3_0/offlineIPP', 
                                                self.offlineIPP_service_callback)
        
        # Wait to get the waypoints from the offline IPP planner
        # Stop service after receiving the waypoints from the offline IPP planner
        while not rclpy.ok() and self.waypoints is None:
            rclpy.spin_once(self, timeout_sec=1.0)
            
        del self.srv

        # Init the sgp models for online IPP and parameter estimation
        self.init_sgp_models()
        
        # Sync the waypoints with the trajectory planner
        self.sync_waypoints()

        self.get_logger().info(self.ns+'OnlineIPP: Initial waypoints synced with the trajectory planner')

        
        # Setup the subscribers
        self.create_subscription(SensorData,
                         self.ns+'sensor_data',
                         self.data_callback, qos_profile)
                                                                   
        self.create_subscription(Int32, 
                         self.ns+'current_waypoint', 
                         self.current_waypoint_callback, qos_profile)

        # Setup the timer to update the parameters and waypoints
        self.timer = self.create_timer(5.0, self.update_with_data)

        rclpy.spin(self)

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
        self.transformer = IPPTransform(n_dim=2,
                                          num_robots=1)
        self.IPP_model, _ = continuous_sgp(self.num_waypoints, 
                                           self.X_train,
                                           likelihood_variance,
                                           kernel,
                                           self.transformer,
                                           max_steps=0,
                                           Xu_init=self.waypoints)
        # Initialize the OSGPR model
        xx = np.linspace(-1.5, 1.5, 25)
        yy = np.linspace(-1.5, 1.5, 25)
        X_train = np.array(np.meshgrid(xx, yy)).T.reshape(-1, 2)
        self.param_model = init_osgpr(X_train, num_inducing=40)

    '''
    Callback to get the current waypoint. If the robot has reached a waypoint and 
    is heading to the next waypoint, update the parameters and the future waypoints.  
    '''
    def current_waypoint_callback(self, msg):
        if msg.data == self.num_waypoints:
            self.get_logger().info(self.ns+'OnlineIPP: Mission complete')
            rclpy.shutdown()
        elif msg.data > self.current_waypoint:
            self.update_with_data(force_update=True)
        self.current_waypoint = msg.data
        ##print(self.current_waypoint) #For testing

    def data_callback(self, msg):
        # Use data only when the vechicle is moving (avoids failed cholskey decomposition in OSGPR)
        if self.current_waypoint > 0 and self.current_waypoint != self.num_waypoints:
            # Append the new data to the buffers
            self.data_X.append(msg.x)
            self.data_y.append(msg.y)

    def sync_waypoints(self):
        # Send the new waypoints to the trajectory planner and 
        # update the current waypoint from the service
        
        waypoints_service = self.create_client(Waypoints, 'waypoints')
        request = Waypoints.Request()

        try:
            while not waypoints_service.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('service not avaliable, waiting...')

            for waypoint in self.waypoints:
                request.waypoints.waypoints.append(Point(x=waypoint[0],
                                                        y=waypoint[1]))
            
            future = waypoints_service.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                self.get_logger().info(f'Service call successful')
            else:
                self.get_logger().info(f'Service call failed: {future.exeception()}')
        except Exception as e:
            print(self.ns+f': Service call failed: {e}')

    def update_with_data(self, timer=None, force_update=False):
        # Update the parameters and waypoints if the buffer is full and 
        # empty the buffer after updating 
        if len(self.data_X) > self.buffer_size or (force_update and \
              len(self.data_X) > self.num_param_inducing):
            
            self.get_logger().info(self.ns+'OnlineIPP: Updating parameters and waypoints')

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

            # Sync the waypoints with the trajectory planner
            self.sync_waypoints()
            self.get_logger().info(self.ns+'OnlineIPP: Updated waypoints synced with the trajectory planner')

    def update_waypoints(self, current_waypoint):
        """Update the IPP solution."""

        # Freeze the visited inducing points
        Xu_visited = self.waypoints.copy()[:current_waypoint]
        Xu_visited = np.array(Xu_visited).reshape(1, -1, 2)
        self.transformer.update_Xu_fixed(Xu_visited)

        # Get the new inducing points for the path
        self.IPP_model.update(self.param_model.likelihood.variance,
                              self.param_model.kernel)
        optimize_model(self.IPP_model, max_steps=100, 
                       kernel_grad=False, 
                       lr=1e-2, optimizer='adam')

        self.waypoints = self.IPP_model.inducing_variable.Z
        self.waypoints = self.IPP_model.transformer.expand(self.waypoints).numpy()

    def update_param(self, X_new, y_new):
        """Update the OSGPR parameters."""
        # Get the new inducing points for the path
        self.param_model.update((X_new, y_new))
        optimize_model(self.param_model, optimizer='scipy')


def main(args=None):
    # Start the online IPP mission
    rclpy.init(args=args)
    online_ipp = OnlineIPP()
    rclpy.spin(online_ipp)
    rclpy.shutdown()
    ### OnlineIPP()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)

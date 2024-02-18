#! /usr/bin/env python3

import gpflow
import numpy as np
from sgp_ipp.utils.tsp import run_tsp
from sgp_ipp.utils.sensor_placement import *
from sgp_ipp.models.osgpr import OSGPR_VFE
from sgp_ipp.models.transformations import IPPTransformer

from ros_sgp_ipp.srv import Waypoints, WaypointsResponse
from geometry_msgs.msg import Point, PoseStamped
from ros_sgp_ipp.msg import RSSI, WaypointsList
from std_msgs.msg import Int32
import message_filters
import rospy


class OnlineIPP:
    """
    Class to create an online IPP mission.

    Note: Make sure the number of waypoints is small enough so that 
    the parameters update and waypoints updates are fast enough to 
    reach the robot before it reaches the next waypoint.

    Args:
        X_train (np.ndarray): The training data for the IPP model, 
                              used to approximate the bounds of the environment.
        num_waypoints (int): The number of waypoints/inducing points for the IPP model.
        num_param_inducing (int): The number of inducing points for the OSGPR model.
        buffer_size (int): The size of the buffers to store the sensor data.
    """
    def __init__(self, X_train, 
                 num_waypoints=10, 
                 num_param_inducing=40,
                 buffer_size=100):
        super().__init__()

        # Setup the ROS node
        rospy.init_node('online_ipp', anonymous=True)                      
        rospy.loginfo('Initializing online IPP mission')

        # setup variables
        self.waypoints = None
        self.X_train = X_train
        self.num_waypoints = num_waypoints
        self.num_param_inducing = num_param_inducing

        # Setup the data buffers and the current waypoint
        self.data_X = []
        self.data_y = []
        self.buffer_size = buffer_size
        self.current_waypoint = 0
        self.mean = None
        self.std = None

        # Setup the service to receive the waypoints
        self.waypoint_service = rospy.Service('online_waypoints', 
                                              Waypoints, 
                                              self.waypoint_service_callback)
        
        # Wait to get the waypoints from the offline IPP planner
        while not rospy.is_shutdown() and self.waypoints is None:
            rospy.sleep(1)
        # Stop service after receiving the waypoints from the offline IPP planner
        del self.waypoint_service
        
        # Setup the subscribers
        ns = rospy.get_namespace()
        pose_subscriber = message_filters.Subscriber('/vrpn_client_node'+ns+'pose', 
                                                     PoseStamped)
        rssi_subscriber = message_filters.Subscriber('/rssi', 
                                                     RSSI)
        data_subscriber = message_filters.ApproximateTimeSynchronizer([pose_subscriber, 
                                                                       rssi_subscriber], 
                                                                       10, 0.1, 
                                                                       allow_headerless=True)
        data_subscriber.registerCallback(self.data_callback)

        rospy.Subscriber('/current_waypoint', Int32, self.current_waypoint_callback)

        # Setup the timer to update the parameters and waypoints
        self.timer = rospy.Timer(rospy.Duration(5), self.update_with_data)

        # Init the sgp models for online IPP and parameter estimation
        xx = np.linspace(-2, 2, 25)
        yy = np.linspace(-2, 2, 25)
        self.X_train = np.array(np.meshgrid(xx, yy)).T.reshape(-1, 2)
        self.init_sgp_models()
        
        # Sync the waypoints with the trajectory planner
        self.sync_waypoints()
        rospy.loginfo('OnlineIPP: Initial waypoints synced with the trajectory planner')

        rospy.spin()

    def init_sgp_models(self):
        # Initialize random SGP parameters
        likelihood_variance = 1e-4
        kernel = gpflow.kernels.RBF(variance=1.0, 
                                    lengthscales=1.0)

        # Get the initial IPP solution
        self.transformer = IPPTransformer(n_dim=2,
                                          num_robots=1)
        self.IPP_model, _ = get_aug_sgp_sol(self.num_waypoints, 
                                            self.X_train,
                                            likelihood_variance,
                                            kernel,
                                            self.transformer,
                                            num_steps=0,
                                            Xu_init=self.waypoints)

        # Initilize a SGPR model with random parameters for the OSGPR
        # The X_train and y_train are not used to optimize the kernel parameters
        # but they effect the initial inducing point locations, i.e., limits them
        # to the bounds of the data
        Z_init = get_inducing_pts(self.X_train, self.num_param_inducing)
        init_param = gpflow.models.SGPR((self.X_train, np.zeros((self.X_train.shape[0], 1))),
                                        kernel=kernel,
                                        inducing_variable=Z_init, 
                                        noise_variance=likelihood_variance)
        
        # Initialize the OSGPR model using the parameters from the SGPR model
        # The X_train and y_train here will be overwritten in the online phase
        X_train = np.array([[0, 0], [0, 0]])
        y_train = np.array([0, 0]).reshape(-1, 1)
        Zopt = init_param.inducing_variable.Z.numpy()
        mu, Su = init_param.predict_f(Zopt, full_cov=True)
        Kaa = init_param.kernel(Zopt)
        self.param_model = OSGPR_VFE((X_train, y_train),
                                     init_param.kernel,
                                     mu, Su[0], Kaa,
                                     Zopt, Zopt)
        self.param_model.likelihood.variance.assign(init_param.likelihood.variance)

        del init_param

    '''
    Service callback to receive the waypoints and return the current waypoint

    Args:
        req: Request containing the waypoints
    Returns:
        WaypointsResponse: Response containing the current waypoint
    '''
    def waypoint_service_callback(self, req):
        waypoints = req.waypoints.waypoints
        self.waypoints = []
        for i in range(len(waypoints)):
            self.waypoints.append([waypoints[i].x, waypoints[i].y])
        rospy.loginfo('OnlineIPP: Waypoints received')
        return WaypointsResponse(True)

    '''
    Callback to get the current waypoint. If the robot has reached a waypoint and 
    is heading to the next waypoint, update the parameters and the future waypoints.  
    '''
    def current_waypoint_callback(self, msg):
        if msg.data > self.current_waypoint:
            if msg.data == len(self.waypoints):
                rospy.signal_shutdown('OnlineIPP: Mission complete')
                return
            else:
                self.update_with_data(force_update=True)
        self.current_waypoint = msg.data

    def data_callback(self, pose_msg, rssi_msg):
        # Append the new data to the buffers
        self.data_X.append([pose_msg.pose.position.x, 
                            pose_msg.pose.position.y])
        self.data_y.append(rssi_msg.rssi)

    def sync_waypoints(self):
        # Send the new waypoints to the trajectory planner and 
        # update the current waypoint from the service
        rospy.wait_for_service('waypoints')
        try:
            waypoint_service = rospy.ServiceProxy('waypoints', Waypoints)
            waypoints = WaypointsList()
            for waypoint in self.waypoints:
                waypoints.waypoints.append(Point(x=waypoint[0],
                                                 y=waypoint[1]))
            success = waypoint_service(waypoints)
        except rospy.ServiceException as e:
            print(f'Service call failed: {e}')

    def update_with_data(self, timer=None, force_update=False):
        # Update the parameters and waypoints if the buffer is full and 
        # empty the buffer after updating 
        if len(self.data_X) > self.buffer_size or (force_update and \
              len(self.data_X) > self.num_waypoints):
            
            rospy.loginfo('OnlineIPP: Updating parameters and waypoints')

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
            rospy.loginfo('OnlineIPP: Updated waypoints synced with the trajectory planner')

    def update_waypoints(self, current_waypoint):
        """Update the IPP solution."""

        # Freeze the visited inducing points
        Xu_visited = self.waypoints.copy()[:current_waypoint]
        self.transformer.update(Xu_visited)

        # Get the new inducing points for the path
        self.IPP_model.update(self.param_model.likelihood.variance,
                              self.param_model.kernel)
        optimize_model(self.IPP_model, num_steps=100, 
                       kernel_grad=False, 
                       lr=1e-2, opt='adam')

        self.waypoints = self.IPP_model.inducing_variable.Z
        self.waypoints = self.IPP_model.transformer.expand(self.waypoints).numpy()

    def update_param(self, X_new, y_new):
        """Update the OSGPR parameters."""

        # Get the new inducing points for the path
        y_new = self.preprocess_data(y_new)
        self.param_model.update((X_new, y_new))
        optimize_model(self.param_model, opt='scipy')

    def preprocess_data(self, y, alpha=0.1):
        """Normalize the data and return the normalized data."""
        if self.mean is None:
            self.mean = np.mean(y)
            self.std = np.std(y)
        else:
            self.mean = (1 - alpha) * self.mean + alpha * np.mean(y)
            self.std = (1 - alpha) * self.std + alpha * np.std(y)

        # Sanity check to avoid spreading the data too much
        if self.std < 1e-2:
            std = 1
        else:
            std = self.std

        y = (y - self.mean) / std

        return y


def main():

    # Define the extent of the environment
    xx = np.linspace(-2, 2, 25)
    yy = np.linspace(-2, 2, 25)
    X_train = np.array(np.meshgrid(xx, yy)).T.reshape(-1, 2)

    # Start the online IPP mission
    OnlineIPP(X_train)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)

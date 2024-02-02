#! /usr/bin/env python3

import gpflow
import numpy as np
from sgp_ipp.utils.tsp import run_tsp
from sgp_ipp.utils.sensor_placement import *
from sgp_ipp.models.osgpr import OSGPR_VFE
from sgp_ipp.models.transformations import FixedInducingTransformer

from gazebo_msgs.msg import ModelStates
from ros_sgp_ipp.srv import Waypoints
from geometry_msgs.msg import Point
from ros_sgp_ipp.msg import RSSI
import message_filters
import rospy


class OnlineIPP:
    """Class to create an online IPP mission."""

    def __init__(self, X_train, 
                 num_placements=20, 
                 num_param_inducing=40,
                 buffer_size=5000):
        super().__init__('online_ipp')

        self.num_placements = num_placements

        # Initialize random SGP parameters
        likelihood_variance = 1e-4
        kernel = gpflow.kernels.RBF(variance=1.0, 
                                    lengthscales=1.0)

        # Get the initial IPP solution
        self.transformer = FixedInducingTransformer()
        self.IPP_model, _ = get_aug_sgp_sol(num_placements, 
                                            X_train,
                                            likelihood_variance,
                                            kernel,
                                            self.transformer)

        # Reorder the inducing points to match the tsp solution
        self.waypoints = self.IPP_model.inducing_variable.Z.numpy()
        self.waypoints = self.waypoints[run_tsp(self.waypoints)[0]]
        self.IPP_model.inducing_variable.Z.assign(self.waypoints)

        # Initilize a SGPR model with random parameters for the OSGPR
        # The X_train and y_train are not used to optimize the kernel parameters
        # but they effect the initial inducing point locations, i.e., limits them
        # to the bounds of the data
        Z_init = get_inducing_pts(X_train, num_param_inducing)
        init_param = gpflow.models.SGPR((X_train, np.zeros((X_train.shape[0], 1))),
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
        self.online_param = OSGPR_VFE((X_train, y_train),
                                       init_param.kernel,
                                       mu, Su[0], Kaa,
                                       Zopt, Zopt)
        self.online_param.likelihood.variance.assign(init_param.likelihood.variance)

        del init_param

        # Setup the data buffers and the current waypoint
        self.data_X = []
        self.data_y = []
        self.buffer_size = buffer_size
        self.current_waypoint = 0

        # Setup the ROS node
        rospy.init_node('online_ipp', anonymous=True)                      
        
        # Setup the subscribers
        pose_subscriber = message_filters.Subscriber('/gazebo/model_states', 
                                                     ModelStates)
        rssi_subscriber = message_filters.Subscriber('/rssi', 
                                                     RSSI)
        data_subscriber = message_filters.ApproximateTimeSynchronizer([pose_subscriber, 
                                                                       rssi_subscriber], 
                                                                       10, 0.1, 
                                                                       allow_headerless=True)
        data_subscriber.registerCallback(self.data_callback)

        # Setup the timer to update the parameters and waypoints
        self.timer = rospy.Timer(rospy.Duration(5), self.update_with_data)

        # Sync the waypoints with the trajectory planner
        self.sync_waypoints()

        rospy.spin()

    def data_callback(self, pose_msg, rssi_msg):
        # Append the new data to the buffers
        self.data_X.append([pose_msg.pose[1].position.x, 
                            pose_msg.pose[1].position.y])
        self.data_y.append(rssi_msg.rssi)

    def sync_waypoints(self):
        # Send the new waypoints to the trajectory planner and 
        # update the current waypoint from the service
        rospy.wait_for_service('waypoints')
        try:
            waypoint_service = rospy.ServiceProxy('waypoints', Waypoints)
            waypoints = Waypoints()
            for waypoint in self.waypoints:
                waypoints.waypoints.append(Point(x=waypoint[0],
                                                 y=waypoint[1],
                                                 z=0))
            response = waypoint_service(waypoints)
            self.current_waypoint = response.current_waypoint
        except rospy.ServiceException as e:
            print(f'Service call failed: {e}')

    def update_with_data(self):
        # Update the parameters and waypoints if the buffer is full and 
        # empty the buffer after updating 
        if len(self.data_X) >= self.buffer_size:
            self.update_param(np.array(self.data_X),
                              np.array(self.data_y))
            self.update_waypoints(self.current_waypoint)

            # Empty the data buffers
            self.data_X = []
            self.data_y = []

            # Sync the waypoints with the trajectory planner
            self.sync_waypoints()

    def update_waypoints(self, current_waypoint):
        """Update the IPP solution."""

        # Freeze the visited inducing points
        Xu_visited = self.waypoints.copy()[:current_waypoint]
        self.transformer.update(Xu_visited)

        # Get the new inducing points for the path
        self.IPP_model.update(self.online_param.likelihood.variance,
                              self.online_param.kernel)
        optimize_model(self.IPP_model, num_steps=100, 
                       kernel_grad=False, 
                       lr=1e-2, opt='adam')

        self.waypoints = self.IPP_model.inducing_variable.Z
        self.waypoints = self.IPP_model.transformer.expand(self.waypoints).numpy()

    def update_param(self, X_new, y_new):
        """Update the OSGPR parameters."""

        # Get the new inducing points for the path
        self.online_param.update((X_new, y_new))
        optimize_model(self.online_param, opt='scipy')

def main(args=None):
    print('Starting online IPP mission')

    # Define the extent of the environment
    xx = np.linspace(0, 10, 25)
    yy = np.linspace(0, 10, 25)
    X_train = np.array(np.meshgrid(xx, yy)).T.reshape(-1, 2)

    # Start the online IPP mission
    mission = OnlineIPP(X_train)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)

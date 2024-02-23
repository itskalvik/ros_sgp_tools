#! /usr/bin/env python3

import gpflow
import numpy as np
from sgptools.utils.tsp import run_tsp
from sgptools.utils.sensor_placement import *
from sgptools.models.transformations import IPPTransformer
from sklearn.neighbors import KNeighborsClassifier

from ros_sgp_ipp.msg import OfflineIPPData
from ros_sgp_ipp.srv import OfflineIPP
from geometry_msgs.msg import Point
import rospy

import matplotlib.pyplot as plt
np.random.seed(2021)

class offlineIPP:
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
        super().__init__()

        # Setup the ROS node
        rospy.init_node('offline_ipp', anonymous=True)
        rospy.loginfo('Initializing offline IPP mission')

        self.X_train = X_train
        self.num_waypoints = num_waypoints
        self.num_robots = num_robots

        # Get initial solution paths
        self.compute_init_paths()

        # Sync the waypoints with the trajectory planner
        self.sync_waypoints()
        rospy.loginfo('OfflineIPP: Initial waypoints synced with the trajectory planner')
        rospy.loginfo('Shutting down offline IPP node')

    def compute_init_paths(self):
        # Initialize random SGP parameters
        likelihood_variance = 1e-4
        kernel = gpflow.kernels.RBF(variance=1.0, 
                                    lengthscales=1.0)

        # Get the initial IPP solution
        transformer = IPPTransformer(n_dim=2, 
                                     num_robots=self.num_robots)
        # Sample uniform random initial waypoints and compute initial paths
        Xu_init = get_inducing_pts(self.X_train, self.num_waypoints*self.num_robots)
        path_idx, _ = run_tsp(Xu_init, num_vehicles=self.num_robots)
        Xu_init =  [Xu_init[path] for path in path_idx]
        Xu_init = np.concatenate(Xu_init, axis=0)

        # Optimize the SGP
        IPP_model, _ = get_aug_sgp_sol(self.num_waypoints, 
                                       self.X_train,
                                       likelihood_variance,
                                       kernel,
                                       transformer,
                                       Xu_init=Xu_init)

        # Generate new paths from optimized waypoints
        self.waypoints = IPP_model.inducing_variable.Z.numpy()
        path_idx, _  = run_tsp(self.waypoints, num_vehicles=self.num_robots)
        self.waypoints =  [self.waypoints[path] for path in path_idx]

        # Print path lengths
        rospy.loginfo('OfflineIPP: Initial IPP solution found')
        path_lengths = transformer.distance(np.concatenate(self.waypoints, axis=0)).numpy()
        msg = 'Initial path lengths: '
        for path_length in path_lengths:
            msg += f'{path_length:.2f} '
        rospy.loginfo(msg)

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
            rospy.wait_for_service(service)
            try:
                offline_ipp_service = rospy.ServiceProxy(service, OfflineIPP)
                service_data = OfflineIPPData()
                for waypoint in self.waypoints[robot_idx]:
                    service_data.waypoints.append(Point(x=waypoint[0],
                                                        y=waypoint[1]))
                for point in self.data[robot_idx]:
                    service_data.x_train.append(Point(x=point[0],
                                                      y=point[1]))
                success = offline_ipp_service(service_data)
            except rospy.ServiceException as e:
                print(f'Service call failed: {e}')


if __name__ == '__main__':

    # Define the extent of the environment
    xx = np.linspace(-2, 2, 25)
    yy = np.linspace(-2, 2, 25)
    X_train = np.array(np.meshgrid(xx, yy)).T.reshape(-1, 2)

    # Get model parameters
    if rospy.has_param('/num_waypoints'):
        num_waypoints=rospy.get_param('/num_waypoints')
    else:
        num_waypoints=10

    if rospy.has_param('/num_robots'):
        num_robots=rospy.get_param('/num_robots')
    else:
        num_robots=1

    # Start the offline IPP mission
    offlineIPP(X_train, 
               num_waypoints=num_waypoints, 
               num_robots=num_robots)
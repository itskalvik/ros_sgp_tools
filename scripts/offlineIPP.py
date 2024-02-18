#! /usr/bin/env python3

import gpflow
import numpy as np
from sgp_ipp.utils.tsp import run_tsp
from sgp_ipp.utils.sensor_placement import *
from sgp_ipp.models.transformations import IPPTransformer

from visualization_msgs.msg import Marker
from ros_sgp_ipp.msg import WaypointsList
from ros_sgp_ipp.srv import Waypoints
from geometry_msgs.msg import Point
import rospy

import matplotlib.pyplot as plt


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

        self.marker_pub = rospy.Publisher("/visualization_marker", 
                                          Marker, 
                                          queue_size = 2)
        self.colors = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

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
        Xu_init = get_inducing_pts(X_train, self.num_waypoints*self.num_robots)
        path_idx, _ = run_tsp(Xu_init, num_vehicles=self.num_robots)
        Xu_init =  [Xu_init[path] for path in path_idx]
        Xu_init = np.concatenate(Xu_init, axis=0)

        # Optimize the SGP
        IPP_model, _ = get_aug_sgp_sol(self.num_waypoints, 
                                       X_train,
                                       likelihood_variance,
                                       kernel,
                                       transformer,
                                       Xu_init=Xu_init)

        # Generate new paths from optimized waypoints
        self.waypoints = IPP_model.inducing_variable.Z.numpy()
        path_idx, _  = run_tsp(self.waypoints, num_vehicles=self.num_robots)
        self.waypoints =  [self.waypoints[path] for path in path_idx]
        self.waypoints = np.concatenate(self.waypoints, axis=0)
        
        rospy.loginfo('OfflineIPP: Initial IPP solution found')
        path_lengths = transformer.distance(self.waypoints).numpy()

        msg = 'Initial path lengths: '
        for path_length in path_lengths:
            msg += f'{path_length:.2f} '
        rospy.loginfo(msg)

        self.plot_paths(self.waypoints.reshape(self.num_robots, -1, 2),
                        X_train,
                        "Solution")

        '''
        paths = self.waypoints.reshape(self.num_robots, -1, 2)
        for i, path in enumerate(paths):
            for waypoint in path:
                self.publish_marker(waypoint, i)
        '''
        
    def plot_paths(self, paths, candidates=None, title=None):
        plt.figure()
        for path in paths:
            plt.plot(path[:, 0], path[:, 1], 
                        c='r', label='Path', zorder=0, marker='o')
            if candidates is not None:
                plt.scatter(candidates[:, 0], candidates[:, 1], 
                            c='k', s=1, label='Candidates', zorder=1)
        if title is not None:
            plt.title(title)
        plt.savefig('/home/kalvik/test.png')

    def publish_marker(self, position, color_idx=0):
        marker = Marker()
        marker.header.frame_id = "/base_footprint"
        marker.header.stamp = rospy.Time.now()

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        marker.type = 2
        marker.id = 0
        marker.action = 0

        # Set the scale of the marker
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        # Set the color
        marker.color.r = self.colors[color_idx][0]
        marker.color.g = self.colors[color_idx][1]
        marker.color.b = self.colors[color_idx][2]
        marker.color.a = 1.0

        # Set the pose of the marker
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = 0.0

        marker.scale.x = 1
        marker.scale.y = 1
        marker.scale.z = 1

        marker.lifetime = rospy.Duration(200)

        self.marker_pub.publish(marker)

    def sync_waypoints(self):
        # Send the new waypoints to the trajectory planner and 
        # update the current waypoint from the service
        sol_waypoints = self.waypoints.reshape(self.num_robots, -1, 2)
        for robot_idx in range(self.num_robots):
            service = f'tb3_{robot_idx}/online_waypoints'
            rospy.wait_for_service(service)
            try:
                waypoint_service = rospy.ServiceProxy(service, Waypoints)
                waypoints_list = WaypointsList()
                for waypoint in sol_waypoints[robot_idx]:
                    waypoints_list.waypoints.append(Point(x=waypoint[0],
                                                          y=waypoint[1]))
                success = waypoint_service(waypoints_list)
            except rospy.ServiceException as e:
                print(f'Service call failed: {e}')


if __name__ == '__main__':

    # Define the extent of the environment
    xx = np.linspace(-2, 2, 25)
    yy = np.linspace(-2, 2, 25)
    X_train = np.array(np.meshgrid(xx, yy)).T.reshape(-1, 2)

    # Start the offline IPP mission
    offlineIPP(X_train)
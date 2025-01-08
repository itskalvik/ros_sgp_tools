#! /usr/bin/env python3

from ros_sgp_ipp.srv import Waypoints, WaypointsResponse
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Int32
import tf_transformations
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.timer import Timer

from controllers import *

class TrajectoryPlanner(Node):
    '''
    Initialize the trajectory planner with the maximum linear and angular velocity,
    distance and angle tolerance to the goal.

    Args:
        distance_tolerance: Distance tolerance to the goal
        angle_tolerance: Angle tolerance to the goal
        update_rate: Frequency of trajectory commands
    '''
    def __init__(self, 
                 distance_tolerance=0.1,
                 angle_tolerance=0.1,
                 update_rate=30):
        super().__init__('trajectory_planner')

        self.distance_tolerance = distance_tolerance
        self.angle_tolerance = angle_tolerance

        # Initialize the publisher and subscriber
        self.control_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.current_waypoint_publisher = self.create_publisher(Int32, 'current_waypoint', 10)
        self.pose_subscriber = self.create_subscription(PoseStamped, '/vrpn_client_node/pose', 
                                                        self.position_callback, 10)

        # Setup the service to receive the waypoints
        self.waypoint_service = self.create_service(Waypoints, 'waypoints', 
                                                   self.waypoint_service_callback)

        # Setup the timer to update the parameters and waypoints
        self.timer = self.create_timer(1.0 / update_rate, self.visit_waypoints)
        self.current_waypoint_timer = self.create_timer(1.0 / update_rate, 
                                                        self.publish_current_waypoint)

        # Initialize the position and goal
        self.position = np.array([0, 0, 0])
        self.control_cmd = Twist()
        self.waypoints = []
        self.current_waypoint = -1

        # Create the trajectory controller
        self.get_control_cmd = create_hybrid_unicycle_pose_controller(linear_velocity_gain=2.0,
                                                                      angular_velocity_gain=0.6,
                                                                      position_error=0.1, 
                                                                      position_epsilon=0.3, 
                                                                      rotation_error=0.1)

        self.get_logger().info('Trajectory Planner: initialized, waiting for waypoints')

    '''
    Move the robot to a goal position using the Single-Integrator Dynamics 
    and Unicycle Control Commands mapping. 

    Args:
        goal: Goal position [x, y]      
    '''
    def move2goal(self, goal):
        goal.append(0.0)
        rotation_complete = False
        while np.linalg.norm(self.position[0:2] - goal[0:2]) > self.distance_tolerance \
            and rclpy.ok():

            # Compute best approach angle to goal
            goal[2] = np.arctan2(goal[1] - self.position[1], goal[0] - self.position[0])

            # Rotate the robot towards the goal first
            if not rotation_complete:
                error_angle = goal[2] - self.position[2]
                # Wrap the angle to [-π, π]
                error_angle = np.arctan2(np.sin(error_angle), np.cos(error_angle))

                if np.abs(error_angle) > self.angle_tolerance:
                    control_cmd = [[0.], [error_angle * 0.6]]
                else:
                    rotation_complete = True
                    continue
            # Move the robot to the goal
            else:      
                control_cmd = self.get_control_cmd(np.array(self.position).reshape(-1, 1), 
                                                   np.array(goal).reshape(-1, 1))

            # Publish the control command
            self.control_cmd.linear.x = control_cmd[0][0]
            self.control_cmd.angular.z = control_cmd[1][0]
            self.control_publisher.publish(self.control_cmd)

            rclpy.spin_once(self)

        # Stop the robot
        self.control_cmd.linear.x = 0
        self.control_cmd.angular.z = 0
        self.control_publisher.publish(self.control_cmd)

    def publish_current_waypoint(self):
        self.current_waypoint_publisher.publish(Int32(data=self.current_waypoint))

    '''
    Service callback to receive the waypoints and return the current waypoint

    Args:
        req: Request containing the waypoints
    Returns:
        WaypointsResponse: Response containing the current waypoint
    '''
    def waypoint_service_callback(self, req, response):
        waypoints = req.waypoints.waypoints
        self.waypoints = []
        for wp in waypoints:
            self.waypoints.append([wp.x, wp.y])
        self.get_logger().info('Trajectory Planner: Waypoints received')
        response.success = True
        return response

    '''
    Visit the waypoints in the waypoints list and empty the list after visiting all of them 
    '''
    def visit_waypoints(self):
        # Visit the waypoints if there are any
        if len(self.waypoints) == 0:
            return
        
        self.get_logger().info('Trajectory Planner: Visiting waypoints')
        for i in range(len(self.waypoints)):
            self.current_waypoint = i + 1
            self.move2goal(self.waypoints[i])
            self.get_logger().info(f'Trajectory Planner: Reached Waypoint {i+1}')

        # Shutdown after visiting the waypoints 
        self.get_logger().info('Trajectory Planner: All waypoints visited')
        rclpy.shutdown()

    def position_callback(self, msg):
        q = [msg.pose.orientation.x,
             msg.pose.orientation.y,
             msg.pose.orientation.z,
             msg.pose.orientation.w]
        (roll, pitch, yaw) = tf_transformations.euler_from_quaternion(q)
        self.position = np.array([msg.pose.position.x, 
                                  msg.pose.position.y, 
                                  yaw])


def main(args=None):
    rclpy.init(args=args)
    tp = TrajectoryPlanner()
    rclpy.spin(tp)


if __name__ == '__main__':
    main()

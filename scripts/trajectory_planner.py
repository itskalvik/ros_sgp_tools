#! /usr/bin/env python3

from ros_sgp_ipp.srv import Waypoints, WaypointsResponse
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
import tf.transformations
import numpy as np
import rospy


class TrajectoryPlanner:
    def __init__(self, 
                 max_linear_velocity=0.5,
                 max_angular_velocity=0.5,
                 distance_tolerance=0.05,
                 angle_tolerance=0.01):
        self.distance_tolerance = distance_tolerance
        self.angle_tolerance = angle_tolerance
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity

        # Initialize the node, publisher, and subscriber
        rospy.init_node('trajectory_planner', anonymous=True)
        self.control_publisher = rospy.Publisher('/cmd_vel', Twist, 
                                                 queue_size=10)
        self.pose_subscriber = rospy.Subscriber('/gazebo/model_states', 
                                                ModelStates, 
                                                self.position_callback)
        
        # Setup the service to send the waypoints
        self.waypoint_service = rospy.Service('waypoints', 
                                              Waypoints, 
                                              self.waypoint_service_callback)

        # Setup the timer to update the parameters and waypoints
        self.timer = rospy.Timer(rospy.Duration(5), self.visit_waypoints)

        # Initialize the position and goal
        self.position = np.array([0, 0, 0])
        self.control_cmd = Twist()
        self.waypoints = []
        self.current_waypoint = 0
        
        rospy.spin()

    def waypoint_service_callback(self, req):
        waypoints = req.waypoints
        self.waypoints = []
        for waypoint in waypoints:
            self.waypoints.append([waypoint.x, waypoint.y])
        return WaypointsResponse(self.current_waypoint)

    def visit_waypoints(self):
        for i in range(len(self.waypoints)):
            print(f'Visiting waypoint {i}')
            print(f'Current position: {self.position}')
            print(f'Goal position: {self.waypoints[i]}')
        
            self.current_waypoint = i
            self.move2goal(self.waypoints[i])

        # Empty the waypoints after visiting all of them
        self.waypoints = []

    '''
    Move the robot to the goal position
    '''
    def move2goal(self, goal):
        while np.linalg.norm(self.position[0:2] - goal[0:2]) > self.distance_tolerance:
            # Move the robot to the goal
            control_cmd = self.get_control_cmd(self.position, goal)
            self.control_cmd.linear.x = control_cmd[0]
            self.control_cmd.angular.z = control_cmd[1]
            self.control_publisher.publish(self.control_cmd)

            self.rate.sleep()

        # Stop the robot
        self.control_cmd.linear.x = 0
        self.control_cmd.angular.z = 0
        self.control_publisher.publish(self.control_cmd)

    '''
    Mapping Single-Integrator Dynamics to Unicycle Control Commands
    https://liwanggt.github.io/files/Robotarium_CSM_Impact.pdf (Page 14)

    Args:
        state: [x, y, θ]
        goal: [x, y, θ]
        l: Lookahead distance
    '''
    def get_control_cmd(self, state, goal, l=-0.05):
        if len(goal) == 2:
            goal = np.array([goal[0], goal[1], state[2]])

        x_diff = goal - state
        s_diff = x_diff[0:2] + l * x_diff[2] * np.array([-np.sin(state[2]), 
                                                          np.cos(state[2])])
        R_inv = np.array([[       np.cos(state[2]),       np.sin(state[2])],
                          [-(1/l)*np.sin(state[2]), (1/l)*np.cos(state[2])]])
        sol = np.dot(R_inv, s_diff)

        # Limiting the linear and angular velocity
        sol[0] = np.clip(sol[0], 
                         -self.max_linear_velocity, 
                         self.max_linear_velocity)
        sol[1] = np.clip(sol[1], 
                         -self.max_angular_velocity, 
                         self.max_angular_velocity)
        
        return sol

    def position_callback(self, msg):
        q = [msg.pose[1].orientation.x,
             msg.pose[1].orientation.y,
             msg.pose[1].orientation.z,
             msg.pose[1].orientation.w]
        theta = tf.transformations.euler_from_quaternion(q)
        self.position = np.array([msg.pose[1].position.x, 
                                  msg.pose[1].position.y, 
                                  theta[2]])


if __name__ == '__main__':
    tp = TrajectoryPlanner()

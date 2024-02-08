#! /usr/bin/env python3

from ros_sgp_ipp.srv import Waypoints, WaypointsResponse
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist, PoseStamped
from math import remainder
import tf.transformations
import numpy as np
import rospy


'''
Trajectory planner class to move a diff-drive/skid-drive robot to waypoints using the
Single-Integrator Dynamics and Unicycle Control Commands mapping.
'''
class TrajectoryPlanner:
    '''
    Initialize the trajectory planner with the maximum linear and angular velocity,
    distance and angle tolerance to the goal.

    Args:
        max_linear_velocity: Maximum linear velocity of the robot
        max_angular_velocity: Maximum angular velocity of the robot
        distance_tolerance: Distance tolerance to the goal
        angle_tolerance: Angle tolerance to the goal
    '''
    def __init__(self, 
                 max_linear_velocity=0.6,
                 max_angular_velocity=0.5,
                 distance_tolerance=0.05,
                 angle_tolerance=0.05):
        self.distance_tolerance = distance_tolerance
        self.angle_tolerance = angle_tolerance
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity

        # Initialize the node, publisher, and subscriber
        rospy.init_node('trajectory_planner', anonymous=True)
        self.control_publisher = rospy.Publisher('/cmd_vel', Twist, 
                                                 queue_size=10)
        self.current_waypoint_publisher = rospy.Publisher('/current_waypoint',
                                                          Int32,
                                                          queue_size=10)
        self.pose_subscriber = rospy.Subscriber('/vrpn_client_node/Robot1/pose', 
                                                PoseStamped, 
                                                self.position_callback)
        
        # Setup the service to send the waypoints
        self.waypoint_service = rospy.Service('waypoints', 
                                              Waypoints, 
                                              self.waypoint_service_callback)

        # Setup the timer to update the parameters and waypoints
        self.timer = rospy.Timer(rospy.Duration(5), self.visit_waypoints)
        self.current_waypoint_timer = rospy.Timer(rospy.Duration(1), 
                                                  self.publish_current_waypoint)

        # Initialize the position and goal
        self.position = np.array([0, 0, 0])
        self.control_cmd = Twist()
        self.waypoints = []
        self.current_waypoint = -1
        
        self.rate = rospy.Rate(10)

        rospy.loginfo('Trajectory planner initialized, waiting for waypoints')
        rospy.spin()

    def publish_current_waypoint(self, timer):
        self.current_waypoint_publisher.publish(self.current_waypoint)

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
        rospy.loginfo('Waypoints received')
        return WaypointsResponse(True)

    '''
    Visit the waypoints in the waypoints list and empty the list after visiting all of them 
    '''
    def visit_waypoints(self, timer):
        # Visit the waypoints if there are any
        if len(self.waypoints) == 0:
            return
        
        rospy.loginfo('Visiting waypoints')
        for i in range(len(self.waypoints)):
            self.current_waypoint = i+1
            self.move2goal(self.waypoints[i])
            rospy.loginfo(f'Reached Waypoint {i+1}')

        # Empty the waypoints after visiting all of them
        self.waypoints = []
        self.current_waypoint = -1
        rospy.loginfo('All waypoints visited, waiting for new waypoints')

    '''
    Move the robot to a goal position using the Single-Integrator Dynamics 
    and Unicycle Control Commands mapping. 

    Args:
        goal: Goal position [x, y]      
    '''
    def move2goal(self, goal):
        rotation_complete = False
        while np.linalg.norm(self.position[0:2] - goal[0:2]) > self.distance_tolerance:
            # Rotate the robot towards the goal first
            if not rotation_complete:
                # Calculate the angle to the goal
                error_angle = np.arctan2(goal[1] - self.position[1], 
                                         goal[0] - self.position[0])
                error_angle -= self.position[2]
                # Normalize the angle to [-π, π]
                error_angle = remainder(error_angle, 2*np.pi)
                if np.abs(error_angle) > self.angle_tolerance:
                    control_cmd = self.get_control_cmd(self.position,
                                                       [self.position[0], 
                                                        self.position[1], 
                                                        self.position[2] + error_angle])
                else:
                    rotation_complete = True
                    continue
            # Move the robot to the goal
            else:      
                control_cmd = self.get_control_cmd(self.position, goal)

            # Publish the control command
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
        state: [x, y, θ] Current state of the robot
        goal: [x, y, θ] Goal position
        l: Lookahead distance
    Returns:
        sol: [v, ω] Control commands [linear velocity, angular velocity]
    '''
    def get_control_cmd(self, state, goal, l=0.05):
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
        (roll,pitch,yaw) = tf.transformations.euler_from_quaternion(q)
        self.position = np.array([msg.pose.position.x, 
                                  msg.pose.position.y, 
                                  yaw])


if __name__ == '__main__':
    tp = TrajectoryPlanner()

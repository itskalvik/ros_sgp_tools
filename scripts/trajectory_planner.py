#! /usr/bin/env python3

from ros_sgp_ipp.srv import Waypoints, WaypointsResponse
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Int32
import tf.transformations
import numpy as np
import rospy

from controllers import *


'''
Trajectory planner class to move a diff-drive/skid-drive robot to waypoints using the
Single-Integrator Dynamics and Unicycle Control Commands mapping.
'''
class TrajectoryPlanner:
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
        self.distance_tolerance = distance_tolerance
        self.angle_tolerance = angle_tolerance

        # Initialize the node, publisher, and subscriber
        rospy.init_node('trajectory_planner', anonymous=True)

        self.ns = rospy.get_namespace()
        self.control_publisher = rospy.Publisher('cmd_vel', Twist, 
                                                 queue_size=10)
        self.current_waypoint_publisher = rospy.Publisher('current_waypoint',
                                                          Int32,
                                                          queue_size=10)
        self.pose_subscriber = rospy.Subscriber('/vrpn_client_node'+self.ns+'pose',
                                                PoseStamped, 
                                                self.position_callback)
        
        # Setup the service to receive the waypoints
        self.waypoint_service = rospy.Service('waypoints', 
                                              Waypoints, 
                                              self.waypoint_service_callback)

        # Setup the timer to update the parameters and waypoints
        self.timer = rospy.Timer(rospy.Duration(update_rate), self.visit_waypoints)
        self.current_waypoint_timer = rospy.Timer(rospy.Duration(update_rate), 
                                                  self.publish_current_waypoint)

        # Initialize the position and goal
        self.position = np.array([0, 0, 0])
        self.control_cmd = Twist()
        self.waypoints = []
        self.current_waypoint = -1
        
        # Create the trajectory controller
        self.get_control_cmd = create_hybrid_unicycle_pose_controller()
        
        rospy.loginfo(self.ns+'Trajectory Planner: initialized, waiting for waypoints')

        # Keep alive until waypoints are received and then send vel commands at update rate
        self.rate = rospy.Rate(update_rate)
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
        rospy.loginfo(self.ns+'Trajectory Planner: Waypoints received')
        return WaypointsResponse(True)

    '''
    Visit the waypoints in the waypoints list and empty the list after visiting all of them 
    '''
    def visit_waypoints(self, timer):
        # Visit the waypoints if there are any
        if len(self.waypoints) == 0:
            return
        
        rospy.loginfo(self.ns+'Trajectory Planner: Visiting waypoints')
        for i in range(len(self.waypoints)):
            self.current_waypoint = i+1
            self.move2goal(self.waypoints[i])
            rospy.loginfo(self.ns+f'Trajectory Planner: Reached Waypoint {i+1}')

        # Shutdown after visiting the waypoints 
        rospy.loginfo(self.ns+'Trajectory Planner: All waypoints visited')
        rospy.signal_shutdown(self.ns+'Trajectory Planner: Received shutdown signal')

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
            and not rospy.is_shutdown():
            # Compute best approach angle to goal
            goal[2] = np.arctan2(goal[1] - self.position[1], goal[0] - self.position[0])

            # Rotate the robot towards the goal first
            if not rotation_complete:
                error_angle = goal[2] - self.position[2]
                # Wrap the angle to [-π, π]
                error_angle = np.arctan2(np.sin(error_angle),np.cos(error_angle))

                if np.abs(error_angle) > self.angle_tolerance:
                    control_cmd = [[0.], [error_angle*0.5]]
                else:
                    rotation_complete = True
                    rospy.loginfo(self.ns+'Rotation complete')
                    continue
            # Move the robot to the goal
            else:      
                control_cmd = self.get_control_cmd(np.array(self.position).reshape(-1, 1), 
                                                   np.array(goal).reshape(-1, 1))

            # Publish the control command
            self.control_cmd.linear.x = control_cmd[0][0]
            self.control_cmd.angular.z = control_cmd[1][0]
            self.control_publisher.publish(self.control_cmd)

            self.rate.sleep()

        # Stop the robot
        self.control_cmd.linear.x = 0
        self.control_cmd.angular.z = 0
        self.control_publisher.publish(self.control_cmd)

    def position_callback(self, msg):
        q = [msg.pose.orientation.x,
             msg.pose.orientation.y,
             msg.pose.orientation.z,
             msg.pose.orientation.w]
        (roll,pitch,yaw) = tf.transformations.euler_from_quaternion(q)
        self.position = np.array([msg.pose.position.x, 
                                  msg.pose.position.y, 
                                  yaw])


if __name__ == '__main__':
    tp = TrajectoryPlanner()

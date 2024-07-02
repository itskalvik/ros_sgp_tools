#! /usr/bin/env python3

from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode, CommandBool
from geographic_msgs.msg import GeoPoseStamped
from sensor_msgs.msg import NavSatFix

import rclpy
from rclpy.node import Node

import numpy as np

class PathPlanner(Node):

    def __init__(self):
        super().__init__('path_planner')

        # Create QoS profiles

        # STATE_QOS used for state topics, like ~/state, ~/mission/waypoints etc.
        STATE_QOS = rclpy.qos.QoSProfile(
            depth=10, 
            durability=rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        # SENSOR_QOS used for most of sensor streams
        SENSOR_QOS = rclpy.qos.qos_profile_sensor_data

        # Create subscribers
        self.vehicle_state_subscriber = self.create_subscription(
            State, '/mavros/state', self.vehicle_state_callback, STATE_QOS)
        self.vehicle_pose_subscriber = self.create_subscription(
            NavSatFix, '/mavros/global_position/global', 
            self.vehicle_position_callback, SENSOR_QOS)
        
        # Create publishers
        self.setpoint_position_publisher = self.create_publisher(
            GeoPoseStamped, '/mavros/setpoint_position/global', SENSOR_QOS)

        # Create service clients
        self.set_mode_client = self.create_client(
            SetMode, '/mavros/set_mode')
        while not self.set_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Set mode service not available, waiting again...')
        self.get_logger().info('Set mode service available')

        self.arm_client = self.create_client(
            CommandBool, '/mavros/cmd/arming')
        while not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Arming service not available, waiting again...')
        self.get_logger().info('Arming service available')

        # Initialize variables
        self.vehicle_state = State()
        self.vehicle_position = np.array([0., 0.])
        self.arm_request = CommandBool.Request()
        self.set_mode_request = SetMode.Request()
        self.setpoint_position = GeoPoseStamped()
        self.waypoint_id = 0
        self.last_request = self.get_clock().now().to_msg().sec

        # Create a timer to publish control commands
        self.timer = self.create_timer(0.1, self.timer_callback)

    def at_waypoint(self, waypoint, tolerance=0.000005):
        """Check if the vehicle is at the waypoint."""
        dist = np.linalg.norm(self.vehicle_position - np.array(waypoint))
        if dist < tolerance:
            return True
        else:
            return False

    def vehicle_state_callback(self, vehicle_state):
        """Callback function for vehicle_state topic subscriber."""
        self.vehicle_state = vehicle_state

    def vehicle_position_callback(self, position):
        """Callback function for vehicle position topic subscriber."""
        self.vehicle_position = np.array([position.latitude, 
                                          position.longitude])

    def arm(self):
        """Send an arm command to the vehicle."""
        self.arm_request.value = True
        self.arm_client.call_async(self.arm_request)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.arm_request.value = False
        self.arm_client.call_async(self.arm_request)
        self.get_logger().info('Disarm command sent')

    def engage_guided_mode(self):
        """Send a command to the vehicle to enter GUIDED mode."""
        if self.set_mode_client.wait_for_service(timeout_sec=0.5):
            self.set_mode_request.custom_mode = "GUIDED"
            self.set_mode_client.call_async(self.set_mode_request)
            self.get_logger().info('Engage GUIDED mode command sent')
            
    def set_waypoint(self, latitude, longitude):
        """Set the next waypoint of the vehicle."""
        self.setpoint_position.header.stamp = self.get_clock().now().to_msg()
        self.setpoint_position.pose.position.latitude = latitude
        self.setpoint_position.pose.position.longitude = longitude
        self.setpoint_position_publisher.publish(self.setpoint_position)

    def timer_callback(self) -> None:
        """Callback function for the timer."""

        # Engage GUIDED mode
        if self.vehicle_state.mode != "GUIDED" and \
            (self.get_clock().now().to_msg().sec - self.last_request) > 5.0:
            self.engage_guided_mode()
            self.last_request = self.get_clock().now().to_msg().sec

        # Arm the vehicle
        if not self.vehicle_state.armed and \
            self.vehicle_state.mode == "GUIDED" and \
            (self.get_clock().now().to_msg().sec - self.last_request) > 5.0:
            self.arm()
            self.last_request = self.get_clock().now().to_msg().sec

        # Set the position setpoint
        if self.waypoint_id == 0:
            if self.at_waypoint([35.30684387683425, -80.7360063599907]):
                self.waypoint_id = 1
                self.get_logger().info("Waypoint 1 reached")
            self.set_waypoint(35.30684387683425, -80.7360063599907)
        if self.waypoint_id == 1:
            if self.at_waypoint([35.30684275566786, -80.73612370299257]):
                self.waypoint_id = 2
                self.get_logger().info("Waypoint 2 reached")
            self.set_waypoint(35.30684275566786, -80.73612370299257)
        if self.waypoint_id == 2:
            if self.at_waypoint([35.30679876645213, -80.73623439122146]):
                self.waypoint_id = 3
                self.get_logger().info("Waypoint 3 reached")
            self.set_waypoint(35.30679876645213, -80.73623439122146)
        if self.waypoint_id == 3:
            if self.at_waypoint([35.30674267884529, -80.73600329951549]):
                self.waypoint_id = 4
                self.get_logger().info("Waypoint 4 reached")
                self.disarm()
                exit()
            self.set_waypoint(35.30674267884529, -80.73600329951549)


def main(args=None):
    rclpy.init(args=args)
    path_planner = PathPlanner()
    rclpy.spin(path_planner)
    rclpy.shutdown()
    

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
